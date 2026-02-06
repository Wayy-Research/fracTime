"""
CloudConditionedLSTM forecasting model.

Uses FracTime's simulation cloud as a conditioning input to an LSTM trained
on historical data. The cloud provides a probabilistic prior based on fractal
properties, while the LSTM learns to refine forecasts from historical patterns.

Key innovation: FiLM (Feature-wise Linear Modulation) layers condition the
LSTM hidden states using encoded cloud statistics, allowing the fractal
simulation to modulate learned patterns without replacing them.
"""

from __future__ import annotations

import numpy as np
import warnings
from typing import Dict, Optional, Tuple

from .._numba import (
    compute_hurst_rs,
    compute_rolling_volatility,
    compute_box_dimension,
)
from ..simulator import Simulator
from ..result import ForecastResult

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None


# Try to import scipy for skew/kurtosis
try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_stats = None


def _compute_skew(x: np.ndarray) -> float:
    """Compute skewness, with fallback if scipy not available."""
    try:
        if SCIPY_AVAILABLE:
            result = scipy_stats.skew(x, nan_policy="omit")
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return float(result)
        # Manual computation
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std < 1e-8:
            return 0.0
        result = np.nanmean(((x - mean) / std) ** 3)
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return float(result)
    except Exception:
        return 0.0


def _compute_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis, with fallback if scipy not available."""
    try:
        if SCIPY_AVAILABLE:
            result = scipy_stats.kurtosis(x, nan_policy="omit")
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return float(result)
        # Manual computation (excess kurtosis)
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std < 1e-8:
            return 0.0
        result = np.nanmean(((x - mean) / std) ** 4) - 3.0
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return float(result)
    except Exception:
        return 0.0


class CloudEncoder(nn.Module if PYTORCH_AVAILABLE else object):
    """
    Encode simulation cloud into conditioning features.

    For each time step in the cloud, computes:
    - Moments: mean, std, skew, kurtosis
    - Quantiles: 5%, 25%, 75%, 95%

    Output shape: (n_steps, 8) per cloud, then projected to cloud_encoding_dim.
    """

    def __init__(self, n_steps: int, cloud_encoding_dim: int = 32):
        if PYTORCH_AVAILABLE:
            super(CloudEncoder, self).__init__()
            self.n_steps = n_steps
            self.cloud_encoding_dim = cloud_encoding_dim

            # 8 statistics per step: mean, std, skew, kurtosis, q5, q25, q75, q95
            self.stats_per_step = 8

            # Project statistics to encoding dimension
            self.fc = nn.Linear(self.stats_per_step, cloud_encoding_dim)
            self.norm = nn.LayerNorm(cloud_encoding_dim)

    def encode_cloud_numpy(self, cloud: np.ndarray) -> np.ndarray:
        """
        Encode cloud to statistics using numpy (for preprocessing).

        Args:
            cloud: Shape (n_paths, n_steps) - price paths

        Returns:
            Shape (n_steps, 8) - statistics per step
        """
        n_paths, n_steps = cloud.shape
        stats = np.zeros((n_steps, 8))

        for t in range(n_steps):
            values = cloud[:, t]
            stats[t, 0] = np.mean(values)
            stats[t, 1] = np.std(values)
            stats[t, 2] = _compute_skew(values)
            stats[t, 3] = _compute_kurtosis(values)
            stats[t, 4] = np.percentile(values, 5)
            stats[t, 5] = np.percentile(values, 25)
            stats[t, 6] = np.percentile(values, 75)
            stats[t, 7] = np.percentile(values, 95)

        return stats

    def forward(self, cloud_stats: torch.Tensor) -> torch.Tensor:
        """
        Project cloud statistics to encoding dimension.

        Args:
            cloud_stats: Shape (batch, n_steps, 8)

        Returns:
            Shape (batch, n_steps, cloud_encoding_dim)
        """
        # Project each step's statistics
        x = self.fc(cloud_stats)  # (batch, n_steps, cloud_encoding_dim)
        x = self.norm(x)
        return torch.relu(x)


class FiLMLayer(nn.Module if PYTORCH_AVAILABLE else object):
    """
    Feature-wise Linear Modulation layer.

    Conditions LSTM hidden states using cloud encoding:
        h' = gamma(cloud_encoding) * h + beta(cloud_encoding)

    This allows the cloud to modulate learned features without replacing them.
    """

    def __init__(self, hidden_size: int, cloud_encoding_dim: int):
        if PYTORCH_AVAILABLE:
            super(FiLMLayer, self).__init__()
            self.hidden_size = hidden_size

            # Generate gamma and beta from cloud encoding
            # gamma centered at 1, beta centered at 0
            self.gamma_fc = nn.Linear(cloud_encoding_dim, hidden_size)
            self.beta_fc = nn.Linear(cloud_encoding_dim, hidden_size)

            # Initialize weights small so gamma = 1 + w*x starts near 1
            # and beta = w*x starts near 0
            nn.init.normal_(self.gamma_fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.gamma_fc.bias)
            nn.init.zeros_(self.beta_fc.weight)
            nn.init.zeros_(self.beta_fc.bias)

    def forward(self, h: torch.Tensor, cloud_encoding: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            h: Hidden state, shape (batch, seq_len, hidden_size)
            cloud_encoding: Shape (batch, seq_len, cloud_encoding_dim)

        Returns:
            Modulated hidden state, shape (batch, seq_len, hidden_size)
        """
        # Generate modulation parameters
        gamma = 1.0 + self.gamma_fc(cloud_encoding)  # Center at 1
        beta = self.beta_fc(cloud_encoding)

        # Apply modulation
        return gamma * h + beta


class CloudConditionedLSTMNetwork(nn.Module if PYTORCH_AVAILABLE else object):
    """
    LSTM with FiLM conditioning from cloud encoder.

    Architecture:
        - Input: [log_returns, volatility, H, D] - 4 features
        - Cloud Encoder: (n_paths, n_steps) -> (n_steps, cloud_encoding_dim)
        - LSTM Layer 1 + FiLM: 64 units
        - LSTM Layer 2 + FiLM: 32 units
        - Output: Dense (1 unit) - predict next day return
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        cloud_encoding_dim: int = 32,
        n_forecast_steps: int = 10,
        dropout: float = 0.2,
    ):
        if PYTORCH_AVAILABLE:
            super(CloudConditionedLSTMNetwork, self).__init__()
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.cloud_encoding_dim = cloud_encoding_dim

            # Cloud encoder
            self.cloud_encoder = CloudEncoder(
                n_steps=n_forecast_steps,
                cloud_encoding_dim=cloud_encoding_dim,
            )

            # Layer 1: LSTM with 64 units
            self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size_1,
                num_layers=1,
                batch_first=True,
            )

            # FiLM layer 1
            self.film1 = FiLMLayer(hidden_size_1, cloud_encoding_dim)

            # Dropout after layer 1
            self.dropout1 = nn.Dropout(dropout)

            # Layer 2: LSTM with 32 units
            self.lstm2 = nn.LSTM(
                input_size=hidden_size_1,
                hidden_size=hidden_size_2,
                num_layers=1,
                batch_first=True,
            )

            # FiLM layer 2
            self.film2 = FiLMLayer(hidden_size_2, cloud_encoding_dim)

            # Dropout after layer 2
            self.dropout2 = nn.Dropout(dropout)

            # Output layer
            self.fc = nn.Linear(hidden_size_2, 1)

    def forward(
        self,
        x: torch.Tensor,
        cloud_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with cloud conditioning.

        Args:
            x: Input tensor, shape (batch, seq_len, 4)
            cloud_stats: Cloud statistics, shape (batch, n_steps, 8)

        Returns:
            Output tensor, shape (batch, 1)
        """
        seq_len = x.shape[1]

        # Encode cloud
        cloud_encoding = self.cloud_encoder(cloud_stats)  # (batch, n_steps, enc_dim)

        # Use first step encoding for all lookback steps
        # (cloud is for future steps, we use aggregate for conditioning)
        cloud_mean = cloud_encoding.mean(dim=1, keepdim=True)  # (batch, 1, enc_dim)
        cloud_broadcast = cloud_mean.expand(
            -1, seq_len, -1
        )  # (batch, seq_len, enc_dim)

        # LSTM Layer 1
        lstm1_out, _ = self.lstm1(x)  # (batch, seq_len, hidden_size_1)

        # FiLM conditioning
        lstm1_out = self.film1(lstm1_out, cloud_broadcast)
        lstm1_out = self.dropout1(lstm1_out)

        # LSTM Layer 2
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, seq_len, hidden_size_2)

        # FiLM conditioning
        lstm2_out = self.film2(lstm2_out, cloud_broadcast)

        # Take last time step
        last_output = lstm2_out[:, -1, :]  # (batch, hidden_size_2)
        last_output = self.dropout2(last_output)

        # Output layer
        out = self.fc(last_output)  # (batch, 1)

        return out


class CloudConditionedLSTMForecaster:
    """
    LSTM forecaster conditioned on FracTime simulation cloud.

    The key innovation is using the simulation cloud (generated from fractal
    properties) to condition the LSTM via FiLM layers. This combines:
    - LSTM's ability to learn temporal patterns from historical data
    - Simulator's probabilistic prior based on fractal properties

    Args:
        lookback: Sequence length for LSTM (default 30)
        n_cloud_paths: Number of paths in simulation cloud (default 500)
        hidden_size_1: First LSTM layer size (default 64)
        hidden_size_2: Second LSTM layer size (default 32)
        cloud_encoding_dim: Cloud encoder output dimension (default 32)
        cloud_weight: Weight for cloud in final output (default 0.3)
        dropout: Dropout rate (default 0.2)
        epochs: Maximum training epochs (default 100)
        batch_size: Training batch size (default 32)
        validation_split: Fraction of data for validation (default 0.2)
        patience: Early stopping patience (default 10)
        learning_rate: Adam learning rate (default 0.001)
        hurst_window: Window for computing rolling Hurst (default 63)
        verbose: 0=silent, 1=progress (default 0)

    Example:
        >>> model = CloudConditionedLSTMForecaster(lookback=30)
        >>> model.fit(prices)
        >>> result = model.predict(n_steps=10)
        >>> result.forecast  # Point forecast
        >>> result.paths     # Combined LSTM + cloud paths
    """

    def __init__(
        self,
        lookback: int = 30,
        n_cloud_paths: int = 500,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        cloud_encoding_dim: int = 32,
        cloud_weight: float = 0.3,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
        learning_rate: float = 0.001,
        hurst_window: int = 63,
        verbose: int = 0,
    ):
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not installed. Install with: uv pip install torch"
            )

        if not 0.0 <= cloud_weight <= 1.0:
            raise ValueError(
                f"cloud_weight must be between 0 and 1, got {cloud_weight}"
            )

        self.lookback = lookback
        self.n_cloud_paths = n_cloud_paths
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.cloud_encoding_dim = cloud_encoding_dim
        self.cloud_weight = cloud_weight
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.learning_rate = learning_rate
        self.hurst_window = hurst_window
        self.verbose = verbose

        self.model: Optional[CloudConditionedLSTMNetwork] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalization parameters
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None
        self.cloud_stats_mean: Optional[np.ndarray] = None
        self.cloud_stats_std: Optional[np.ndarray] = None

        # Training history
        self.train_losses: list = []
        self.val_losses: list = []

        # Stored data for prediction
        self.prices: Optional[np.ndarray] = None
        self.last_features: Optional[np.ndarray] = None
        self.simulator: Optional[Simulator] = None

    def _compute_fractal_features(
        self, prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute rolling fractal features: [log_returns, volatility, H, D].

        Args:
            prices: Price series

        Returns:
            Tuple of (log_returns, volatility, hurst, fractal_dim) arrays
        """
        n = len(prices)

        # Log returns
        log_returns = np.zeros(n)
        log_returns[1:] = np.diff(np.log(prices))

        # Rolling volatility (non-annualized for consistency)
        volatility = compute_rolling_volatility(
            prices, self.hurst_window, annualize=False
        )

        # Rolling Hurst exponent
        hurst = np.zeros(n)
        for i in range(self.hurst_window, n):
            segment = prices[i - self.hurst_window : i]
            min_lag = max(10, self.hurst_window // 10)
            max_lag = self.hurst_window // 2
            hurst[i] = compute_hurst_rs(segment, min_lag, max_lag)

        # Fill initial values with first computed value
        if self.hurst_window < n:
            hurst[: self.hurst_window] = hurst[self.hurst_window]

        # Rolling fractal dimension
        fractal_dim = np.zeros(n)
        for i in range(self.hurst_window, n):
            segment = prices[i - self.hurst_window : i]
            # Scale prices to [0, 1]
            p_min, p_max = np.min(segment), np.max(segment)
            if p_max - p_min > 1e-10:
                scaled = (segment - p_min) / (p_max - p_min)
            else:
                scaled = np.zeros_like(segment)
            fractal_dim[i] = compute_box_dimension(
                scaled, 2, min(30, self.hurst_window // 4), 1
            )

        # Fill initial values
        if self.hurst_window < n:
            fractal_dim[: self.hurst_window] = fractal_dim[self.hurst_window]

        return log_returns, volatility, hurst, fractal_dim

    def _create_sequences_with_clouds(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        hurst: np.ndarray,
        fractal_dim: np.ndarray,
        n_forecast_steps: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create multivariate sequences with pre-generated cloud statistics.

        Args:
            returns: Log returns series
            volatility: Rolling volatility series
            hurst: Rolling Hurst exponent series
            fractal_dim: Rolling fractal dimension series
            n_forecast_steps: Number of forecast steps for cloud

        Returns:
            X: Input sequences, shape (n_samples, lookback, 4)
            y: Target values, shape (n_samples,)
            cloud_stats: Cloud statistics, shape (n_samples, n_forecast_steps, 8)
        """
        # Stack features: (n, 4)
        features = np.column_stack([returns, volatility, hurst, fractal_dim])

        X, y, cloud_stats_list = [], [], []

        # Start after hurst_window to ensure valid fractal features
        start_idx = max(self.hurst_window, self.lookback)

        for i in range(start_idx, len(features) - 1):
            # Input: lookback steps of 4 features
            X.append(features[i - self.lookback : i])
            # Target: next step's return
            y.append(returns[i + 1])

            # Generate cloud for this window
            window_prices = self.prices[: i + 1]
            try:
                sim = Simulator(window_prices)
                cloud = sim.generate(n_paths=self.n_cloud_paths, steps=n_forecast_steps)
                encoder = CloudEncoder(n_forecast_steps, self.cloud_encoding_dim)
                cloud_stats = encoder.encode_cloud_numpy(cloud)
            except Exception:
                # Fallback: use neutral statistics
                cloud_stats = np.zeros((n_forecast_steps, 8))
                cloud_stats[:, 0] = window_prices[-1]  # Mean at current price
                cloud_stats[:, 1] = np.std(window_prices[-21:])  # Recent vol

            cloud_stats_list.append(cloud_stats)

        return np.array(X), np.array(y), np.array(cloud_stats_list)

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        self.feature_means = np.mean(X, axis=(0, 1))
        self.feature_stds = np.std(X, axis=(0, 1))
        self.feature_stds = np.where(self.feature_stds < 1e-8, 1.0, self.feature_stds)
        return (X - self.feature_means) / self.feature_stds

    def _normalize_targets(self, y: np.ndarray) -> np.ndarray:
        """Normalize target values."""
        self.target_mean = np.mean(y)
        self.target_std = np.std(y)
        if self.target_std < 1e-8:
            self.target_std = 1.0
            warnings.warn("Zero standard deviation in targets. Using std=1.0")
        return (y - self.target_mean) / self.target_std

    def _normalize_cloud_stats(self, cloud_stats: np.ndarray) -> np.ndarray:
        """Normalize cloud statistics, handling NaN values from skew/kurtosis."""
        # Replace NaN with 0 before normalization (skew/kurtosis can produce NaN)
        cloud_stats = np.nan_to_num(cloud_stats, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape to (n_samples * n_steps, 8) for computing stats
        orig_shape = cloud_stats.shape
        flat = cloud_stats.reshape(-1, 8)
        self.cloud_stats_mean = np.nanmean(flat, axis=0)
        self.cloud_stats_std = np.nanstd(flat, axis=0)
        self.cloud_stats_std = np.where(
            self.cloud_stats_std < 1e-8, 1.0, self.cloud_stats_std
        )
        # Replace any remaining NaN in mean/std
        self.cloud_stats_mean = np.nan_to_num(self.cloud_stats_mean, nan=0.0)
        self.cloud_stats_std = np.nan_to_num(self.cloud_stats_std, nan=1.0)

        normalized = (flat - self.cloud_stats_mean) / self.cloud_stats_std
        # Final cleanup
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.reshape(orig_shape)

    def _denormalize_predictions(self, y: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return y * self.target_std + self.target_mean

    def fit(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        n_forecast_steps: int = 10,
        **kwargs,
    ) -> "CloudConditionedLSTMForecaster":
        """
        Fit the CloudConditionedLSTM model to historical prices.

        Args:
            prices: Historical price series
            dates: Optional date series (for compatibility, not used)
            n_forecast_steps: Number of forecast steps for cloud generation
            **kwargs: Additional arguments (for compatibility)

        Returns:
            self: Fitted model
        """
        self.prices = np.asarray(prices).flatten()

        # Validate data length
        min_required = self.hurst_window + self.lookback + 20
        if len(self.prices) < min_required:
            warnings.warn(
                f"Short time series ({len(self.prices)} points). "
                f"Need at least {min_required}. Reducing windows."
            )
            self.hurst_window = max(20, len(self.prices) // 5)
            self.lookback = max(10, len(self.prices) // 10)

        # Create simulator for later use
        self.simulator = Simulator(self.prices)

        # Compute fractal features
        log_returns, volatility, hurst, fractal_dim = self._compute_fractal_features(
            self.prices
        )

        # Create sequences with cloud statistics
        if self.verbose > 0:
            print("Generating clouds for training sequences...")

        X, y, cloud_stats = self._create_sequences_with_clouds(
            log_returns, volatility, hurst, fractal_dim, n_forecast_steps
        )

        if len(X) < 10:
            raise ValueError(
                f"Not enough data for training. Need at least "
                f"{self.hurst_window + self.lookback + 10} points, "
                f"got {len(self.prices)}"
            )

        # Normalize
        X_norm = self._normalize_features(X)
        y_norm = self._normalize_targets(y)
        cloud_stats_norm = self._normalize_cloud_stats(cloud_stats)

        # Store last features for prediction
        features = np.column_stack([log_returns, volatility, hurst, fractal_dim])
        self.last_features = features[-self.lookback :]

        # Split into train/validation
        n_train = int(len(X_norm) * (1 - self.validation_split))
        X_train, X_val = X_norm[:n_train], X_norm[n_train:]
        y_train, y_val = y_norm[:n_train], y_norm[n_train:]
        cloud_train, cloud_val = cloud_stats_norm[:n_train], cloud_stats_norm[n_train:]

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)
        cloud_train_t = torch.FloatTensor(cloud_train).to(self.device)

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)
        cloud_val_t = torch.FloatTensor(cloud_val).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_t, cloud_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Build model
        self.model = CloudConditionedLSTMNetwork(
            input_size=4,
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            cloud_encoding_dim=self.cloud_encoding_dim,
            n_forecast_steps=n_forecast_steps,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        self.train_losses = []
        self.val_losses = []

        try:
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_cloud, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X, batch_cloud)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                self.train_losses.append(train_loss)

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t, cloud_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                    self.val_losses.append(val_loss)

                if self.verbose > 0:
                    print(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if self.verbose > 0:
                        print(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break

        except Exception as e:
            warnings.warn(
                f"CloudConditionedLSTM training failed ({type(e).__name__}): {e}. "
                "Using fallback."
            )
            # Fallback: simpler model
            self.hidden_size_1 = 32
            self.hidden_size_2 = 16
            self.model = CloudConditionedLSTMNetwork(
                input_size=4,
                hidden_size_1=self.hidden_size_1,
                hidden_size_2=self.hidden_size_2,
                cloud_encoding_dim=self.cloud_encoding_dim,
                n_forecast_steps=n_forecast_steps,
                dropout=0.0,
            ).to(self.device)

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(min(20, self.epochs)):
                self.model.train()
                for batch_X, batch_cloud, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X, batch_cloud)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        return self

    def predict(
        self,
        n_steps: int = 10,
        n_simulations: int = 100,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate forecast with uncertainty via Monte Carlo dropout.

        Combines LSTM predictions (via MC dropout) with simulation cloud
        for final output.

        Args:
            n_steps: Number of steps to forecast
            n_simulations: Number of MC simulations for LSTM uncertainty
            **kwargs: Additional arguments (for compatibility)

        Returns:
            ForecastResult with:
                - forecast: Point forecast (weighted median)
                - paths: Combined LSTM + cloud paths
                - lower/upper: Confidence bounds
        """
        if self.model is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        # Generate fresh cloud from simulator
        cloud = self.simulator.generate(n_paths=self.n_cloud_paths, steps=n_steps)
        encoder = CloudEncoder(n_steps, self.cloud_encoding_dim)
        cloud_stats = encoder.encode_cloud_numpy(cloud)

        # Handle NaN values (from skew/kurtosis edge cases)
        cloud_stats = np.nan_to_num(cloud_stats, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize cloud stats
        cloud_stats_norm = (cloud_stats - self.cloud_stats_mean) / self.cloud_stats_std
        cloud_stats_norm = np.nan_to_num(
            cloud_stats_norm, nan=0.0, posinf=0.0, neginf=0.0
        )
        cloud_stats_t = torch.FloatTensor(cloud_stats_norm).unsqueeze(0).to(self.device)

        # Generate LSTM forecasts using MC dropout
        lstm_forecasts = np.zeros((n_simulations, n_steps))

        for sim in range(n_simulations):
            # Enable/disable dropout for uncertainty
            if sim > 0:
                self.model.train()  # Enable dropout
            else:
                self.model.eval()  # Deterministic for first sim

            # Start with last known features
            current_features = self.last_features.copy()
            last_price = self.prices[-1]
            forecast_sim = []

            with torch.no_grad():
                for step in range(n_steps):
                    # Normalize features
                    features_norm = (
                        current_features - self.feature_means
                    ) / self.feature_stds

                    # Prepare input tensor
                    X_input = (
                        torch.FloatTensor(features_norm).unsqueeze(0).to(self.device)
                    )

                    # Predict next return (normalized)
                    pred_norm = self.model(X_input, cloud_stats_t).cpu().numpy()[0, 0]

                    # Denormalize
                    pred_return = self._denormalize_predictions(np.array([pred_norm]))[
                        0
                    ]
                    forecast_sim.append(pred_return)

                    # Update price
                    last_price = last_price * np.exp(pred_return)

                    # Update features for next step (roll forward)
                    new_feature = np.array(
                        [
                            pred_return,
                            current_features[-1, 1],  # Keep last volatility
                            current_features[-1, 2],  # Keep last Hurst
                            current_features[-1, 3],  # Keep last fractal_dim
                        ]
                    )
                    current_features = np.vstack([current_features[1:], new_feature])

            lstm_forecasts[sim] = forecast_sim

        # Set model back to eval mode
        self.model.eval()

        # Convert LSTM return forecasts to price forecasts
        lstm_price_forecasts = np.zeros_like(lstm_forecasts)
        last_price = self.prices[-1]

        for sim in range(n_simulations):
            cumulative_returns = np.cumsum(lstm_forecasts[sim])
            lstm_price_forecasts[sim] = last_price * np.exp(cumulative_returns)

        # Combine LSTM paths with cloud paths
        # LSTM weight: (1 - cloud_weight), Cloud weight: cloud_weight
        n_lstm_paths = int(n_simulations * (1 - self.cloud_weight))
        n_cloud_paths = n_simulations - n_lstm_paths

        # Sample from cloud paths (with replacement if we need more than available)
        replace = n_cloud_paths > self.n_cloud_paths
        cloud_sample_idx = np.random.choice(
            self.n_cloud_paths, size=n_cloud_paths, replace=replace
        )
        cloud_sample = cloud[cloud_sample_idx]

        # Combine paths
        combined_paths = np.vstack([lstm_price_forecasts[:n_lstm_paths], cloud_sample])

        # Create probability weights (uniform for now)
        probabilities = np.ones(len(combined_paths)) / len(combined_paths)

        return ForecastResult(
            _paths=combined_paths,
            _probabilities=probabilities,
            metadata={
                "model_name": "CloudConditionedLSTM",
                "n_lstm_paths": n_lstm_paths,
                "n_cloud_paths": n_cloud_paths,
                "cloud_weight": self.cloud_weight,
                "params": self.get_model_params(),
            },
        )

    def get_model_params(self) -> Dict:
        """Get fitted model parameters."""
        if self.model is None:
            return {}

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        params = {
            "lookback": self.lookback,
            "n_cloud_paths": self.n_cloud_paths,
            "hidden_size_1": self.hidden_size_1,
            "hidden_size_2": self.hidden_size_2,
            "cloud_encoding_dim": self.cloud_encoding_dim,
            "cloud_weight": self.cloud_weight,
            "dropout": self.dropout,
            "hurst_window": self.hurst_window,
            "total_params": total_params,
        }

        # Add training history
        if self.train_losses:
            params["final_loss"] = float(self.train_losses[-1])
            params["final_val_loss"] = (
                float(self.val_losses[-1]) if self.val_losses else None
            )
            params["epochs_trained"] = len(self.train_losses)

        return params

    def __repr__(self) -> str:
        return (
            f"CloudConditionedLSTMForecaster("
            f"lookback={self.lookback}, "
            f"hidden=[{self.hidden_size_1}, {self.hidden_size_2}], "
            f"cloud_weight={self.cloud_weight}, "
            f"hurst_window={self.hurst_window})"
        )
