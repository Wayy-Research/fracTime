"""
FracTime-LSTM hybrid forecasting model.

Implements the FracTime-LSTM architecture from Section 3.3 of the paper.
The key innovation is providing Hurst exponent and fractal dimension as
explicit features, giving the network an "inductive bias" for long-range
dependence.

The fractal features allow the LSTM to modulate its gates based on the
current fractal regime:
    - When H_t > 0.5 (long memory): preserve cell state
    - When H_t < 0.5 (anti-persistent): reset more frequently
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


class FractalLSTMNetwork(nn.Module if PYTORCH_AVAILABLE else object):
    """
    PyTorch LSTM network with fractal feature inputs.

    Architecture (from paper Section 3.3):
        - Input: [log_returns, volatility_t, H_t, D_t] - 4 features
        - Layer 1: LSTM (64 units, tanh, return_sequences=True)
        - Dropout: 0.2
        - Layer 2: LSTM (32 units, tanh)
        - Dropout: 0.2
        - Output: Dense (1 unit, linear) - predict next day return
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout: float = 0.2,
    ):
        if PYTORCH_AVAILABLE:
            super(FractalLSTMNetwork, self).__init__()
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2

            # Layer 1: LSTM with 64 units
            self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size_1,
                num_layers=1,
                batch_first=True,
            )

            # Dropout after layer 1
            self.dropout1 = nn.Dropout(dropout)

            # Layer 2: LSTM with 32 units
            self.lstm2 = nn.LSTM(
                input_size=hidden_size_1,
                hidden_size=hidden_size_2,
                num_layers=1,
                batch_first=True,
            )

            # Dropout after layer 2
            self.dropout2 = nn.Dropout(dropout)

            # Output layer: Dense (1 unit, linear)
            self.fc = nn.Linear(hidden_size_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, 4)
               where 4 = [log_returns, volatility, hurst, fractal_dim]

        Returns:
            Output tensor of shape (batch, 1)
        """
        # Layer 1: LSTM
        lstm1_out, _ = self.lstm1(x)  # (batch, seq_len, hidden_size_1)
        lstm1_out = self.dropout1(lstm1_out)

        # Layer 2: LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, seq_len, hidden_size_2)

        # Take the last time step
        last_output = lstm2_out[:, -1, :]  # (batch, hidden_size_2)
        last_output = self.dropout2(last_output)

        # Output layer
        out = self.fc(last_output)  # (batch, 1)

        return out


class FractalLSTMForecaster:
    """
    LSTM with fractal features as inputs (FracTime-LSTM hybrid).

    The key innovation is providing Hurst exponent and fractal dimension
    as explicit features, giving the network an "inductive bias" for
    long-range dependence. This allows the LSTM to learn regime-specific
    dynamics without having to discover these patterns from raw data alone.

    Input features (computed rolling):
        - log_returns: Log returns of the price series
        - volatility_t: Rolling volatility
        - H_t: Rolling Hurst exponent
        - D_t: Rolling fractal dimension

    Args:
        lookback: Sequence length for LSTM (default 30)
        hidden_size_1: First LSTM layer size (default 64)
        hidden_size_2: Second LSTM layer size (default 32)
        dropout: Dropout rate (default 0.2)
        epochs: Maximum training epochs (default 100)
        batch_size: Training batch size (default 32)
        validation_split: Fraction of data for validation (default 0.2)
        patience: Early stopping patience (default 10)
        learning_rate: Adam learning rate (default 0.001)
        hurst_window: Window for computing rolling Hurst (default 63)
        verbose: 0=silent, 1=progress (default 0)

    Example:
        >>> model = FractalLSTMForecaster(lookback=30, hurst_window=63)
        >>> model.fit(prices)
        >>> forecast = model.predict(n_steps=10)
        >>> print(forecast['forecast'])  # Point forecast
        >>> print(forecast['upper'])     # 97.5% CI
    """

    def __init__(
        self,
        lookback: int = 30,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
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

        self.lookback = lookback
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.learning_rate = learning_rate
        self.hurst_window = hurst_window
        self.verbose = verbose

        self.model: Optional[FractalLSTMNetwork] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Normalization parameters (per feature)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None

        # Training history
        self.train_losses: list = []
        self.val_losses: list = []

        # Stored data for prediction
        self.prices: Optional[np.ndarray] = None
        self.last_features: Optional[np.ndarray] = None

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
        volatility = compute_rolling_volatility(prices, self.hurst_window, annualize=False)

        # Rolling Hurst exponent
        hurst = np.zeros(n)
        for i in range(self.hurst_window, n):
            segment = prices[i - self.hurst_window:i]
            min_lag = max(10, self.hurst_window // 10)
            max_lag = self.hurst_window // 2
            hurst[i] = compute_hurst_rs(segment, min_lag, max_lag)

        # Fill initial values with first computed value
        if self.hurst_window < n:
            hurst[:self.hurst_window] = hurst[self.hurst_window]

        # Rolling fractal dimension
        fractal_dim = np.zeros(n)
        for i in range(self.hurst_window, n):
            segment = prices[i - self.hurst_window:i]
            # Scale prices to [0, 1]
            p_min, p_max = np.min(segment), np.max(segment)
            if p_max - p_min > 1e-10:
                scaled = (segment - p_min) / (p_max - p_min)
            else:
                scaled = np.zeros_like(segment)
            fractal_dim[i] = compute_box_dimension(scaled, 2, min(30, self.hurst_window // 4), 1)

        # Fill initial values
        if self.hurst_window < n:
            fractal_dim[:self.hurst_window] = fractal_dim[self.hurst_window]

        return log_returns, volatility, hurst, fractal_dim

    def _create_sequences(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        hurst: np.ndarray,
        fractal_dim: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multivariate sequences for LSTM training.

        Args:
            returns: Log returns series
            volatility: Rolling volatility series
            hurst: Rolling Hurst exponent series
            fractal_dim: Rolling fractal dimension series

        Returns:
            X: Input sequences of shape (n_samples, lookback, 4)
            y: Target values of shape (n_samples,)
        """
        # Stack features: (n, 4)
        features = np.column_stack([returns, volatility, hurst, fractal_dim])

        X, y = [], []

        # Start after hurst_window to ensure valid fractal features
        start_idx = max(self.hurst_window, self.lookback)

        for i in range(start_idx, len(features) - 1):
            # Input: lookback steps of 4 features
            X.append(features[i - self.lookback:i])
            # Target: next step's return
            y.append(returns[i + 1])

        return np.array(X), np.array(y)

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score normalization.

        Computes mean and std per feature across all samples and time steps.
        """
        # X shape: (n_samples, lookback, n_features)
        # Compute stats across samples and time
        self.feature_means = np.mean(X, axis=(0, 1))
        self.feature_stds = np.std(X, axis=(0, 1))

        # Prevent division by zero
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

    def _denormalize_predictions(self, y: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return y * self.target_std + self.target_mean

    def fit(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'FractalLSTMForecaster':
        """
        Fit the FractalLSTM model to historical prices.

        Args:
            prices: Historical price series
            dates: Optional date series (for compatibility, not used)
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

        # Compute fractal features
        log_returns, volatility, hurst, fractal_dim = self._compute_fractal_features(
            self.prices
        )

        # Create sequences
        X, y = self._create_sequences(log_returns, volatility, hurst, fractal_dim)

        if len(X) < 10:
            raise ValueError(
                f"Not enough data for training. Need at least "
                f"{self.hurst_window + self.lookback + 10} points, got {len(self.prices)}"
            )

        # Normalize
        X_norm = self._normalize_features(X)
        y_norm = self._normalize_targets(y)

        # Store last features for prediction
        features = np.column_stack([log_returns, volatility, hurst, fractal_dim])
        self.last_features = features[-self.lookback:]

        # Split into train/validation
        n_train = int(len(X_norm) * (1 - self.validation_split))
        X_train, X_val = X_norm[:n_train], X_norm[n_train:]
        y_train, y_val = y_norm[:n_train], y_norm[n_train:]

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Build model
        self.model = FractalLSTMNetwork(
            input_size=4,  # [returns, volatility, hurst, fractal_dim]
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        self.train_losses = []
        self.val_losses = []

        try:
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                self.train_losses.append(train_loss)

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
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
            warnings.warn(f"FractalLSTM training failed: {e}. Using fallback.")
            # Fallback: simpler model
            self.hidden_size_1 = 32
            self.hidden_size_2 = 16
            self.model = FractalLSTMNetwork(
                input_size=4,
                hidden_size_1=self.hidden_size_1,
                hidden_size_2=self.hidden_size_2,
                dropout=0.0,
            ).to(self.device)

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(min(20, self.epochs)):
                self.model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

        return self

    def predict(
        self,
        n_steps: int = 10,
        n_simulations: int = 100,
        **kwargs
    ) -> Dict:
        """
        Generate forecast with uncertainty via Monte Carlo dropout.

        The MC dropout approach provides uncertainty quantification by
        running multiple forward passes with dropout enabled, giving
        a distribution of predictions.

        Args:
            n_steps: Number of steps to forecast
            n_simulations: Number of MC simulations for uncertainty
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Dictionary with:
                - forecast: Point forecast (mean of simulations)
                - mean: Same as forecast
                - std: Standard deviation of simulations
                - lower: 2.5th percentile (95% CI lower)
                - upper: 97.5th percentile (95% CI upper)
                - model_name: 'FractalLSTM'
                - params: Model parameters
        """
        if self.model is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        # Generate forecasts using MC dropout
        all_forecasts = np.zeros((n_simulations, n_steps))

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
                        (current_features - self.feature_means) / self.feature_stds
                    )

                    # Prepare input tensor
                    X_input = (
                        torch.FloatTensor(features_norm).unsqueeze(0).to(self.device)
                    )

                    # Predict next return (normalized)
                    pred_norm = self.model(X_input).cpu().numpy()[0, 0]

                    # Denormalize
                    pred_return = self._denormalize_predictions(
                        np.array([pred_norm])
                    )[0]
                    forecast_sim.append(pred_return)

                    # Update price
                    last_price = last_price * np.exp(pred_return)

                    # Update features for next step (roll forward)
                    new_feature = np.array([
                        pred_return,
                        current_features[-1, 1],  # Keep last volatility
                        current_features[-1, 2],  # Keep last Hurst
                        current_features[-1, 3],  # Keep last fractal_dim
                    ])
                    current_features = np.vstack([current_features[1:], new_feature])

            all_forecasts[sim] = forecast_sim

        # Set model back to eval mode
        self.model.eval()

        # Convert return forecasts to price forecasts
        all_price_forecasts = np.zeros_like(all_forecasts)
        last_price = self.prices[-1]

        for sim in range(n_simulations):
            cumulative_returns = np.cumsum(all_forecasts[sim])
            all_price_forecasts[sim] = last_price * np.exp(cumulative_returns)

        # Calculate statistics
        forecast = np.mean(all_price_forecasts, axis=0)
        std = np.std(all_price_forecasts, axis=0)
        lower = np.percentile(all_price_forecasts, 2.5, axis=0)
        upper = np.percentile(all_price_forecasts, 97.5, axis=0)

        return {
            'forecast': forecast,
            'mean': forecast,
            'std': std,
            'lower': lower,
            'upper': upper,
            'model_name': 'FractalLSTM',
            'params': self.get_model_params(),
        }

    def get_model_params(self) -> Dict:
        """Get fitted model parameters."""
        if self.model is None:
            return {}

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        params = {
            'lookback': self.lookback,
            'hidden_size_1': self.hidden_size_1,
            'hidden_size_2': self.hidden_size_2,
            'dropout': self.dropout,
            'hurst_window': self.hurst_window,
            'total_params': total_params,
        }

        # Add training history
        if self.train_losses:
            params['final_loss'] = float(self.train_losses[-1])
            params['final_val_loss'] = (
                float(self.val_losses[-1]) if self.val_losses else None
            )
            params['epochs_trained'] = len(self.train_losses)

        return params

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance based on input gradients.

        Returns importance scores for each feature:
        [log_returns, volatility, hurst, fractal_dim]

        Note: This is a simple gradient-based importance measure.
        For production use, consider SHAP or integrated gradients.
        """
        if self.model is None:
            raise ValueError("Model must be fitted first.")

        # Must be in train mode for LSTM backward pass with cuDNN
        self.model.train()

        # Use last features as input
        features_norm = (self.last_features - self.feature_means) / self.feature_stds
        X_input = torch.FloatTensor(features_norm).unsqueeze(0).to(self.device)
        X_input.requires_grad_(True)

        # Forward pass
        output = self.model(X_input)

        # Backward pass
        output.backward()

        # Get gradient magnitudes per feature (average over time)
        gradients = X_input.grad.cpu().numpy()[0]
        importance = np.mean(np.abs(gradients), axis=0)

        # Set back to eval mode
        self.model.eval()

        # Normalize to sum to 1
        importance = importance / (importance.sum() + 1e-8)

        return {
            'log_returns': float(importance[0]),
            'volatility': float(importance[1]),
            'hurst': float(importance[2]),
            'fractal_dim': float(importance[3]),
        }

    def __repr__(self) -> str:
        return (
            f"FractalLSTMForecaster(lookback={self.lookback}, "
            f"hidden=[{self.hidden_size_1}, {self.hidden_size_2}], "
            f"hurst_window={self.hurst_window})"
        )
