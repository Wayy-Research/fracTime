"""
Long Short-Term Memory (LSTM) neural network baseline model.

Implements deep learning-based time series forecasting using PyTorch.
"""

import numpy as np
import warnings
from typing import Dict, Optional, Tuple

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


class LSTMNetwork(nn.Module if PYTORCH_AVAILABLE else object):
    """PyTorch LSTM network architecture."""

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        if PYTORCH_AVAILABLE:
            super(LSTMNetwork, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

            # Output layers
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Take the last time step
        last_output = lstm_out[:, -1, :]

        # Pass through dense layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class LSTMForecaster:
    """
    LSTM neural network forecasting model using PyTorch.

    Uses recurrent neural networks with LSTM cells to capture
    long-term dependencies in time series data.

    Args:
        lookback: Number of time steps to use as input (sequence length)
        hidden_size: Number of LSTM units in each layer
        num_layers: Number of LSTM layers (1-3 recommended)
        dropout: Dropout rate for regularization (0.0-0.5)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        patience: Early stopping patience (epochs)
        learning_rate: Initial learning rate
        verbose: Training verbosity (0=silent, 1=epoch info)

    Example:
        >>> model = LSTMForecaster(lookback=30, hidden_size=50, num_layers=2)
        >>> model.fit(prices)
        >>> forecast = model.predict(n_steps=10)
    """

    def __init__(
        self,
        lookback: int = 30,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
        learning_rate: float = 0.001,
        verbose: int = 0
    ):
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not installed. Install with: uv pip install torch"
            )

        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_mean = None
        self.scaler_std = None
        self.prices = None
        self.train_losses = []
        self.val_losses = []

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for supervised learning.

        Args:
            data: 1D array of values

        Returns:
            X: Input sequences (n_samples, lookback)
            y: Target values (n_samples,)
        """
        X, y = [], []

        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])

        return np.array(X), np.array(y)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using z-score normalization."""
        self.scaler_mean = np.mean(data)
        self.scaler_std = np.std(data)

        if self.scaler_std == 0:
            self.scaler_std = 1.0
            warnings.warn("Zero standard deviation in data. Using std=1.0")

        return (data - self.scaler_mean) / self.scaler_std

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return data * self.scaler_std + self.scaler_mean

    def fit(self, prices: np.ndarray, **kwargs) -> 'LSTMForecaster':
        """
        Fit the LSTM model to historical prices.

        Args:
            prices: Historical price series
            **kwargs: Additional arguments

        Returns:
            self: Fitted model
        """
        self.prices = np.asarray(prices).flatten()

        # Handle edge cases
        if len(self.prices) < self.lookback + 20:
            warnings.warn(
                f"Short time series ({len(self.prices)} points) for lookback={self.lookback}. "
                "Reducing lookback."
            )
            self.lookback = max(5, len(self.prices) // 4)

        # Normalize data
        normalized_prices = self._normalize(self.prices)

        # Create sequences
        X, y = self._create_sequences(normalized_prices)

        if len(X) < 10:
            raise ValueError(
                f"Not enough data for training. Need at least {self.lookback + 10} points, "
                f"got {len(self.prices)}"
            )

        # Split into train/validation
        n_train = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)
        X_val = torch.FloatTensor(X_val).unsqueeze(-1).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
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
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    self.val_losses.append(val_loss)

                if self.verbose > 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if self.verbose > 0:
                        print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break

        except Exception as e:
            warnings.warn(f"LSTM training failed: {e}. Using fallback configuration.")
            # Fallback: simpler model
            self.num_layers = 1
            self.hidden_size = 20
            self.model = LSTMNetwork(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.0
            ).to(self.device)

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            for epoch in range(min(20, self.epochs)):
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

        return self

    def predict(self, n_steps: int = 10, n_simulations: int = 100, **kwargs) -> Dict:
        """
        Generate forecast for n_steps ahead.

        Uses Monte Carlo dropout for uncertainty quantification.

        Args:
            n_steps: Number of steps to forecast
            n_simulations: Number of Monte Carlo simulations for uncertainty
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Dictionary with forecast results
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        self.model.eval()

        # Get last lookback points
        last_sequence = self.prices[-self.lookback:]
        normalized_sequence = (last_sequence - self.scaler_mean) / self.scaler_std

        # Initialize forecasts storage for Monte Carlo simulation
        all_forecasts = np.zeros((n_simulations, n_steps))

        # Monte Carlo dropout for uncertainty estimation
        for sim in range(n_simulations):
            current_sequence = normalized_sequence.copy()
            forecast_sim = []

            # Enable dropout for uncertainty estimation (except first simulation)
            if sim > 0:
                self.model.train()  # Enable dropout
            else:
                self.model.eval()  # Deterministic

            with torch.no_grad():
                for step in range(n_steps):
                    # Prepare input
                    X_input = torch.FloatTensor(current_sequence[-self.lookback:]).unsqueeze(0).unsqueeze(-1).to(self.device)

                    # Predict next step
                    pred = self.model(X_input).cpu().numpy()[0, 0]
                    forecast_sim.append(pred)

                    # Update sequence for next prediction
                    current_sequence = np.append(current_sequence, pred)

            # Denormalize forecast
            all_forecasts[sim] = self._denormalize(np.array(forecast_sim))

        # Set back to eval mode
        self.model.eval()

        # Calculate statistics from Monte Carlo samples
        forecast = np.mean(all_forecasts, axis=0)
        mean = forecast
        std = np.std(all_forecasts, axis=0)
        lower = np.percentile(all_forecasts, 2.5, axis=0)
        upper = np.percentile(all_forecasts, 97.5, axis=0)

        return {
            'forecast': forecast,
            'mean': mean,
            'std': std,
            'lower': lower,
            'upper': upper,
            'model_name': 'LSTM',
            'params': {
                'lookback': self.lookback,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }

    def get_model_params(self) -> Dict:
        """Get fitted model parameters."""
        if self.model is None:
            return {}

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        params = {
            'lookback': self.lookback,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'total_params': total_params,
        }

        # Add training history if available
        if self.train_losses:
            params['final_loss'] = float(self.train_losses[-1])
            params['final_val_loss'] = float(self.val_losses[-1]) if self.val_losses else None
            params['epochs_trained'] = len(self.train_losses)

        return params

    def plot_training_history(self):
        """
        Plot training history (loss curves).

        Requires matplotlib.
        """
        if not self.train_losses:
            print("No training history available. Model must be fitted first.")
            return

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            epochs = range(1, len(self.train_losses) + 1)
            ax.plot(epochs, self.train_losses, label='Training Loss', marker='o', markersize=3)

            if self.val_losses:
                ax.plot(epochs, self.val_losses, label='Validation Loss', marker='s', markersize=3)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('LSTM Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib not installed. Install with: uv pip install matplotlib")
