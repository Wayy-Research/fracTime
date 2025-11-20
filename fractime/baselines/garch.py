"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) baseline forecaster.

GARCH models volatility clustering - periods of high/low volatility tend to cluster.
This is particularly relevant for financial time series.

Uses the arch package for GARCH estimation.
"""

import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    arch_model = None


class GARCHForecaster:
    """
    GARCH baseline forecaster for volatility modeling.

    GARCH(p, q) models conditional variance:
    σ²_t = ω + Σα_i ε²_{t-i} + Σβ_j σ²_{t-j}

    For forecasting prices, combines GARCH variance with mean model.

    Parameters:
        p: GARCH lag order (default 1)
        q: ARCH lag order (default 1)
        mean_model: Mean model type ('Constant', 'Zero', 'AR', default 'Constant')
        vol_model: Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH', default 'GARCH')
        dist: Distribution for innovations ('normal', 't', 'skewt', default 'normal')

    Example:
        >>> from fractime.baselines import GARCHForecaster
        >>> from fractime.backtesting import WalkForwardValidator
        >>>
        >>> model = GARCHForecaster(p=1, q=1)
        >>> validator = WalkForwardValidator(model)
        >>> results = validator.run(prices, dates)
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean_model: str = 'Constant',
        vol_model: str = 'GARCH',
        dist: str = 'normal'
    ):
        if not ARCH_AVAILABLE:
            raise ImportError(
                "arch not installed. Install with: uv pip install arch"
            )

        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.dist = dist

        # Model state
        self.model = None
        self.fitted_model = None
        self.prices_history = None
        self.dates_history = None
        self.returns_history = None
        self.is_fitted = False

    def fit(self, prices: np.ndarray, dates: Optional[np.ndarray] = None) -> None:
        """
        Fit GARCH model to price data.

        Args:
            prices: Historical prices
            dates: Optional dates (not used by GARCH but kept for API compatibility)
        """
        self.prices_history = np.asarray(prices)
        self.dates_history = dates

        # Compute log returns (GARCH is fit on returns, not levels)
        self.returns_history = np.diff(np.log(self.prices_history)) * 100  # Percentage returns

        # Build and fit GARCH model
        self.model = arch_model(
            self.returns_history,
            mean=self.mean_model,
            vol=self.vol_model,
            p=self.p,
            q=self.q,
            dist=self.dist
        )

        # Fit with reduced verbosity
        self.fitted_model = self.model.fit(disp='off', show_warning=False)
        self.is_fitted = True

    def predict(
        self,
        n_steps: Optional[int] = None,
        end_date: Optional[Any] = None,
        confidence: float = 0.95,
        n_paths: int = 1000
    ) -> Dict:
        """
        Generate forecast with confidence intervals.

        GARCH forecasts volatility, then we simulate price paths using:
        - Mean return from fitted model
        - Conditional variance from GARCH
        - Bootstrap/simulation for multi-step forecasts

        Args:
            n_steps: Number of steps ahead to forecast
            end_date: Alternative to n_steps
            confidence: Confidence level for intervals (default 0.95)
            n_paths: Number of simulated paths (default 1000)

        Returns:
            Dictionary with:
                - forecast: Point forecast (median of simulated paths)
                - lower: Lower confidence bound
                - upper: Upper confidence bound
                - mean: Mean of simulated paths
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Determine forecast horizon
        if n_steps is None and end_date is not None:
            if self.dates_history is None:
                raise ValueError("Cannot use end_date without dates in training data")
            n_steps = self._compute_steps_to_date(end_date)

        if n_steps is None:
            n_steps = 1

        # Get GARCH variance forecast
        garch_forecast = self.fitted_model.forecast(horizon=n_steps, method='simulation')

        # Extract mean and variance forecasts
        mean_return = self.fitted_model.params['mu'] if 'mu' in self.fitted_model.params else 0.0
        variance_forecast = garch_forecast.variance.values[-1, :]

        # Simulate price paths
        last_price = self.prices_history[-1]
        simulated_paths = self._simulate_price_paths(
            last_price, mean_return, variance_forecast, n_paths, n_steps
        )

        # Compute statistics from simulated paths
        forecast_median = np.median(simulated_paths[:, -1])
        forecast_mean = np.mean(simulated_paths[:, -1])

        # Confidence intervals
        alpha = 1 - confidence
        lower = np.percentile(simulated_paths[:, -1], alpha/2 * 100)
        upper = np.percentile(simulated_paths[:, -1], (1 - alpha/2) * 100)

        # For multi-step, return full forecast trajectory
        if n_steps > 1:
            forecast_trajectory = np.median(simulated_paths, axis=0)
            lower_trajectory = np.percentile(simulated_paths, alpha/2 * 100, axis=0)
            upper_trajectory = np.percentile(simulated_paths, (1 - alpha/2) * 100, axis=0)
        else:
            forecast_trajectory = np.array([forecast_median])
            lower_trajectory = np.array([lower])
            upper_trajectory = np.array([upper])

        return {
            'forecast': forecast_trajectory,
            'mean': np.mean(simulated_paths, axis=0),
            'lower': lower_trajectory,
            'upper': upper_trajectory,
            'model_params': self._get_model_params()
        }

    def _simulate_price_paths(
        self,
        initial_price: float,
        mean_return: float,
        variance_forecast: np.ndarray,
        n_paths: int,
        n_steps: int
    ) -> np.ndarray:
        """
        Simulate price paths using GARCH forecasted volatility.

        Args:
            initial_price: Starting price
            mean_return: Mean return (from fitted model)
            variance_forecast: Conditional variance forecast for each step
            n_paths: Number of paths to simulate
            n_steps: Number of steps ahead

        Returns:
            Array of shape (n_paths, n_steps) with simulated prices
        """
        paths = np.zeros((n_paths, n_steps))
        log_price = np.log(initial_price)

        for path_idx in range(n_paths):
            current_log_price = log_price

            for step in range(n_steps):
                # Sample return from distribution with GARCH variance
                std_dev = np.sqrt(variance_forecast[step]) / 100  # Convert back from percentage
                return_shock = np.random.normal(mean_return / 100, std_dev)

                # Update log price
                current_log_price += return_shock

                # Store price
                paths[path_idx, step] = np.exp(current_log_price)

        return paths

    def _compute_steps_to_date(self, end_date: Any) -> int:
        """Compute number of steps from last training date to end_date."""
        if self.dates_history is None:
            raise ValueError("Cannot compute steps without dates")

        last_date = self.dates_history[-1]

        # Convert to datetime if needed
        if isinstance(last_date, np.datetime64):
            last_date = last_date.astype('datetime64[D]').astype(datetime)
        if isinstance(end_date, np.datetime64):
            end_date = end_date.astype('datetime64[D]').astype(datetime)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        delta = (end_date - last_date).days
        return max(1, delta)

    def _get_model_params(self) -> Dict:
        """Get fitted model parameters for tracking."""
        if self.fitted_model is None:
            return {}

        return {
            'p': self.p,
            'q': self.q,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'loglikelihood': self.fitted_model.loglikelihood,
        }

    def get_parameter_summary(self) -> Dict:
        """Get model parameter summary (for compatibility with backtesting)."""
        return self._get_model_params()

    def __repr__(self) -> str:
        if self.is_fitted and self.fitted_model is not None:
            return f"GARCHForecaster(p={self.p}, q={self.q}, AIC={self.fitted_model.aic:.2f})"
        else:
            return f"GARCHForecaster(p={self.p}, q={self.q}, not fitted)"
