"""
Exponential Smoothing (ETS) baseline model.

Implements Error-Trend-Seasonality models using statsmodels.
"""

import numpy as np
import warnings
from typing import Dict, Optional

# Try to import statsmodels
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    ExponentialSmoothing = None


class ETSForecaster:
    """
    Exponential Smoothing (ETS) forecasting model.

    This is a classic time series forecasting method that decomposes
    the series into Error, Trend, and Seasonality components.

    Args:
        trend: Type of trend component ('add', 'mul', or None)
        seasonal: Type of seasonal component ('add', 'mul', or None)
        seasonal_periods: Number of periods in seasonal cycle (None for no seasonality)
        damped_trend: Whether to use a damped trend
        initialization_method: Method for initializing the parameters

    Example:
        >>> model = ETSForecaster(trend='add', seasonal=None)
        >>> model.fit(prices)
        >>> forecast = model.predict(n_steps=10)
    """

    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False,
        initialization_method: str = 'estimated'
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels not installed. Install with: uv pip install statsmodels"
            )

        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method
        self.model = None
        self.fitted_model = None
        self.prices = None

    def fit(self, prices: np.ndarray, **kwargs) -> 'ETSForecaster':
        """
        Fit the ETS model to historical prices.

        Args:
            prices: Historical price series
            **kwargs: Additional arguments passed to ExponentialSmoothing.fit()

        Returns:
            self: Fitted model
        """
        self.prices = np.asarray(prices).flatten()

        # Handle edge cases
        if len(self.prices) < 10:
            warnings.warn(
                f"Very short time series ({len(self.prices)} points). "
                "ETS may not perform well."
            )

        try:
            # Create the model
            self.model = ExponentialSmoothing(
                self.prices,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
                initialization_method=self.initialization_method
            )

            # Fit the model with error handling
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.fitted_model = self.model.fit(**kwargs)

        except Exception as e:
            warnings.warn(f"ETS fitting failed: {e}. Using fallback.")
            # Fallback to simpler model
            self.model = ExponentialSmoothing(
                self.prices,
                trend=None,
                seasonal=None
            )
            self.fitted_model = self.model.fit()

        return self

    def predict(self, n_steps: int = 10, **kwargs) -> Dict:
        """
        Generate forecast for n_steps ahead.

        Args:
            n_steps: Number of steps to forecast
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Dictionary with forecast results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Generate forecast
        forecast = self.fitted_model.forecast(steps=n_steps)

        # Generate prediction intervals using simulation
        # ETS prediction intervals from fitted model
        try:
            # Try to get prediction intervals if available
            forecast_obj = self.fitted_model.get_prediction(
                start=len(self.prices),
                end=len(self.prices) + n_steps - 1
            )
            pred_int = forecast_obj.conf_int(alpha=0.05)
            lower = pred_int.iloc[:, 0].values
            upper = pred_int.iloc[:, 1].values
        except:
            # Fallback: use simple expanding confidence intervals
            std = np.std(np.diff(self.prices))
            expanding_std = std * np.sqrt(np.arange(1, n_steps + 1))
            lower = forecast - 1.96 * expanding_std
            upper = forecast + 1.96 * expanding_std

        # Compute mean and std from forecast
        mean = forecast
        std = (upper - lower) / (2 * 1.96)

        return {
            'forecast': forecast,
            'mean': mean,
            'std': std,
            'lower': lower,
            'upper': upper,
            'model_name': 'ETS',
            'params': {
                'trend': self.trend,
                'seasonal': self.seasonal,
                'damped': self.damped_trend
            }
        }

    def get_model_params(self) -> Dict:
        """Get fitted model parameters."""
        if self.fitted_model is None:
            return {}

        params = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'level': self.fitted_model.params.get('smoothing_level'),
            'trend': self.fitted_model.params.get('smoothing_trend'),
            'seasonal': self.fitted_model.params.get('smoothing_seasonal'),
        }

        # Add damping parameter if applicable
        if self.damped_trend:
            params['damping'] = self.fitted_model.params.get('damping_trend')

        return {k: v for k, v in params.items() if v is not None}
