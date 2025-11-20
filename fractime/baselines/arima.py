"""
ARIMA (Auto-Regressive Integrated Moving Average) baseline forecaster.

Uses pmdarima's auto_arima for automatic parameter selection.
This provides a strong classical benchmark for comparison.
"""

import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    auto_arima = None


class ARIMAForecaster:
    """
    ARIMA baseline forecaster with automatic parameter selection.

    Uses auto_arima to find optimal (p, d, q) parameters.
    Compatible with WalkForwardValidator API.

    Parameters:
        seasonal: Whether to fit seasonal ARIMA (default False)
        m: Seasonal period (default 1, ignored if seasonal=False)
        max_p: Maximum AR order (default 5)
        max_q: Maximum MA order (default 5)
        max_d: Maximum differencing order (default 2)
        stepwise: Use stepwise search (faster but less thorough, default True)
        suppress_warnings: Suppress convergence warnings (default True)

    Example:
        >>> from fractime.baselines import ARIMAForecaster
        >>> from fractime.backtesting import WalkForwardValidator
        >>>
        >>> model = ARIMAForecaster()
        >>> validator = WalkForwardValidator(model)
        >>> results = validator.run(prices, dates)
        >>>
        >>> print(f"ARIMA RMSE: {results['metrics']['rmse']:.4f}")
    """

    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        stepwise: bool = True,
        suppress_warnings: bool = True,
        information_criterion: str = 'aic'
    ):
        if not PMDARIMA_AVAILABLE:
            raise ImportError(
                "pmdarima not installed. Install with: uv pip install pmdarima"
            )

        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.information_criterion = information_criterion

        # Model state
        self.model = None
        self.prices_history = None
        self.dates_history = None
        self.is_fitted = False

    def fit(self, prices: np.ndarray, dates: Optional[np.ndarray] = None) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to price data.

        Args:
            prices: Historical prices
            dates: Optional dates (not used by ARIMA but kept for API compatibility)

        Returns:
            self: Fitted forecaster
        """
        self.prices_history = np.asarray(prices)
        self.dates_history = dates

        # Auto ARIMA parameter search
        self.model = auto_arima(
            self.prices_history,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            stepwise=self.stepwise,
            suppress_warnings=self.suppress_warnings,
            error_action='ignore',
            trace=False,
            information_criterion=self.information_criterion
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        n_steps: Optional[int] = None,
        end_date: Optional[Any] = None,
        confidence: float = 0.95,
        n_paths: int = 1000
    ) -> Dict:
        """
        Generate forecast with confidence intervals.

        Args:
            n_steps: Number of steps ahead to forecast
            end_date: Alternative to n_steps (compute steps from dates)
            confidence: Confidence level for intervals (default 0.95)
            n_paths: Number of simulated paths (for compatibility, not used by ARIMA)

        Returns:
            Dictionary with:
                - forecast: Point forecast
                - lower: Lower confidence bound
                - upper: Upper confidence bound
                - mean: Same as forecast (for compatibility)
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

        # Generate forecast with confidence intervals
        forecast, conf_int = self.model.predict(
            n_periods=n_steps,
            return_conf_int=True,
            alpha=1 - confidence
        )

        lower = conf_int[:, 0]
        upper = conf_int[:, 1]

        # Compute standard deviation from confidence interval
        # For 95% CI: upper = mean + 1.96*std, lower = mean - 1.96*std
        # So: std = (upper - lower) / (2 * 1.96)
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        std = (upper - lower) / (2 * z_score)

        return {
            'forecast': forecast,
            'mean': forecast,  # For compatibility
            'std': std,  # For compatibility with other models
            'lower': lower,
            'upper': upper,
            'model_params': self._get_model_params()
        }

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

        # Compute business days difference (approximate)
        delta = (end_date - last_date).days
        return max(1, delta)

    def _get_model_params(self) -> Dict:
        """Get fitted model parameters for tracking."""
        if self.model is None:
            return {}

        return {
            'order': self.model.order,
            'seasonal_order': self.model.seasonal_order if self.seasonal else None,
            'aic': self.model.aic(),
            'bic': self.model.bic(),
        }

    def get_parameter_summary(self) -> Dict:
        """
        Get model parameter summary (for compatibility with backtesting).

        Returns:
            Dictionary with model order and information criteria.
        """
        return self._get_model_params()

    def __repr__(self) -> str:
        if self.is_fitted and self.model is not None:
            return f"ARIMAForecaster(order={self.model.order}, AIC={self.model.aic():.2f})"
        else:
            return "ARIMAForecaster(not fitted)"
