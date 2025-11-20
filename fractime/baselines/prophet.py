"""
Prophet baseline forecaster.

Facebook's Prophet is designed for business forecasting with:
- Strong seasonal components
- Multiple seasonality (daily, weekly, yearly)
- Holiday effects
- Trend changepoints

Good baseline for time series with clear seasonal patterns.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None


class ProphetForecaster:
    """
    Facebook Prophet baseline forecaster.

    Prophet decomposes time series into:
    - Trend (piecewise linear or logistic)
    - Seasonality (Fourier series)
    - Holidays
    - Error term

    Parameters:
        growth: 'linear' or 'logistic' trend (default 'linear')
        changepoint_prior_scale: Flexibility of trend changes (default 0.05)
        seasonality_prior_scale: Flexibility of seasonality (default 10.0)
        yearly_seasonality: Fit yearly seasonality (default 'auto')
        weekly_seasonality: Fit weekly seasonality (default 'auto')
        daily_seasonality: Fit daily seasonality (default 'auto')
        interval_width: Width of uncertainty intervals (default 0.95)

    Example:
        >>> from fractime.baselines import ProphetForecaster
        >>> from fractime.backtesting import WalkForwardValidator
        >>>
        >>> model = ProphetForecaster()
        >>> validator = WalkForwardValidator(model)
        >>> results = validator.run(prices, dates)

    Note:
        Prophet requires dates. If dates are not provided, will generate
        daily dates starting from 2020-01-01.
    """

    def __init__(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: Any = 'auto',
        weekly_seasonality: Any = 'auto',
        daily_seasonality: Any = 'auto',
        interval_width: float = 0.95
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "prophet not installed. Install with: uv pip install prophet"
            )

        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width

        # Model state
        self.model = None
        self.prices_history = None
        self.dates_history = None
        self.is_fitted = False
        self.frequency = None

    def fit(self, prices: np.ndarray, dates: Optional[np.ndarray] = None) -> None:
        """
        Fit Prophet model to price data.

        Args:
            prices: Historical prices
            dates: Dates for each price (required for Prophet)
                  If None, will generate daily dates starting from 2020-01-01
        """
        self.prices_history = np.asarray(prices)

        # Prophet requires dates
        if dates is None:
            # Generate default dates (daily frequency)
            start_date = pd.Timestamp('2020-01-01')
            dates = pd.date_range(start=start_date, periods=len(prices), freq='D')
            self.dates_history = dates.values
        else:
            self.dates_history = dates

        # Convert to DataFrame format required by Prophet
        df = self._prepare_dataframe(self.prices_history, self.dates_history)

        # Initialize and fit Prophet model
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width
        )

        # Suppress logging
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)

        self.model.fit(df)
        self.is_fitted = True

    def predict(
        self,
        n_steps: Optional[int] = None,
        end_date: Optional[Any] = None,
        confidence: Optional[float] = None,
        n_paths: int = 1000
    ) -> Dict:
        """
        Generate forecast with confidence intervals.

        Args:
            n_steps: Number of steps ahead to forecast
            end_date: Alternative to n_steps
            confidence: Confidence level (uses model's interval_width if None)
            n_paths: Not used (Prophet generates intervals analytically)

        Returns:
            Dictionary with:
                - forecast: Point forecast
                - lower: Lower confidence bound
                - upper: Upper confidence bound
                - mean: Same as forecast
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Determine forecast horizon
        if n_steps is None and end_date is not None:
            n_steps = self._compute_steps_to_date(end_date)

        if n_steps is None:
            n_steps = 1

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=n_steps, include_history=False)

        # Generate forecast
        forecast_df = self.model.predict(future)

        # Extract forecasts and confidence intervals
        forecast = forecast_df['yhat'].values
        lower = forecast_df['yhat_lower'].values
        upper = forecast_df['yhat_upper'].values

        return {
            'forecast': forecast,
            'mean': forecast,
            'lower': lower,
            'upper': upper,
            'model_params': self._get_model_params()
        }

    def _prepare_dataframe(self, prices: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
        """
        Convert prices and dates to Prophet's required format.

        Prophet requires DataFrame with columns: 'ds' (dates) and 'y' (values).
        """
        # Convert dates to pandas datetime
        if isinstance(dates[0], np.datetime64):
            dates_pd = pd.to_datetime(dates)
        elif isinstance(dates[0], datetime):
            dates_pd = pd.to_datetime(dates)
        else:
            # Assume dates are already in correct format
            dates_pd = pd.to_datetime(dates)

        df = pd.DataFrame({
            'ds': dates_pd,
            'y': prices
        })

        return df

    def _compute_steps_to_date(self, end_date: Any) -> int:
        """Compute number of steps from last training date to end_date."""
        if self.dates_history is None:
            raise ValueError("Cannot compute steps without dates")

        last_date = pd.to_datetime(self.dates_history[-1])

        # Convert end_date to pandas datetime
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        elif isinstance(end_date, np.datetime64):
            end_date = pd.to_datetime(end_date)
        elif isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)

        # Compute business days difference
        delta = (end_date - last_date).days
        return max(1, delta)

    def _get_model_params(self) -> Dict:
        """Get fitted model parameters for tracking."""
        if self.model is None:
            return {}

        # Extract key changepoint info
        params = {
            'growth': self.growth,
            'n_changepoints': len(self.model.changepoints),
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
        }

        # Add seasonality info if fitted
        if hasattr(self.model, 'seasonalities'):
            params['seasonalities'] = list(self.model.seasonalities.keys())

        return params

    def get_parameter_summary(self) -> Dict:
        """Get model parameter summary (for compatibility with backtesting)."""
        return self._get_model_params()

    def __repr__(self) -> str:
        if self.is_fitted and self.model is not None:
            n_changepoints = len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0
            return f"ProphetForecaster(growth={self.growth}, changepoints={n_changepoints})"
        else:
            return f"ProphetForecaster(growth={self.growth}, not fitted)"
