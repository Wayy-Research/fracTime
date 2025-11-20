"""
Baseline forecasting models for comparison.

This module provides wrappers for classical time series forecasting models:
- ARIMA: Auto-regressive Integrated Moving Average
- GARCH: Generalized Autoregressive Conditional Heteroskedasticity
- Prophet: Facebook's forecasting tool
- ETS: Exponential Smoothing (Error-Trend-Seasonality)
- VAR: Vector Autoregression
- LSTM: Long Short-Term Memory neural network

All models follow the same API as FractalForecaster:
- fit(prices, dates=None)
- predict(n_steps=None, end_date=None)

This makes them compatible with WalkForwardValidator for rigorous comparison.

Example:
    >>> from fractime.baselines import ARIMAForecaster, LSTMForecaster
    >>> from fractime.backtesting import WalkForwardValidator, compare_models
    >>>
    >>> # Compare ARIMA vs LSTM vs Fractal
    >>> arima = ARIMAForecaster()
    >>> lstm = LSTMForecaster(lookback=30, units=50)
    >>> validator = WalkForwardValidator(arima)
    >>> arima_results = validator.run(prices, dates)
"""

from .arima import ARIMAForecaster
from .garch import GARCHForecaster
from .prophet import ProphetForecaster
from .ets import ETSForecaster
from .var import VARForecaster
from .lstm import LSTMForecaster

__all__ = [
    'ARIMAForecaster',
    'GARCHForecaster',
    'ProphetForecaster',
    'ETSForecaster',
    'VARForecaster',
    'LSTMForecaster',
]
