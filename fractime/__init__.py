"""
FracTime: Advanced Time Series Forecasting with Fractal Geometry

A Python package for fractal-based time series forecasting.
"""

from .core import (
    FractalAnalyzer,
    FractalSimulator,
    FractalForecaster,
    PathAnalyzer,
    FractalVisualizer,
    get_yahoo_data,
    plot_forecast,
    run_backtest
)

__version__ = "0.1.0"
__all__ = [
    'FractalAnalyzer',
    'FractalSimulator',
    'FractalForecaster',
    'PathAnalyzer',
    'FractalVisualizer',
    'get_yahoo_data',
    'plot_forecast',
    'run_backtest'
]
