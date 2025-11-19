"""
FracTime: Advanced Time Series Forecasting with Fractal Geometry

A Python package for fractal-based time series forecasting.
"""

from .core import (
    FractalAnalyzer,
    FractalSimulator,
    FractalForecaster,
    plot_forecast,
    plot_forecast_interactive
)

__version__ = "0.1.0"
__all__ = [
    'FractalAnalyzer',
    'FractalSimulator',
    'FractalForecaster',
    'plot_forecast',
    'plot_forecast_interactive'
]
