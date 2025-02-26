"""
FracTime: Advanced Time Series Forecasting with Fractal Geometry

A Python package for time series analysis and forecasting using fractal geometry
and chaos theory principles, inspired by Mandelbrot's work on financial markets.
"""

from .core import (
    FractalAnalyzer,
    FractalSimulator,
    PathAnalyzer,
    FractalVisualizer,
    get_yahoo_data
)

__version__ = "0.1.0"
__all__ = [
    'FractalAnalyzer',
    'FractalSimulator',
    'PathAnalyzer',
    'FractalVisualizer',
    'get_yahoo_data'
]
