"""
FracTime: Advanced Time Series Forecasting with Fractal Geometry

A Python package for fractal-based time series forecasting.

Simple, flat API following the Zen of Python:
- Beautiful is better than ugly
- Explicit is better than implicit
- Simple is better than complex
- Flat is better than nested
- Readability counts
"""

# Core forecasting (top-level imports for simplicity)
from .core import FractalForecaster, plot_forecast, plot_forecast_interactive, print_forecast_summary

# Analysis tools (also available at top level)
from .analysis import FractalAnalyzer, CrossDimensionalAnalyzer

# Simulation (for advanced users)
from .core import FractalSimulator

# Bayesian forecasting (optional, requires PyMC)
try:
    from .bayesian import BayesianFractalForecaster
    _BAYESIAN_AVAILABLE = True
except (ImportError, NameError):
    # PyMC not installed or other import issues
    _BAYESIAN_AVAILABLE = False
    BayesianFractalForecaster = None

__version__ = "0.1.0"

# Top-level API - most commonly used classes and functions
__all__ = [
    # Main forecaster
    'FractalForecaster',

    # Visualization
    'plot_forecast_interactive',
    'plot_forecast',
    'print_forecast_summary',

    # Analysis
    'FractalAnalyzer',
    'CrossDimensionalAnalyzer',

    # Simulation (advanced)
    'FractalSimulator',
]

# Add Bayesian if available
if _BAYESIAN_AVAILABLE:
    __all__.append('BayesianFractalForecaster')
