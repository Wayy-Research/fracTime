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
from .core import FractalForecaster

# Analysis tools (also available at top level)
from .analysis import FractalAnalyzer, CrossDimensionalAnalyzer

# Simulation tools (refactored into separate module)
from .simulation import FractalSimulator, TradingTimeWarper, PathAnalyzer

# Exogenous predictors
from .exogenous import (
    ExogenousHandler,
    ExogenousRegimeModifier,
    ExogenousForecastAdjuster,
    compute_exogenous_fractal_coherence
)

# Visualization tools (refactored into separate module)
from .visualization import (
    FractalVisualizer,
    plot_forecast,
    plot_forecast_interactive,
    print_forecast_summary
)

# Ensemble methods (advanced)
from .ensemble import StackingForecaster, BoostingForecaster

# Utility functions
from .utils import get_yahoo_data

# Bayesian forecasting (optional, requires PyMC)
try:
    from .bayesian import BayesianFractalForecaster
    _BAYESIAN_AVAILABLE = True
except (ImportError, NameError):
    # PyMC not installed or other import issues
    _BAYESIAN_AVAILABLE = False
    BayesianFractalForecaster = None

__version__ = "0.2.0"

# Top-level API - most commonly used classes and functions
__all__ = [
    # Main forecaster
    'FractalForecaster',

    # Visualization
    'plot_forecast_interactive',
    'plot_forecast',
    'print_forecast_summary',
    'FractalVisualizer',

    # Analysis
    'FractalAnalyzer',
    'CrossDimensionalAnalyzer',

    # Simulation (advanced)
    'FractalSimulator',
    'TradingTimeWarper',
    'PathAnalyzer',

    # Exogenous predictors
    'ExogenousHandler',
    'ExogenousRegimeModifier',
    'ExogenousForecastAdjuster',
    'compute_exogenous_fractal_coherence',

    # Ensemble methods (advanced)
    'StackingForecaster',
    'BoostingForecaster',

    # Utilities
    'get_yahoo_data',
]

# Add Bayesian if available
if _BAYESIAN_AVAILABLE:
    __all__.append('BayesianFractalForecaster')
