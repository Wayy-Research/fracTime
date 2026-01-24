"""
FracTime: Fractal-based time series analysis and forecasting.

Simple, composable API:

    >>> import fractime as ft

    # Analyze
    >>> analyzer = ft.Analyzer(prices)
    >>> analyzer.hurst                    # Point estimate
    >>> analyzer.hurst.rolling            # Rolling values
    >>> analyzer.hurst.ci(0.95)           # Confidence interval

    # Forecast
    >>> model = ft.Forecaster(prices)
    >>> result = model.predict(steps=30)
    >>> result.forecast                   # Primary forecast
    >>> result.ci(0.95)                   # Confidence interval

    # Plot
    >>> ft.plot(result)

All components are composable:
    - Analyzer: Compute fractal properties (Hurst, fractal dim, volatility)
    - Simulator: Generate Monte Carlo paths
    - Forecaster: Probabilistic forecasting
    - Ensemble: Combine multiple models
"""

# Core classes
from .analyzer import Analyzer, analyze
from .forecaster import Forecaster, forecast
from .simulator import Simulator
from .ensemble import Ensemble

# Result types
from .result import Metric, AnalysisResult, ForecastResult

# Visualization
from .visualization import plot, plot_forecast

# Bayesian (optional - requires PyMC)
try:
    from .bayesian import BayesianFractalForecaster as BayesianForecaster
    _BAYESIAN_AVAILABLE = True
except (ImportError, ModuleNotFoundError, NameError, Exception):
    _BAYESIAN_AVAILABLE = False
    BayesianForecaster = None

__version__ = "0.4.0"

__all__ = [
    # Core classes
    'Analyzer',
    'Forecaster',
    'Simulator',
    'Ensemble',

    # Convenience functions
    'analyze',
    'forecast',
    'plot',
    'plot_forecast',

    # Result types
    'Metric',
    'AnalysisResult',
    'ForecastResult',
]

# Add Bayesian if available
if _BAYESIAN_AVAILABLE:
    __all__.append('BayesianForecaster')
