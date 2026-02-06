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

    # Regime Detection (NEW in v0.5)
    >>> detector = ft.RegimeDetector(n_regimes=2)
    >>> detector.fit(returns)
    >>> regime, prob = detector.get_current_regime(returns)

    # Regime-Based Strategy (NEW in v0.5)
    >>> strategy = ft.RegimeStrategy(bull_allocation=1.0, bear_allocation=0.3)
    >>> results = strategy.backtest(prices)
    >>> print(f"Sharpe: {results.sharpe:.2f}")

    # Plot
    >>> ft.plot(result)

All components are composable:
    - Analyzer: Compute fractal properties (Hurst, fractal dim, volatility)
    - Simulator: Generate Monte Carlo paths
    - Forecaster: Probabilistic forecasting
    - Ensemble: Combine multiple models
    - RegimeDetector: HMM-based market regime detection
    - RegimeStrategy: Regime-based position sizing
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

# Regime detection and strategy (NEW in v0.5)
from .regime import RegimeDetector
from .strategy import RegimeStrategy, BacktestResult, quick_backtest

# Bayesian (optional - requires PyMC)
try:
    from .bayesian import BayesianFractalForecaster as BayesianForecaster
    _BAYESIAN_AVAILABLE = True
except (ImportError, ModuleNotFoundError, NameError, Exception):
    _BAYESIAN_AVAILABLE = False
    BayesianForecaster = None

__version__ = "0.5.1"

__all__ = [
    # Core classes
    'Analyzer',
    'Forecaster',
    'Simulator',
    'Ensemble',

    # Regime detection and strategy (NEW in v0.5)
    'RegimeDetector',
    'RegimeStrategy',
    'BacktestResult',
    'quick_backtest',

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
