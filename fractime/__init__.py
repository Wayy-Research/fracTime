"""
FracTime: Fractal-based time series analysis and forecasting.

Simple, composable API:

    >>> import fractime as ft

    # Analyze (supports rs, dfa, wavelet methods)
    >>> analyzer = ft.Analyzer(prices, method='dfa')
    >>> analyzer.hurst                    # Point estimate
    >>> analyzer.hurst.rolling            # Rolling values
    >>> analyzer.hurst.ci(0.95)           # Confidence interval

    # Forecast
    >>> model = ft.Forecaster(prices)
    >>> result = model.predict(steps=30)
    >>> result.forecast                   # Primary forecast
    >>> result.ci(0.95)                   # Confidence interval

    # HMM Regime Detection (bull/bear/sideways)
    >>> detector = ft.RegimeDetector(n_regimes=2)
    >>> detector.fit(returns)
    >>> regime, prob = detector.get_current_regime(returns)

    # Hurst Regime Persistence (trending/random/mean-reverting)
    >>> hra = ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
    >>> hra.current_regime                # 'trending'
    >>> hra.recommended_horizon           # 30

    # Regime-Based Strategy
    >>> strategy = ft.RegimeStrategy(bull_allocation=1.0, bear_allocation=0.3)
    >>> results = strategy.backtest(prices)
    >>> print(f"Sharpe: {results.sharpe:.2f}")

    # Plot
    >>> ft.plot(result)

All components are composable:
    - Analyzer: Fractal properties (Hurst via RS/DFA/wavelet, fractal dim, volatility)
    - Forecaster: Probabilistic forecasting via Monte Carlo
    - Simulator: Direct path generation (fBm, pattern, bootstrap)
    - Ensemble: Multi-model combination (average, weighted, stacking, boosting)
    - RegimeDetector: HMM-based market regime detection (bull/bear)
    - RegimeStrategy: Regime-based position sizing and backtesting
    - HurstRegimeAnalyzer: Hurst regime persistence and forecast horizon estimation
    - Advanced analytics: MF-ADCCA, QPL, fractal coherence, FTD, DTW beta
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

# Hurst regime persistence
from .hurst_regime import HurstRegimeAnalyzer

# Advanced analytics (NEW in v0.6)
from .advanced import (
    compute_wavelet_hurst,
    compute_rolling_wavelet_hurst,
    compute_mf_adcca,
    compute_qpl,
    compute_fractal_coherence,
    compute_rolling_fractal_coherence,
    compute_ftd,
    compute_dtw_alignment,
    compute_dtw_beta,
)

# Bayesian (optional - requires PyMC)
try:
    from .bayesian import BayesianFractalForecaster as BayesianForecaster
    _BAYESIAN_AVAILABLE = True
except (ImportError, ModuleNotFoundError, NameError, Exception):
    _BAYESIAN_AVAILABLE = False
    BayesianForecaster = None

__version__ = "0.7.0"

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

    # Hurst regime persistence
    'HurstRegimeAnalyzer',

    # Advanced analytics (NEW in v0.6)
    'compute_wavelet_hurst',
    'compute_rolling_wavelet_hurst',
    'compute_mf_adcca',
    'compute_qpl',
    'compute_fractal_coherence',
    'compute_rolling_fractal_coherence',
    'compute_ftd',
    'compute_dtw_alignment',
    'compute_dtw_beta',

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
