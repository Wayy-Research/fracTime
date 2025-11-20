"""
Backtesting framework for time series forecasting models.

This module provides tools for rigorous walk-forward validation,
accuracy measurement, and overfitting detection.

Key Components:
- WalkForwardValidator: Sequential validation with expanding/rolling windows
- Metrics: RMSE, CRPS, calibration, directional accuracy
- DualPenaltyScorer: Balance accuracy vs overfitting

Example:
    >>> from fractime.backtesting import WalkForwardValidator
    >>> from fractime import FractalForecaster
    >>>
    >>> validator = WalkForwardValidator(FractalForecaster())
    >>> results = validator.run(prices, dates)
    >>>
    >>> print(f"RMSE: {results['metrics']['rmse']:.4f}")
    >>> print(f"Direction Accuracy: {results['metrics']['direction_accuracy']:.2%}")
"""

from .metrics import (
    compute_rmse,
    compute_mae,
    compute_mape,
    compute_direction_accuracy,
    compute_coverage,
    compute_crps,
    ForecastMetrics,
)

from .validator import WalkForwardValidator

from .scoring import DualPenaltyScorer, compare_models

__all__ = [
    # Validator
    'WalkForwardValidator',

    # Metrics
    'ForecastMetrics',
    'compute_rmse',
    'compute_mae',
    'compute_mape',
    'compute_direction_accuracy',
    'compute_coverage',
    'compute_crps',

    # Scoring
    'DualPenaltyScorer',
    'compare_models',
]
