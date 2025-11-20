"""
Model selection framework for automated forecasting.

This module provides tools for:
- Model Registry: Catalog all available models
- Auto-Selection: Automatically select the best model for a dataset
- Statistical Testing: Test significance of performance differences
- Ensemble Methods: Combine multiple models

Example:
    >>> from fractime.selection import AutoSelector
    >>>
    >>> # Automatically select best model
    >>> selector = AutoSelector()
    >>> best_model = selector.select_best(prices, dates)
    >>>
    >>> print(f"Best model: {best_model.name}")
    >>> print(f"Score: {best_model.score:.4f}")
"""

from .registry import ModelRegistry, register_model, get_global_registry
from .selector import AutoSelector, SelectionResult
from .statistical_tests import diebold_mariano_test, model_confidence_set
from .ensemble import EnsembleForecaster, WeightedEnsemble, create_ensemble

__all__ = [
    # Registry
    'ModelRegistry',
    'register_model',
    'get_global_registry',

    # Selection
    'AutoSelector',
    'SelectionResult',

    # Statistical Tests
    'diebold_mariano_test',
    'model_confidence_set',

    # Ensemble
    'EnsembleForecaster',
    'WeightedEnsemble',
    'create_ensemble',
]
