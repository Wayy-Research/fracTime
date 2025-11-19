"""
Bayesian fractal forecasting module.

This module provides Bayesian inference capabilities for fractal time series forecasting,
including:
- PyMC models for fractal parameter estimation
- Multiple inference modes (MCMC, ADVI, hybrid)
- Parameter evolution tracking
- Expanding window analysis
"""

from .models import BayesianFractalModel
from .forecaster import BayesianFractalForecaster

__all__ = [
    'BayesianFractalModel',
    'BayesianFractalForecaster',
]
