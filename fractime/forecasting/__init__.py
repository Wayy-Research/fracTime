"""
FracTime Forecasting Module

This module contains various time series forecasting methods, including:
- Traditional statistical methods (ARIMA, ETS, etc.)
- Fractal forecasting methods (mentioned in the research paper)
- Machine learning methods (Random Forest, XGBoost, etc.)
- Neural network methods (LSTM, etc.)
"""

from .base import BaseForecaster
from .statistical import ARIMAForecaster, SARIMAForecaster, ExponentialSmoothingForecaster
from .fractal import (
    StateTransitionFRSRForecaster,
    FractalProjectionForecaster,
    FractalClassificationForecaster,
    RescaledRangeForecaster,
    FractalInterpolationForecaster,
    FractalReductionForecaster
)
from .ml import (
    RandomForestForecaster,
    XGBoostForecaster,
    SVRForecaster,
    KNNForecaster
)

# Re-export everything
__all__ = [
    # Base classes
    'BaseForecaster',
    
    # Statistical forecasters
    'ARIMAForecaster',
    'SARIMAForecaster',
    'ExponentialSmoothingForecaster',
    
    # Fractal forecasters
    'StateTransitionFRSRForecaster',
    'FractalProjectionForecaster',
    'FractalClassificationForecaster',
    'RescaledRangeForecaster',
    'FractalInterpolationForecaster',
    'FractalReductionForecaster',
    
    # Machine learning forecasters
    'RandomForestForecaster',
    'XGBoostForecaster',
    'SVRForecaster',
    'KNNForecaster'
]