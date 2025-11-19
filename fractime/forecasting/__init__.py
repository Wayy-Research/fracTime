"""
FracTime Forecasting Module

Core fractal-based forecasting methods.

For most use cases, use the unified FractalForecaster from fractime.FractalForecaster
which combines the best fractal techniques automatically.
"""

from .base import BaseForecaster
from .statistical import ARIMAForecaster, ExponentialSmoothingForecaster
from .fractal import (
    StateTransitionFRSRForecaster,
    FractalProjectionForecaster,
    FractalClassificationForecaster
)

# Re-export core classes
__all__ = [
    # Base class
    'BaseForecaster',

    # Statistical baselines (for comparison)
    'ARIMAForecaster',
    'ExponentialSmoothingForecaster',

    # Core fractal forecasters
    'StateTransitionFRSRForecaster',
    'FractalProjectionForecaster',
    'FractalClassificationForecaster'
]

# NOTE: For most use cases, use fractime.FractalForecaster instead
# of these individual forecasters. The unified forecaster combines
# fractal analysis, pattern recognition, and regime detection automatically.