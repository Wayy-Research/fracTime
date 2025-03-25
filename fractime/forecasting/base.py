"""
Base class for all forecasting models in FracTime.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Tuple, Dict


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the interface that all forecasting models must implement.
    It includes methods for fitting, predicting, and evaluating models.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the forecaster.
        
        Args:
            name: Optional name for the forecaster
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        """
        Fit the forecaster to the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: The fitted forecaster
        """
        # Implementation should set self.is_fitted = True when done
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted forecaster.
        
        Args:
            X: Test features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Forecaster has not been fitted yet.")
        pass
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions.
        
        This is a default implementation that recursively calls predict().
        Subclasses should override this if they have a more efficient implementation.
        
        Args:
            X: Features for the starting point
            n_steps: Number of steps to predict ahead
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Forecaster has not been fitted yet.")
        
        # Make sure X is the right shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Initialize the predictions array
        predictions = np.zeros(n_steps)
        
        # Make the first prediction
        current_X = X.copy()
        predictions[0] = self.predict(current_X)[0]
        
        # Recursively predict the rest
        for i in range(1, n_steps):
            # Update X based on the previous prediction
            # This assumes a direct autoregressive structure
            # Subclasses should override this for more complex feature structures
            current_X = self._update_features(current_X, predictions[i-1])
            predictions[i] = self.predict(current_X)[0]
        
        return predictions
    
    def _update_features(self, X: np.ndarray, last_prediction: float) -> np.ndarray:
        """
        Update the features for the next prediction step based on the last prediction.
        
        This is a simple default implementation that assumes the last column is the target
        and other columns are lagged values of the target. Subclasses should override this
        for more complex feature structures.
        
        Args:
            X: Current feature array
            last_prediction: Last prediction value
            
        Returns:
            Updated feature array
        """
        # Default implementation: assume features are lagged values
        new_X = X.copy()
        n_features = X.shape[1]
        
        # Shift features (assuming they are ordered from most recent to oldest)
        for i in range(n_features - 1, 0, -1):
            new_X[0, i] = new_X[0, i-1]
        
        # Set the most recent value to the last prediction
        new_X[0, 0] = last_prediction
        
        return new_X
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {}
    
    def set_params(self, **params) -> 'BaseForecaster':
        """
        Set the parameters of the forecaster.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The forecaster with updated parameters
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self