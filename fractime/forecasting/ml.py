"""
Machine learning forecasting methods for time series.

This module includes machine learning forecasting methods such as:
- Random Forest
- XGBoost
- SVR (Support Vector Regression)
- KNN (K-Nearest Neighbors)
"""

import numpy as np
from typing import Any, List, Optional, Union, Tuple, Dict
import warnings

from .base import BaseForecaster

# Optional imports for ML models
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. ML forecasters will not work.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost not installed. XGBoostForecaster will not work.")


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest forecaster for time series.
    
    This is a wrapper around scikit-learn's RandomForestRegressor.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **kwargs):
        """
        Initialize the Random Forest forecaster.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            **kwargs: Additional arguments to pass to RandomForestRegressor
        """
        super().__init__(name=f"RandomForest(n={n_estimators}, depth={max_depth})")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestForecaster")
        
        # Initialize the model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestForecaster':
        """
        Fit the Random Forest model to the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: The fitted forecaster
        """
        # Fit the model
        self.model.fit(X, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Random Forest model.
        
        Args:
            X: Test features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            **self.kwargs
        }


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost forecaster for time series.
    
    This is a wrapper around XGBoost's XGBRegressor.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, **kwargs):
        """
        Initialize the XGBoost forecaster.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of the trees
            learning_rate: Learning rate (shrinkage)
            **kwargs: Additional arguments to pass to XGBRegressor
        """
        super().__init__(name=f"XGBoost(n={n_estimators}, depth={max_depth}, lr={learning_rate})")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoostForecaster")
        
        # Initialize the model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostForecaster':
        """
        Fit the XGBoost model to the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: The fitted forecaster
        """
        # Fit the model
        self.model.fit(X, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted XGBoost model.
        
        Args:
            X: Test features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            **self.kwargs
        }


class SVRForecaster(BaseForecaster):
    """
    Support Vector Regression forecaster for time series.
    
    This is a wrapper around scikit-learn's SVR.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1, **kwargs):
        """
        Initialize the SVR forecaster.
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            epsilon: Epsilon in the epsilon-SVR model
            **kwargs: Additional arguments to pass to SVR
        """
        super().__init__(name=f"SVR(kernel={kernel}, C={C}, epsilon={epsilon})")
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.kwargs = kwargs
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SVRForecaster")
        
        # Initialize the model
        self.model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRForecaster':
        """
        Fit the SVR model to the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: The fitted forecaster
        """
        # Fit the model
        self.model.fit(X, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted SVR model.
        
        Args:
            X: Test features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            **self.kwargs
        }


class KNNForecaster(BaseForecaster):
    """
    K-Nearest Neighbors forecaster for time series.
    
    This is a wrapper around scikit-learn's KNeighborsRegressor.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', **kwargs):
        """
        Initialize the KNN forecaster.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform', 'distance')
            **kwargs: Additional arguments to pass to KNeighborsRegressor
        """
        super().__init__(name=f"KNN(n={n_neighbors}, weights={weights})")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.kwargs = kwargs
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for KNNForecaster")
        
        # Initialize the model
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNForecaster':
        """
        Fit the KNN model to the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: The fitted forecaster
        """
        # Fit the model
        self.model.fit(X, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted KNN model.
        
        Args:
            X: Test features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            **self.kwargs
        }