"""
Statistical forecasting methods for time series.

This module includes traditional statistical forecasting methods such as:
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)
- Exponential Smoothing
"""

import numpy as np
from typing import Any, List, Optional, Union, Tuple, Dict
import warnings

from .base import BaseForecaster

# Optional imports for statistical models
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. Statistical forecasters will not work.")


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.
    
    This is a wrapper around statsmodels ARIMA implementation.
    """
    
    def __init__(self, p: int = 1, d: int = 0, q: int = 0, **kwargs):
        """
        Initialize the ARIMA forecaster.
        
        Args:
            p: Order of the AR model (number of lag observations)
            d: Degree of differencing
            q: Order of the MA model (size of the moving average window)
            **kwargs: Additional arguments to pass to statsmodels ARIMA
        """
        super().__init__(name=f"ARIMA({p},{d},{q})")
        self.p = p
        self.d = d
        self.q = q
        self.kwargs = kwargs
        self.model = None
        self.result = None
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMAForecaster")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ARIMAForecaster':
        """
        Fit the ARIMA model to the training data.
        
        Args:
            X: Training features (not used in this model)
            y: Training target (time series values)
            
        Returns:
            self: The fitted forecaster
        """
        # ARIMA uses only the target values, not features
        # Initialize the model
        self.model = sm.tsa.ARIMA(y, order=(self.p, self.d, self.q), **self.kwargs)
        
        # Fit the model
        self.result = self.model.fit()
        
        # Store the last values for prediction
        self.last_y = y
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted ARIMA model.
        
        Args:
            X: Test features (not used in this model)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Make predictions
        forecast = self.result.forecast(steps=n_periods)
        
        return forecast
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted ARIMA model.
        
        Args:
            X: Test features (not used in this model)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions
        forecast = self.result.forecast(steps=n_steps)
        
        return forecast
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            **self.kwargs
        }


class SARIMAForecaster(BaseForecaster):
    """
    SARIMA (Seasonal ARIMA) forecaster.
    
    This is a wrapper around statsmodels SARIMAX implementation.
    """
    
    def __init__(self, p: int = 1, d: int = 0, q: int = 0, 
                 P: int = 0, D: int = 0, Q: int = 0, s: int = 0, **kwargs):
        """
        Initialize the SARIMA forecaster.
        
        Args:
            p: Order of the AR model (non-seasonal)
            d: Degree of differencing (non-seasonal)
            q: Order of the MA model (non-seasonal)
            P: Order of the AR model (seasonal)
            D: Degree of differencing (seasonal)
            Q: Order of the MA model (seasonal)
            s: Length of the seasonal cycle
            **kwargs: Additional arguments to pass to statsmodels SARIMAX
        """
        super().__init__(name=f"SARIMA({p},{d},{q})({P},{D},{Q}){s}")
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.kwargs = kwargs
        self.model = None
        self.result = None
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMAForecaster")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SARIMAForecaster':
        """
        Fit the SARIMA model to the training data.
        
        Args:
            X: Training features (not used in this model)
            y: Training target (time series values)
            
        Returns:
            self: The fitted forecaster
        """
        # SARIMA uses only the target values, not features
        # Initialize the model
        order = (self.p, self.d, self.q)
        seasonal_order = (self.P, self.D, self.Q, self.s) if self.s > 0 else None
        
        self.model = sm.tsa.SARIMAX(y, order=order, seasonal_order=seasonal_order, **self.kwargs)
        
        # Fit the model
        self.result = self.model.fit(disp=False)
        
        # Store the last values for prediction
        self.last_y = y
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted SARIMA model.
        
        Args:
            X: Test features (not used in this model)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Make predictions
        forecast = self.result.forecast(steps=n_periods)
        
        return forecast
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted SARIMA model.
        
        Args:
            X: Test features (not used in this model)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions
        forecast = self.result.forecast(steps=n_steps)
        
        return forecast
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'P': self.P,
            'D': self.D,
            'Q': self.Q,
            's': self.s,
            **self.kwargs
        }


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Exponential Smoothing forecaster.
    
    This is a wrapper around statsmodels ExponentialSmoothing implementation.
    """
    
    def __init__(self, trend: Optional[str] = None, seasonal: Optional[str] = None, 
                 seasonal_periods: Optional[int] = None, **kwargs):
        """
        Initialize the Exponential Smoothing forecaster.
        
        Args:
            trend: Type of trend component ('add', 'mul', 'additive', 'multiplicative', None)
            seasonal: Type of seasonal component ('add', 'mul', 'additive', 'multiplicative', None)
            seasonal_periods: Number of periods in a complete seasonal cycle
            **kwargs: Additional arguments to pass to statsmodels ExponentialSmoothing
        """
        name_parts = ["ES"]
        if trend:
            name_parts.append(f"trend={trend}")
        if seasonal:
            name_parts.append(f"seasonal={seasonal}")
        if seasonal_periods:
            name_parts.append(f"periods={seasonal_periods}")
        
        super().__init__(name="-".join(name_parts))
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.kwargs = kwargs
        self.model = None
        self.result = None
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ExponentialSmoothingForecaster")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExponentialSmoothingForecaster':
        """
        Fit the Exponential Smoothing model to the training data.
        
        Args:
            X: Training features (not used in this model)
            y: Training target (time series values)
            
        Returns:
            self: The fitted forecaster
        """
        # ExponentialSmoothing uses only the target values, not features
        # Initialize the model
        self.model = sm.tsa.ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            **self.kwargs
        )
        
        # Fit the model
        self.result = self.model.fit()
        
        # Store the last values for prediction
        self.last_y = y
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Exponential Smoothing model.
        
        Args:
            X: Test features (not used in this model)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Make predictions
        forecast = self.result.forecast(steps=n_periods)
        
        return forecast
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted Exponential Smoothing model.
        
        Args:
            X: Test features (not used in this model)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions
        forecast = self.result.forecast(steps=n_steps)
        
        return forecast
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the forecaster.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            **self.kwargs
        }