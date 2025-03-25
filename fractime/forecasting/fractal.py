"""
Fractal forecasting methods for time series.

This module includes fractal-based forecasting methods such as:
- State Transition-Fitted Residual Scale Ratio (ST-FRSR)
- Fractal Projection Algorithm
- Fractal Classification Scheme
- Rescaled Range (R/S) Analysis-based forecasting
- Fractal Interpolation
- Fractal Reduction with Binary Gate Logic
"""

import numpy as np
from typing import Any, List, Optional, Union, Tuple, Dict
import warnings
from scipy import stats
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class StateTransitionFRSRForecaster(BaseForecaster):
    """
    State Transition-Fitted Residual Scale Ratio (ST-FRSR) forecaster.
    
    This model combines a state transition model for predicting extreme events
    with a scale ratio analysis to capture the scaling symmetries in the time series.
    
    Reference: From the research paper, this forecasts intraday exchange rate futures contracts.
    """
    
    def __init__(self, n_states: int = 3, window_size: int = 10, 
                 transition_threshold: float = 0.95, **kwargs):
        """
        Initialize the ST-FRSR forecaster.
        
        Args:
            n_states: Number of states in the model
            window_size: Size of the window for computing scale ratios
            transition_threshold: Threshold for state transitions
            **kwargs: Additional parameters
        """
        super().__init__(name=f"ST-FRSR(states={n_states}, window={window_size})")
        self.n_states = n_states
        self.window_size = window_size
        self.transition_threshold = transition_threshold
        self.kwargs = kwargs
        
        # Model components will be initialized during fitting
        self.state_kmeans = KMeans(n_clusters=n_states)
        self.scaler = StandardScaler()
        self.transition_matrix = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StateTransitionFRSRForecaster':
        """
        Fit the ST-FRSR model to the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: The fitted forecaster
        """
        # Extract features for state identification
        self.feature_names = ['volatility', 'trend', 'scale_ratio']
        
        # Compute features for the time series
        features = self._compute_features(y)
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Cluster the features into states
        self.state_kmeans.fit(scaled_features)
        states = self.state_kmeans.predict(scaled_features)
        
        # Compute state transition matrix
        self.transition_matrix = self._compute_transition_matrix(states)
        
        # Compute scale ratios for each state
        self.state_scale_ratios = self._compute_state_scale_ratios(y, states)
        
        # Store the last state and values for prediction
        self.last_state = states[-1]
        self.last_values = y[-self.window_size:]
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted ST-FRSR model.
        
        Args:
            X: Test features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Initialize predictions
        predictions = np.zeros(n_periods)
        
        # Current state and values
        current_state = self.last_state
        current_values = self.last_values.copy()
        
        # Make predictions for each period
        for i in range(n_periods):
            # Determine the next state based on transition probabilities
            next_state = self._predict_next_state(current_state)
            
            # Get the scale ratio for the current state
            scale_ratio = self.state_scale_ratios[current_state]
            
            # Compute the next value based on the scale ratio
            next_value = self._predict_with_scale_ratio(current_values, scale_ratio)
            
            # Store the prediction
            predictions[i] = next_value
            
            # Update the current state and values
            current_state = next_state
            current_values = np.append(current_values[1:], next_value)
        
        return predictions
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted ST-FRSR model.
        
        Args:
            X: Test features (only the first row is used)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Initialize predictions
        predictions = np.zeros(n_steps)
        
        # Current state and values
        current_state = self.last_state
        current_values = self.last_values.copy()
        
        # Make predictions for each step
        for i in range(n_steps):
            # Determine the next state based on transition probabilities
            next_state = self._predict_next_state(current_state)
            
            # Get the scale ratio for the current state
            scale_ratio = self.state_scale_ratios[current_state]
            
            # Compute the next value based on the scale ratio
            next_value = self._predict_with_scale_ratio(current_values, scale_ratio)
            
            # Store the prediction
            predictions[i] = next_value
            
            # Update the current state and values
            current_state = next_state
            current_values = np.append(current_values[1:], next_value)
        
        return predictions
    
    def _compute_features(self, y: np.ndarray) -> np.ndarray:
        """
        Compute features for state identification.
        
        Args:
            y: Time series values
            
        Returns:
            Features array with shape (n_samples - window_size, n_features)
        """
        n_samples = len(y)
        features = np.zeros((n_samples - self.window_size, 3))
        
        for i in range(self.window_size, n_samples):
            window = y[i-self.window_size:i]
            
            # Compute volatility (standard deviation of returns)
            returns = np.diff(window) / window[:-1]
            volatility = np.std(returns)
            
            # Compute trend (slope of linear fit)
            x = np.arange(self.window_size)
            slope, _, _, _, _ = stats.linregress(x, window)
            
            # Compute scale ratio (ratio of rates of change at proximate separation distances)
            scale_ratio = self._compute_scale_ratio(window)
            
            # Store features
            features[i-self.window_size] = [volatility, slope, scale_ratio]
        
        return features
    
    def _compute_scale_ratio(self, window: np.ndarray) -> float:
        """
        Compute the scale ratio for a window of time series data.
        
        Args:
            window: Window of time series values
            
        Returns:
            Scale ratio
        """
        # Compute rates of change at different separation distances
        rate1 = np.diff(window[::2]) / 2
        rate2 = np.diff(window[::1])
        
        # Compute the ratio of rates of change
        if len(rate1) > 0 and len(rate2) > 0:
            ratio = np.mean(np.abs(rate1)) / (np.mean(np.abs(rate2)) + 1e-10)
        else:
            ratio = 1.0
        
        return ratio
    
    def _compute_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Compute the state transition matrix.
        
        Args:
            states: Array of state indices
            
        Returns:
            Transition matrix with shape (n_states, n_states)
        """
        n_states = self.n_states
        transition_matrix = np.zeros((n_states, n_states))
        
        # Count transitions
        for i in range(len(states) - 1):
            transition_matrix[states[i], states[i+1]] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_matrix, 
            row_sums, 
            out=np.zeros_like(transition_matrix), 
            where=row_sums != 0
        )
        
        return transition_matrix
    
    def _compute_state_scale_ratios(self, y: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Compute the average scale ratio for each state.
        
        Args:
            y: Time series values
            states: Array of state indices
            
        Returns:
            Array of scale ratios for each state
        """
        n_states = self.n_states
        state_scale_ratios = np.zeros(n_states)
        state_counts = np.zeros(n_states)
        
        # Compute scale ratios for each window
        for i in range(self.window_size, len(y)):
            window = y[i-self.window_size:i]
            state = states[i-self.window_size]
            
            scale_ratio = self._compute_scale_ratio(window)
            
            # Accumulate scale ratios for each state
            state_scale_ratios[state] += scale_ratio
            state_counts[state] += 1
        
        # Compute average scale ratio for each state
        state_scale_ratios = np.divide(
            state_scale_ratios, 
            state_counts, 
            out=np.ones_like(state_scale_ratios), 
            where=state_counts != 0
        )
        
        return state_scale_ratios
    
    def _predict_next_state(self, current_state: int) -> int:
        """
        Predict the next state based on transition probabilities.
        
        Args:
            current_state: Current state index
            
        Returns:
            Predicted next state
        """
        transition_probs = self.transition_matrix[current_state]
        
        # Check if the highest probability exceeds the threshold
        if np.max(transition_probs) > self.transition_threshold:
            # Deterministic transition to the most likely state
            next_state = np.argmax(transition_probs)
        else:
            # Probabilistic transition based on transition probabilities
            next_state = np.random.choice(self.n_states, p=transition_probs)
        
        return next_state
    
    def _predict_with_scale_ratio(self, values: np.ndarray, scale_ratio: float) -> float:
        """
        Predict the next value using the scale ratio.
        
        Args:
            values: Current window of values
            scale_ratio: Scale ratio for the current state
            
        Returns:
            Predicted next value
        """
        # Compute the change at different scales
        half_window = len(values) // 2
        
        # Compute the change over the full window
        full_change = values[-1] - values[0]
        
        # Compute the change over the second half of the window
        half_change = values[-1] - values[half_window]
        
        # Predict the change for the next point using the scale ratio
        next_change = half_change * scale_ratio
        
        # Predict the next value
        next_value = values[-1] + next_change
        
        return next_value


class FractalProjectionForecaster(BaseForecaster):
    """
    Fractal Projection Algorithm forecaster.
    
    This model identifies recurring patterns in the time series and projects them
    forward in time to generate a forecast.
    
    Reference: From the research paper, this is useful for limited historical data
    and for time series with semi-periodic behavior and recursive substructures.
    """
    
    def __init__(self, pattern_length: int = 10, similarity_threshold: float = 0.8, 
                 smoothing: float = 0.5, **kwargs):
        """
        Initialize the Fractal Projection forecaster.
        
        Args:
            pattern_length: Length of the pattern to look for
            similarity_threshold: Threshold for pattern similarity
            smoothing: Smoothing factor for blending multiple pattern matches
            **kwargs: Additional parameters
        """
        super().__init__(name=f"FractalProjection(length={pattern_length}, threshold={similarity_threshold})")
        self.pattern_length = pattern_length
        self.similarity_threshold = similarity_threshold
        self.smoothing = smoothing
        self.kwargs = kwargs
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FractalProjectionForecaster':
        """
        Fit the Fractal Projection model to the training data.
        
        Args:
            X: Training features (not used in this model)
            y: Training target (time series values)
            
        Returns:
            self: The fitted forecaster
        """
        # Store the time series
        self.time_series = y
        
        # Identify the most recent pattern
        self.recent_pattern = y[-self.pattern_length:]
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Fractal Projection model.
        
        Args:
            X: Test features (not used in this model)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Find similar patterns in the historical data
        similar_patterns = self._find_similar_patterns()
        
        # If no similar patterns found, use naive projection
        if len(similar_patterns) == 0:
            # Simple trend extrapolation
            slope = (self.recent_pattern[-1] - self.recent_pattern[0]) / self.pattern_length
            predictions = np.array([self.recent_pattern[-1] + (i+1) * slope for i in range(n_periods)])
            return predictions
        
        # Initialize predictions
        predictions = np.zeros(n_periods)
        
        # Weight the contributions of each similar pattern
        similarity_weights = np.array([similarity for _, similarity in similar_patterns])
        similarity_weights = similarity_weights / np.sum(similarity_weights)
        
        # Generate predictions from each similar pattern and blend them
        for i in range(n_periods):
            # For each prediction step
            step_prediction = 0.0
            total_weight = 0.0
            
            for (pattern_idx, similarity) in similar_patterns:
                # Get the subsequent value from the historical pattern
                if pattern_idx + self.pattern_length + i < len(self.time_series):
                    subsequent_value = self.time_series[pattern_idx + self.pattern_length + i]
                    step_prediction += subsequent_value * similarity
                    total_weight += similarity
            
            # Normalize if we got predictions
            if total_weight > 0:
                predictions[i] = step_prediction / total_weight
            else:
                # Fallback to simple trend extrapolation
                slope = (self.recent_pattern[-1] - self.recent_pattern[0]) / self.pattern_length
                predictions[i] = self.recent_pattern[-1] + (i+1) * slope
        
        # Apply smoothing
        if n_periods > 1:
            predictions = gaussian_filter1d(predictions, sigma=self.smoothing)
        
        return predictions
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted Fractal Projection model.
        
        Args:
            X: Test features (only the first row is used)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Create a dummy X with appropriate shape
        dummy_X = np.zeros((n_steps, X.shape[1] if X.ndim > 1 else 1))
        
        # Use the predict method to generate forecasts
        return self.predict(dummy_X)
    
    def _find_similar_patterns(self) -> List[Tuple[int, float]]:
        """
        Find patterns in the historical data that are similar to the recent pattern.
        
        Returns:
            List of tuples (pattern_index, similarity_score)
        """
        similar_patterns = []
        
        # Loop through the time series to find similar patterns
        for i in range(len(self.time_series) - self.pattern_length - 1):
            # Extract the candidate pattern
            candidate = self.time_series[i:i+self.pattern_length]
            
            # Compute similarity score (normalized cross-correlation)
            similarity = self._compute_similarity(self.recent_pattern, candidate)
            
            # If similarity is above threshold, add to the list
            if similarity > self.similarity_threshold:
                similar_patterns.append((i, similarity))
        
        # Sort by similarity (descending)
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patterns
    
    def _compute_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Compute the similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (normalized cross-correlation)
        """
        # Normalize the patterns
        pattern1_norm = (pattern1 - np.mean(pattern1)) / (np.std(pattern1) + 1e-10)
        pattern2_norm = (pattern2 - np.mean(pattern2)) / (np.std(pattern2) + 1e-10)
        
        # Compute cross-correlation
        correlation = np.correlate(pattern1_norm, pattern2_norm, mode='valid')[0] / len(pattern1)
        
        # Convert to similarity score (0 to 1)
        similarity = (correlation + 1) / 2
        
        return similarity


class FractalClassificationForecaster(BaseForecaster):
    """
    Fractal Classification Scheme forecaster.
    
    This model transforms the time series into a sequence of classes based on a
    fractal scheme and then predicts the next class in the sequence.
    
    Reference: From the research paper, this was tested on monthly average temperature
    and showed more accurate forecasts compared to the naive method.
    """
    
    def __init__(self, n_classes: int = 4, window_size: int = 5, **kwargs):
        """
        Initialize the Fractal Classification forecaster.
        
        Args:
            n_classes: Number of classes to divide the data into
            window_size: Size of the window for computing features
            **kwargs: Additional parameters
        """
        super().__init__(name=f"FractalClassification(classes={n_classes}, window={window_size})")
        self.n_classes = n_classes
        self.window_size = window_size
        self.kwargs = kwargs
        
        # Model components will be initialized during fitting
        self.kmeans = KMeans(n_clusters=n_classes)
        self.scaler = StandardScaler()
        self.transition_matrix = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FractalClassificationForecaster':
        """
        Fit the Fractal Classification model to the training data.
        
        Args:
            X: Training features (not used in this model)
            y: Training target (time series values)
            
        Returns:
            self: The fitted forecaster
        """
        # Compute features for each window
        features = self._compute_window_features(y)
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Cluster the features into classes
        self.kmeans.fit(scaled_features)
        classes = self.kmeans.predict(scaled_features)
        
        # Compute class centroids in the original space
        self.class_centroids = np.zeros((self.n_classes, 1))
        for i in range(self.n_classes):
            class_indices = np.where(classes == i)[0]
            if len(class_indices) > 0:
                class_start_indices = class_indices + self.window_size
                class_start_indices = class_start_indices[class_start_indices < len(y)]
                if len(class_start_indices) > 0:
                    self.class_centroids[i] = np.mean(y[class_start_indices])
                else:
                    self.class_centroids[i] = np.mean(y)
            else:
                self.class_centroids[i] = np.mean(y)
        
        # Compute the transition matrix
        self.transition_matrix = self._compute_transition_matrix(classes)
        
        # Store the class sequence and last window
        self.class_sequence = classes
        self.last_window = y[-self.window_size:]
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Fractal Classification model.
        
        Args:
            X: Test features (not used in this model)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Get the most recent class
        recent_features = self._compute_window_features(self.last_window)
        recent_features_scaled = self.scaler.transform(recent_features)
        current_class = self.kmeans.predict(recent_features_scaled)[0]
        
        # Initialize predictions
        predictions = np.zeros(n_periods)
        
        # Make predictions for each period
        for i in range(n_periods):
            # Predict the next class
            next_class = self._predict_next_class(current_class)
            
            # Get the centroid value for the predicted class
            predictions[i] = self.class_centroids[next_class][0]
            
            # Update the current class for the next iteration
            current_class = next_class
            
            # If we're making more predictions, we'd need to update the window
            # However, this is a simplified implementation
        
        return predictions
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted Fractal Classification model.
        
        Args:
            X: Test features (only the first row is used)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Create a dummy X with appropriate shape
        dummy_X = np.zeros((n_steps, X.shape[1] if X.ndim > 1 else 1))
        
        # Use the predict method to generate forecasts
        return self.predict(dummy_X)
    
    def _compute_window_features(self, y: np.ndarray) -> np.ndarray:
        """
        Compute features for each window of the time series.
        
        Args:
            y: Time series values
            
        Returns:
            Feature matrix
        """
        if len(y) < self.window_size:
            # Handle the case where y is too short
            y_extended = np.pad(y, (self.window_size - len(y), 0), mode='edge')
            windows = np.array([y_extended])
        else:
            # Create rolling windows
            windows = np.array([y[i:i+self.window_size] for i in range(len(y) - self.window_size + 1)])
        
        # Compute features for each window
        n_windows = windows.shape[0]
        features = np.zeros((n_windows, 4))
        
        for i in range(n_windows):
            window = windows[i]
            
            # Compute statistical features
            mean = np.mean(window)
            std = np.std(window)
            skew = stats.skew(window) if len(window) > 2 else 0
            
            # Compute fractal dimension (simplified Higuchi method)
            if len(window) > 2:
                diff = np.diff(window)
                fd = 1 + np.log(len(diff)) / np.log(np.sum(np.abs(diff)) / (np.max(window) - np.min(window)))
            else:
                fd = 1.0
            
            # Store features
            features[i] = [mean, std, skew, fd]
        
        return features
    
    def _compute_transition_matrix(self, classes: np.ndarray) -> np.ndarray:
        """
        Compute the transition matrix between classes.
        
        Args:
            classes: Sequence of class indices
            
        Returns:
            Transition matrix
        """
        n_classes = self.n_classes
        transition_matrix = np.zeros((n_classes, n_classes))
        
        # Count transitions
        for i in range(len(classes) - 1):
            transition_matrix[classes[i], classes[i+1]] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_matrix, 
            row_sums, 
            out=np.zeros_like(transition_matrix), 
            where=row_sums != 0
        )
        
        return transition_matrix
    
    def _predict_next_class(self, current_class: int) -> int:
        """
        Predict the next class based on the current class.
        
        Args:
            current_class: Current class index
            
        Returns:
            Predicted next class index
        """
        # Get the transition probabilities for the current class
        transition_probs = self.transition_matrix[current_class]
        
        # If all probabilities are zero, return the current class
        if np.sum(transition_probs) == 0:
            return current_class
        
        # Otherwise, sample from the transition probabilities
        next_class = np.random.choice(self.n_classes, p=transition_probs)
        
        return next_class


class RescaledRangeForecaster(BaseForecaster):
    """
    Rescaled Range (R/S) Analysis-based forecaster.
    
    This model uses R/S analysis to estimate the Hurst exponent and then uses
    this information to inform the forecast.
    
    Reference: From the research paper, R/S analysis is primarily known for estimating
    the Hurst exponent and fractal dimension, but can also be used for forecasting.
    """
    
    def __init__(self, window_size: int = 10, n_lags: int = 5, **kwargs):
        """
        Initialize the Rescaled Range forecaster.
        
        Args:
            window_size: Size of the window for R/S analysis
            n_lags: Number of lag values to use in the prediction model
            **kwargs: Additional parameters
        """
        super().__init__(name=f"RescaledRange(window={window_size}, lags={n_lags})")
        self.window_size = window_size
        self.n_lags = n_lags
        self.kwargs = kwargs
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RescaledRangeForecaster':
        """
        Fit the Rescaled Range model to the training data.
        
        Args:
            X: Training features (used for lag values if available)
            y: Training target (time series values)
            
        Returns:
            self: The fitted forecaster
        """
        # Compute the Hurst exponent using R/S analysis
        self.hurst_exponent = self._compute_hurst_exponent(y)
        
        # Store the last values for prediction
        self.last_values = y[-self.n_lags:]
        
        # Determine the forecasting approach based on the Hurst exponent
        if self.hurst_exponent > 0.5:
            # Trending (persistent) - use trend extrapolation
            self.forecast_method = 'trend'
            
            # Compute weights based on the Hurst exponent (more weight to recent values for higher H)
            self.weights = np.power(np.arange(1, self.n_lags + 1) / self.n_lags, 2 - 2 * (self.hurst_exponent - 0.5))
            self.weights = self.weights / np.sum(self.weights)
            
        elif self.hurst_exponent < 0.5:
            # Mean-reverting (anti-persistent) - use mean reversion
            self.forecast_method = 'mean_reversion'
            
            # Compute the mean and standard deviation
            self.mean = np.mean(y)
            self.std = np.std(y)
            
            # Compute the reversion strength (stronger for lower H)
            self.reversion_strength = 1 - 2 * self.hurst_exponent
            
        else:
            # Random walk (Hurst exponent ≈ 0.5) - use simple average
            self.forecast_method = 'random_walk'
            
            # Equal weights for all lag values
            self.weights = np.ones(self.n_lags) / self.n_lags
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Rescaled Range model.
        
        Args:
            X: Test features (not used in this model)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Initialize predictions
        predictions = np.zeros(n_periods)
        current_values = self.last_values.copy()
        
        # Make predictions based on the forecasting method
        for i in range(n_periods):
            if self.forecast_method == 'trend':
                # Weighted average of trend continuation
                diffs = np.diff(np.append([current_values[0]], current_values))
                weighted_diff = np.sum(diffs * self.weights)
                next_value = current_values[-1] + weighted_diff
                
            elif self.forecast_method == 'mean_reversion':
                # Mean reversion with a strength based on the Hurst exponent
                deviation = current_values[-1] - self.mean
                next_value = current_values[-1] - self.reversion_strength * deviation
                
            else:  # random_walk
                # Simple weighted average
                next_value = np.sum(current_values * self.weights)
            
            # Store the prediction
            predictions[i] = next_value
            
            # Update current values for next prediction
            current_values = np.append(current_values[1:], next_value)
        
        return predictions
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted Rescaled Range model.
        
        Args:
            X: Test features (only the first row is used)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Create a dummy X with appropriate shape
        dummy_X = np.zeros((n_steps, X.shape[1] if X.ndim > 1 else 1))
        
        # Use the predict method to generate forecasts
        return self.predict(dummy_X)
    
    def _compute_hurst_exponent(self, y: np.ndarray) -> float:
        """
        Compute the Hurst exponent using Rescaled Range (R/S) analysis.
        
        Args:
            y: Time series values
            
        Returns:
            Hurst exponent
        """
        n = len(y)
        if n < 20:  # R/S analysis requires a reasonably long series
            return 0.5  # Default to random walk for very short series
        
        # Calculate the array of the mean of the rescaled range
        rs_values = []
        window_sizes = [self.window_size]
        
        # If the time series is long enough, use multiple window sizes
        if n >= 4 * self.window_size:
            max_window = min(n // 4, 100)  # Limit to avoid excessive computation
            window_sizes = np.unique(np.logspace(
                np.log10(self.window_size), 
                np.log10(max_window), 
                min(10, n // self.window_size)
            ).astype(int))
        
        for w in window_sizes:
            rs = []
            # Split the time series into non-overlapping windows
            n_windows = n // w
            for i in range(n_windows):
                window = y[i*w:(i+1)*w]
                
                # Calculate mean and standard deviation
                mean = np.mean(window)
                std = np.std(window)
                
                # Calculate cumulative deviation from the mean
                cumsum = np.cumsum(window - mean)
                
                # Calculate range (max - min of cumulative sum)
                r = np.max(cumsum) - np.min(cumsum)
                
                # Calculate rescaled range (R/S)
                if std > 0:
                    rs.append(r / std)
                
            # Calculate the mean of R/S values for this window size
            if rs:
                rs_values.append(np.mean(rs))
            else:
                rs_values.append(1.0)  # Default if standard deviation is zero
        
        # If we have at least 2 different window sizes, use log-log regression
        if len(window_sizes) >= 2:
            log_w = np.log10(window_sizes)
            log_rs = np.log10(rs_values)
            
            # Linear regression on log-log plot
            slope, _, _, _, _ = stats.linregress(log_w, log_rs)
            
            # The Hurst exponent is the slope of the regression line
            hurst = slope
        else:
            # If only one window size, use a direct calculation
            # This is a simplified approach
            rs_value = rs_values[0]
            hurst = np.log(rs_value) / np.log(window_sizes[0])
        
        # Clamp the Hurst exponent to the valid range [0, 1]
        hurst = max(0.0, min(1.0, hurst))
        
        return hurst


class FractalInterpolationForecaster(BaseForecaster):
    """
    Fractal Interpolation forecaster.
    
    This model uses fractal interpolation functions to enhance the data quality
    and improve forecasting accuracy.
    
    Reference: From the research paper, fractal interpolation can be used as a
    preprocessing step to enhance machine learning models for forecasting.
    """
    
    def __init__(self, base_model: Any, n_points: int = 5, scale_factor: float = 0.5, **kwargs):
        """
        Initialize the Fractal Interpolation forecaster.
        
        Args:
            base_model: The base forecasting model to enhance
            n_points: Number of interpolation points between original data points
            scale_factor: Vertical scaling factor for the fractal interpolation
            **kwargs: Additional parameters
        """
        name = f"FractalInterpolation({base_model.__class__.__name__}, n={n_points}, s={scale_factor})"
        super().__init__(name=name)
        self.base_model = base_model
        self.n_points = n_points
        self.scale_factor = scale_factor
        self.kwargs = kwargs
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FractalInterpolationForecaster':
        """
        Fit the Fractal Interpolation model to the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: The fitted forecaster
        """
        # Apply fractal interpolation to enhance the training data
        X_enhanced, y_enhanced = self._fractal_interpolate(X, y)
        
        # Fit the base model on the enhanced data
        self.base_model.fit(X_enhanced, y_enhanced)
        
        # Store the original data for later use
        self.X_orig = X
        self.y_orig = y
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Fractal Interpolation model.
        
        Args:
            X: Test features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make predictions using the base model
        predictions = self.base_model.predict(X)
        
        return predictions
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted Fractal Interpolation model.
        
        Args:
            X: Test features (only the first row is used)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Use the base model's predict_many method if available
        if hasattr(self.base_model, 'predict_many'):
            return self.base_model.predict_many(X, n_steps)
        
        # Otherwise, use the default implementation
        return super().predict_many(X, n_steps)
    
    def _fractal_interpolate(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fractal interpolation to enhance the training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Tuple of enhanced features and target
        """
        # Number of samples
        n_samples = len(y)
        if n_samples < 2:
            return X, y  # Not enough data for interpolation
        
        # Create interpolation points
        n_enhanced = (n_samples - 1) * (self.n_points + 1) + 1
        
        # Create indices for the original and enhanced data
        orig_indices = np.arange(n_samples)
        enhanced_indices = np.linspace(0, n_samples - 1, n_enhanced)
        
        # Interpolate the target values using a spline
        y_spline = interpolate.splrep(orig_indices, y, k=3)
        y_smooth = interpolate.splev(enhanced_indices, y_spline)
        
        # Add fractal noise to the interpolated values
        y_enhanced = y_smooth.copy()
        for i in range(1, n_enhanced - 1):
            if i % (self.n_points + 1) != 0:  # Skip the original points
                # Find the surrounding original points
                left_idx = (i // (self.n_points + 1))
                right_idx = left_idx + 1
                
                # Compute the relative position
                t = (i % (self.n_points + 1)) / (self.n_points + 1)
                
                # Compute the fractal offset
                fractal_offset = self.scale_factor * np.sin(np.pi * t) * (y[right_idx] - y[left_idx])
                
                # Apply the offset to add fractal roughness
                y_enhanced[i] += fractal_offset
        
        # Interpolate each feature column
        X_enhanced = np.zeros((n_enhanced, X.shape[1]))
        for j in range(X.shape[1]):
            X_spline = interpolate.splrep(orig_indices, X[:, j], k=1)
            X_enhanced[:, j] = interpolate.splev(enhanced_indices, X_spline)
        
        # Ensure the original points are preserved exactly
        for i in range(n_samples):
            enhanced_idx = i * (self.n_points + 1)
            X_enhanced[enhanced_idx] = X[i]
            y_enhanced[enhanced_idx] = y[i]
        
        return X_enhanced, y_enhanced


class FractalReductionForecaster(BaseForecaster):
    """
    Fractal Reduction forecaster with Binary Gate Logic.
    
    This model decomposes a scalar time series into a collection of parallel
    binary time series, forecasts each binary series, and then reconstructs the
    original series.
    
    Reference: From the research paper, this approach claims advantages like
    greater flexibility, ease of interpretation, numerical stability, and
    outcome determinism.
    """
    
    def __init__(self, n_levels: int = 8, gate_type: str = 'AND', **kwargs):
        """
        Initialize the Fractal Reduction forecaster.
        
        Args:
            n_levels: Number of binary levels to use
            gate_type: Type of binary gate to use ('AND', 'OR', 'XOR')
            **kwargs: Additional parameters
        """
        super().__init__(name=f"FractalReduction(levels={n_levels}, gate={gate_type})")
        self.n_levels = n_levels
        self.gate_type = gate_type.upper()
        self.kwargs = kwargs
        
        # Validate gate type
        if self.gate_type not in ['AND', 'OR', 'XOR']:
            raise ValueError("Gate type must be one of 'AND', 'OR', 'XOR'")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FractalReductionForecaster':
        """
        Fit the Fractal Reduction model to the training data.
        
        Args:
            X: Training features (not used in this model)
            y: Training target (time series values)
            
        Returns:
            self: The fitted forecaster
        """
        # Normalize the time series to [0, 1]
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        if self.y_max > self.y_min:
            y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        else:
            y_norm = np.ones_like(y) * 0.5
        
        # Decompose the time series into binary series
        self.binary_series = self._decompose_to_binary(y_norm)
        
        # Compute transition probabilities for each binary series
        self.transition_probs = self._compute_transition_probs()
        
        # Store the last values for each binary series
        self.last_binary_values = np.array([series[-1] for series in self.binary_series])
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Fractal Reduction model.
        
        Args:
            X: Test features (not used in this model)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Get the number of predictions to make
        n_periods = X.shape[0]
        
        # Initialize predictions (binary and scalar)
        binary_predictions = np.zeros((self.n_levels, n_periods), dtype=np.int8)
        predictions = np.zeros(n_periods)
        
        # Current binary values
        current_binary = self.last_binary_values.copy()
        
        # Make predictions for each period
        for i in range(n_periods):
            # Predict the next value for each binary series
            for j in range(self.n_levels):
                # Get the transition probabilities for the current state
                trans_prob = self.transition_probs[j][current_binary[j]]
                
                # Predict the next binary value
                if np.random.random() < trans_prob:
                    binary_predictions[j, i] = 1
                else:
                    binary_predictions[j, i] = 0
            
            # Convert the binary predictions to a scalar value
            predictions[i] = self._binary_to_scalar(binary_predictions[:, i])
            
            # Update the current binary values
            current_binary = binary_predictions[:, i]
        
        # Denormalize the predictions
        predictions = predictions * (self.y_max - self.y_min) + self.y_min
        
        return predictions
    
    def predict_many(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Make multi-step predictions using the fitted Fractal Reduction model.
        
        Args:
            X: Test features (only the first row is used)
            n_steps: Number of steps to forecast
            
        Returns:
            Predictions for n_steps ahead
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Create a dummy X with appropriate shape
        dummy_X = np.zeros((n_steps, X.shape[1] if X.ndim > 1 else 1))
        
        # Use the predict method to generate forecasts
        return self.predict(dummy_X)
    
    def _decompose_to_binary(self, y_norm: np.ndarray) -> List[np.ndarray]:
        """
        Decompose a normalized time series into binary series.
        
        Args:
            y_norm: Normalized time series
            
        Returns:
            List of binary series
        """
        binary_series = []
        
        # Create binary thresholds
        thresholds = np.linspace(0, 1, self.n_levels + 1)[1:-1]
        
        # Create binary series for each threshold
        for threshold in thresholds:
            binary = (y_norm >= threshold).astype(np.int8)
            binary_series.append(binary)
        
        return binary_series
    
    def _compute_transition_probs(self) -> List[np.ndarray]:
        """
        Compute transition probabilities for each binary series.
        
        Returns:
            List of transition probability arrays
        """
        transition_probs = []
        
        for series in self.binary_series:
            # Count transitions from 0 to 1 and from 1 to 1
            count_0_to_1 = 0
            count_0_total = 0
            count_1_to_1 = 0
            count_1_total = 0
            
            for i in range(len(series) - 1):
                if series[i] == 0:
                    count_0_total += 1
                    if series[i+1] == 1:
                        count_0_to_1 += 1
                else:
                    count_1_total += 1
                    if series[i+1] == 1:
                        count_1_to_1 += 1
            
            # Compute transition probabilities
            p_0_to_1 = count_0_to_1 / count_0_total if count_0_total > 0 else 0.5
            p_1_to_1 = count_1_to_1 / count_1_total if count_1_total > 0 else 0.5
            
            transition_probs.append(np.array([p_0_to_1, p_1_to_1]))
        
        return transition_probs
    
    def _binary_to_scalar(self, binary_values: np.ndarray) -> float:
        """
        Convert binary values to a scalar value.
        
        Args:
            binary_values: Binary values
            
        Returns:
            Scalar value
        """
        # Implement different gate types
        if self.gate_type == 'AND':
            # Weight the binary values and take the product of non-zero values
            weights = np.linspace(0, 1, self.n_levels)
            weighted_sum = 0
            for i, val in enumerate(binary_values):
                weighted_sum += weights[i] * val
            
            # Normalize to [0, 1]
            return weighted_sum / np.sum(weights) if np.sum(weights) > 0 else 0.5
            
        elif self.gate_type == 'OR':
            # Take the maximum weight where the binary value is 1
            weights = np.linspace(0, 1, self.n_levels)
            max_weight = 0
            for i, val in enumerate(binary_values):
                if val == 1:
                    max_weight = max(max_weight, weights[i])
            
            return max_weight
            
        elif self.gate_type == 'XOR':
            # Take the XOR of all binary values and scale to [0, 1]
            xor_value = np.bitwise_xor.reduce(binary_values)
            
            # Weight the XOR value by the proportion of 1s
            proportion_of_ones = np.mean(binary_values)
            
            return xor_value * proportion_of_ones
        
        else:
            # Default to a simple average
            return np.mean(binary_values)