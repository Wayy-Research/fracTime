import numpy as np
import polars as pl
import requests
from io import StringIO
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit, float64, int64
from numba.typed import List
from typing import Tuple, List as PyList, Dict
import warnings
import yfinance as yf
import pandas as pd
from fractime.optimization import compute_box_dimension_safe
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

def get_yahoo_data(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """Get historical price data from Yahoo Finance."""
    try:
        # If no end date is provided, use current date
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Get data from yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Ensure we have the expected columns
        if 'Date' not in data.columns or 'Close' not in data.columns:
            raise ValueError(f"Required columns not found in data for {symbol}")
            
        # Handle any missing values
        data = data.dropna(subset=['Close'])
        
        # Return the processed dataframe
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        raise

@njit
def compute_rs(returns: np.ndarray, lag: int) -> float:
    """Compute R/S value for a given lag."""
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0.0
        
    cumsum = np.cumsum(returns - mean)
    r = np.max(cumsum) - np.min(cumsum)
    return r / std if std > 0 else 0.0

@njit
def linear_regression(x: np.ndarray, y: np.ndarray) -> float:
    """Simple linear regression, returns slope."""
    n = len(x)
    if n < 2:
        return 0.0
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    return slope

@njit
def compute_hurst_exponent(prices: np.ndarray, min_lag: int, max_lag: int) -> float:
    """Optimized Hurst exponent calculation."""
    returns = np.diff(np.log(prices))
    tau = np.empty(max_lag - min_lag)
    rs_values = np.empty(max_lag - min_lag)
    
    for i, lag in enumerate(range(min_lag, max_lag)):
        rs_current = np.empty((len(returns) - lag) // lag)
        
        for j in range(0, len(returns) - lag, lag):
            rs_current[j // lag] = compute_rs(returns[j:j+lag], lag)
            
        valid_rs = rs_current[rs_current > 0]
        if len(valid_rs) > 0:
            tau[i] = np.log(lag)
            rs_values[i] = np.log(np.mean(valid_rs))
        else:
            tau[i] = 0
            rs_values[i] = 0
    
    # Filter out zero values
    mask = (tau > 0) & (rs_values > 0)
    if np.sum(mask) > 1:
        return linear_regression(tau[mask], rs_values[mask])
    return 0.5

@njit
def compute_box_dimension(scaled_prices: np.ndarray, min_window: int, max_window: int, step: int) -> float:
    """Optimized box-counting dimension calculation."""
    dimensions = np.empty((max_window - min_window) // step)
    
    for i, scale in enumerate(range(min_window, max_window, step)):
        boxes = np.ceil(scaled_prices * scale)
        unique_boxes = len(np.unique(boxes))
        dimensions[i] = np.log(unique_boxes) / np.log(scale)
    
    return np.mean(dimensions)

@njit
def compute_box_dimension_safe(scaled_prices: np.ndarray, min_window: int, max_window: int, step: int) -> float:
    """Box-counting dimension calculation with safety checks."""
    if step <= 0:
        step = 1
        
    num_scales = (max_window - min_window) // step
    if num_scales <= 0:
        return 1.5  # Default value
        
    dimensions = np.empty(num_scales)
    valid_count = 0
    
    for i, scale in enumerate(range(min_window, max_window, step)):
        if scale <= 0:  # Skip invalid scales
            continue
            
        boxes = np.ceil(scaled_prices * scale)
        unique_boxes = len(np.unique(boxes))
        
        if unique_boxes > 0 and scale > 0:  # Safety check
            dimensions[valid_count] = np.log(unique_boxes) / np.log(scale)
            valid_count += 1
    
    if valid_count > 0:
        return np.mean(dimensions[:valid_count])
    else:
        return 1.5  # Default value

class FractalAnalyzer:
    """Analyzes fractal properties of time series data."""
    
    def __init__(self):
        """Initialize with empty cache for performance."""
        self.cache = {}  # Cache for expensive computations
    
    def analyze_patterns(self, prices: np.ndarray, full_analysis=True) -> dict:
        """Analyze with caching and selective feature computation."""
        # Generate a cache key based on the first/last/middle values and length
        if len(prices) > 3:
            cache_key = f"{len(prices)}_{prices[0]:.2f}_{prices[-1]:.2f}_{prices[len(prices)//2]:.2f}"
            
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        results = {
            'hurst': self.compute_hurst(prices),
            'fractal_dim': self.compute_fractal_dimension(prices)  # Changed key to match existing code
        }
        
        # Only compute expensive metrics when requested or for small datasets
        if full_analysis or len(prices) < 1000:
            results['self_similar_patterns'] = self._find_patterns(prices)
        else:
            # Use simplified patterns for backtesting
            results['self_similar_patterns'] = self._find_simple_patterns(prices)
        
        # Cache the results for future use
        if len(prices) > 3:
            self.cache[cache_key] = results
            
            # Limit cache size to prevent memory issues
            if len(self.cache) > 100:
                # Remove a random key to keep cache size reasonable
                self.cache.pop(next(iter(self.cache)))
        
        return results
    
    def _find_simple_patterns(self, prices: np.ndarray) -> list:
        """Faster pattern detection for backtesting."""
        patterns = []
        returns = np.diff(np.log(prices))
        
        # Use fewer window sizes and skip many positions
        window_sizes = [10, 20, 50]
        max_patterns = 20
        
        for window in window_sizes:
            if len(prices) < window * 2:
                continue
                
            skip_size = max(1, len(prices) // (max_patterns * 2))
            
            for i in range(0, len(prices) - window * 2, skip_size):
                if len(patterns) >= max_patterns:
                    break
                    
                pattern1_returns = returns[i:i+window-1]
                pattern1_vol = np.std(pattern1_returns)
                
                if pattern1_vol < 1e-8:
                    continue
                
                patterns.append({
                    'start': i,
                    'length': window,
                    'returns': pattern1_returns,
                    'volatility': pattern1_vol,
                    'similarity': 0.8,  # Default value
                    'fractal_dim': 1.5  # Default value
                })
        
        return patterns
    
    def _find_patterns(self, prices: np.ndarray) -> list:
        """Optimized pattern detection."""
        from fractime.optimization import compute_pattern_similarities
        
        patterns = []
        returns = np.diff(np.log(prices))
        
        # Use fewer window sizes to reduce computation
        min_window = 10
        max_window = min(250, len(prices)//3)
        window_step = max(1, (max_window - min_window) // 10)
        window_sizes = range(min_window, max_window, window_step)
        
        # Pre-compute volatilities
        volatilities = {}
        for window in window_sizes:
            rolling_vols = np.array([np.std(returns[i:i+window-1]) for i in range(len(returns)-window+1)])
            volatilities[window] = rolling_vols
        
        # Sample fewer starting points
        for window in window_sizes:
            if len(patterns) >= 50:  # Limit total patterns
                break
                
            step_size = max(1, window // 4)  # Skip positions to reduce computation
            
            for i in range(0, len(prices)-window*2, step_size):
                # Use pre-computed volatilities
                if i >= len(volatilities[window]) or i+window >= len(volatilities[window]):
                    continue
                    
                pattern1_vol = volatilities[window][i]
                pattern2_vol = volatilities[window][i+window]
                
                # Skip zero volatility patterns
                if pattern1_vol < 1e-8 or pattern2_vol < 1e-8:
                    continue
                
                # Get pattern returns
                pattern1_returns = returns[i:i+window-1]
                pattern2_returns = returns[i+window:i+window*2-1]
                
                # Use Numba-optimized similarity calculation
                similarity = compute_pattern_similarities(
                    pattern1_returns, pattern2_returns, pattern1_vol, pattern2_vol
                )
                
                if similarity > 0.8:  # Only keep strong correlations
                    patterns.append({
                        'start': i,
                        'length': window,
                        'returns': pattern1_returns,
                        'volatility': pattern1_vol,
                        'similarity': similarity,
                        'fractal_dim': self.compute_fractal_dimension(
                            prices[i:i+window], 
                            quick_mode=True
                        )
                    })
        
        return patterns
    
    def compute_fractal_dimension(self, prices: np.ndarray, quick_mode=False) -> float:
        """Compute fractal dimension, optionally using a faster approximation."""
        try:
            if len(prices) < 10:  # Not enough points for meaningful calculation
                return 1.5  # Return a reasonable default value
                
            if quick_mode:
                # Fast approximation using fewer box sizes
                r_min = 2
                r_max = min(10, len(prices)//4)
                step = 2
            else:
                # Full computation
                r_min = 2
                r_max = min(20, len(prices)//4) 
                step = 1
                
            # Ensure we have at least one scale value
            if r_min >= r_max:
                return 1.5
                
            # Transform prices for scaling
            try:
                scaled_prices = StandardScaler().fit_transform(prices.reshape(-1, 1)).ravel()
            except:
                # If scaling fails, use min-max normalization
                min_price = np.min(prices)
                max_price = np.max(prices)
                if max_price == min_price:  # Avoid division by zero
                    return 1.0  # Straight line has dimension 1
                scaled_prices = (prices - min_price) / (max_price - min_price)
                
            # Safe computation of fractal dimension
            return compute_box_dimension_safe(scaled_prices, r_min, r_max, step)
        except Exception as e:
            print(f"Error computing fractal dimension: {e}")
            return 1.5  # Reasonable default
    
    def get_patterns(self, prices: np.ndarray, max_patterns=20) -> list:
        """Extract patterns efficiently with sampling."""
        window_sizes = [10, 20, 30]  # Different pattern lengths to extract
        patterns = []
        
        # For longer series, use sampling to avoid excessive patterns
        if len(prices) > 1000:
            skip_factor = len(prices) // 500
        else:
            skip_factor = 1
            
        for window in window_sizes:
            # Step back with larger jumps for efficiency
            step_size = max(1, window // 2)
            
            for i in range(len(prices) - window, 0, -step_size * skip_factor):
                if i >= window:
                    # Extract the pattern segment
                    pattern = prices[i-window:i]
                    if len(pattern) == window:  # Ensure we have a complete pattern
                        patterns.append(pattern)
                        
                # Stop if we have enough patterns
                if len(patterns) >= max_patterns // len(window_sizes):
                    break
        
        # Print some info about patterns found
        print(f"Extracted {len(patterns)} patterns from price data")
            
        return patterns

    def compute_hurst(self, prices: np.ndarray) -> float:
        """Compute the Hurst exponent for a price series."""
        if len(prices) < 20:
            # Not enough data for reliable calculation
            return 0.5
        
        try:
            # Use the existing compute_hurst_exponent function
            min_lag = 10
            max_lag = min(250, len(prices) // 2)
            return compute_hurst_exponent(prices, min_lag, max_lag)
        except Exception as e:
            print(f"Error computing Hurst exponent: {e}")
            # Return 0.5 as a default (random walk)
            return 0.5

class FractalSimulator:
    """Generates paths based on fractal patterns and historical distributions."""
    
    def __init__(self, prices: np.ndarray, analyzer: FractalAnalyzer):
        self.prices = prices
        self.analyzer = analyzer
        self.patterns = None
        self.hurst = None
        self._analyze()
        
        # Prepare sampled data for faster simulations
        if len(self.prices) > 1000:
            # Create downsampled version for faster regime matching
            sampling_rate = len(self.prices) // 1000
            self.sampled_prices = self.prices[::sampling_rate]
            print(f"Created downsampled data: {len(self.sampled_prices)} points")
        else:
            self.sampled_prices = self.prices
    
    def _analyze(self):
        """Perform initial analysis."""
        results = self.analyzer.analyze_patterns(self.prices)
        self.patterns = results['self_similar_patterns']
        self.hurst = results['hurst']
    
    def _compute_volatility_regimes(self, returns: np.ndarray) -> Dict:
        """Compute fractal volatility regimes across multiple scales."""
        # Multiple timeframes for fractal analysis
        timeframes = [5, 21, 63, 126]  # daily, weekly, monthly, quarterly
        
        # Ensure we have enough data
        if len(returns) < max(timeframes):
            # Fall back to smaller timeframes if needed
            timeframes = [tf for tf in timeframes if tf < len(returns)]
            if not timeframes:
                timeframes = [5]  # Minimum timeframe
        
        # Compute volatility at each scale
        vol_states = {}
        for tf in timeframes:
            try:
                # Rolling volatility for this timeframe with safety checks
                rolling_vol = np.array([
                    max(1e-8, np.std(returns[max(0, i-tf):i]))  # Ensure non-zero
                    for i in range(tf, len(returns))
                ])
                
                if len(rolling_vol) < 3:  # Need minimum points for analysis
                    continue
                    
                # Ensure we have valid data before computing
                if np.all(rolling_vol == rolling_vol[0]):  # All values same
                    vol_hurst = 0.5
                    vol_fractal_dim = 1.0
                else:
                    # Compute Hurst exponent of volatility series
                    vol_hurst = compute_hurst_exponent(
                        rolling_vol, 
                        min(5, tf//4),
                        min(50, len(rolling_vol)//3)
                    )
                    
                    # Compute fractal dimension of volatility
                    scaled_vol = StandardScaler().fit_transform(rolling_vol.reshape(-1, 1)).ravel()
                    vol_fractal_dim = compute_box_dimension(
                        scaled_vol,
                        min(5, tf//4),
                        min(tf, len(rolling_vol)//2),
                        2
                    )
                
                # Prepare features for clustering with safety checks
                vol_features = []
                for i in range(len(rolling_vol)-tf):
                    feature_set = [
                        rolling_vol[i],  # Current vol
                        np.log(rolling_vol[i+1] / rolling_vol[i]) if rolling_vol[i] > 0 else 0,  # Vol change
                        max(1e-8, np.std(rolling_vol[i:i+tf]))  # Vol of vol
                    ]
                    vol_features.append(feature_set)
                
                vol_features = np.array(vol_features)
                if len(vol_features) > 0:
                    # Normalize features
                    scaler = StandardScaler()
                    vol_features = scaler.fit_transform(vol_features)
                    
                    # Cluster with minimum 2 clusters if enough data
                    n_clusters = min(3, len(vol_features) // 5)
                    if n_clusters < 2:
                        n_clusters = 2
                    
                    kmeans = KMeans(n_clusters=n_clusters)
                    regime_labels = kmeans.fit_predict(vol_features)
                    regime_centers = kmeans.cluster_centers_
                    
                    vol_states[tf] = {
                        'current': rolling_vol[-1],
                        'history': rolling_vol,
                        'hurst': vol_hurst,
                        'fractal_dim': vol_fractal_dim,
                        'regime_labels': regime_labels,
                        'regime_centers': regime_centers,
                        'current_regime': regime_labels[-1] if len(regime_labels) > 0 else 0
                    }
                    
            except Exception as e:
                print(f"Warning: Error computing volatility regime for timeframe {tf}: {e}")
                continue
        
        # Ensure we have at least one valid state
        if not vol_states:
            # Create a simple fallback state
            vol_states[5] = {
                'current': max(1e-8, np.std(returns[-5:])),
                'history': np.array([max(1e-8, np.std(returns))]),
                'hurst': 0.5,
                'fractal_dim': 1.0,
                'regime_labels': np.array([0]),
                'regime_centers': np.array([[0, 0, 0]]),
                'current_regime': 0
            }
        
        return vol_states

    def simulate_paths(
        self,
        n_steps: int,
        n_paths: int = 1000,
        pattern_weight: float = 0.3,
        cloud_paths: int = 200,
        preserve_volatility: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """Generate paths using regime-matched sampling based on recent volatility."""
        # Get historical returns
        historical_returns = np.diff(np.log(self.prices))
        
        # Define lookback window as 2x the forecast horizon
        lookback_window = 2 * n_steps
        
        # Get recent volatility regime
        recent_returns = historical_returns[-lookback_window:]
        recent_vol = np.std(recent_returns)
        
        # Find similar volatility regimes in history using multiple metrics
        regime_windows = []
        recent_skew = stats.skew(recent_returns)
        recent_kurt = stats.kurtosis(recent_returns)
        
        for i in range(len(historical_returns) - lookback_window):
            window_returns = historical_returns[i:i+lookback_window]
            window_vol = np.std(window_returns)
            window_skew = stats.skew(window_returns)
            window_kurt = stats.kurtosis(window_returns)
            
            # Calculate similarities using multiple metrics
            vol_similarity = abs(window_vol - recent_vol) / recent_vol
            skew_similarity = abs(window_skew - recent_skew)
            kurt_similarity = abs(window_kurt - recent_kurt)
            
            # Combine similarities with weights
            total_similarity = (0.6 * vol_similarity + 
                              0.25 * skew_similarity + 
                              0.15 * kurt_similarity)
            
            # If combined similarity is good enough, include this window
            if total_similarity < 0.3:  # Start with stricter threshold
                regime_windows.append(i)
        
        # Ensure we have enough similar windows, if not, gradually relax constraint
        similarity_threshold = 0.3
        while len(regime_windows) < 20 and similarity_threshold < 1.0:
            similarity_threshold += 0.1  # More gradual relaxation
            regime_windows = []
            for i in range(len(historical_returns) - lookback_window):
                window_returns = historical_returns[i:i+lookback_window]
                window_vol = np.std(window_returns)
                window_skew = stats.skew(window_returns)
                window_kurt = stats.kurtosis(window_returns)
                
                vol_similarity = abs(window_vol - recent_vol) / recent_vol
                skew_similarity = abs(window_skew - recent_skew)
                kurt_similarity = abs(window_kurt - recent_kurt)
                
                total_similarity = (0.6 * vol_similarity + 
                                  0.25 * skew_similarity + 
                                  0.15 * kurt_similarity)
                
                if total_similarity < similarity_threshold:
                    regime_windows.append(i)
        
        # Initialize paths array
        paths = np.zeros((n_paths, n_steps))
        
        # Generate paths by sampling from similar regimes
        for i in range(n_paths):
            # Randomly select a similar regime window
            if regime_windows:
                start_idx = np.random.choice(regime_windows)
                # Get n_steps returns from this regime
                regime_returns = historical_returns[start_idx:start_idx+lookback_window]
                # Randomly select a continuous segment of length n_steps
                segment_start = np.random.randint(0, len(regime_returns) - n_steps)
                path_returns = regime_returns[segment_start:segment_start+n_steps]
            else:
                # Fallback to recent returns if no similar regimes found
                path_returns = np.random.choice(recent_returns, size=n_steps, replace=True)
            
            # Convert returns to price path
            paths[i] = self.prices[-1] * np.exp(np.cumsum(path_returns))
        
        # Cluster and analyze paths
        scaler = StandardScaler()
        scaled_paths = scaler.fit_transform(paths)
        
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(scaled_paths)
        
        # Calculate centroids and probabilities
        centroids = []
        cluster_sizes = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_paths = paths[labels == i]
            centroids.append(np.mean(cluster_paths, axis=0))
            cluster_sizes[i] = len(cluster_paths)
        
        # Compare with patterns across timeframes and compute scores
        cluster_scores = np.zeros(n_clusters)
        pattern_matches = []
        
        # Find similar patterns in historical data
        patterns = self.analyzer.get_patterns(self.prices)
        
        # Get pattern similarity - fix the call to match our updated method
        try:
            # Check what type of patterns we're working with
            if isinstance(patterns, dict):
                # Handle dictionary of patterns by timeframe (original structure)
                similarities_by_timeframe = {}
                for timeframe, timeframe_patterns in patterns.items():
                    # Convert each timeframe's patterns to array form if needed
                    pattern_arrays = []
                    for pattern in timeframe_patterns:
                        if 'start' in pattern and 'length' in pattern:
                            # Extract the actual price segment for this pattern
                            start = pattern['start']
                            length = pattern['length']
                            if start + length <= len(self.prices):
                                pattern_arrays.append(self.prices[start:start+length])
                
                    # Compute similarity for this timeframe's patterns
                    if pattern_arrays:
                        similarities_by_timeframe[timeframe] = np.mean(
                            self._compute_path_pattern_similarity(self.prices, pattern_arrays)
                        )
                    else:
                        similarities_by_timeframe[timeframe] = 0.0
                
                # Calculate weighted average across timeframes (original logic)
                weighted_similarity = (0.5 * similarities_by_timeframe.get('daily', 0) +
                                     0.3 * similarities_by_timeframe.get('weekly', 0) +
                                     0.2 * similarities_by_timeframe.get('monthly', 0))
            else:
                # Handle direct list of pattern arrays (new structure)
                similarities = self._compute_path_pattern_similarity(self.prices, patterns)
                weighted_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0
        except Exception as e:
            print(f"Error computing pattern similarity: {e}")
            weighted_similarity = 0.0  # Fallback to zero similarity
        
        cluster_scores = weighted_similarity * np.ones(n_clusters)
        
        # Calculate final probabilities
        size_probs = cluster_sizes / n_paths
        combined_scores = (1 - pattern_weight) * size_probs + pattern_weight * cluster_scores
        cluster_probs = combined_scores / np.sum(combined_scores)
        
        # Find most likely path
        most_likely_cluster = np.argmax(cluster_probs)
        most_likely_path = centroids[most_likely_cluster]
        
        # Generate probability cloud around most likely path using similar regime sampling
        n_cloud_paths = cloud_paths  # Store the number since cloud_paths will be overwritten
        cloud_paths = np.zeros((n_cloud_paths, n_steps))
        
        # Calculate path probabilities based on regime similarity
        path_probabilities = np.zeros(n_cloud_paths)
        
        for i in range(cloud_paths.shape[0]):
            if regime_windows:
                # Sample from similar regime windows but add more noise
                start_idx = np.random.choice(regime_windows)
                regime_returns = historical_returns[start_idx:start_idx+lookback_window]
                segment_start = np.random.randint(0, len(regime_returns) - n_steps)
                base_returns = regime_returns[segment_start:segment_start+n_steps]
                
                # Add noise scaled by the regime's volatility
                noise_scale = np.std(regime_returns) * 0.3  # 30% of regime volatility
                cloud_returns = base_returns + np.random.normal(0, noise_scale, size=len(base_returns))
            else:
                # Fallback to sampling from recent returns with noise
                cloud_returns = np.random.choice(recent_returns, size=n_steps, replace=True)
            
            cloud_paths[i] = self.prices[-1] * np.exp(np.cumsum(cloud_returns))
            
            # Calculate probability based on multiple factors:
            # 1. Volatility similarity to recent regime
            path_vol = np.std(cloud_returns)
            vol_similarity = abs(path_vol - recent_vol) / recent_vol
            
            # 2. Return distribution similarity
            path_skew = stats.skew(cloud_returns)
            recent_skew = stats.skew(recent_returns)
            skew_similarity = abs(path_skew - recent_skew)
            
            # Combine similarities with weights
            total_similarity = (0.7 * vol_similarity + 0.3 * skew_similarity)
            path_probabilities[i] = np.exp(-2 * total_similarity)  # Less aggressive decay
        for i, path in enumerate(cloud_paths):
            distance = np.mean(np.abs(path - most_likely_path))
            path_probabilities[i] = np.exp(-distance / np.std(most_likely_path))
        
        # Normalize probabilities
        path_probabilities = path_probabilities / np.sum(path_probabilities)
        
        # Before returning paths, ensure volatility matches historical data
        if preserve_volatility:
            # Calculate historical volatility (day-to-day changes)
            hist_diffs = np.diff(np.log(self.prices))
            hist_std = np.std(hist_diffs)
            
            # Calculate forecast volatility
            forecast_diffs = np.diff(np.log(paths), axis=1)
            forecast_std = np.mean([np.std(path_diffs) for path_diffs in forecast_diffs])
            
            # If forecast is too smooth, add appropriate noise
            if forecast_std < 0.8 * hist_std:  # Allow some smoothing, but not too much
                print(f"Adjusting volatility from {forecast_std:.5f} to {hist_std:.5f}")
                volatility_factor = hist_std / forecast_std
                
                # Add scaled noise to maintain proper volatility
                for i in range(paths.shape[0]):
                    for j in range(1, paths.shape[1]):
                        # Generate noise with same distribution as historical data
                        noise = np.random.choice(hist_diffs) * 0.5  # Scale down slightly for stability
                        # Apply noise multiplicatively
                        paths[i, j] *= np.exp(noise)
        
        return paths, {
            'labels': labels,
            'cluster_weights': cluster_scores,
            'cluster_sizes': cluster_sizes,
            'pattern_matches': pattern_matches,
            'centroids': centroids,
            'cluster_probs': cluster_probs,
            'most_likely_path': most_likely_path,
            'probability_cloud': cloud_paths,
            'path_probabilities': path_probabilities
        }

    def _compute_path_pattern_similarity(self, prices: np.ndarray, patterns: list) -> np.ndarray:
        """Compute similarity between a price path and known patterns."""
        n_patterns = len(patterns)
        
        # If no patterns, return zero similarities
        if n_patterns == 0:
            return np.zeros(0)
        
        similarities = np.zeros(n_patterns)
        
        # For each pattern
        for i, pattern in enumerate(patterns):
            # If pattern is empty or too short, skip it
            if len(pattern) == 0 or len(pattern) < 2:
                similarities[i] = 0
                continue
            
            # Skip if prices array is too short
            if len(prices) < 2:
                similarities[i] = 0
                continue
            
            # Normalize pattern to [0, 1] range
            pat_min = np.min(pattern)
            pat_max = np.max(pattern)
            
            # Avoid division by zero
            if pat_max == pat_min:
                pat_norm = np.zeros_like(pattern)
            else:
                pat_norm = (pattern - pat_min) / (pat_max - pat_min)
            
            # Ensure pattern is long enough
            min_segment = 10  # Minimum segment length for correlation
            
            # Use maximum possible segment length
            segment_len = min(min_segment, len(pattern), len(prices))
            
            # Get segment of prices of the same length as pattern
            segment = prices[-segment_len:]
            
            # Ensure pat_norm is the right length too
            pat_norm = pat_norm[-segment_len:]
            
            # Normalize segment to [0, 1] range
            seg_min = np.min(segment)
            seg_max = np.max(segment)
            
            # Avoid division by zero
            if seg_max == seg_min:
                seg_norm = np.zeros_like(segment)
            else:
                seg_norm = (segment - seg_min) / (seg_max - seg_min)
            
            # Compute correlation between normalized segment and pattern
            # Add error handling for the correlation coefficient calculation
            try:
                if len(seg_norm) > 1 and len(pat_norm) > 1:
                    corr = np.corrcoef(seg_norm, pat_norm)[0,1]
                    if np.isnan(corr):
                        corr = 0  # Handle NaN correlations
                else:
                    corr = 0  # Not enough data for correlation
                
                # Scale to similarity: 1 is perfect match, 0 is no match
                similarities[i] = max(0, corr)  # Only positive correlations count as similarity
            except Exception as e:
                print(f"Error in correlation calculation: {e}")
                print(f"Segment shape: {seg_norm.shape}, Pattern shape: {pat_norm.shape}")
                similarities[i] = 0
        
        return similarities

    @staticmethod
    def compute_box_dimension(data: np.ndarray, min_size: int, max_size: int, step: int) -> float:
        """Compute the box-counting fractal dimension of a time series.
        
        Args:
            data: Input time series
            min_size: Minimum box size
            max_size: Maximum box size
            step: Step size for box scaling
            
        Returns:
            Estimated fractal dimension
        """
        sizes = range(min_size, max_size + 1, step)
        counts = []
        
        for size in sizes:
            # Count number of boxes needed to cover the curve
            boxes = set()
            for i in range(len(data) - 1):
                # Scale to box coordinates
                x = i // size
                y = int(data[i] / (max(data) - min(data)) * size)
                boxes.add((x, y))
            
            counts.append(len(boxes))
        
        # Compute dimension from log-log plot
        log_sizes = np.log([1/s for s in sizes])
        log_counts = np.log(counts)
        
        # Linear regression
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return slope

    def analyze_path_distributions(self, paths: np.ndarray) -> Dict:
        """Analyze the distribution of simulated paths."""
        n_paths, n_steps = paths.shape
        
        # Calculate return distributions at different horizons
        distributions = {}
        for step in [1, 5, 10, n_steps-1]:  # Different horizons
            returns = np.log(paths[:,step] / paths[:,0])
            distributions[step] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'skew': stats.skew(returns),
                'kurt': stats.kurtosis(returns),
                'quantiles': np.percentile(returns, [1, 25, 50, 75, 99])
            }
        
        # Find most common path shapes
        scaled_paths = StandardScaler().fit_transform(paths)
        kmeans = KMeans(n_clusters=5).fit(scaled_paths)
        
        # Calculate cluster statistics
        clusters = {}
        for i in range(5):
            cluster_paths = paths[kmeans.labels_ == i]
            clusters[i] = {
                'size': len(cluster_paths),
                'mean_path': np.mean(cluster_paths, axis=0),
                'std_path': np.std(cluster_paths, axis=0)
            }
        
        return {
            'distributions': distributions,
            'clusters': clusters
        }

    @staticmethod
    @njit
    def _generate_fbm(n: int, hurst: float) -> np.ndarray:
        """Generate fractional Brownian motion using a simplified method."""
        # Generate Gaussian noise
        noise = np.random.normal(0, 1, n)
        
        # Create time increments
        dt = 1.0 / n
        t = np.arange(n) * dt
        
        # Initialize fBm array
        fbm = np.zeros(n)
        
        # Compute fBm using direct method
        for i in range(1, n):
            # Power-law correlations
            increments = noise[:i] * np.power(t[i] - t[:i], hurst - 0.5)
            fbm[i] = fbm[i-1] + np.sum(increments) * np.sqrt(dt)
        
        # Normalize
        fbm = fbm / np.std(fbm)
        return fbm

    def simulate_paths_fast(self, n_steps, n_paths=100):
        """Faster path simulation for backtesting with fewer paths."""
        # Simplified version with fewer paths and calculations
        historical_returns = np.diff(np.log(self.prices))
        recent_returns = historical_returns[-min(len(historical_returns), 30):]
        
        # Use simple sampling with bootstrapping instead of complex regime matching
        sampled_indices = np.random.choice(
            len(recent_returns), 
            size=(n_paths, n_steps), 
            replace=True
        )
        
        # Generate paths based on sampled returns
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.prices[-1]  # Start with last price
        
        for i in range(n_steps):
            # Use the sampled returns for each path
            step_returns = recent_returns[sampled_indices[:, i]]
            paths[:, i+1] = paths[:, i] * np.exp(step_returns)
        
        # Calculate key statistics
        mean_path = np.mean(paths, axis=0)
        median_path = np.median(paths, axis=0)
        upper_95 = np.percentile(paths, 95, axis=0)
        lower_5 = np.percentile(paths, 5, axis=0)
        
        # Find most likely path (closest to mean)
        path_diffs = np.sum((paths - mean_path) ** 2, axis=1)
        most_likely_idx = np.argmin(path_diffs)
        most_likely_path = paths[most_likely_idx]
        
        # Simple analysis with mean path and percentiles
        path_analysis = {
            'mean_path': mean_path,
            'median_path': median_path,
            'most_likely_path': most_likely_path,
            'upper_95': upper_95,
            'lower_5': lower_5
        }
        
        return paths, path_analysis
        
    def simulate_paths_gpu(self, n_steps, n_paths=1000):
        """GPU-accelerated path simulation."""
        from fractime.optimization import try_import_cupy
        
        # Try to import cupy for GPU acceleration
        cp = try_import_cupy()
        
        if cp is None:
            print("GPU acceleration not available, falling back to CPU")
            return self.simulate_paths_fast(n_steps, n_paths)
        
        try:
            # Calculate returns
            historical_returns = np.diff(np.log(self.prices))
            recent_returns = historical_returns[-min(len(historical_returns), 30):]
            
            # Move data to GPU
            recent_returns_gpu = cp.array(recent_returns)
            
            # Generate paths on GPU
            paths_gpu = cp.zeros((n_paths, n_steps + 1))
            paths_gpu[:, 0] = self.prices[-1]
            
            # Generate random indices on GPU for bootstrapping
            indices = cp.random.randint(0, len(recent_returns), (n_paths, n_steps))
            
            # Use GPU for path generation
            for i in range(n_steps):
                returns = recent_returns_gpu[indices[:, i]]
                paths_gpu[:, i+1] = paths_gpu[:, i] * cp.exp(returns)
            
            # Move results back to CPU
            paths = cp.asnumpy(paths_gpu)
            
            # Compute statistics on CPU using NumPy
            mean_path = np.mean(paths, axis=0)
            median_path = np.median(paths, axis=0)
            upper_95 = np.percentile(paths, 95, axis=0)
            lower_5 = np.percentile(paths, 5, axis=0)
            
            # Find most likely path (closest to mean)
            path_diffs = np.sum((paths - mean_path) ** 2, axis=1)
            most_likely_idx = np.argmin(path_diffs)
            most_likely_path = paths[most_likely_idx]
            
            # Analysis results
            path_analysis = {
                'mean_path': mean_path,
                'median_path': median_path, 
                'most_likely_path': most_likely_path,
                'upper_95': upper_95,
                'lower_5': lower_5
            }
            
            return paths, path_analysis
            
        except Exception as e:
            print(f"GPU simulation failed: {e}, falling back to CPU")
            return self.simulate_paths_fast(n_steps, n_paths)

class PathAnalyzer:
    """Analyzes and clusters simulation paths."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters)
    
    def analyze_paths(self, paths: np.ndarray) -> Dict:
        """Cluster paths and identify representative trajectories."""
        # Scale paths for clustering - no transpose needed initially
        scaled_paths = self.scaler.fit_transform(paths)
        
        # Perform clustering
        labels = self.kmeans.fit_predict(scaled_paths)
        
        # Find most central path for each cluster
        centroids = self.kmeans.cluster_centers_
        representative_paths = []
        cluster_sizes = []
        
        for i in range(self.n_clusters):
            cluster_paths = paths[labels == i]
            cluster_sizes.append(len(cluster_paths))
            
            # Find path closest to centroid
            distances = np.linalg.norm(
                self.scaler.transform(cluster_paths) - centroids[i],
                axis=1
            )
            representative_paths.append(cluster_paths[np.argmin(distances)])
        
        return {
            'labels': labels,
            'representative_paths': representative_paths,
            'cluster_sizes': cluster_sizes
        }

class FractalVisualizer:
    """Creates interactive visualizations of fractal analysis and simulations."""
    
    @staticmethod
    def plot_analysis_and_forecast(
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray
    ) -> go.Figure:
        """Create comprehensive visualization with return distribution comparison."""
        paths, path_analysis = simulation_results
        
        # For very large path counts, use density visualization instead of individual paths
        use_density_plot = paths.shape[0] > 5000
        
        # Create future dates for forecast
        last_date = dates[-1]
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=paths.shape[1]+1,
            freq='B'
        )
        
        # Create subplots with proper specifications
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Forecast with Probability Cloud',
                'Return Distribution Comparison',
                'Historical Pattern Matches',
                'Fractal Statistics'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],  # First row: price plot and distribution
                [{"type": "xy"}, {"type": "table"}]  # Second row: patterns and stats
            ],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot historical prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=historical_prices,
                name='Historical',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        if use_density_plot:
            # DENSITY VISUALIZATION APPROACH - with professional styling and preserved volatility
            
            # First, select a few representative paths to show the jaggedness
            num_sample_paths = 3  # Show a few sample paths
            if paths.shape[0] > num_sample_paths:
                # Get some diverse paths by sampling from different percentiles
                sample_indices = []
                for p in np.linspace(25, 75, num_sample_paths):
                    # Find path closest to this percentile at the end point
                    target_value = np.percentile(paths[:, -1], p)
                    idx = np.argmin(np.abs(paths[:, -1] - target_value))
                    sample_indices.append(idx)
                
                sample_paths = paths[sample_indices]
            
            # Calculate percentiles at each time step
            percentiles = [5, 25, 50, 75, 95]
            percentile_paths = np.zeros((len(percentiles), paths.shape[1]))
            
            for t in range(paths.shape[1]):
                percentile_paths[:, t] = np.percentile(paths[:, t], percentiles)
            
            # Plot the 90% confidence band with shading
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[4]]),  # 95th
                    name='95th percentile',
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[0]]),  # 5th
                    name='5th percentile',
                    fill='tonexty',
                    fillcolor='rgba(255,82,82,0.15)',  # Very light red
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[3]]),  # 75th
                    name='75th percentile',
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[1]]),  # 25th
                    name='25th percentile',
                    fill='tonexty',
                    fillcolor='rgba(255,82,82,0.3)',  # Medium light red
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[2]]),  # 50th
                    name='Median forecast',
                    line=dict(color='rgb(255,82,82)', width=2)  # Solid red line
                ),
                row=1, col=1
            )
            
            # Add sample jagged paths to show volatility
            for i, path in enumerate(sample_paths):
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=np.concatenate([[historical_prices[-1]], path]),
                        name=f'Sample path {i+1}',
                        opacity=0.7,
                        line=dict(color='rgba(255,82,82,0.7)', width=0.7, dash='dot'),
                        showlegend=(i==0)  # Only show one in legend
                    ),
                    row=1, col=1
                )
            
            # Plot most likely path if available and different from median
            if 'most_likely_path' in path_analysis:
                most_likely = path_analysis['most_likely_path']
                # Only add if it's significantly different from the median
                median_path = percentile_paths[2]
                if np.mean(np.abs(most_likely - median_path)) > 0.01 * np.mean(median_path):
                    if 'cluster_probs' in path_analysis:
                        prob = path_analysis["cluster_probs"][np.argmax(path_analysis["cluster_probs"])]
                    else:
                        prob = np.max(path_analysis['path_probabilities'])
                        
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_dates,
                            y=np.concatenate([[historical_prices[-1]], most_likely]),
                            name=f'Most likely path ({prob:.0%})',
                            line=dict(color='rgb(128,0,0)', width=2, dash='dot')  # Darker red, dotted
                        ),
                        row=1, col=1
                    )
        else:
            # INDIVIDUAL PATHS APPROACH - for when there are fewer paths
            # Plot all paths with probability-based coloring
            cloud_paths = paths
            path_probs = path_analysis['path_probabilities']
            
            # Sort paths by probability
            sort_idx = np.argsort(path_probs)
            cloud_paths = cloud_paths[sort_idx]
            path_probs = path_probs[sort_idx]
            
            # Find 95th percentile probability to identify top paths
            prob_95th = np.percentile(path_probs, 95)
            
            # Plot paths from lowest to highest probability
            for i, path in enumerate(cloud_paths):
                prob = path_probs[i]
                
                # Aggressive exponential scaling to emphasize only the highest probability paths
                # Scale relative to 95th percentile for better highlighting
                rel_prob = (prob / prob_95th) ** 3  # Cube for more aggressive scaling
                
                if prob > prob_95th:
                    # Top 5% paths: Purple to Red transition
                    r = 1.0
                    b = max(0.0, 1.0 - (prob - prob_95th) / (np.max(path_probs) - prob_95th))
                    opacity = 0.8
                    width = 2.0
                else:
                    # Bottom 95% paths: Light blue with very low opacity
                    r = 0.0
                    b = 1.0
                    opacity = 0.03 + 0.07 * rel_prob  # Very subtle
                    width = 0.5
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=np.concatenate([[historical_prices[-1]], path]),
                        name=f'Path (p={prob:.1%})',
                        line=dict(
                            color=f'rgba({int(r*255)},0,{int(b*255)},{opacity})', 
                            width=width
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Plot most likely path if available
            if 'most_likely_path' in path_analysis:
                most_likely = path_analysis['most_likely_path']
                if 'cluster_probs' in path_analysis:
                    prob = path_analysis["cluster_probs"][np.argmax(path_analysis["cluster_probs"])]
                else:
                    prob = np.max(path_probs)
                    
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=np.concatenate([[historical_prices[-1]], most_likely]),
                        name=f'Most Likely Path ({prob:.0%})',
                        line=dict(color='red', width=3)
                    ),
                    row=1, col=1
                )
        
        # 2. Return distribution comparison
        historical_returns = np.diff(np.log(historical_prices))
        all_forecast_returns = np.diff(np.log(paths), axis=1).flatten()

        # Plot return distributions
        fig.add_trace(
            go.Histogram(
                x=historical_returns,
                name='Historical Returns',
                nbinsx=30,
                opacity=0.7,
                histnorm='probability',
                marker_color='blue'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(
                x=all_forecast_returns,
                name='Forecast Returns',
                nbinsx=30,
                opacity=0.7,
                histnorm='probability',
                marker_color='red'
            ),
            row=1, col=2
        )

        # Calculate distribution statistics
        hist_mean = np.mean(historical_returns)
        hist_std = np.std(historical_returns)
        fore_mean = np.mean(all_forecast_returns)
        fore_std = np.std(all_forecast_returns)

        # Add distribution statistics annotation
        stats_text = (
            f"Historical: μ={hist_mean:.3%}, σ={hist_std:.3%}<br>"
            f"Forecast: μ={fore_mean:.3%}, σ={fore_std:.3%}"
        )

        fig.add_annotation(
            text=stats_text,
            xref="x2", yref="y2",
            x=min(min(historical_returns), min(all_forecast_returns)),
            y=0.9,
            showarrow=False,
            font=dict(size=10)
        )

        # Update return distribution layout
        fig.update_xaxes(
            title_text="Returns",
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Probability",
            row=1, col=2
        )

        # 3. Add pattern matches if available
        if 'pattern_matches' in path_analysis and len(path_analysis['pattern_matches']) > 0:
            pattern_matches = path_analysis['pattern_matches']
            for i, pattern in enumerate(pattern_matches[:5]):  # Show up to 5 patterns
                match_start = pattern['historical_idx']
                match_end = match_start + pattern['segment_length']
                match_pattern = historical_prices[match_start:match_end]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates[match_start:match_end],
                        y=match_pattern,
                        name=f'Pattern {i+1}',
                        line=dict(dash='dot')
                    ),
                    row=2, col=1
                )

        # 4. Statistics table
        prob = 0
        if 'cluster_probs' in path_analysis and len(path_analysis['cluster_probs']) > 0:
            prob = max(path_analysis['cluster_probs'])
        elif 'path_probabilities' in path_analysis:
            prob = max(path_analysis['path_probabilities'])

        stats_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Hurst Exponent', f"{analysis_results['hurst']:.3f}", '>0.5 suggests trend persistence'],
            ['Fractal Dimension', f"{analysis_results['fractal_dim']:.3f}", 'higher = more volatile'],
            ['Pattern Matches', str(len(path_analysis.get('pattern_matches', []))), 'matching forecast paths'],
            ['Forecast Paths', f"{paths.shape[0]:,}", 'simulated trajectories'],
            ['Forecast Horizon', f"{paths.shape[1]} steps", 'prediction length']
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>', '<b>Interpretation</b>'],
                    font=dict(size=12),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*stats_data)),
                    font=dict(size=11),
                    align='left'
                )
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            title_text="Fractal Pattern Analysis and Forecast",
            title_x=0.5
        )

        # Add explanatory annotations
        fig.add_annotation(
            text="Gray cloud shows possible paths<br>Red line shows most likely trajectory",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10)
        )

        fig.add_annotation(
            text="Historical patterns matching<br>forecast trajectories",
            xref="paper", yref="paper",
            x=0.02, y=0.45,
            showarrow=False,
            font=dict(size=10)
        )

        return fig

    def plot_quantum_analysis(self, prices, quantum_results, dates=None):
        """Plot quantum analysis results."""
        fig = make_subplots(
            rows=3, cols=1, 
            subplot_titles=[
                "Price History", 
                "Quantum Price Levels", 
                "Multidimensional Fractal Analysis"
            ],
            vertical_spacing=0.1,
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # Add price history
        if dates is not None:
            fig.add_trace(go.Scatter(x=dates, y=prices, name="Price"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(y=prices, name="Price"), row=1, col=1)
        
        # Add quantum price levels
        qpl = quantum_results['price_levels']['levels']
        for level in qpl:
            fig.add_trace(
                go.Scatter(
                    x=[0, len(prices)], 
                    y=[level['price'], level['price']], 
                    name=f"QPL: {level['price']:.2f}",
                    line=dict(dash="dash", width=1, color=f"rgba(255, 0, 0, {level['probability']:.2f})")
                ),
                row=2, col=1
            )
        
        # Add hurst exponent heatmap
        multi_results = quantum_results['multi_dimensional']
        fig.add_trace(
            go.Heatmap(
                z=multi_results['cross_correlations'],
                x=['Price', 'Volume'],
                y=['Price', 'Volume'],
                colorscale='Viridis',
                name="Cross-Correlations"
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text="Quantum Fractal Analysis"
        )
        
        return fig

    def plot_high_density_forecast(
        self,
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray
    ) -> go.Figure:
        """Create high-performance density visualization showing sample paths with percentile coloring."""
        paths, path_analysis = simulation_results
        
        # Create future dates for forecast
        last_date = dates[-1]
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=paths.shape[1]+1,
            freq='B'
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=historical_prices,
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )
        
        # Calculate percentiles at each time step
        percentiles = [5, 25, 50, 75, 95]
        percentile_paths = np.zeros((len(percentiles), paths.shape[1]))
        
        for t in range(paths.shape[1]):
            percentile_paths[:, t] = np.percentile(paths[:, t], percentiles)
        
        # Define color scale
        def get_color(percentile, opacity=0.15):
            """Get color based on percentile (blue->green->red)."""
            if percentile < 50:
                # Blue (0,0,255) to Green (0,255,0)
                ratio = percentile / 50
                r = 0
                g = int(255 * ratio)
                b = int(255 * (1 - ratio))
            else:
                # Green (0,255,0) to Red (255,0,0)
                ratio = (percentile - 50) / 50
                r = int(255 * ratio)
                g = int(255 * (1 - ratio))
                b = 0
            return f'rgba({r},{g},{b},{opacity})'
        
        # Add sample paths - limit to 100 for performance
        max_samples = min(100, paths.shape[0])
        sample_indices = np.linspace(0, paths.shape[0]-1, max_samples, dtype=int)
        
        # Calculate final values for each path to determine percentiles
        final_values = paths[sample_indices, -1]
        ranks = np.argsort(np.argsort(final_values)) / len(final_values) * 100
        
        # Plot sampled paths
        for i, idx in enumerate(sample_indices):
            percentile = ranks[i]
            color = get_color(percentile)
            path = paths[idx]
            
            # No legend entries for individual paths
            fig.add_trace(
                go.Scatter(  # Use regular Scatter instead of Scattergl for compatibility
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], path]),
                    name=f'Path {i+1}',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='none'  # Disable hover for performance
                )
            )
        
        # Add shaded areas for percentile bands
        # 5-95 percentile band
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[4]]),  # 95th
                name='95th percentile',
                line=dict(color='rgba(255,0,0,0.5)', width=1.5, dash='dot'),
                showlegend=True
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[0]]),  # 5th
                name='5th percentile',
                fill='tonexty',
                fillcolor='rgba(150,150,150,0.2)',  # Light gray
                line=dict(color='rgba(0,0,255,0.5)', width=1.5, dash='dot'),
                showlegend=True
            )
        )
        
        # 25-75 percentile band (interquartile range)
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[3]]),  # 75th
                name='75th percentile',
                line=dict(color='rgba(255,0,0,0.8)', width=1.5),
                showlegend=True
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[1]]),  # 25th
                name='25th percentile',
                fill='tonexty',
                fillcolor='rgba(150,150,150,0.4)',  # Darker gray
                line=dict(color='rgba(0,0,255,0.8)', width=1.5),
                showlegend=True
            )
        )
        
        # Median path
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[2]]),  # 50th
                name='Median forecast',
                line=dict(color='purple', width=2),
                showlegend=True
            )
        )
        
        # Add most likely path if available
        if 'most_likely_path' in path_analysis:
            most_likely = path_analysis['most_likely_path']
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], most_likely]),
                    name='Most likely path',
                    line=dict(color='black', width=2.5),
                    showlegend=True
                )
            )
        
        # Configure layout
        fig.update_layout(
            title="High Density Path Visualization",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

def run_backtest(
    symbols: list, 
    sample_count: int, 
    start_date: str,
    end_date: str,
    forecast_horizon: int,
    metrics: list,
    benchmarks: list,
    progress_callback=None,
    status_callback=None,
    cancellation_callback=None,
    parallel=True,
    max_workers=4,
    shared_cancellation_flag=None
) -> dict:
    """Run a comprehensive backtest of the fractal forecasting model with parallel processing."""
    symbol_results = {}
    all_samples = []
    
    # Use a shared flag if provided, otherwise create a new one
    cancellation_flag = shared_cancellation_flag if shared_cancellation_flag is not None else [False]
    
    # Keep track of total samples processed
    total_samples = 0
    
    if parallel:
        message = f"Starting backtest with {len(symbols)} symbols, {sample_count} samples each"
        print(message)
        if status_callback:
            try:
                status_callback(message)
            except:
                print("Status callback failed - likely a session state issue")
            
        message = f"Running in parallel mode with up to {max_workers} workers"
        print(message)
        if status_callback:
            try:
                status_callback(message)
            except:
                print("Status callback failed - likely a session state issue")
        
        # Create a wrapper for each symbol that includes the required parameters
        def process_with_params(symbol):
            return process_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                forecast_horizon=forecast_horizon,
                sample_count=sample_count,
                metrics=metrics,
                benchmarks=benchmarks,
                status_callback=None,  # Don't pass the status_callback to the worker
                progress_callback=None,  # Don't pass the progress_callback to the worker
                cancellation_flag=cancellation_flag
            )
            
        # Process symbols in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            # Start all workers
            future_results = {executor.submit(process_with_params, symbol): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in as_completed(future_results):
                try:
                    symbol = future_results[future]
                    result = future.result()
                    
                    # Check for cancellation
                    if cancellation_flag[0]:
                        break
                    
                    # Store valid results
                    if result is not None and 'samples' in result and len(result['samples']) > 0:
                        symbol_results[symbol] = result
                        all_samples.extend(result['samples'])
                        total_samples += len(result['samples'])
                        print(f"Added {len(result['samples'])} samples from {symbol}")
                        
                except Exception as e:
                    print(f"Error processing {future_results[future]}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Sequential processing
        for symbol in symbols:
            if cancellation_callback and cancellation_callback():
                break
                
            try:
                result = process_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    forecast_horizon=forecast_horizon,
                    sample_count=sample_count,
                    metrics=metrics,
                    benchmarks=benchmarks,
                    status_callback=status_callback,
                    progress_callback=progress_callback,
                    cancellation_flag=None  # No shared flag needed for sequential
                )
                
                if result is not None and 'samples' in result and len(result['samples']) > 0:
                    symbol_results[symbol] = result
                    all_samples.extend(result['samples'])
                    total_samples += len(result['samples'])
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                if status_callback:
                    try:
                        status_callback(f"Error processing {symbol}: {e}")
                    except:
                        print("Status callback failed - likely a session state issue")
    
    # Process the final results
    message = f"Backtest complete with {total_samples} total samples across {len(symbol_results)} symbols"
    print(message)
    
    if status_callback:
        try:
            status_callback(message)
        except:
            print("Status callback failed - likely session state issue")
    
    # Calculate aggregate metrics if we have samples
    if len(all_samples) > 0:
        aggregate_metrics = aggregate_sample_metrics(all_samples, metrics, benchmarks)
    else:
        aggregate_metrics = {}
    
    # Return the results
    return {
        'aggregate_metrics': aggregate_metrics,
        'symbol_results': symbol_results
    }

def backtest_symbol(
    symbol: str, 
    prices: np.ndarray, 
    dates: np.ndarray, 
    sample_count: int, 
    forecast_horizon: int,
    metrics: list,
    benchmarks: list,
    cancellation_callback=None
) -> dict:
    """Run backtest for a single symbol."""
    samples = []
    
    print(f"Starting backtest for {symbol} with {sample_count} samples")
    
    # Determine valid range for test windows
    min_test_start = 252  # Require at least 1 year of data
    max_test_start = len(prices) - forecast_horizon
    
    if max_test_start <= min_test_start:
        print(f"Not enough data for {symbol}, need at least {min_test_start + forecast_horizon} points")
        return {'samples': [], 'aggregated_metrics': {}}
    
    for i in range(sample_count):
        # Check for cancellation
        if cancellation_callback and cancellation_callback():
            print(f"Backtest cancelled after {len(samples)} samples")
            break
            
        try:
            # Randomly select a test start point
            test_start = np.random.randint(min_test_start, max_test_start)
            
            # Split data into train/test
            train_prices = prices[:test_start]  # Changed from idx to test_start
            train_dates = dates[:test_start]    # Changed from idx to test_start
            test_prices = prices[test_start:test_start+forecast_horizon]
            test_dates = dates[test_start:test_start+forecast_horizon]
            
            if len(test_prices) < forecast_horizon:
                continue  # Skip if not enough test data
            
            # Initialize analyzer and simulator
            analyzer = FractalAnalyzer()
            simulator = FractalSimulator(train_prices, analyzer)
            
            # Generate forecast paths and calculate representative path
            paths, path_analysis = simulator.simulate_paths_fast(n_steps=forecast_horizon, n_paths=100)
            forecast_path = path_analysis['most_likely_path']
            
            # Verify forecast path shape
            print(f"Sample {i}: Forecast path shape {forecast_path.shape}, Test prices shape {test_prices.shape}")
            
            # Generate benchmark forecasts
            benchmark_forecasts = {}
            
            if 'Random Walk' in benchmarks:
                # Last price + random normal noise based on historical volatility
                hist_returns = np.diff(np.log(train_prices[-30:]))  # Use last 30 days for volatility
                daily_vol = np.std(hist_returns)
                
                rw_returns = np.random.normal(0, daily_vol, size=forecast_horizon)
                rw_forecast = train_prices[-1] * np.exp(np.cumsum(rw_returns))
                benchmark_forecasts['Random Walk'] = rw_forecast
            
            if 'Simple Moving Average' in benchmarks:
                # 5-day SMA continuation
                window = 5
                sma = np.mean(train_prices[-window:])
                sma_forecast = np.ones(forecast_horizon) * sma
                benchmark_forecasts['Simple Moving Average'] = sma_forecast
                
            if 'ARIMA' in benchmarks or 'SARIMA' in benchmarks:
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    from pmdarima import auto_arima  # We'll need to add this to requirements.txt
                    
                    # Let auto_arima find the best parameters
                    train_series = pd.Series(train_prices[-max(60, forecast_horizon*3):])
                    
                    # For shorter samples, use simpler models
                    if len(train_series) < 100:
                        model = auto_arima(train_series, start_p=0, start_q=0,
                                   max_p=2, max_q=2, m=5,
                                   seasonal=True, d=1, D=1, trace=False,
                                   error_action='ignore',  
                                   suppress_warnings=True, 
                                   stepwise=True)
                    else:
                        model = auto_arima(train_series, seasonal=True, m=5,
                                   error_action='ignore',  
                                   suppress_warnings=True)
                                   
                    # Generate forecast
                    arima_forecast = model.predict(n_periods=forecast_horizon)
                    benchmark_forecasts['ARIMA'] = arima_forecast
                except Exception as e:
                    print(f"Error with ARIMA benchmark: {e}")
                    # Fallback to simpler model if auto_arima fails
                    benchmark_forecasts['ARIMA'] = np.ones(forecast_horizon) * train_prices[-1]
            
            # Calculate performance metrics for fractal model
            fractal_metrics = calculate_forecast_metrics(test_prices, forecast_path, metrics)
            print(f"Sample {i}: Calculated metrics: {fractal_metrics}")
            
            sample_results = {
                'symbol': symbol,
                'start_date': train_dates[-1],
                'end_date': test_dates[-1],
                'train_prices': train_prices,
                'test_prices': test_prices,
                'forecast_path': forecast_path,
                'benchmark_forecasts': benchmark_forecasts,
                'metrics': fractal_metrics,
                'benchmark_metrics': {}
            }
            
            # Calculate benchmark metrics
            for name, forecast in benchmark_forecasts.items():
                bench_metrics = calculate_forecast_metrics(test_prices, forecast, metrics)
                sample_results['benchmark_metrics'][name] = bench_metrics
                print(f"Sample {i}: {name} metrics: {bench_metrics}")
                
            samples.append(sample_results)
        except Exception as e:
            print(f"Error processing sample {i} for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Completed {len(samples)} valid samples for {symbol}")
    
    # Aggregate metrics across all samples for this symbol
    symbol_metrics = aggregate_sample_metrics(samples, metrics, benchmarks)
    
    return {
        'samples': samples,
        'aggregated_metrics': symbol_metrics
    }

def calculate_forecast_metrics(actual: np.ndarray, forecast: np.ndarray, metrics: list) -> dict:
    """Calculate performance metrics for a forecast."""
    results = {}
    
    # Ensure arrays are the same length
    min_len = min(len(actual), len(forecast))
    actual = actual[:min_len]
    forecast = forecast[:min_len]
    
    if 'MAPE' in metrics:
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        results['MAPE'] = mape
    
    if 'RMSE' in metrics:
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        results['RMSE'] = rmse
    
    if 'Direction Accuracy' in metrics:
        # Direction prediction accuracy
        actual_dirs = np.sign(np.diff(actual))
        forecast_dirs = np.sign(np.diff(forecast))
        
        # Count correct direction predictions
        correct_dirs = np.sum(actual_dirs == forecast_dirs)
        total_dirs = len(actual_dirs)
        
        direction_accuracy = correct_dirs / total_dirs * 100 if total_dirs > 0 else 0
        results['Direction Accuracy'] = direction_accuracy
    
    return results

def aggregate_sample_metrics(samples: list, metrics: list, benchmarks: list) -> dict:
    """Aggregate metrics across all samples for a symbol."""
    # Initialize with proper structure even if no samples
    aggregated = {
        'fractal_model': {
            metric: {'mean': 0, 'median': 0, 'std': 0, 'win_rate': 0} 
            for metric in metrics
        },
        'benchmarks': {
            benchmark: {
                metric: {'mean': 0, 'median': 0, 'std': 0} 
                for metric in metrics
            } 
            for benchmark in benchmarks
        }
    }
    
    # If no samples, return the initialized structure
    if not samples:
        return aggregated
    
    # Reset values to be calculated from samples
    for metric in metrics:
        aggregated['fractal_model'][metric] = {
            'mean': 0, 'median': 0, 'std': 0, 'win_rate': 0
        }
        
    # Collect all metrics
    for metric in metrics:
        fractal_values = [s['metrics'].get(metric, np.nan) for s in samples]
        aggregated['fractal_model'][metric] = {
            'mean': np.nanmean(fractal_values),
            'median': np.nanmedian(fractal_values),
            'std': np.nanstd(fractal_values),
            'win_rate': 0  # Will calculate after getting benchmark metrics
        }
        
        # Collect benchmark metrics
        for benchmark in benchmarks:
            bench_values = [s['benchmark_metrics'].get(benchmark, {}).get(metric, np.nan) 
                           for s in samples]
            
            aggregated['benchmarks'][benchmark][metric] = {
                'mean': np.nanmean(bench_values),
                'median': np.nanmedian(bench_values),
                'std': np.nanstd(bench_values)
            }
            
            # Calculate win rate against benchmark
            if metric in ['MAPE', 'RMSE']:  # Lower is better
                wins = sum(fv < bv for fv, bv in zip(fractal_values, bench_values) 
                          if not np.isnan(fv) and not np.isnan(bv))
            else:  # Higher is better
                wins = sum(fv > bv for fv, bv in zip(fractal_values, bench_values)
                          if not np.isnan(fv) and not np.isnan(bv))
                
            total = sum(1 for fv, bv in zip(fractal_values, bench_values)
                       if not np.isnan(fv) and not np.isnan(bv))
            
            if total > 0:
                aggregated['fractal_model'][metric]['win_rate'] = wins / total * 100
            
    return aggregated

def aggregate_backtest_results(symbol_results: dict, metrics: list, benchmarks: list) -> dict:
    """Aggregate results across all symbols."""
    all_samples = []
    
    for symbol, results in symbol_results.items():
        all_samples.extend(results['samples'])
    
    return aggregate_sample_metrics(all_samples, metrics, benchmarks)

def process_symbol(
    symbol,
    start_date,
    end_date,
    forecast_horizon,
    sample_count,
    metrics,
    benchmarks,
    status_callback=None,
    progress_callback=None,
    cancellation_flag=None
):
    """Worker function for parallel processing that doesn't access st.session_state."""
    try:
        # Set default cancellation flag if none provided
        if cancellation_flag is None:
            cancellation_flag = [False]
            
        # Get full historical data
        full_data = get_yahoo_data(symbol, start_date, end_date)
        prices = full_data['Close'].to_numpy()
        dates = full_data['Date'].to_numpy()
        
        if len(prices) < forecast_horizon * 2:
            print(f"Not enough data for {symbol}, skipping")
            if status_callback:
                status_callback(f"Not enough data for {symbol}, skipping")
            return None
        
        # Define a local cancellation callback that uses the shared flag
        def local_cancellation_check():
            return cancellation_flag[0] if cancellation_flag else False
            
        # Process the symbol
        symbol_results = backtest_symbol(
            symbol, prices, dates, sample_count, forecast_horizon, metrics, benchmarks,
            cancellation_callback=local_cancellation_check
        )
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(symbol, len(symbol_results['samples']))
            
        return symbol_results
        
    except Exception as e:
        print(f"Error backtesting {symbol}: {e}")
        if status_callback:
            status_callback(f"Error backtesting {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    # Get data
    symbol = "^GSPC"  # S&P 500
    data = get_yahoo_data(symbol, "2020-01-01")
    prices = data['Close'].to_numpy()
    
    # Initialize components
    analyzer = FractalAnalyzer()
    simulator = FractalSimulator(prices, analyzer)
    path_analyzer = PathAnalyzer()
    visualizer = FractalVisualizer()
    
    # Run analysis and simulation
    analysis_results = analyzer.analyze_patterns(prices)
    paths, path_analysis = simulator.simulate_paths_fast(n_steps=30, n_paths=100)
    path_analysis = path_analyzer.analyze_paths(paths)
    
    # Visualize results
    fig = visualizer.plot_analysis_and_forecast(
        prices,
        (paths, path_analysis),
        analysis_results,
        data['Date'].to_numpy()
    )
    fig.show()