import numpy as np
import polars as pl
import requests
from io import StringIO
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from numba import njit, float64, int64
from numba.typed import List
from typing import Tuple, List as PyList, Dict
import warnings
import yfinance as yf
import pandas as pd
from fractime.optimization import compute_box_dimension_safe
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

def _ensure_numpy_array(data):
    """Convert Polars Series or other array-like to NumPy array."""
    if data is None:
        return None
    if isinstance(data, pl.Series):
        return data.to_numpy()
    if isinstance(data, (list, tuple)):
        return np.array(data)
    if isinstance(data, pd.Series):
        return data.to_numpy()
    return np.asarray(data)

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

class CrossDimensionalAnalyzer:
    """
    Analyzes fractal correlations across multiple dimensions (e.g., price and volume).
    
    This class implements the concept that different market dimensions (price, volume,
    volatility) exhibit fractal correlations that can provide deeper insights into
    market regimes and transitions.
    """
    
    def __init__(self):
        """Initialize cross-dimensional analyzer."""
        self.dimensions = {}
        self.correlation_matrix = None
        self.fractal_coherence = None
        self.regime_states = None
    
    def add_dimension(self, name: str, data: np.ndarray) -> None:
        """
        Add a market dimension for cross-analysis.

        Args:
            name: Dimension name (e.g., "price", "volume")
            data: Time series data for this dimension
        """
        # Convert to NumPy array if needed (e.g., from Polars Series)
        self.dimensions[name] = _ensure_numpy_array(data)
        self._invalidate_cache()
    
    def _invalidate_cache(self) -> None:
        """Reset cached computations when dimensions change."""
        self.correlation_matrix = None
        self.fractal_coherence = None
        self.regime_states = None
    
    def compute_fractal_dimensions(self) -> Dict[str, float]:
        """Compute fractal dimension for each data dimension."""
        fractal_dims = {}
        analyzer = FractalAnalyzer()
        
        for name, data in self.dimensions.items():
            # Calculate fractal dimension using existing analyzer
            fractal_dims[name] = analyzer.compute_fractal_dimension(data)
        
        return fractal_dims
    
    def compute_hurst_exponents(self) -> Dict[str, float]:
        """Compute Hurst exponent for each data dimension."""
        hurst_exponents = {}
        analyzer = FractalAnalyzer()
        
        for name, data in self.dimensions.items():
            # Calculate Hurst exponent using existing analyzer
            hurst_exponents[name] = analyzer.compute_hurst(data)
        
        return hurst_exponents
    
    def compute_cross_correlation(self, window_size: int = 20) -> np.ndarray:
        """
        Compute cross-correlation matrix between dimensions at different scales.
        
        Args:
            window_size: Size of rolling window for local correlation analysis
            
        Returns:
            Correlation matrix (dimensions × dimensions)
        """
        # Get dimension names in consistent order
        dim_names = list(self.dimensions.keys())
        n_dims = len(dim_names)
        
        if n_dims < 2:
            raise ValueError("Need at least 2 dimensions for cross-correlation analysis")
            
        # Pre-compute returns/changes for each dimension
        changes = {}
        for name in dim_names:
            # Get log returns for price-like series, or percentage changes for others
            data = self.dimensions[name]
            changes[name] = np.diff(np.log(data)) if np.min(data) > 0 else np.diff(data) / data[:-1]
            
        # Find the smallest valid length across all dimensions
        min_length = min(len(changes[name]) for name in dim_names)
        
        # Compute rolling correlation matrix
        n_windows = min_length - window_size + 1
        if n_windows <= 0:
            raise ValueError(f"Window size {window_size} is too large for data length {min_length}")
            
        # Initialize correlation matrices at different scales
        # We'll consider 3 scales: short-term, medium-term, and long-term
        scales = [window_size, window_size*2, window_size*4]
        scale_weights = [0.5, 0.3, 0.2]  # Weighting for different scales
        
        cross_corr_by_scale = {}
        
        for scale in scales:
            if min_length <= scale:
                continue  # Skip scales that are too large
                
            # Calculate correlation for this scale
            corr_matrix = np.zeros((n_dims, n_dims))
            
            for i, name1 in enumerate(dim_names):
                change1 = changes[name1][:min_length]
                
                for j, name2 in enumerate(dim_names):
                    if i == j:
                        corr_matrix[i, j] = 1.0  # Self-correlation
                        continue
                        
                    change2 = changes[name2][:min_length]
                    
                    try:
                        # Calculate correlation with safety checks
                        if np.std(change1) > 0 and np.std(change2) > 0:
                            corr = np.corrcoef(change1, change2)[0, 1]
                            if not np.isnan(corr):
                                corr_matrix[i, j] = corr
                    except Exception:
                        pass  # Keep default 0 correlation on error
                        
            cross_corr_by_scale[scale] = corr_matrix
        
        # Combine scales with weights
        if not cross_corr_by_scale:
            # If no valid scales, return identity matrix
            self.correlation_matrix = np.eye(n_dims)
            return self.correlation_matrix
            
        # Combine with weights
        combined_matrix = np.zeros((n_dims, n_dims))
        total_weight = 0
        
        for idx, scale in enumerate(scales):
            if scale in cross_corr_by_scale:
                weight = scale_weights[idx]
                combined_matrix += weight * cross_corr_by_scale[scale]
                total_weight += weight
                
        if total_weight > 0:
            combined_matrix /= total_weight
            
        # Ensure diagonal is 1
        np.fill_diagonal(combined_matrix, 1.0)
        
        self.correlation_matrix = combined_matrix
        return combined_matrix
    
    def compute_fractal_coherence(self, window_sizes=[5, 21, 63]) -> Dict:
        """
        Compute fractal coherence between dimensions.
        
        Fractal coherence measures how consistent the fractal properties
        are between different dimensions over time. Higher coherence
        suggests stronger cross-dimensional coupling.
        
        Args:
            window_sizes: List of window sizes for multi-scale analysis
            
        Returns:
            Dictionary of coherence metrics
        """
        dim_names = list(self.dimensions.keys())
        n_dims = len(dim_names)
        
        if n_dims < 2:
            return {'coherence': 0}
            
        # Compute local Hurst exponents and fractal dimensions at different scales
        local_hurst = {}
        local_fractal_dim = {}
        
        for name in dim_names:
            data = self.dimensions[name]
            local_hurst[name] = {}
            local_fractal_dim[name] = {}
            
            for window in window_sizes:
                if len(data) <= window:
                    continue
                    
                # Calculate rolling Hurst exponent
                h_values = []
                fd_values = []
                analyzer = FractalAnalyzer()
                
                for i in range(window, len(data), window//2):  # Use overlapping windows
                    segment = data[i-window:i]
                    h_values.append(analyzer.compute_hurst(segment))
                    fd_values.append(analyzer.compute_fractal_dimension(segment, quick_mode=True))
                
                local_hurst[name][window] = np.array(h_values)
                local_fractal_dim[name][window] = np.array(fd_values)
        
        # Calculate coherence at each scale
        coherence_by_scale = {}
        
        for window in window_sizes:
            # Check if we have data for this window
            valid_dims = [name for name in dim_names 
                         if window in local_hurst[name] and len(local_hurst[name][window]) > 0]
            
            if len(valid_dims) < 2:
                continue
                
            # Calculate pairwise correlations in fractal metrics
            h_corr = np.zeros((len(valid_dims), len(valid_dims)))
            fd_corr = np.zeros((len(valid_dims), len(valid_dims)))
            
            for i, name1 in enumerate(valid_dims):
                for j, name2 in enumerate(valid_dims):
                    if i >= j:  # Only calculate upper triangle
                        continue
                        
                    h1 = local_hurst[name1][window]
                    h2 = local_hurst[name2][window]
                    
                    fd1 = local_fractal_dim[name1][window]
                    fd2 = local_fractal_dim[name2][window]
                    
                    # Ensure same length
                    min_len = min(len(h1), len(h2))
                    if min_len > 1:
                        h1, h2 = h1[:min_len], h2[:min_len]
                        try:
                            h_corr[i, j] = abs(np.corrcoef(h1, h2)[0, 1])
                        except:
                            h_corr[i, j] = 0
                            
                    min_len = min(len(fd1), len(fd2))
                    if min_len > 1:
                        fd1, fd2 = fd1[:min_len], fd2[:min_len]
                        try:
                            fd_corr[i, j] = abs(np.corrcoef(fd1, fd2)[0, 1])
                        except:
                            fd_corr[i, j] = 0
            
            # Make symmetric
            h_corr = h_corr + h_corr.T
            fd_corr = fd_corr + fd_corr.T
            np.fill_diagonal(h_corr, 1.0)
            np.fill_diagonal(fd_corr, 1.0)
            
            # Combine Hurst and fractal dimension coherence
            coherence = 0.5 * h_corr + 0.5 * fd_corr
            coherence_by_scale[window] = coherence
            
        # Calculate average coherence across scales
        if not coherence_by_scale:
            return {'coherence': 0}
            
        # Combine coherence across scales with weights favoring smaller scales
        weights = {window: 1.0/window for window in coherence_by_scale}
        total_weight = sum(weights.values())
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted average of coherence matrices
        avg_coherence = sum(weights[w] * coherence_by_scale[w] for w in coherence_by_scale)
        
        # Calculate overall coherence as average of off-diagonal elements
        off_diag_mask = ~np.eye(avg_coherence.shape[0], dtype=bool)
        overall_coherence = np.mean(avg_coherence[off_diag_mask])
        
        self.fractal_coherence = {
            'by_scale': coherence_by_scale,
            'average': avg_coherence,
            'overall': overall_coherence
        }
        
        return self.fractal_coherence
    
    def identify_regimes(self, n_regimes: int = 3) -> Dict:
        """
        Identify market regimes based on cross-dimensional fractal properties.
        
        Args:
            n_regimes: Number of regime states to identify
            
        Returns:
            Dictionary with regime information
        """
        # Ensure we have computed necessary metrics
        if self.correlation_matrix is None:
            self.compute_cross_correlation()
            
        if self.fractal_coherence is None:
            self.compute_fractal_coherence()
            
        dim_names = list(self.dimensions.keys())
        n_dims = len(dim_names)
        
        # Extract features for regime identification
        features = []
        
        # Ensure at least one dimension (typically price) has fractal metrics
        for name in dim_names:
            data = self.dimensions[name]
            analyzer = FractalAnalyzer()
            
            # Calculate in smaller windows to capture regime changes
            window_size = min(63, len(data) // 5)
            window_size = max(window_size, 20)  # Ensure at least 20 points
            
            # Calculate rolling window metrics
            hurst_values = []
            fractal_dim_values = []
            
            for i in range(window_size, len(data), window_size // 2):
                segment = data[i-window_size:i]
                hurst_values.append(analyzer.compute_hurst(segment))
                fractal_dim_values.append(analyzer.compute_fractal_dimension(segment, quick_mode=True))
                
            if not hurst_values:  # No valid windows
                continue
                
            # Use metrics from most recent window
            features.append(hurst_values[-1])
            features.append(fractal_dim_values[-1])
            
        # Add cross-correlation information
        if n_dims > 1:
            # Flatten upper triangle of correlation matrix
            corr_features = []
            for i in range(n_dims):
                for j in range(i+1, n_dims):
                    corr_features.append(self.correlation_matrix[i, j])
            features.extend(corr_features)
            
        # Add coherence if available
        if isinstance(self.fractal_coherence, dict) and 'overall' in self.fractal_coherence:
            features.append(self.fractal_coherence['overall'])
            
        # Ensure we have features
        if not features:
            return {
                'regime': 0,
                'n_regimes': 1,
                'confidence': 1.0
            }
            
        # Normalize features
        features = np.array(features).reshape(1, -1)
        
        # Determine current regime based on these features
        # We need historical data for this, but as an approximation,
        # we can categorize the current state
        
        # Use simple thresholds for now
        # In a more complete implementation, we would use clustering or HMM
        # on historical feature vectors
        
        # For now, let's use a simple rule-based approach
        if len(features[0]) >= 2:  # At least Hurst and fractal dimension 
            hurst = features[0][0]
            fractal_dim = features[0][1]
            
            if hurst > 0.6:  # Strong trend persistence
                regime = 0  # Trending regime
                confidence = min(1.0, (hurst - 0.6) * 5)  # Scale confidence
            elif hurst < 0.4:  # Mean reversion
                regime = 1  # Mean-reverting regime
                confidence = min(1.0, (0.4 - hurst) * 5)
            else:  # Random walk-like
                regime = 2  # Random walk regime
                mid_point = 0.5
                confidence = 1.0 - abs(hurst - mid_point) * 5
                
            # Adjust confidence based on fractal dimension
            # Higher fractal dimension typically indicates higher volatility
            volatility_factor = min(1.0, max(0.0, (fractal_dim - 1.0) / 0.5))
            
            # Crosscheck with coherence if available
            if len(features[0]) > 2 and features[0][2] > 0.7:
                # High coherence strengthens confidence
                confidence *= 1.2
                confidence = min(1.0, confidence)
        else:
            # Not enough features
            regime = 0
            confidence = 0.5
            
        self.regime_states = {
            'regime': regime,
            'n_regimes': n_regimes,
            'confidence': confidence,
            'features': features[0].tolist(),
            'feature_names': ['hurst', 'fractal_dim'] + 
                            [f'corr_{i}_{j}' for i in range(n_dims) for j in range(i+1, n_dims)] +
                            (['coherence'] if self.fractal_coherence else [])
        }
        
        return self.regime_states
    
    def analyze_dimensions(self) -> Dict:
        """
        Run complete cross-dimensional fractal analysis.
        
        Returns:
            Comprehensive analysis results
        """
        results = {
            'dimensions': list(self.dimensions.keys()),
            'fractal_dimensions': self.compute_fractal_dimensions(),
            'hurst_exponents': self.compute_hurst_exponents(),
            'cross_correlation': self.compute_cross_correlation().tolist(),
        }
        
        # Add fractal coherence if we have multiple dimensions
        if len(self.dimensions) > 1:
            coherence = self.compute_fractal_coherence()
            results['fractal_coherence'] = {
                'overall': coherence['overall']
            }
            
            # Add regime identification
            regime_info = self.identify_regimes()
            results['regime'] = regime_info
            
        return results


class FractalAnalyzer:
    """Analyzes fractal properties of time series data."""
    
    def __init__(self):
        """Initialize with empty cache for performance."""
        self.cache = {}  # Cache for expensive computations
    
    def analyze_patterns(self, prices: np.ndarray, full_analysis=True) -> dict:
        """Analyze with caching and selective feature computation."""
        # Convert to NumPy array if needed (e.g., from Polars Series)
        prices = _ensure_numpy_array(prices)

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
        # Convert to NumPy array if needed (e.g., from Polars Series)
        prices = _ensure_numpy_array(prices)

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
        # Convert to NumPy array if needed (e.g., from Polars Series)
        prices = _ensure_numpy_array(prices)

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

    def __init__(self, prices: np.ndarray, analyzer: FractalAnalyzer, volumes: np.ndarray = None):
        # Convert to NumPy arrays if needed (e.g., from Polars Series)
        self.prices = _ensure_numpy_array(prices)
        self.analyzer = analyzer
        self.patterns = None
        self.hurst = None
        self.volumes = _ensure_numpy_array(volumes)
        self.time_warper = TradingTimeWarper()
        self.cross_dim_analyzer = None
        self.quantum_generator = None
        self.quantum_levels = None
        self._analyze()
        
        # Set up trading time warping if we have volumes
        if self.volumes is not None and len(self.volumes) == len(self.prices):
            self.time_map = self.time_warper.compute_time_transformation(self.prices, self.volumes)
            print("Trading time mapping computed with price and volume data")
            
            # Initialize cross-dimensional analyzer if we have volumes
            self.cross_dim_analyzer = CrossDimensionalAnalyzer()
            self.cross_dim_analyzer.add_dimension("price", self.prices)
            self.cross_dim_analyzer.add_dimension("volume", self.volumes)
            
            # Perform initial cross-dimensional analysis
            self.cross_dim_results = self.cross_dim_analyzer.analyze_dimensions()
            print("Cross-dimensional fractal analysis completed")
        else:
            # Just use prices for time warping if no volumes
            self.time_map = self.time_warper.compute_time_transformation(self.prices)
            print("Trading time mapping computed with price data only")
        
        # Initialize quantum price level generator (optional feature)
        try:
            from fractime.quantum import QuantumPriceLevelGenerator
            self.quantum_generator = QuantumPriceLevelGenerator(energy_levels=5)
            self.quantum_levels = self.quantum_generator.generate_price_levels(self.prices)
            print("Quantum price levels generated")
        except ImportError:
            print("Quantum module not available, skipping quantum price levels")
            self.quantum_generator = None
            self.quantum_levels = None
        
        # Prepare sampled data for faster simulations
        if len(self.prices) > 1000:
            # Create downsampled version for faster regime matching
            sampling_rate = len(self.prices) // 1000
            self.sampled_prices = self.prices[::sampling_rate]
            if self.volumes is not None:
                self.sampled_volumes = self.volumes[::sampling_rate]
            print(f"Created downsampled data: {len(self.sampled_prices)} points")
        else:
            self.sampled_prices = self.prices
            if self.volumes is not None:
                self.sampled_volumes = self.volumes
    
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
        preserve_volatility: bool = True,
        use_trading_time: bool = True,  # Parameter for trading time warping
        warping_alpha: float = 0.5,     # Parameter for warping intensity
        enable_time_forecast: bool = True,  # Whether to forecast time dilation into the future
        use_cross_dim: bool = True,  # Parameter for cross-dimensional filtering
        use_quantum_levels: bool = True,  # Parameter for quantum price level filtering
        quantum_influence: float = 0.5  # How strongly quantum levels influence path selection (0-1)
    ) -> Tuple[np.ndarray, Dict]:
        """Generate paths using regime-matched sampling based on trading time transformation."""
        # Get historical returns
        historical_returns = np.diff(np.log(self.prices))
        
        # Define lookback window as 2x the forecast horizon
        lookback_window = 2 * n_steps
        
        # Get recent volatility regime
        recent_returns = historical_returns[-lookback_window:]
        recent_vol = np.std(recent_returns)
        
        # Update time warper settings if changed
        if warping_alpha != self.time_warper.alpha:
            self.time_warper.alpha = warping_alpha
            # Recompute time warping
            if self.volumes is not None and len(self.volumes) == len(self.prices):
                self.time_map = self.time_warper.compute_time_transformation(self.prices, self.volumes)
            else:
                self.time_map = self.time_warper.compute_time_transformation(self.prices)
            
        # Find similar volatility regimes in history using multiple metrics
        regime_windows = []
        recent_skew = stats.skew(recent_returns)
        recent_kurt = stats.kurtosis(recent_returns)
        
        # If we're using trading time, incorporate dilation factors into regime matching
        if use_trading_time:
            # Get recent time dilation factors
            recent_dilation = self.time_map['dilation_factors'][-lookback_window:]
            recent_dilation_mean = np.mean(recent_dilation)
            recent_dilation_std = np.std(recent_dilation)
            
            for i in range(len(historical_returns) - lookback_window):
                window_returns = historical_returns[i:i+lookback_window]
                window_vol = np.std(window_returns)
                window_skew = stats.skew(window_returns)
                window_kurt = stats.kurtosis(window_returns)
                
                # Get time dilation for this window
                window_dilation = self.time_map['dilation_factors'][i:i+lookback_window]
                window_dilation_mean = np.mean(window_dilation)
                window_dilation_std = np.std(window_dilation)
                
                # Calculate similarities using multiple metrics
                vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                skew_similarity = abs(window_skew - recent_skew)
                kurt_similarity = abs(window_kurt - recent_kurt)
                
                # Time dilation similarity (both mean level and variability)
                dilation_mean_sim = abs(window_dilation_mean - recent_dilation_mean) / max(recent_dilation_mean, 1e-8)
                dilation_std_sim = abs(window_dilation_std - recent_dilation_std) / max(recent_dilation_std, 1e-8)
                
                # Combine similarities with weights (add time dilation factors)
                total_similarity = (
                    0.4 * vol_similarity + 
                    0.15 * skew_similarity + 
                    0.15 * kurt_similarity +
                    0.2 * dilation_mean_sim +
                    0.1 * dilation_std_sim
                )
                
                # If combined similarity is good enough, include this window
                if total_similarity < 0.3:  # Start with stricter threshold
                    regime_windows.append(i)
        else:
            # Original regime matching without time warping
            for i in range(len(historical_returns) - lookback_window):
                window_returns = historical_returns[i:i+lookback_window]
                window_vol = np.std(window_returns)
                window_skew = stats.skew(window_returns)
                window_kurt = stats.kurtosis(window_returns)
                
                # Calculate similarities using multiple metrics
                vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                skew_similarity = abs(window_skew - recent_skew)
                kurt_similarity = abs(window_kurt - recent_kurt)
                
                # Combine similarities with weights
                total_similarity = (
                    0.6 * vol_similarity + 
                    0.25 * skew_similarity + 
                    0.15 * kurt_similarity
                )
                
                # If combined similarity is good enough, include this window
                if total_similarity < 0.3:  # Start with stricter threshold
                    regime_windows.append(i)
        
        # Ensure we have enough similar windows, if not, gradually relax constraint
        similarity_threshold = 0.3
        while len(regime_windows) < 20 and similarity_threshold < 1.0:
            similarity_threshold += 0.1  # More gradual relaxation
            regime_windows = []
            
            if use_trading_time:
                # With time warping
                for i in range(len(historical_returns) - lookback_window):
                    window_returns = historical_returns[i:i+lookback_window]
                    window_vol = np.std(window_returns)
                    window_skew = stats.skew(window_returns)
                    window_kurt = stats.kurtosis(window_returns)
                    
                    # Get time dilation for this window
                    window_dilation = self.time_map['dilation_factors'][i:i+lookback_window]
                    window_dilation_mean = np.mean(window_dilation)
                    window_dilation_std = np.std(window_dilation)
                    
                    # Calculate similarities
                    vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                    skew_similarity = abs(window_skew - recent_skew)
                    kurt_similarity = abs(window_kurt - recent_kurt)
                    dilation_mean_sim = abs(window_dilation_mean - recent_dilation_mean) / max(recent_dilation_mean, 1e-8)
                    dilation_std_sim = abs(window_dilation_std - recent_dilation_std) / max(recent_dilation_std, 1e-8)
                    
                    # Combined similarity
                    total_similarity = (
                        0.4 * vol_similarity + 
                        0.15 * skew_similarity + 
                        0.15 * kurt_similarity +
                        0.2 * dilation_mean_sim +
                        0.1 * dilation_std_sim
                    )
                    
                    if total_similarity < similarity_threshold:
                        regime_windows.append(i)
            else:
                # Original approach
                for i in range(len(historical_returns) - lookback_window):
                    window_returns = historical_returns[i:i+lookback_window]
                    window_vol = np.std(window_returns)
                    window_skew = stats.skew(window_returns)
                    window_kurt = stats.kurtosis(window_returns)
                    
                    vol_similarity = abs(window_vol - recent_vol) / max(recent_vol, 1e-8)
                    skew_similarity = abs(window_skew - recent_skew)
                    kurt_similarity = abs(window_kurt - recent_kurt)
                    
                    total_similarity = (
                        0.6 * vol_similarity + 
                        0.25 * skew_similarity + 
                        0.15 * kurt_similarity
                    )
                    
                    if total_similarity < similarity_threshold:
                        regime_windows.append(i)
        
        # Initialize paths array
        paths = np.zeros((n_paths, n_steps))
        
        # For trading time approach, store dilation factors for each path
        if use_trading_time and enable_time_forecast:
            path_dilation_factors = np.zeros((n_paths, n_steps))
        
        # Generate paths by sampling from similar regimes
        for i in range(n_paths):
            # Randomly select a similar regime window
            if regime_windows:
                start_idx = np.random.choice(regime_windows)
                # Get n_steps returns from this regime
                regime_returns = historical_returns[start_idx:start_idx+lookback_window]
                # Randomly select a continuous segment of length n_steps
                # Handle edge case where regime_returns length equals n_steps
                max_start = max(0, len(regime_returns) - n_steps)
                if max_start == 0:
                    segment_start = 0
                else:
                    segment_start = np.random.randint(0, max_start)
                path_returns = regime_returns[segment_start:segment_start+n_steps]
                
                # For trading time approach, get corresponding dilation factors
                if use_trading_time and enable_time_forecast:
                    dilation_start = start_idx + segment_start
                    dilation_end = dilation_start + n_steps
                    if dilation_end <= len(self.time_map['dilation_factors']):
                        path_dilation_factors[i] = self.time_map['dilation_factors'][dilation_start:dilation_end]
                    else:
                        # Use repeating recent values if we go beyond available factors
                        recent_dilation = self.time_map['dilation_factors'][-lookback_window//2:]
                        path_dilation_factors[i] = np.random.choice(recent_dilation, size=n_steps)
            else:
                # Fallback to recent returns if no similar regimes found
                path_returns = np.random.choice(recent_returns, size=n_steps, replace=True)
                
                # For trading time approach, fallback to recent dilation factors
                if use_trading_time and enable_time_forecast:
                    recent_dilation = self.time_map['dilation_factors'][-lookback_window//2:]
                    path_dilation_factors[i] = np.random.choice(recent_dilation, size=n_steps)
            
            # Convert returns to price path
            paths[i] = self.prices[-1] * np.exp(np.cumsum(path_returns))
        
        # If using trading time, resample paths to account for time dilation
        if use_trading_time and enable_time_forecast:
            # Apply trading time transformations to each path
            warped_paths = np.zeros_like(paths)
            
            for i in range(n_paths):
                # Create a custom time warper for this path's forecasted time dilation
                path_warper = TradingTimeWarper(alpha=self.time_warper.alpha)
                
                # Create a simple mapping of clock time to trading time for this path
                dilation_factors = path_dilation_factors[i]
                trading_time = np.cumsum(dilation_factors)
                
                # Normalize to same total duration
                trading_time = trading_time * (n_steps - 1) / trading_time[-1]
                
                # Create a time map just for this path
                path_time_map = {
                    'trading_time_values': trading_time,
                    'clock_time_indices': np.arange(n_steps),
                    'dilation_factors': dilation_factors
                }
                
                # Use our custom transformation on the path
                x_orig = np.arange(n_steps)
                y_orig = paths[i]
                
                # Create uniform grid in trading time
                x_uniform = np.linspace(0, trading_time[-1], n_steps)
                
                # Resample onto uniform trading time grid (essentially stretching/compressing time)
                warped_paths[i] = np.interp(x_uniform, trading_time, y_orig)
            
            # Replace original paths with time-warped paths
            paths = warped_paths
            
        # Apply cross-dimensional filtering if enabled and we have volume data
        cross_dim_weights = None
        cross_dim_regime = None
        
        if use_cross_dim and self.cross_dim_analyzer is not None and self.volumes is not None:
            # We'll score each path based on how well it matches the cross-dimensional
            # fractal properties of the historical data
            
            cross_dim_weights = np.ones(n_paths)
            
            # Get current regime information
            cross_dim_regime = self.cross_dim_results.get('regime', {})
            current_regime = cross_dim_regime.get('regime', 0)
            regime_confidence = cross_dim_regime.get('confidence', 0.5)
            
            # Get cross correlation between price and volume
            price_vol_corr = self.cross_dim_results.get('cross_correlation', [[1, 0], [0, 1]])[0][1]
            
            # Perform analysis on each path
            for i in range(n_paths):
                path = paths[i]
                
                # Create a fictional volume path that preserves the cross correlation
                if hasattr(self, 'sampled_volumes') and self.sampled_volumes is not None:
                    # Create a volume series that correlates with this path's price
                    # at approximately the same level as in the historical data
                    
                    # Compute path returns
                    path_returns = np.diff(np.log(path))
                    
                    # Measure price/volume correlation for this path
                    # Check if we can use the simulated volume from the same regime
                    if regime_windows and i < len(regime_windows):
                        # Get historical volume data from this regime
                        start_idx = regime_windows[i % len(regime_windows)]
                        if start_idx + n_steps < len(self.volumes):
                            # Extract a segment of historical volume
                            sim_volume = self.volumes[start_idx:start_idx+n_steps-1]
                            
                            # Calculate log changes
                            vol_changes = np.diff(np.log(sim_volume + 1))  # Add 1 to avoid log(0)
                            
                            # Calculate correlation
                            if len(vol_changes) > 1 and len(path_returns) > 1:
                                try:
                                    # Calculate correlation between price and volume paths
                                    path_corr = np.corrcoef(path_returns, vol_changes)[0, 1]
                                    
                                    # Calculate correlation similarity score
                                    # Compare to historical price-volume correlation
                                    corr_similarity = 1.0 - min(1.0, abs(path_corr - price_vol_corr))
                                    
                                    # Weight paths by similarity to historical correlation patterns
                                    # Higher weights for paths that preserve the price-volume relationship
                                    cross_dim_weights[i] = 0.2 + 0.8 * corr_similarity
                                except:
                                    # Keep default weight on error
                                    pass
        
        # Cluster and analyze paths
        scaler = StandardScaler()
        scaled_paths = scaler.fit_transform(paths)

        # Handle NaN/Inf that can occur with constant values
        if not np.all(np.isfinite(scaled_paths)):
            # Replace NaN/Inf with zeros (constant paths get zero variance)
            scaled_paths = np.nan_to_num(scaled_paths, nan=0.0, posinf=0.0, neginf=0.0)

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
        
        # Start with standard pattern-based scoring
        combined_scores = (1 - pattern_weight) * size_probs + pattern_weight * cluster_scores
        
        # Incorporate cross-dimensional weights if available
        # Apply quantum price level filtering if enabled
        quantum_weights = None
        
        if use_quantum_levels and self.quantum_generator is not None:
            # Filter paths based on how well they respect quantum price levels
            quantum_weights = self.quantum_generator.filter_paths_by_levels(
                paths, influence_strength=quantum_influence
            )
        
        # Apply cross-dimensional filtering if available
        if cross_dim_weights is not None:
            # Calculate average cross-dimensional weight for each cluster
            cluster_cross_weights = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    # Average cross-dimensional weight for this cluster
                    cluster_cross_weights[i] = np.mean(cross_dim_weights[cluster_indices])
                else:
                    cluster_cross_weights[i] = 1.0  # Default if no paths in cluster
            
            # Normalize to [0, 1] if needed
            if np.sum(cluster_cross_weights) > 0:
                cluster_cross_weights = cluster_cross_weights / np.max(cluster_cross_weights)
            else:
                cluster_cross_weights = np.ones(n_clusters)
            
            # Blend in cross-dimensional weights (give them 30% influence)
            cross_dim_weight = 0.3
            combined_scores = (1 - cross_dim_weight) * combined_scores + cross_dim_weight * cluster_cross_weights
            
        # Apply quantum level filtering if available
        if quantum_weights is not None:
            # Calculate average quantum weight for each cluster
            cluster_quantum_weights = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    # Average quantum weight for this cluster
                    cluster_quantum_weights[i] = np.mean(quantum_weights[cluster_indices])
                else:
                    cluster_quantum_weights[i] = 1.0  # Default if no paths in cluster
            
            # Normalize to [0, 1] if needed
            if np.sum(cluster_quantum_weights) > 0:
                cluster_quantum_weights = cluster_quantum_weights / np.max(cluster_quantum_weights)
            else:
                cluster_quantum_weights = np.ones(n_clusters)
            
            # Blend in quantum weights (give them 30% influence)
            q_weight = 0.3
            combined_scores = (1 - q_weight) * combined_scores + q_weight * cluster_quantum_weights
            
        # Normalize probabilities
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
            vol_similarity = abs(path_vol - recent_vol) / max(recent_vol, 1e-8)
            
            # 2. Return distribution similarity
            path_skew = stats.skew(cloud_returns)
            path_kurt = stats.kurtosis(cloud_returns)
            skew_similarity = abs(path_skew - recent_skew)
            kurt_similarity = abs(path_kurt - recent_kurt)
            
            # 3. For trading time, add time dilation similarity
            if use_trading_time and enable_time_forecast:
                # Compare path's implied time dilation with recent dilation
                dilation_similarity = 0.0
                # We would calculate this from the path's characteristics, but as an approximation:
                volatility_ratio = path_vol / recent_vol
                # Higher volatility implies faster trading time
                implied_dilation = np.power(volatility_ratio, self.time_warper.alpha)
                dilation_similarity = abs(implied_dilation - recent_dilation_mean) / max(recent_dilation_mean, 1e-8)
                
                # Combine similarities with trading time component
                total_similarity = (
                    0.5 * vol_similarity + 
                    0.2 * skew_similarity +
                    0.1 * kurt_similarity +
                    0.2 * dilation_similarity
                )
            else:
                # Original similarity calculation
                total_similarity = (0.7 * vol_similarity + 0.3 * skew_similarity)
                
            # Calculate probability - less aggressive decay    
            path_probabilities[i] = np.exp(-2 * total_similarity)
            
        # Distance-based probabilities (alternative approach)
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
            'path_probabilities': path_probabilities,
            'trading_time_enabled': use_trading_time,
            'time_dilation_factors': self.time_map['dilation_factors'][-lookback_window:] if use_trading_time else None,
            'cross_dim_enabled': use_cross_dim and cross_dim_weights is not None,
            'cross_dim_regime': cross_dim_regime,
            'cross_dim_weights': cross_dim_weights.tolist() if cross_dim_weights is not None else None,
            'quantum_levels_enabled': use_quantum_levels,
            'quantum_levels': self.quantum_levels,
            'quantum_weights': quantum_weights.tolist() if quantum_weights is not None else None
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

class TradingTimeWarper:
    """
    Transforms between clock time and trading time based on volatility regimes.
    
    This class implements Mandelbrot's concept that markets operate on their own
    time scale that flows faster during high volatility and slower during low volatility.
    """
    
    def __init__(self, alpha: float = 0.5, min_scale: float = 0.1, max_scale: float = 10.0):
        """
        Initialize trading time warper.
        
        Args:
            alpha: Scaling factor that controls transformation intensity (0.5 default)
            min_scale: Minimum scaling factor to prevent extreme compression (0.1 default)
            max_scale: Maximum scaling factor to prevent extreme expansion (10.0 default)
        """
        self.alpha = alpha
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.time_map = None
        self.inverse_map = None
        
    def compute_time_transformation(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        Compute transformation between clock time and trading time.

        Args:
            prices: Historical price series
            volumes: Optional volume series (if available)

        Returns:
            Dictionary with time mapping information
        """
        # Convert to NumPy arrays if needed (e.g., from Polars Series)
        prices = _ensure_numpy_array(prices)
        volumes = _ensure_numpy_array(volumes)

        n = len(prices)

        # Calculate returns and volatility
        log_returns = np.diff(np.log(prices))
        
        # Use rolling windows to estimate local volatility
        window_sizes = [5, 21, 63]  # Day, week, month in trading days
        volatility_series = {}
        
        for window in window_sizes:
            # Compute rolling volatility with window
            vol = np.zeros(n)
            for i in range(window, n):
                vol[i] = np.std(log_returns[i-window:i]) * np.sqrt(252)  # Annualized

            # Fill initial points with first valid value
            # Handle edge case where window >= n
            if window < n:
                vol[:window] = vol[window]
            else:
                # If window is >= data length, compute volatility over all available data
                vol[:] = np.std(log_returns) * np.sqrt(252)
            volatility_series[window] = vol
        
        # Combine volatilities with different weights (emphasize recent)
        weights = {5: 0.6, 21: 0.3, 63: 0.1}
        combined_vol = np.zeros(n)
        
        for window, vol in volatility_series.items():
            combined_vol += weights[window] * vol
            
        # Normalize volatility to have mean 1
        relative_vol = combined_vol / np.mean(combined_vol)
        
        # If volumes are provided, incorporate them
        if volumes is not None and len(volumes) == n:
            # Normalize volumes to have mean 1
            relative_vol_normalized = volumes / np.mean(volumes)
            # Combined activity measure (volatility and volume)
            activity = np.sqrt(relative_vol * relative_vol_normalized)
        else:
            # Just use volatility if volume not available
            activity = relative_vol
        
        # Apply power transformation with bounds
        time_dilation = np.power(activity, self.alpha)
        time_dilation = np.clip(time_dilation, self.min_scale, self.max_scale)
        
        # Compute cumulative trading time
        trading_time = np.cumsum(time_dilation)
        
        # Normalize to span same overall duration
        trading_time = trading_time * (n - 1) / trading_time[-1]
        
        # Store the mapping
        self.time_map = {
            'clock_to_trading': trading_time,
            'trading_time_values': trading_time,
            'clock_time_indices': np.arange(n),
            'dilation_factors': time_dilation,
            'activity': activity
        }
        
        return self.time_map
    
    def transform_to_trading_time(self, series: np.ndarray) -> np.ndarray:
        """
        Transform a time series from clock time to trading time.
        
        Args:
            series: Time series in clock time
            
        Returns:
            Time series resampled to trading time
        """
        if self.time_map is None:
            raise ValueError("Must call compute_time_transformation first")
            
        # Get original points
        x_orig = np.arange(len(series))
        y_orig = series
        
        # Get trading time points
        x_trading = self.time_map['trading_time_values']
        
        # Create uniform grid in trading time
        x_uniform = np.linspace(0, x_trading[-1], len(series))
        
        # Resample onto uniform trading time grid
        y_trading = np.interp(x_uniform, x_trading, y_orig)
        
        return y_trading
    
    def transform_path_to_trading_time(self, path: np.ndarray) -> np.ndarray:
        """
        Transform a path from clock time to trading time.
        
        Args:
            path: Path in clock time
            
        Returns:
            Path resampled to trading time
        """
        return self.transform_to_trading_time(path)
    
    def transform_paths_to_trading_time(self, paths: np.ndarray) -> np.ndarray:
        """
        Transform multiple paths from clock time to trading time.
        
        Args:
            paths: Array of paths in clock time (n_paths, n_steps)
            
        Returns:
            Array of paths in trading time
        """
        warped_paths = np.zeros_like(paths)
        
        for i in range(paths.shape[0]):
            warped_paths[i] = self.transform_path_to_trading_time(paths[i])
            
        return warped_paths
    
    def transform_to_clock_time(self, trading_series: np.ndarray) -> np.ndarray:
        """
        Transform a time series from trading time to clock time.
        
        Args:
            trading_series: Time series in trading time
            
        Returns:
            Time series resampled to clock time
        """
        if self.time_map is None:
            raise ValueError("Must call compute_time_transformation first")
            
        # Create uniform trading time grid (assuming trading_series is on uniform grid)
        x_trading_uniform = np.linspace(0, self.time_map['trading_time_values'][-1], len(trading_series))
        
        # Get original time points
        x_clock = np.arange(len(self.time_map['trading_time_values']))
        x_trading = self.time_map['trading_time_values']
        
        # Resample from uniform trading time back to clock time
        resampled = np.interp(x_trading, x_trading_uniform, trading_series)
        
        return resampled


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

# FractalVisualizer class moved to visualization module
# Import from fractime.visualization for interactive visualizations

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


class FractalForecaster:
    """
    Unified fractal-based forecasting model.

    Combines fractal analysis, pattern recognition, and regime detection
    to create accurate time series forecasts.

    Supports exogenous predictors for conditioning forecasts on external
    variables like economic indicators, market indices, or correlated series.
    """

    def __init__(
        self,
        lookback: int = 252,
        method: str = 'rs',
        min_scale: int = 10,
        max_scale: int = 100,
        use_exogenous: bool = False,
        exog_max_lags: int = 10,
        exog_min_correlation: float = 0.1,
        exog_adjustment_strength: float = 0.3
    ):
        """
        Initialize the forecaster.

        Args:
            lookback: Number of historical periods to use for analysis
            method: Hurst exponent calculation method ('rs' or 'dfa')
            min_scale: Minimum scale for fractal analysis
            max_scale: Maximum scale for fractal analysis
            use_exogenous: Whether to use exogenous predictors
            exog_max_lags: Maximum lags for exogenous variable analysis
            exog_min_correlation: Minimum correlation to include exog variable
            exog_adjustment_strength: How strongly exogenous vars affect paths (0-1)
        """
        self.lookback = lookback
        self.method = method
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.analyzer = FractalAnalyzer()
        self.simulator = None
        self.prices_history = None
        self.dates_history = None
        self.frequency = None
        self.hurst = None
        self.fractal_dim = None

        # Exogenous predictors
        self.use_exogenous = use_exogenous
        self.exog_max_lags = exog_max_lags
        self.exog_min_correlation = exog_min_correlation
        self.exog_adjustment_strength = exog_adjustment_strength
        self.exog_handler = None
        self.exog_regime_modifier = None
        self.exog_data = None
        self.exog_names = None

    def _infer_frequency(self, dates: np.ndarray) -> str:
        """
        Infer the frequency of the time series from dates.

        Args:
            dates: Array of datetime objects or strings

        Returns:
            Frequency string ('D', 'H', 'min', etc.)
        """
        # Convert to polars Series for datetime operations
        from datetime import datetime

        if isinstance(dates[0], str):
            dates_pl = pl.Series(dates).str.to_datetime()
        elif isinstance(dates[0], datetime):
            # Convert Python datetime objects to numpy datetime64 first
            dates_np = np.array(dates, dtype='datetime64[ns]')
            dates_pl = pl.Series(dates_np)
        else:
            # Convert numpy datetime64 or other to polars
            dates_pl = pl.Series(dates)
            if dates_pl.dtype != pl.Datetime:
                try:
                    dates_pl = dates_pl.cast(pl.Datetime)
                except:
                    # Fallback: convert to numpy datetime64 first
                    dates_np = np.array(dates, dtype='datetime64[ns]')
                    dates_pl = pl.Series(dates_np)

        # Calculate difference between consecutive dates
        diffs = dates_pl.diff().drop_nulls()

        # Get the most common difference (mode)
        if len(diffs) == 0:
            return 'D'  # Default to daily

        # Convert to seconds for easy comparison
        # Polars duration is in microseconds by default
        diff_seconds = diffs[0].total_seconds() if hasattr(diffs[0], 'total_seconds') else (diffs[0] / 1_000_000)

        # Infer frequency based on typical difference
        if diff_seconds < 3600:
            return 'min'
        elif diff_seconds < 86400:
            return 'H'
        else:
            return 'D'

    def fit(
        self,
        prices: np.ndarray,
        dates: np.ndarray = None,
        exogenous = None
    ):
        """
        Fit the forecaster to historical data.

        Args:
            prices: Historical price series
            dates: Optional datetime array matching prices length
            exogenous: Optional exogenous variables (array, DataFrame, or dict)
                      Each column/key is a separate exogenous predictor
        """
        prices = _ensure_numpy_array(prices)
        self.prices_history = prices[-self.lookback:] if len(prices) > self.lookback else prices

        # Handle dates if provided
        if dates is not None:
            dates = _ensure_numpy_array(dates)
            self.dates_history = dates[-self.lookback:] if len(dates) > self.lookback else dates
            self.frequency = self._infer_frequency(self.dates_history)
        else:
            self.dates_history = None
            self.frequency = None

        # Analyze fractal properties
        self.hurst = self.analyzer.compute_hurst(self.prices_history)
        self.fractal_dim = self.analyzer.compute_fractal_dimension(self.prices_history)

        # Handle exogenous variables
        if exogenous is not None or self.use_exogenous:
            self._fit_exogenous(exogenous)

        # Initialize simulator
        self.simulator = FractalSimulator(self.prices_history, self.analyzer)

        return self

    def _fit_exogenous(self, exogenous):
        """
        Fit exogenous variable handler and regime modifier.

        Args:
            exogenous: Exogenous data (array, DataFrame, or dict)
        """
        if exogenous is None:
            return

        from .exogenous import ExogenousHandler, ExogenousRegimeModifier

        # Initialize and fit exogenous handler
        self.exog_handler = ExogenousHandler(
            max_lags=self.exog_max_lags,
            min_correlation=self.exog_min_correlation,
            use_differences=True,
            scale_features=True
        )
        self.exog_handler.fit(self.prices_history, exogenous)

        # Store exogenous data
        import pandas as pd
        if isinstance(exogenous, pd.DataFrame):
            self.exog_data = exogenous.values[-self.lookback:] if len(exogenous) > self.lookback else exogenous.values
            self.exog_names = list(exogenous.columns)
        elif isinstance(exogenous, dict):
            self.exog_names = list(exogenous.keys())
            self.exog_data = np.column_stack([
                exogenous[name][-self.lookback:] if len(exogenous[name]) > self.lookback else exogenous[name]
                for name in self.exog_names
            ])
        else:
            exogenous = _ensure_numpy_array(exogenous)
            if exogenous.ndim == 1:
                exogenous = exogenous.reshape(-1, 1)
            self.exog_data = exogenous[-self.lookback:] if len(exogenous) > self.lookback else exogenous
            self.exog_names = [f'exog_{i}' for i in range(self.exog_data.shape[1])]

        # Fit regime modifier
        try:
            X, target_aligned = self.exog_handler.get_feature_matrix(self.prices_history)
            if X.shape[1] > 0:  # Have valid features
                target_returns = np.diff(np.log(target_aligned))
                X = X[1:]  # Align with returns

                self.exog_regime_modifier = ExogenousRegimeModifier(n_regimes=3)
                self.exog_regime_modifier.fit(target_returns, X)
        except Exception as e:
            import warnings
            warnings.warn(f"Could not fit exogenous regime modifier: {e}")
            self.exog_regime_modifier = None

    def get_exogenous_summary(self):
        """
        Get summary of exogenous variable analysis.

        Returns:
            Dictionary with exogenous analysis results, or None if not fitted
        """
        if self.exog_handler is None:
            return None
        return self.exog_handler.get_summary()

    def _remove_duplicate_paths(self, paths: np.ndarray, probabilities: np.ndarray = None,
                                 add_noise: bool = True, noise_scale: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove duplicate paths and optionally add small noise to near-duplicates.

        Args:
            paths: Array of paths (n_paths x n_steps)
            probabilities: Optional probability weights (will be preserved for unique paths)
            add_noise: Whether to add small noise to break ties
            noise_scale: Scale of noise relative to price level (default 1e-6 = 0.0001%)

        Returns:
            Tuple of (unique_paths, unique_probabilities)
        """
        # Find unique paths
        unique_paths, unique_indices, inverse_indices, counts = np.unique(
            paths, axis=0, return_index=True, return_inverse=True, return_counts=True
        )

        # If probabilities provided, aggregate probabilities for duplicates
        if probabilities is not None:
            unique_probs = np.zeros(len(unique_paths))
            for i, idx in enumerate(unique_indices):
                # Sum probabilities of all paths that map to this unique path
                duplicate_mask = inverse_indices == i
                unique_probs[i] = np.sum(probabilities[duplicate_mask])
        else:
            unique_probs = None

        # Add small noise if requested to break near-duplicates
        if add_noise and len(unique_paths) < len(paths) * 0.9:  # Only if >10% duplicates
            # Add noise proportional to each path's values
            for i in range(len(unique_paths)):
                noise = np.random.normal(0, noise_scale * np.abs(unique_paths[i]), size=unique_paths.shape[1])
                unique_paths[i] += noise

        return unique_paths, unique_probs

    def _calculate_path_probabilities(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate probability weights for each path based on fractal similarity
        to historical patterns and multi-scale consistency.

        Args:
            paths: Array of simulated paths (n_paths x n_steps)

        Returns:
            Array of probability weights (n_paths,) summing to 1.0
        """
        n_paths = len(paths)
        scores = np.zeros(n_paths)

        # Get historical patterns for comparison
        hist_returns = np.diff(np.log(self.prices_history))
        hist_vol = np.std(hist_returns)

        for i, path in enumerate(paths):
            path_score = 0.0

            # 1. Hurst consistency - paths matching historical Hurst get higher weight
            path_returns = np.diff(np.log(path))
            if len(path_returns) >= 20:
                try:
                    path_hurst = self.analyzer.compute_hurst(path)
                    hurst_similarity = 1.0 - abs(path_hurst - self.hurst)
                    path_score += hurst_similarity * 0.3
                except:
                    path_score += 0.15  # Neutral score if calculation fails

            # 2. Volatility similarity - paths with similar volatility to history
            path_vol = np.std(path_returns) if len(path_returns) > 0 else hist_vol
            vol_ratio = min(path_vol, hist_vol) / max(path_vol, hist_vol)
            path_score += vol_ratio * 0.3

            # 3. Pattern similarity - check multi-scale patterns
            # Compare short, medium, and long-term trends
            if len(path) >= 10:
                # Short-term (last 5 steps)
                path_short_trend = np.mean(path_returns[-5:]) if len(path_returns) >= 5 else 0
                hist_short_trend = np.mean(hist_returns[-5:]) if len(hist_returns) >= 5 else 0
                short_similarity = 1.0 - min(abs(path_short_trend - hist_short_trend) / (hist_vol + 1e-8), 1.0)
                path_score += short_similarity * 0.2

                # Medium-term trend consistency
                if len(path) >= 15:
                    path_med_trend = np.mean(path_returns[-15:-5]) if len(path_returns) >= 15 else 0
                    hist_med_trend = np.mean(hist_returns[-15:-5]) if len(hist_returns) >= 15 else 0
                    med_similarity = 1.0 - min(abs(path_med_trend - hist_med_trend) / (hist_vol + 1e-8), 1.0)
                    path_score += med_similarity * 0.2

            scores[i] = max(path_score, 0.1)  # Ensure minimum score

        # Normalize to probabilities
        probabilities = scores / np.sum(scores)
        return probabilities

    def _parse_period(self, period: str) -> int:
        """
        Parse period string to number of steps.

        Args:
            period: Period string like '7d', '2w', '1M', '3m' (minute), '12h'

        Returns:
            Number of steps based on inferred frequency
        """
        import re

        if self.frequency is None:
            raise ValueError("Cannot use period without dates. Call fit() with dates first.")

        # Parse period string
        match = re.match(r'(\d+)([a-zA-Z]+)', period)
        if not match:
            raise ValueError(f"Invalid period format: {period}. Use format like '7d', '2w', '1M'")

        value = int(match.group(1))
        unit = match.group(2).lower()

        # Map to steps based on frequency
        freq_upper = self.frequency.upper() if isinstance(self.frequency, str) else 'D'

        # Daily frequency
        if freq_upper == 'D' or freq_upper.startswith('D'):
            if unit in ['d', 'day', 'days']:
                return value
            elif unit in ['w', 'week', 'weeks']:
                return value * 7
            elif unit in ['m', 'month', 'months']:
                return value * 30
            elif unit in ['y', 'year', 'years']:
                return value * 365
        # Hourly frequency
        elif freq_upper == 'H' or freq_upper.startswith('H'):
            if unit in ['h', 'hour', 'hours']:
                return value
            elif unit in ['d', 'day', 'days']:
                return value * 24
            elif unit in ['w', 'week', 'weeks']:
                return value * 24 * 7
        # Minute frequency
        elif freq_upper == 'MIN' or freq_upper.startswith('T'):
            if unit in ['m', 'min', 'minute', 'minutes']:
                return value
            elif unit in ['h', 'hour', 'hours']:
                return value * 60
            elif unit in ['d', 'day', 'days']:
                return value * 60 * 24

        raise ValueError(f"Cannot convert period '{period}' with frequency '{self.frequency}'")

    def _calculate_steps_to_date(self, end_date: str) -> int:
        """
        Calculate number of steps from last historical date to end_date.

        Args:
            end_date: Target date as string (e.g., '2025-11-27')

        Returns:
            Number of steps
        """
        if self.dates_history is None:
            raise ValueError("Cannot use end_date without historical dates. Call fit() with dates first.")

        # Convert last date to polars datetime (handle different input types)
        last_date_value = self.dates_history[-1]

        # Handle numpy datetime64, pandas Timestamp, datetime, or string
        if isinstance(last_date_value, np.datetime64):
            # Convert numpy datetime64 to datetime
            last_date = pl.from_numpy(np.array([last_date_value], dtype='datetime64[ns]')).item()
        elif isinstance(last_date_value, str):
            last_date = pl.Series([last_date_value]).str.to_datetime().item()
        else:
            # datetime or other types
            last_date = pl.Series([last_date_value]).cast(pl.Datetime).item()

        # Parse target date
        target_date = pl.Series([end_date]).str.to_datetime().item()

        if target_date <= last_date:
            raise ValueError(f"end_date ({target_date}) must be after last historical date ({last_date})")

        # Generate date range and count steps using polars
        if self.frequency == 'D':
            interval = '1d'
        elif self.frequency == 'H':
            interval = '1h'
        elif self.frequency == 'min':
            interval = '1m'
        else:
            interval = '1d'  # Default

        date_range = pl.datetime_range(start=last_date, end=target_date, interval=interval, eager=True)
        n_steps = len(date_range) - 1  # Exclude the start date

        return n_steps

    def predict(self, n_steps: int = None, end_date: str = None, period: str = None,
                n_paths: int = 1000, confidence: float = 0.95):
        """
        Generate forecast with uncertainty quantification and path probabilities.

        Provide exactly ONE of: n_steps, end_date, or period.

        Args:
            n_steps: Number of steps ahead to forecast (mutually exclusive with end_date/period)
            end_date: Forecast until this date (e.g., '2025-11-27') - requires dates in fit()
            period: Forecast period (e.g., '7d', '2w', '1M') - requires dates in fit()
            n_paths: Number of Monte Carlo paths to simulate (default 1000)
            confidence: Confidence level for intervals (default 0.95)

        Returns:
            Dictionary with:
                'forecast': np.ndarray - Median forecast
                'weighted_forecast': np.ndarray - Probability-weighted forecast
                'mean': np.ndarray - Mean forecast
                'lower': np.ndarray - Lower confidence bound (simple quantile)
                'upper': np.ndarray - Upper confidence bound (simple quantile)
                'weighted_lower': np.ndarray - Probability-weighted lower CI
                'weighted_upper': np.ndarray - Probability-weighted upper CI
                'std': np.ndarray - Standard deviation
                'paths': np.ndarray - All simulated paths (n_paths x n_steps)
                'probabilities': np.ndarray - Probability weight for each path
                'dates': np.ndarray - Forecast dates (only if dates provided to fit())

        Examples:
            >>> # Method 1: Specify number of steps
            >>> result = forecaster.predict(n_steps=30)

            >>> # Method 2: Forecast to specific date (requires dates in fit())
            >>> forecaster.fit(prices, dates=dates)
            >>> result = forecaster.predict(end_date='2025-11-27')

            >>> # Method 3: Forecast for a period (requires dates in fit())
            >>> result = forecaster.predict(period='7d')
        """
        if self.simulator is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle constant/near-constant prices edge case
        price_std = np.std(self.prices_history)
        if price_std < 1e-10:  # Essentially constant
            # Return simple forecast with the constant value
            constant_value = self.prices_history[-1]

            # Still need to determine n_steps
            if end_date is not None:
                n_steps = self._calculate_steps_to_date(end_date)
            elif period is not None:
                n_steps = self._parse_period(period)
            elif n_steps is None:
                raise ValueError("Must provide one of: n_steps, end_date, or period")

            forecast = np.full(n_steps, constant_value)
            return {
                'forecast': forecast,
                'weighted_forecast': forecast,
                'mean': forecast,
                'lower': forecast,
                'upper': forecast,
                'weighted_lower': forecast,
                'weighted_upper': forecast,
                'std': np.zeros(n_steps),
                'paths': np.tile(forecast, (1, 1)),
                'probabilities': np.array([1.0]),
                'model_name': 'Fractal (constant series)',
            }

        # Validate arguments - exactly one must be provided
        args_provided = sum([n_steps is not None, end_date is not None, period is not None])
        if args_provided == 0:
            raise ValueError("Must provide one of: n_steps, end_date, or period")
        if args_provided > 1:
            raise ValueError("Provide only ONE of: n_steps, end_date, or period")

        # Calculate n_steps from end_date or period if needed
        if end_date is not None:
            n_steps = self._calculate_steps_to_date(end_date)
        elif period is not None:
            n_steps = self._parse_period(period)

        # Generate paths (single simulation run)
        paths, metadata = self.simulator.simulate_paths(
            n_steps=n_steps,
            n_paths=n_paths,
            pattern_weight=0.4,
            use_trading_time=True
        )

        # Diagnostic: Check for duplicate paths
        unique_paths, unique_indices, counts = np.unique(
            paths, axis=0, return_index=True, return_counts=True
        )
        n_duplicates = np.sum(counts > 1)

        if n_duplicates > 0:
            import warnings
            pct_duplicates = (n_duplicates / len(paths)) * 100
            warnings.warn(
                f"Found {n_duplicates} duplicate paths ({pct_duplicates:.1f}% of {len(paths)} paths). "
                f"This may indicate an issue with path generation. "
                f"Consider using more n_paths or checking random state.",
                UserWarning
            )

        # Calculate path probabilities based on fractal similarity
        probabilities = self._calculate_path_probabilities(paths)

        # Apply exogenous adjustments if fitted
        if self.exog_handler is not None and self.exog_regime_modifier is not None:
            try:
                # Get current exogenous features
                X, _ = self.exog_handler.get_feature_matrix(self.prices_history)
                if X.shape[1] > 0:
                    # Use most recent feature vector
                    current_exog = X[-1:, :]

                    # Adjust probabilities based on exogenous regime
                    probabilities = self.exog_regime_modifier.adjust_path_probabilities(
                        paths,
                        probabilities,
                        current_exog,
                        adjustment_strength=self.exog_adjustment_strength
                    )
            except Exception as e:
                import warnings
                warnings.warn(f"Could not apply exogenous adjustment: {e}")

        # Calculate statistics
        alpha = (1 - confidence) / 2
        lower_pct = alpha * 100
        upper_pct = (1 - alpha) * 100

        # Probability-weighted forecast (in addition to median)
        weighted_forecast = np.average(paths, axis=0, weights=probabilities)

        # Calculate probability-weighted confidence intervals
        weighted_lower = np.zeros(n_steps)
        weighted_upper = np.zeros(n_steps)

        for t in range(n_steps):
            # Get values at this time step
            values = paths[:, t]

            # Sort by value
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_probs = probabilities[sorted_indices]

            # Calculate cumulative probability
            cumsum_probs = np.cumsum(sorted_probs)

            # Find weighted quantiles
            lower_idx = np.searchsorted(cumsum_probs, alpha)
            upper_idx = np.searchsorted(cumsum_probs, 1 - alpha)

            # Ensure indices are within bounds
            lower_idx = min(lower_idx, len(sorted_values) - 1)
            upper_idx = min(upper_idx, len(sorted_values) - 1)

            weighted_lower[t] = sorted_values[lower_idx]
            weighted_upper[t] = sorted_values[upper_idx]

        # Generate forecast dates if historical dates were provided
        result = {
            'forecast': np.median(paths, axis=0),
            'weighted_forecast': weighted_forecast,
            'mean': np.mean(paths, axis=0),
            'lower': np.percentile(paths, lower_pct, axis=0),
            'upper': np.percentile(paths, upper_pct, axis=0),
            'weighted_lower': weighted_lower,  # New: probability-weighted CI
            'weighted_upper': weighted_upper,  # New: probability-weighted CI
            'std': np.std(paths, axis=0),
            'paths': paths,
            'probabilities': probabilities
        }

        # Add forecast dates if historical dates available
        if self.dates_history is not None:
            # Convert last date to polars datetime (handle different input types)
            last_date_value = self.dates_history[-1]

            # Handle numpy datetime64, pandas Timestamp, datetime, or string
            if isinstance(last_date_value, np.datetime64):
                # Convert numpy datetime64 to polars
                last_date = pl.from_numpy(np.array([last_date_value], dtype='datetime64[ns]')).item()
            elif isinstance(last_date_value, str):
                last_date = pl.Series([last_date_value]).str.to_datetime().item()
            else:
                # datetime or other types
                last_date = pl.Series([last_date_value]).cast(pl.Datetime).item()

            # Map frequency to polars interval and compute end date
            if self.frequency == 'D':
                interval = '1d'
                from datetime import timedelta
                end_date_calc = last_date + timedelta(days=n_steps)
            elif self.frequency == 'H':
                interval = '1h'
                from datetime import timedelta
                end_date_calc = last_date + timedelta(hours=n_steps)
            elif self.frequency == 'min':
                interval = '1m'
                from datetime import timedelta
                end_date_calc = last_date + timedelta(minutes=n_steps)
            else:
                interval = '1d'  # Default
                from datetime import timedelta
                end_date_calc = last_date + timedelta(days=n_steps)

            # Generate forecast dates using polars
            all_dates = pl.datetime_range(
                start=last_date,
                end=end_date_calc,
                interval=interval,
                eager=True
            )
            # Skip the first date (which is the last historical date)
            forecast_dates = all_dates.slice(1, n_steps)

            result['dates'] = forecast_dates.to_numpy()

        return result


def plot_forecast(prices: np.ndarray,
                 forecast: np.ndarray = None,
                 paths: np.ndarray = None,
                 confidence_intervals: dict = None,
                 title: str = "Fractal Forecast",
                 dates: np.ndarray = None,
                 show_patterns: bool = False):
    """
    Simple, clean plotting function for forecasts.

    Args:
        prices: Historical price data
        forecast: Point forecast (optional)
        paths: Simulated paths array (optional, shape: n_paths x n_steps)
        confidence_intervals: Dict with 'lower' and 'upper' bounds (optional)
        title: Plot title
        dates: Date array for x-axis (optional)
        show_patterns: Show path distribution (default False)

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta

    prices = _ensure_numpy_array(prices)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare x-axis
    n_hist = len(prices)
    if dates is not None:
        dates = _ensure_numpy_array(dates)
        x_hist = dates[-n_hist:]

        # Generate future dates
        if forecast is not None or paths is not None:
            n_forecast = len(forecast) if forecast is not None else paths.shape[1]
            last_date = x_hist[-1]

            # Check if date is datetime-like (datetime, np.datetime64)
            if isinstance(last_date, (datetime, np.datetime64)):
                # Use polars for date range generation
                last_date_pl = pl.Series([last_date]).cast(pl.Datetime)[0]
                x_forecast = pl.datetime_range(
                    start=last_date_pl,
                    end=None,
                    interval='1d',
                    eager=True
                ).slice(1, n_forecast).to_numpy()
            else:
                x_forecast = np.arange(n_forecast) + n_hist
    else:
        x_hist = np.arange(n_hist)
        if forecast is not None or paths is not None:
            n_forecast = len(forecast) if forecast is not None else paths.shape[1]
            x_forecast = np.arange(n_forecast) + n_hist

    # Plot historical prices
    ax.plot(x_hist, prices, 'k-', linewidth=2, label='Historical', alpha=0.8)

    # Plot paths if provided
    if paths is not None:
        paths = _ensure_numpy_array(paths)
        if show_patterns:
            # Show all paths with transparency
            for i in range(min(100, len(paths))):
                ax.plot(x_forecast, paths[i], 'b-', alpha=0.05, linewidth=0.5)
        else:
            # Show percentile bands
            p10 = np.percentile(paths, 10, axis=0)
            p90 = np.percentile(paths, 90, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p75 = np.percentile(paths, 75, axis=0)

            ax.fill_between(x_forecast, p10, p90, alpha=0.2, color='blue', label='80% Range')
            ax.fill_between(x_forecast, p25, p75, alpha=0.3, color='blue', label='50% Range')

    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        lower = _ensure_numpy_array(confidence_intervals['lower'])
        upper = _ensure_numpy_array(confidence_intervals['upper'])
        ax.fill_between(x_forecast, lower, upper, alpha=0.3, color='green', label='95% CI')

    # Plot forecast
    if forecast is not None:
        forecast = _ensure_numpy_array(forecast)
        ax.plot(x_forecast, forecast, 'r-', linewidth=2, label='Forecast', alpha=0.8)

        # Connect historical to forecast
        ax.plot([x_hist[-1], x_forecast[0]], [prices[-1], forecast[0]],
                'r--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Format dates if using datetime
    if dates is not None and isinstance(x_hist[0], (datetime, np.datetime64)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

    plt.tight_layout()
    return fig


# print_forecast_summary moved to visualization module
# Import from fractime.visualization for the print function


# Example usage
if __name__ == "__main__":
    from fractime.visualization import FractalVisualizer

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