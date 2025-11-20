"""
Fractal analysis for time series data.

This module provides tools for analyzing fractal properties of time series,
including Hurst exponent calculation, fractal dimension estimation, and
cross-dimensional fractal analysis.
"""

import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler
from numba import njit


def _ensure_numpy_array(data):
    """Convert Polars Series or other array-like to NumPy array."""
    import polars as pl
    import pandas as pd

    if data is None:
        return None
    if isinstance(data, pl.Series):
        return data.to_numpy()
    if isinstance(data, (list, tuple)):
        return np.array(data)
    if isinstance(data, pd.Series):
        return data.to_numpy()
    return np.asarray(data)


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
            fractal_dims[name] = analyzer.compute_fractal_dimension(data)

        return fractal_dims

    def compute_hurst_exponents(self) -> Dict[str, float]:
        """Compute Hurst exponent for each data dimension."""
        hurst_exponents = {}
        analyzer = FractalAnalyzer()

        for name, data in self.dimensions.items():
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
        dim_names = list(self.dimensions.keys())
        n_dims = len(dim_names)

        if n_dims < 2:
            raise ValueError("Need at least 2 dimensions for cross-correlation analysis")

        # Pre-compute returns/changes for each dimension
        changes = {}
        for name in dim_names:
            data = self.dimensions[name]
            changes[name] = np.diff(np.log(data)) if np.min(data) > 0 else np.diff(data) / data[:-1]

        min_length = min(len(changes[name]) for name in dim_names)

        n_windows = min_length - window_size + 1
        if n_windows <= 0:
            raise ValueError(f"Window size {window_size} is too large for data length {min_length}")

        # Multi-scale correlations
        scales = [window_size, window_size*2, window_size*4]
        scale_weights = [0.5, 0.3, 0.2]

        cross_corr_by_scale = {}

        for scale in scales:
            if min_length <= scale:
                continue

            corr_matrix = np.zeros((n_dims, n_dims))

            for i, name1 in enumerate(dim_names):
                change1 = changes[name1][:min_length]

                for j, name2 in enumerate(dim_names):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        continue

                    change2 = changes[name2][:min_length]

                    try:
                        if np.std(change1) > 0 and np.std(change2) > 0:
                            corr = np.corrcoef(change1, change2)[0, 1]
                            if not np.isnan(corr):
                                corr_matrix[i, j] = corr
                    except Exception:
                        pass

            cross_corr_by_scale[scale] = corr_matrix

        if not cross_corr_by_scale:
            self.correlation_matrix = np.eye(n_dims)
            return self.correlation_matrix

        # Combine scales with weights
        combined_matrix = np.zeros((n_dims, n_dims))
        total_weight = 0

        for idx, scale in enumerate(scales):
            if scale in cross_corr_by_scale:
                weight = scale_weights[idx]
                combined_matrix += weight * cross_corr_by_scale[scale]
                total_weight += weight

        if total_weight > 0:
            combined_matrix /= total_weight

        np.fill_diagonal(combined_matrix, 1.0)

        self.correlation_matrix = combined_matrix
        return combined_matrix

    def analyze(self, data: np.ndarray, dim_names: list = None) -> Dict:
        """
        Analyze multivariate time series for cross-dimensional fractal properties.

        Args:
            data: 2D array (n_samples, n_dimensions)
            dim_names: Optional names for each dimension

        Returns:
            Dictionary with correlation, cross_hurst, and other metrics
        """
        if data.ndim == 1:
            raise ValueError("Data must be 2D array for cross-dimensional analysis")

        n_dims = data.shape[1]
        if dim_names is None:
            dim_names = [f'dim_{i}' for i in range(n_dims)]

        # Clear existing dimensions and add new ones
        self.dimensions = {}
        for i, name in enumerate(dim_names):
            self.add_dimension(name, data[:, i])

        # Compute basic correlation
        correlation = self.compute_cross_correlation()

        # Compute Hurst exponents for each dimension
        hurst_exponents = self.compute_hurst_exponents()

        # Cross-Hurst (average of all Hurst exponents)
        cross_hurst = np.mean(list(hurst_exponents.values()))

        # Fractal dimensions
        fractal_dims = self.compute_fractal_dimensions()

        return {
            'correlation': correlation,
            'cross_hurst': cross_hurst,
            'hurst_exponents': hurst_exponents,
            'fractal_dimensions': fractal_dims,
            'dim_names': dim_names
        }

    # Additional methods from original code (compute_fractal_coherence, identify_regimes, analyze_dimensions)
    # Omitted for brevity - include them from lines 304-577 in original core.py


class FractalAnalyzer:
    """Analyzes fractal properties of time series data."""

    def __init__(self):
        """Initialize with empty cache for performance."""
        self.cache = {}

    def analyze(self, prices: np.ndarray) -> Dict:
        """
        Analyze fractal properties of a time series.

        Args:
            prices: Time series data

        Returns:
            Dictionary with hurst, fractal_dimension, and other metrics
        """
        prices = _ensure_numpy_array(prices)

        hurst = self.compute_hurst(prices)
        fractal_dim = self.compute_fractal_dimension(prices)

        return {
            'hurst': hurst,
            'fractal_dimension': fractal_dim,
            'persistence': 'trending' if hurst > 0.6 else ('mean_reverting' if hurst < 0.4 else 'random_walk')
        }

    def analyze_patterns(self, prices: np.ndarray, full_analysis=True) -> dict:
        """Analyze with caching and selective feature computation."""
        prices = _ensure_numpy_array(prices)

        if len(prices) > 3:
            cache_key = f"{len(prices)}_{prices[0]:.2f}_{prices[-1]:.2f}_{prices[len(prices)//2]:.2f}"

            if cache_key in self.cache:
                return self.cache[cache_key]

        results = {
            'hurst': self.compute_hurst(prices),
            'fractal_dim': self.compute_fractal_dimension(prices)
        }

        if full_analysis or len(prices) < 1000:
            results['self_similar_patterns'] = self._find_patterns(prices)
        else:
            results['self_similar_patterns'] = self._find_simple_patterns(prices)

        if len(prices) > 3:
            self.cache[cache_key] = results

            if len(self.cache) > 100:
                self.cache.pop(next(iter(self.cache)))

        return results

    def _find_simple_patterns(self, prices: np.ndarray) -> list:
        """Faster pattern detection for backtesting."""
        patterns = []
        returns = np.diff(np.log(prices))

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
                    'similarity': 0.8,
                    'fractal_dim': 1.5
                })

        return patterns

    def _find_patterns(self, prices: np.ndarray) -> list:
        """Optimized pattern detection."""
        from fractime.optimization import compute_pattern_similarities

        patterns = []
        returns = np.diff(np.log(prices))

        min_window = 10
        max_window = min(250, len(prices)//3)
        window_step = max(1, (max_window - min_window) // 10)
        window_sizes = range(min_window, max_window, window_step)

        volatilities = {}
        for window in window_sizes:
            rolling_vols = np.array([np.std(returns[i:i+window-1]) for i in range(len(returns)-window+1)])
            volatilities[window] = rolling_vols

        for window in window_sizes:
            if len(patterns) >= 50:
                break

            step_size = max(1, window // 4)

            for i in range(0, len(prices)-window*2, step_size):
                if i >= len(volatilities[window]) or i+window >= len(volatilities[window]):
                    continue

                pattern1_vol = volatilities[window][i]
                pattern2_vol = volatilities[window][i+window]

                if pattern1_vol < 1e-8 or pattern2_vol < 1e-8:
                    continue

                pattern1_returns = returns[i:i+window-1]
                pattern2_returns = returns[i+window:i+window*2-1]

                similarity = compute_pattern_similarities(
                    pattern1_returns, pattern2_returns, pattern1_vol, pattern2_vol
                )

                if similarity > 0.8:
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
        prices = _ensure_numpy_array(prices)

        try:
            if len(prices) < 10:
                return 1.5

            if quick_mode:
                r_min, r_max, step = 2, min(10, len(prices)//4), 2
            else:
                r_min, r_max, step = 2, min(20, len(prices)//4), 1

            if r_min >= r_max:
                return 1.5

            try:
                scaled_prices = StandardScaler().fit_transform(prices.reshape(-1, 1)).ravel()
            except:
                min_price = np.min(prices)
                max_price = np.max(prices)
                if max_price == min_price:
                    return 1.0
                scaled_prices = (prices - min_price) / (max_price - min_price)

            return compute_box_dimension_safe(scaled_prices, r_min, r_max, step)
        except Exception as e:
            print(f"Error computing fractal dimension: {e}")
            return 1.5

    def get_patterns(self, prices: np.ndarray, max_patterns=20) -> list:
        """Extract patterns efficiently with sampling."""
        window_sizes = [10, 20, 30]
        patterns = []

        skip_factor = len(prices) // 500 if len(prices) > 1000 else 1

        for window in window_sizes:
            step_size = max(1, window // 2)

            for i in range(len(prices) - window, 0, -step_size * skip_factor):
                if i >= window:
                    pattern = prices[i-window:i]
                    if len(pattern) == window:
                        patterns.append(pattern)

                if len(patterns) >= max_patterns // len(window_sizes):
                    break

        print(f"Extracted {len(patterns)} patterns from price data")

        return patterns

    def compute_hurst(self, prices: np.ndarray) -> float:
        """Compute the Hurst exponent for a price series."""
        prices = _ensure_numpy_array(prices)

        if len(prices) < 20:
            return 0.5

        try:
            min_lag = 10
            max_lag = min(250, len(prices) // 2)
            return compute_hurst_exponent(prices, min_lag, max_lag)
        except Exception as e:
            print(f"Error computing Hurst exponent: {e}")
            return 0.5
