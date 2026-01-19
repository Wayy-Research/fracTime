"""
Numba-accelerated internal functions for fractal analysis.

This module contains all JIT-compiled functions for performance-critical
computations. These are internal and should not be used directly by users.
"""

import numpy as np
from numba import njit, prange


# =============================================================================
# Basic Statistics
# =============================================================================

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

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-10:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denom


@njit
def compute_std(arr: np.ndarray) -> float:
    """Compute standard deviation."""
    n = len(arr)
    if n < 2:
        return 0.0
    mean = np.mean(arr)
    return np.sqrt(np.sum((arr - mean) ** 2) / n)


# =============================================================================
# Hurst Exponent (Rescaled Range)
# =============================================================================

@njit
def compute_rs(returns: np.ndarray, lag: int) -> float:
    """Compute R/S (Rescaled Range) value for a given lag."""
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0.0

    cumsum = np.cumsum(returns - mean)
    r = np.max(cumsum) - np.min(cumsum)
    return r / std if std > 0 else 0.0


@njit
def compute_hurst_rs(prices: np.ndarray, min_lag: int, max_lag: int) -> float:
    """
    Compute Hurst exponent using Rescaled Range (R/S) analysis.

    Args:
        prices: Price series (must be positive)
        min_lag: Minimum lag for analysis
        max_lag: Maximum lag for analysis

    Returns:
        Hurst exponent (0-1). H > 0.5 indicates persistence (trending),
        H < 0.5 indicates anti-persistence (mean-reverting),
        H = 0.5 indicates random walk.
    """
    # Handle edge cases
    if len(prices) < max_lag:
        max_lag = len(prices) // 2
    if max_lag <= min_lag:
        return 0.5

    returns = np.diff(np.log(prices))
    n_lags = max_lag - min_lag
    tau = np.empty(n_lags)
    rs_values = np.empty(n_lags)

    for i, lag in enumerate(range(min_lag, max_lag)):
        n_segments = (len(returns) - lag) // lag
        if n_segments <= 0:
            tau[i] = 0
            rs_values[i] = 0
            continue

        rs_current = np.empty(n_segments)

        for j in range(n_segments):
            start = j * lag
            rs_current[j] = compute_rs(returns[start:start + lag], lag)

        valid_rs = rs_current[rs_current > 0]
        if len(valid_rs) > 0:
            tau[i] = np.log(lag)
            rs_values[i] = np.log(np.mean(valid_rs))
        else:
            tau[i] = 0
            rs_values[i] = 0

    # Filter out zero values and perform regression
    mask = (tau > 0) & (rs_values > 0)
    if np.sum(mask) > 1:
        return linear_regression(tau[mask], rs_values[mask])
    return 0.5


# =============================================================================
# Hurst Exponent (DFA - Detrended Fluctuation Analysis)
# =============================================================================

@njit
def compute_hurst_dfa(prices: np.ndarray, min_scale: int, max_scale: int) -> float:
    """
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).

    Args:
        prices: Price series
        min_scale: Minimum scale (window size)
        max_scale: Maximum scale (window size)

    Returns:
        Hurst exponent via DFA
    """
    if len(prices) < max_scale:
        max_scale = len(prices) // 2
    if max_scale <= min_scale:
        return 0.5

    # Compute cumulative sum of deviations
    returns = np.diff(np.log(prices))
    mean_ret = np.mean(returns)
    y = np.cumsum(returns - mean_ret)

    # Compute F(n) for different scales
    scales = []
    fluctuations = []

    for scale in range(min_scale, max_scale):
        n_segments = len(y) // scale
        if n_segments < 2:
            continue

        f_sum = 0.0
        count = 0

        for i in range(n_segments):
            start = i * scale
            end = start + scale
            segment = y[start:end]

            # Linear detrending
            x = np.arange(scale, dtype=np.float64)
            slope = linear_regression(x, segment)
            intercept = np.mean(segment) - slope * np.mean(x)
            trend = slope * x + intercept

            # Compute variance of residuals
            residuals = segment - trend
            f_sum += np.sum(residuals ** 2)
            count += scale

        if count > 0:
            f_n = np.sqrt(f_sum / count)
            if f_n > 0:
                scales.append(np.log(scale))
                fluctuations.append(np.log(f_n))

    if len(scales) < 2:
        return 0.5

    scales_arr = np.array(scales)
    fluct_arr = np.array(fluctuations)
    return linear_regression(scales_arr, fluct_arr)


# =============================================================================
# Fractal Dimension (Box-Counting)
# =============================================================================

@njit
def compute_box_dimension(
    scaled_prices: np.ndarray,
    min_window: int,
    max_window: int,
    step: int
) -> float:
    """
    Compute box-counting fractal dimension.

    Args:
        scaled_prices: Prices scaled to [0, 1]
        min_window: Minimum box size
        max_window: Maximum box size
        step: Step size between box sizes

    Returns:
        Fractal dimension (typically 1.0-2.0 for time series)
    """
    if step <= 0:
        step = 1

    num_scales = (max_window - min_window) // step
    if num_scales <= 0:
        return 1.5

    dimensions = np.empty(num_scales)
    valid_count = 0

    for scale in range(min_window, max_window, step):
        if scale <= 1:
            continue

        boxes = np.ceil(scaled_prices * scale)
        unique_boxes = len(np.unique(boxes))

        if unique_boxes > 0:
            dimensions[valid_count] = np.log(unique_boxes) / np.log(scale)
            valid_count += 1

    if valid_count > 0:
        return np.mean(dimensions[:valid_count])
    return 1.5


# =============================================================================
# Volatility
# =============================================================================

@njit
def compute_rolling_volatility(
    prices: np.ndarray,
    window: int,
    annualize: bool = True
) -> np.ndarray:
    """
    Compute rolling volatility.

    Args:
        prices: Price series
        window: Rolling window size
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        Rolling volatility series
    """
    n = len(prices)
    if n < window + 1:
        return np.zeros(n)

    returns = np.diff(np.log(prices))
    vol = np.zeros(n)

    for i in range(window, n):
        window_returns = returns[i - window:i]
        vol[i] = np.std(window_returns)

    # Fill initial points
    if window < n:
        vol[:window] = vol[window]

    if annualize:
        vol *= np.sqrt(252)

    return vol


@njit(parallel=True)
def compute_rolling_volatility_multi(
    returns: np.ndarray,
    window_sizes: np.ndarray
) -> np.ndarray:
    """
    Compute rolling volatility for multiple window sizes in parallel.

    Args:
        returns: Return series
        window_sizes: Array of window sizes

    Returns:
        2D array of shape (len(window_sizes), len(returns))
    """
    n_windows = len(window_sizes)
    n = len(returns)
    result = np.zeros((n_windows, n))

    for w in prange(n_windows):
        window = window_sizes[w]
        for i in range(window, n):
            window_returns = returns[i - window:i]
            result[w, i] = np.std(window_returns)

        # Fill initial points
        if window < n:
            result[w, :window] = result[w, window]

    return result


# =============================================================================
# Pattern Similarity
# =============================================================================

@njit
def compute_pattern_similarity(
    pattern1: np.ndarray,
    pattern2: np.ndarray
) -> float:
    """
    Compute normalized correlation between two patterns.

    Args:
        pattern1: First pattern
        pattern2: Second pattern

    Returns:
        Correlation coefficient (-1 to 1)
    """
    std1 = np.std(pattern1)
    std2 = np.std(pattern2)

    if std1 < 1e-8 or std2 < 1e-8:
        return 0.0

    p1_norm = (pattern1 - np.mean(pattern1)) / std1
    p2_norm = (pattern2 - np.mean(pattern2)) / std2

    return np.mean(p1_norm * p2_norm)


@njit
def find_similar_patterns(
    returns: np.ndarray,
    pattern: np.ndarray,
    min_similarity: float = 0.5
) -> np.ndarray:
    """
    Find historical patterns similar to the given pattern.

    Args:
        returns: Historical returns
        pattern: Pattern to match
        min_similarity: Minimum similarity threshold

    Returns:
        Array of (start_index, similarity) pairs
    """
    pattern_len = len(pattern)
    n = len(returns)
    max_matches = n - pattern_len

    if max_matches <= 0:
        return np.empty((0, 2))

    matches = np.empty((max_matches, 2))
    count = 0

    for i in range(max_matches):
        historical = returns[i:i + pattern_len]
        sim = compute_pattern_similarity(historical, pattern)

        if sim >= min_similarity:
            matches[count, 0] = i
            matches[count, 1] = sim
            count += 1

    return matches[:count]


# =============================================================================
# Fractional Brownian Motion
# =============================================================================

@njit
def generate_fbm(n: int, hurst: float) -> np.ndarray:
    """
    Generate fractional Brownian motion path.

    Args:
        n: Number of steps
        hurst: Hurst exponent

    Returns:
        fBm path of length n
    """
    noise = np.random.normal(0, 1, n)
    dt = 1.0 / n
    t = np.arange(n) * dt

    fbm = np.zeros(n)
    fbm[0] = 0

    for i in range(1, n):
        weights = np.zeros(i)
        for j in range(i):
            weights[j] = ((i - j) ** (hurst - 0.5) - (i - j - 1) ** (hurst - 0.5))
        fbm[i] = np.sum(weights * noise[:i])

    # Scale by sqrt(dt)
    fbm *= np.sqrt(dt)

    return fbm


@njit
def generate_paths_from_pattern(
    last_price: float,
    returns_history: np.ndarray,
    pattern_indices: np.ndarray,
    pattern_weights: np.ndarray,
    pattern_len: int,
    n_steps: int,
    n_paths: int,
    volatility: float
) -> np.ndarray:
    """
    Generate price paths based on historical patterns.

    Args:
        last_price: Starting price
        returns_history: Historical returns
        pattern_indices: Starting indices of similar patterns
        pattern_weights: Weights for each pattern
        pattern_len: Length of pattern used for matching
        n_steps: Number of steps to forecast
        n_paths: Number of paths to generate
        volatility: Base volatility for noise

    Returns:
        Paths array of shape (n_paths, n_steps)
    """
    paths = np.zeros((n_paths, n_steps))
    n_patterns = len(pattern_indices)

    for p in range(n_paths):
        # Select pattern based on weights
        r = np.random.random()
        cumsum = 0.0
        selected = 0
        for i in range(n_patterns):
            cumsum += pattern_weights[i]
            if r <= cumsum:
                selected = i
                break

        # Get returns following the pattern
        start_idx = int(pattern_indices[selected]) + pattern_len

        price = last_price
        for t in range(n_steps):
            # Use historical return if available, else random
            if start_idx + t < len(returns_history):
                base_return = returns_history[start_idx + t]
            else:
                base_return = np.random.normal(0, volatility)

            # Add some noise
            noise = np.random.normal(0, volatility * 0.5)
            ret = base_return + noise

            price = price * np.exp(ret)
            paths[p, t] = price

    return paths


# =============================================================================
# Bootstrap
# =============================================================================

@njit
def bootstrap_hurst(
    prices: np.ndarray,
    n_samples: int,
    min_lag: int,
    max_lag: int,
    block_size: int
) -> np.ndarray:
    """
    Bootstrap Hurst exponent using block bootstrap.

    Args:
        prices: Price series
        n_samples: Number of bootstrap samples
        min_lag: Minimum lag for Hurst calculation
        max_lag: Maximum lag for Hurst calculation
        block_size: Size of blocks for block bootstrap

    Returns:
        Array of bootstrapped Hurst values
    """
    n = len(prices)
    results = np.empty(n_samples)
    n_blocks = n // block_size

    for s in range(n_samples):
        # Block bootstrap
        boot_prices = np.empty(n)
        idx = 0

        while idx < n:
            # Random starting block
            block_start = np.random.randint(0, n - block_size + 1)
            block_end = min(block_start + block_size, n)
            copy_len = min(block_end - block_start, n - idx)

            boot_prices[idx:idx + copy_len] = prices[block_start:block_start + copy_len]
            idx += copy_len

        results[s] = compute_hurst_rs(boot_prices[:n], min_lag, max_lag)

    return results


@njit
def bootstrap_fractal_dim(
    prices: np.ndarray,
    n_samples: int,
    min_window: int,
    max_window: int,
    step: int,
    block_size: int
) -> np.ndarray:
    """
    Bootstrap fractal dimension using block bootstrap.

    Args:
        prices: Price series
        n_samples: Number of bootstrap samples
        min_window: Minimum box size
        max_window: Maximum box size
        step: Step size
        block_size: Size of blocks for block bootstrap

    Returns:
        Array of bootstrapped fractal dimension values
    """
    n = len(prices)
    results = np.empty(n_samples)

    # Scale prices once
    p_min = np.min(prices)
    p_max = np.max(prices)
    if p_max - p_min < 1e-10:
        return np.full(n_samples, 1.5)

    for s in range(n_samples):
        # Block bootstrap
        boot_prices = np.empty(n)
        idx = 0

        while idx < n:
            block_start = np.random.randint(0, n - block_size + 1)
            block_end = min(block_start + block_size, n)
            copy_len = min(block_end - block_start, n - idx)

            boot_prices[idx:idx + copy_len] = prices[block_start:block_start + copy_len]
            idx += copy_len

        # Scale bootstrapped prices
        b_min = np.min(boot_prices)
        b_max = np.max(boot_prices)
        if b_max - b_min < 1e-10:
            results[s] = 1.5
        else:
            scaled = (boot_prices - b_min) / (b_max - b_min)
            results[s] = compute_box_dimension(scaled, min_window, max_window, step)

    return results
