"""
Optimization functions and utilities for the FracTime system.
"""
import numpy as np
import pandas as pd
from numba import njit, prange
import cProfile
import pstats
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@njit
def compute_pattern_similarities(pattern1_returns, pattern2_returns, pattern1_vol, pattern2_vol):
    """Compute pattern similarity with JIT compilation."""
    if pattern1_vol < 1e-8 or pattern2_vol < 1e-8:
        return 0.0
        
    pattern1_norm = (pattern1_returns - np.mean(pattern1_returns)) / pattern1_vol
    pattern2_norm = (pattern2_returns - np.mean(pattern2_returns)) / pattern2_vol
    
    # Manual correlation calculation since np.corrcoef isn't supported by numba
    mean1 = np.mean(pattern1_norm)
    mean2 = np.mean(pattern2_norm)
    numerator = np.sum((pattern1_norm - mean1) * (pattern2_norm - mean2))
    denom1 = np.sqrt(np.sum((pattern1_norm - mean1)**2))
    denom2 = np.sqrt(np.sum((pattern2_norm - mean2)**2))
    
    if denom1 * denom2 < 1e-8:
        return 0.0
        
    return numerator / (denom1 * denom2)

@njit(parallel=True)
def compute_rolling_volatilities(returns, window_sizes):
    """Efficiently compute rolling volatilities for multiple window sizes using Numba."""
    result = {}
    
    for window in window_sizes:
        n = len(returns) - window + 1
        volatilities = np.zeros(n)
        
        for i in prange(n):
            window_returns = returns[i:i+window]
            volatilities[i] = np.std(window_returns)
            
        result[window] = volatilities
        
    return result

def try_import_cupy():
    """Attempt to import CuPy for GPU acceleration."""
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None

def benchmark_system(symbol="AAPL", start_date="2020-01-01", n_steps=30, n_paths=1000):
    """Profile the FracTime system to identify bottlenecks."""
    from fractime import get_yahoo_data, FractalAnalyzer, FractalSimulator
    
    print(f"Benchmarking FracTime with {symbol} data...")
    
    # Get sample data
    data = get_yahoo_data(symbol, start_date)
    prices = data['Close'].to_numpy()
    
    # Create a profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the operations to profile
    start_time = time.time()
    analyzer = FractalAnalyzer()
    simulator = FractalSimulator(prices, analyzer)
    
    print(f"Initialization took {time.time() - start_time:.2f} seconds")
    
    # Time the standard simulation
    start_time = time.time()
    paths, _ = simulator.simulate_paths(n_steps=n_steps, n_paths=n_paths)
    standard_time = time.time() - start_time
    print(f"Standard simulation with {n_paths} paths took {standard_time:.2f} seconds")
    
    # Time the fast simulation
    start_time = time.time()
    paths, _ = simulator.simulate_paths_fast(n_steps=n_steps, n_paths=n_paths)
    fast_time = time.time() - start_time
    print(f"Fast simulation with {n_paths} paths took {fast_time:.2f} seconds")
    print(f"Speedup: {standard_time / fast_time:.1f}x")
    
    # Try GPU simulation if available
    try:
        start_time = time.time()
        paths, _ = simulator.simulate_paths_gpu(n_steps=n_steps, n_paths=n_paths)
        gpu_time = time.time() - start_time
        print(f"GPU simulation with {n_paths} paths took {gpu_time:.2f} seconds")
        print(f"GPU speedup vs standard: {standard_time / gpu_time:.1f}x")
    except:
        print("GPU simulation not available or failed")
    
    # Disable profiling
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    print("\nTop 20 most time-consuming functions:")
    stats.print_stats(20)
    
    return stats 

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