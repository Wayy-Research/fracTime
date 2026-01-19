"""
Utility functions for fractime.

Common helper functions used throughout the package.
"""

import numpy as np
import polars as pl
import pandas as pd
import yfinance as yf
from datetime import datetime
from numba import njit


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

        # Filter valid values
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
    """Box-counting dimension calculation."""
    dimensions = []

    for scale in range(min_window, max_window, step):
        boxes = np.ceil(scaled_prices * scale)
        unique_boxes = len(np.unique(boxes))

        if unique_boxes > 0:
            dimensions.append(np.log(unique_boxes) / np.log(scale))

    if len(dimensions) > 0:
        return np.mean(np.array(dimensions))
    else:
        return 1.5
