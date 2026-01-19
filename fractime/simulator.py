"""
Monte Carlo path simulation for fractal time series.

The Simulator class generates price paths based on fractal properties
and historical patterns.

Examples:
    >>> import fractime as ft
    >>> sim = ft.Simulator(prices)
    >>> paths = sim.generate(n_paths=1000, steps=30)
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import polars as pl

from .analyzer import Analyzer, _ensure_numpy, _ensure_dates
from ._numba import (
    compute_hurst_rs,
    compute_rolling_volatility,
    generate_fbm,
    find_similar_patterns,
    generate_paths_from_pattern,
)


class Simulator:
    """
    Monte Carlo path simulator using fractal properties.

    Generates price paths that preserve the fractal characteristics
    (Hurst exponent, volatility clustering) of the historical data.

    Args:
        data: Historical price series
        dates: Optional date series
        analyzer: Pre-computed Analyzer (optional)
        time_warp: Use Mandelbrot's trading time concept
        weights: Custom weights for path generation {'hurst': 0.3, ...}

    Examples:
        Basic usage:
            >>> sim = Simulator(prices)
            >>> paths = sim.generate(n_paths=1000, steps=30)

        With time warping:
            >>> sim = Simulator(prices, time_warp=True)
            >>> paths = sim.generate(n_paths=1000, steps=30)

        Custom weights:
            >>> sim = Simulator(prices, weights={'hurst': 0.5, 'pattern': 0.5})
            >>> paths = sim.generate(n_paths=1000, steps=30)

        From analyzer:
            >>> analyzer = Analyzer(prices)
            >>> sim = Simulator(prices, analyzer=analyzer)
    """

    def __init__(
        self,
        data: Union[np.ndarray, pl.Series],
        dates: Optional[Union[np.ndarray, pl.Series]] = None,
        analyzer: Optional[Analyzer] = None,
        time_warp: bool = False,
        weights: Optional[dict] = None,
    ):
        self._data = _ensure_numpy(data)
        self._dates = _ensure_dates(dates)
        self._time_warp = time_warp

        # Path generation weights
        self._weights = weights or {
            'hurst': 0.4,
            'volatility': 0.3,
            'pattern': 0.3,
        }

        # Create or use provided analyzer
        if analyzer is not None:
            self._analyzer = analyzer
        else:
            self._analyzer = Analyzer(self._data, dates=self._dates)

        # Precompute
        self._returns = np.diff(np.log(self._data))
        self._volatility = self._analyzer.volatility.value
        self._hurst = self._analyzer.hurst.value

        # Time warping
        self._time_scale: Optional[np.ndarray] = None
        if time_warp:
            self._compute_time_warp()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def analyzer(self) -> Analyzer:
        """The analyzer used for fractal analysis."""
        return self._analyzer

    @property
    def hurst(self) -> float:
        """Hurst exponent of historical data."""
        return self._hurst

    @property
    def volatility(self) -> float:
        """Annualized volatility of historical data."""
        return self._volatility

    # =========================================================================
    # Time Warping
    # =========================================================================

    def _compute_time_warp(self):
        """Compute Mandelbrot's trading time transformation."""
        # Rolling volatility at multiple scales
        windows = [5, 21, 63]
        weights = [0.5, 0.3, 0.2]

        n = len(self._data)
        combined_vol = np.zeros(n)

        for window, weight in zip(windows, weights):
            vol = compute_rolling_volatility(self._data, window, annualize=False)
            combined_vol += weight * vol

        # Normalize to mean 1
        mean_vol = np.mean(combined_vol[combined_vol > 0])
        if mean_vol > 0:
            relative_vol = combined_vol / mean_vol
        else:
            relative_vol = np.ones(n)

        # Time flows faster during high volatility
        self._time_scale = np.clip(relative_vol, 0.1, 10.0)

    # =========================================================================
    # Path Generation
    # =========================================================================

    def generate(
        self,
        n_paths: int = 1000,
        steps: int = 30,
        method: str = 'auto',
    ) -> np.ndarray:
        """
        Generate Monte Carlo price paths.

        Args:
            n_paths: Number of paths to generate
            steps: Number of steps per path
            method: Generation method
                - 'auto': Choose best method based on data
                - 'fbm': Fractional Brownian motion
                - 'pattern': Pattern-based generation
                - 'bootstrap': Block bootstrap

        Returns:
            Array of shape (n_paths, steps) with generated paths
        """
        if method == 'auto':
            method = self._choose_method(steps)

        if method == 'fbm':
            return self._generate_fbm(n_paths, steps)
        elif method == 'pattern':
            return self._generate_pattern(n_paths, steps)
        elif method == 'bootstrap':
            return self._generate_bootstrap(n_paths, steps)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _choose_method(self, steps: int) -> str:
        """Choose best generation method based on data characteristics."""
        # Use pattern matching if enough history
        if len(self._returns) > steps * 10:
            return 'pattern'
        # Use fBm for shorter histories
        return 'fbm'

    def _generate_fbm(self, n_paths: int, steps: int) -> np.ndarray:
        """Generate paths using fractional Brownian motion."""
        last_price = self._data[-1]
        daily_vol = self._volatility / np.sqrt(252)

        paths = np.zeros((n_paths, steps))

        for i in range(n_paths):
            # Generate fBm
            fbm = generate_fbm(steps, self._hurst)
            increments = np.diff(np.concatenate([[0], fbm]))

            # Scale by volatility
            if self._time_warp and self._time_scale is not None:
                # Use recent time scale for future
                recent_scale = np.mean(self._time_scale[-21:])
                scaled_vol = daily_vol * recent_scale
            else:
                scaled_vol = daily_vol

            returns = increments * scaled_vol

            # Convert to prices
            price = last_price
            for t in range(steps):
                price = price * np.exp(returns[t])
                paths[i, t] = price

        return paths

    def _generate_pattern(self, n_paths: int, steps: int) -> np.ndarray:
        """Generate paths based on historical pattern matching."""
        # Find similar patterns to recent history
        pattern_len = min(21, len(self._returns) // 10)
        current_pattern = self._returns[-pattern_len:]

        matches = find_similar_patterns(
            self._returns[:-pattern_len],
            current_pattern,
            min_similarity=0.3
        )

        if len(matches) == 0:
            # Fall back to fBm
            return self._generate_fbm(n_paths, steps)

        # Normalize similarities to weights
        similarities = matches[:, 1]
        weights = similarities / np.sum(similarities)

        # Generate paths
        last_price = self._data[-1]
        daily_vol = self._volatility / np.sqrt(252)

        paths = generate_paths_from_pattern(
            last_price,
            self._returns,
            matches[:, 0].astype(np.int64),
            weights,
            pattern_len,
            steps,
            n_paths,
            daily_vol
        )

        return paths

    def _generate_bootstrap(self, n_paths: int, steps: int) -> np.ndarray:
        """Generate paths using block bootstrap."""
        last_price = self._data[-1]
        block_size = min(21, len(self._returns) // 5)

        paths = np.zeros((n_paths, steps))

        for i in range(n_paths):
            # Sample blocks of returns
            path_returns = np.zeros(steps)
            idx = 0

            while idx < steps:
                # Random starting point
                start = np.random.randint(0, len(self._returns) - block_size + 1)
                copy_len = min(block_size, steps - idx)
                path_returns[idx:idx + copy_len] = self._returns[start:start + copy_len]
                idx += copy_len

            # Convert to prices
            price = last_price
            for t in range(steps):
                price = price * np.exp(path_returns[t])
                paths[i, t] = price

        return paths

    def __repr__(self) -> str:
        return (
            f"Simulator("
            f"n={len(self._data)}, "
            f"hurst={self._hurst:.3f}, "
            f"time_warp={self._time_warp})"
        )
