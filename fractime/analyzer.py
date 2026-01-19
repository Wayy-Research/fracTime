"""
Fractal analysis with lazy computation.

The Analyzer class provides fractal analysis of time series data.
All metrics are computed lazily on first access and cached.

Examples:
    >>> import fractime as ft
    >>> analyzer = ft.Analyzer(prices)
    >>> analyzer.hurst                    # Point estimate
    >>> analyzer.hurst.rolling            # Rolling values
    >>> analyzer.hurst.ci(0.95)           # Confidence interval
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import polars as pl

from .result import Metric, AnalysisResult
from ._numba import (
    compute_hurst_rs,
    compute_hurst_dfa,
    compute_box_dimension,
    compute_rolling_volatility,
    bootstrap_hurst,
    bootstrap_fractal_dim,
)


def _ensure_numpy(data) -> np.ndarray:
    """Convert input to numpy array."""
    if data is None:
        raise ValueError("Data cannot be None")
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, pl.Series):
        return data.to_numpy()
    if isinstance(data, pl.DataFrame):
        raise ValueError("Expected Series, got DataFrame. Select a column first.")
    return np.asarray(data)


def _ensure_dates(dates) -> Optional[np.ndarray]:
    """Convert dates to numpy array if provided."""
    if dates is None:
        return None
    if isinstance(dates, np.ndarray):
        return dates
    if isinstance(dates, pl.Series):
        return dates.to_numpy()
    return np.asarray(dates)


class Analyzer:
    """
    Fractal time series analyzer with lazy computation.

    All metrics are computed on first access and cached. This means you
    only pay for what you use - accessing just the Hurst exponent is fast,
    while bootstrap confidence intervals are computed only when requested.

    Args:
        data: Price/value series (numpy array, Polars Series, or dict for multi-dimensional)
        dates: Optional date series (for rolling computations with timestamps)
        method: Hurst calculation method ('rs' for Rescaled Range, 'dfa' for DFA)
        window: Rolling window size (default 63, ~3 months of trading days)
        n_samples: Number of bootstrap samples for uncertainty estimation
        min_scale: Minimum scale for fractal analysis
        max_scale: Maximum scale for fractal analysis

    Examples:
        Basic usage:
            >>> analyzer = Analyzer(prices)
            >>> analyzer.hurst                # 0.67
            >>> analyzer.fractal_dim          # 1.43
            >>> analyzer.regime               # 'trending'

        With rolling analysis:
            >>> analyzer = Analyzer(prices, dates=dates)
            >>> analyzer.hurst.rolling        # DataFrame with date, value

        With uncertainty:
            >>> analyzer = Analyzer(prices)
            >>> analyzer.hurst.ci(0.95)       # (0.61, 0.73)
            >>> analyzer.hurst.std            # 0.04

        Multi-dimensional:
            >>> analyzer = Analyzer({'price': prices, 'volume': volumes})
            >>> analyzer['price'].hurst
            >>> analyzer['volume'].hurst
            >>> analyzer.coherence
    """

    def __init__(
        self,
        data: Union[np.ndarray, pl.Series, dict],
        dates: Optional[Union[np.ndarray, pl.Series]] = None,
        method: str = 'rs',
        window: int = 63,
        n_samples: int = 1000,
        min_scale: int = 10,
        max_scale: int = 100,
    ):
        # Handle multi-dimensional data
        if isinstance(data, dict):
            self._multi_dim = True
            self._dimensions = {k: _ensure_numpy(v) for k, v in data.items()}
            self._data = list(self._dimensions.values())[0]  # Primary dimension
            self._analyzers = {}
        else:
            self._multi_dim = False
            self._dimensions = None
            self._data = _ensure_numpy(data)
            self._analyzers = None

        self._dates = _ensure_dates(dates)
        self._method = method
        self._window = window
        self._n_samples = n_samples
        self._min_scale = min_scale
        self._max_scale = max_scale

        # Lazy caches
        self._hurst: Optional[Metric] = None
        self._fractal_dim: Optional[Metric] = None
        self._volatility: Optional[Metric] = None
        self._regime: Optional[str] = None
        self._regime_probs: Optional[dict] = None
        self._coherence: Optional[Metric] = None
        self._result: Optional[AnalysisResult] = None

    # =========================================================================
    # Multi-dimensional access
    # =========================================================================

    def __getitem__(self, key: str) -> 'Analyzer':
        """Access a specific dimension in multi-dimensional analysis."""
        if not self._multi_dim:
            raise KeyError("Single-dimensional analyzer. Use properties directly.")
        if key not in self._dimensions:
            raise KeyError(f"Dimension '{key}' not found. Available: {list(self._dimensions.keys())}")

        if key not in self._analyzers:
            self._analyzers[key] = Analyzer(
                self._dimensions[key],
                dates=self._dates,
                method=self._method,
                window=self._window,
                n_samples=self._n_samples,
                min_scale=self._min_scale,
                max_scale=self._max_scale,
            )
        return self._analyzers[key]

    @property
    def dimensions(self) -> list[str]:
        """List of dimension names (multi-dimensional only)."""
        if not self._multi_dim:
            return []
        return list(self._dimensions.keys())

    # =========================================================================
    # Hurst Exponent
    # =========================================================================

    @property
    def hurst(self) -> Metric:
        """
        Hurst exponent of the series.

        H > 0.5: Trending (persistent) behavior
        H = 0.5: Random walk
        H < 0.5: Mean-reverting (anti-persistent) behavior
        """
        if self._hurst is None:
            self._hurst = Metric(
                name='hurst',
                compute_point_fn=self._compute_hurst_point,
                compute_rolling_fn=self._compute_hurst_rolling,
                compute_bootstrap_fn=self._compute_hurst_bootstrap,
            )
        return self._hurst

    def _compute_hurst_point(self) -> float:
        """Compute point estimate of Hurst exponent."""
        if self._method == 'dfa':
            return compute_hurst_dfa(self._data, self._min_scale, self._max_scale)
        return compute_hurst_rs(self._data, self._min_scale, self._max_scale)

    def _compute_hurst_rolling(self) -> pl.DataFrame:
        """Compute rolling Hurst exponent."""
        n = len(self._data)
        values = []
        indices = []

        for i in range(self._window, n):
            segment = self._data[i - self._window:i]
            if self._method == 'dfa':
                h = compute_hurst_dfa(segment, self._min_scale, min(self._max_scale, self._window // 2))
            else:
                h = compute_hurst_rs(segment, self._min_scale, min(self._max_scale, self._window // 2))
            values.append(h)
            indices.append(i)

        if self._dates is not None:
            return pl.DataFrame({
                'date': self._dates[indices],
                'value': values
            })
        return pl.DataFrame({
            'index': indices,
            'value': values
        })

    def _compute_hurst_bootstrap(self) -> np.ndarray:
        """Compute bootstrap distribution of Hurst exponent."""
        block_size = max(self._window // 4, 10)
        return bootstrap_hurst(
            self._data,
            self._n_samples,
            self._min_scale,
            min(self._max_scale, len(self._data) // 4),
            block_size
        )

    # =========================================================================
    # Fractal Dimension
    # =========================================================================

    @property
    def fractal_dim(self) -> Metric:
        """
        Fractal (box-counting) dimension of the series.

        D ≈ 1.0: Smooth, simple patterns
        D ≈ 1.5: Moderate complexity (typical for financial time series)
        D ≈ 2.0: Highly complex, space-filling patterns
        """
        if self._fractal_dim is None:
            self._fractal_dim = Metric(
                name='fractal_dim',
                compute_point_fn=self._compute_fractal_dim_point,
                compute_rolling_fn=self._compute_fractal_dim_rolling,
                compute_bootstrap_fn=self._compute_fractal_dim_bootstrap,
            )
        return self._fractal_dim

    def _compute_fractal_dim_point(self) -> float:
        """Compute point estimate of fractal dimension."""
        scaled = self._scale_prices(self._data)
        return compute_box_dimension(scaled, 2, min(50, len(self._data) // 4), 1)

    def _compute_fractal_dim_rolling(self) -> pl.DataFrame:
        """Compute rolling fractal dimension."""
        n = len(self._data)
        values = []
        indices = []

        for i in range(self._window, n):
            segment = self._data[i - self._window:i]
            scaled = self._scale_prices(segment)
            fd = compute_box_dimension(scaled, 2, min(30, self._window // 4), 1)
            values.append(fd)
            indices.append(i)

        if self._dates is not None:
            return pl.DataFrame({
                'date': self._dates[indices],
                'value': values
            })
        return pl.DataFrame({
            'index': indices,
            'value': values
        })

    def _compute_fractal_dim_bootstrap(self) -> np.ndarray:
        """Compute bootstrap distribution of fractal dimension."""
        block_size = max(self._window // 4, 10)
        return bootstrap_fractal_dim(
            self._data,
            self._n_samples,
            2,
            min(50, len(self._data) // 4),
            1,
            block_size
        )

    @staticmethod
    def _scale_prices(prices: np.ndarray) -> np.ndarray:
        """Scale prices to [0, 1] range."""
        p_min = np.min(prices)
        p_max = np.max(prices)
        if p_max - p_min < 1e-10:
            return np.zeros_like(prices)
        return (prices - p_min) / (p_max - p_min)

    # =========================================================================
    # Volatility
    # =========================================================================

    @property
    def volatility(self) -> Metric:
        """
        Annualized volatility of the series.

        Computed as the standard deviation of log returns, annualized by sqrt(252).
        """
        if self._volatility is None:
            self._volatility = Metric(
                name='volatility',
                compute_point_fn=self._compute_volatility_point,
                compute_rolling_fn=self._compute_volatility_rolling,
                compute_bootstrap_fn=self._compute_volatility_bootstrap,
            )
        return self._volatility

    def _compute_volatility_point(self) -> float:
        """Compute point estimate of volatility."""
        returns = np.diff(np.log(self._data))
        return float(np.std(returns) * np.sqrt(252))

    def _compute_volatility_rolling(self) -> pl.DataFrame:
        """Compute rolling volatility."""
        vol = compute_rolling_volatility(self._data, self._window, annualize=True)

        if self._dates is not None:
            return pl.DataFrame({
                'date': self._dates,
                'value': vol
            }).filter(pl.col('value') > 0)
        return pl.DataFrame({
            'index': list(range(len(vol))),
            'value': vol
        }).filter(pl.col('value') > 0)

    def _compute_volatility_bootstrap(self) -> np.ndarray:
        """Compute bootstrap distribution of volatility."""
        returns = np.diff(np.log(self._data))
        results = np.empty(self._n_samples)
        n = len(returns)
        block_size = max(self._window // 4, 10)

        for s in range(self._n_samples):
            # Block bootstrap
            boot_returns = np.empty(n)
            idx = 0
            while idx < n:
                block_start = np.random.randint(0, n - block_size + 1)
                copy_len = min(block_size, n - idx)
                boot_returns[idx:idx + copy_len] = returns[block_start:block_start + copy_len]
                idx += copy_len

            results[s] = np.std(boot_returns) * np.sqrt(252)

        return results

    # =========================================================================
    # Regime Detection
    # =========================================================================

    @property
    def regime(self) -> str:
        """
        Current market regime based on fractal properties.

        Returns one of:
            - 'trending': H > 0.55, persistent behavior expected
            - 'mean_reverting': H < 0.45, mean-reverting behavior expected
            - 'random': 0.45 <= H <= 0.55, random walk behavior
        """
        if self._regime is None:
            self._detect_regime()
        return self._regime

    @property
    def regime_probabilities(self) -> dict[str, float]:
        """Probability distribution over regimes."""
        if self._regime_probs is None:
            self._detect_regime()
        return self._regime_probs

    def _detect_regime(self):
        """Detect market regime from Hurst exponent."""
        h = self.hurst.value

        # Simple regime classification based on Hurst
        if h > 0.55:
            self._regime = 'trending'
            # Probability increases with distance from 0.5
            prob_trend = min(0.95, 0.5 + (h - 0.55) * 2)
            self._regime_probs = {
                'trending': prob_trend,
                'random': (1 - prob_trend) * 0.7,
                'mean_reverting': (1 - prob_trend) * 0.3
            }
        elif h < 0.45:
            self._regime = 'mean_reverting'
            prob_mr = min(0.95, 0.5 + (0.45 - h) * 2)
            self._regime_probs = {
                'mean_reverting': prob_mr,
                'random': (1 - prob_mr) * 0.7,
                'trending': (1 - prob_mr) * 0.3
            }
        else:
            self._regime = 'random'
            # Probability peaks at H=0.5
            distance = abs(h - 0.5)
            prob_random = max(0.4, 0.8 - distance * 4)
            remaining = 1 - prob_random
            self._regime_probs = {
                'random': prob_random,
                'trending': remaining * (0.5 + (h - 0.5) * 2),
                'mean_reverting': remaining * (0.5 - (h - 0.5) * 2)
            }

    # =========================================================================
    # Multi-Dimensional: Coherence
    # =========================================================================

    @property
    def coherence(self) -> Metric:
        """
        Cross-dimensional fractal coherence (multi-dimensional only).

        Measures how consistently fractal properties behave across dimensions.
        High coherence suggests strong cross-dimensional coupling.
        """
        if not self._multi_dim:
            raise ValueError("Coherence requires multi-dimensional data.")

        if self._coherence is None:
            self._coherence = Metric(
                name='coherence',
                compute_point_fn=self._compute_coherence_point,
                compute_rolling_fn=self._compute_coherence_rolling,
                compute_bootstrap_fn=None,  # TODO: implement
            )
        return self._coherence

    def _compute_coherence_point(self) -> float:
        """Compute cross-dimensional coherence."""
        if not self._multi_dim or len(self._dimensions) < 2:
            return 0.0

        # Get Hurst exponents for all dimensions
        hursts = [self[dim].hurst.value for dim in self.dimensions]

        # Coherence is inverse of variance in Hurst values
        variance = np.var(hursts)
        return float(1.0 / (1.0 + variance * 10))

    def _compute_coherence_rolling(self) -> pl.DataFrame:
        """Compute rolling coherence across dimensions."""
        if not self._multi_dim or len(self._dimensions) < 2:
            raise ValueError("Rolling coherence requires multi-dimensional data.")

        n = len(self._data)
        values = []
        indices = []

        for i in range(self._window, n):
            hursts = []
            for dim in self.dimensions:
                segment = self._dimensions[dim][i - self._window:i]
                h = compute_hurst_rs(segment, self._min_scale, min(self._max_scale, self._window // 2))
                hursts.append(h)

            variance = np.var(hursts)
            coherence = 1.0 / (1.0 + variance * 10)
            values.append(coherence)
            indices.append(i)

        if self._dates is not None:
            return pl.DataFrame({
                'date': self._dates[indices],
                'value': values
            })
        return pl.DataFrame({
            'index': indices,
            'value': values
        })

    # =========================================================================
    # Result Object
    # =========================================================================

    @property
    def result(self) -> AnalysisResult:
        """Complete analysis result as a structured object."""
        if self._result is None:
            self._result = AnalysisResult(
                hurst=self.hurst,
                fractal_dim=self.fractal_dim,
                volatility=self.volatility,
                regime=self.regime,
                regime_probabilities=self.regime_probabilities,
            )
        return self._result

    def summary(self) -> str:
        """Text summary of analysis results."""
        return self.result.summary()

    def to_frame(self) -> pl.DataFrame:
        """Export all rolling metrics as Polars DataFrame."""
        return self.result.to_frame()

    def __repr__(self) -> str:
        if self._multi_dim:
            return f"Analyzer(dimensions={self.dimensions})"
        return f"Analyzer(n={len(self._data)}, method='{self._method}')"


# =============================================================================
# Convenience Function
# =============================================================================

def analyze(
    data: Union[np.ndarray, pl.Series, dict],
    dates: Optional[Union[np.ndarray, pl.Series]] = None,
    **kwargs
) -> AnalysisResult:
    """
    Quick fractal analysis of a time series.

    This is a convenience function that creates an Analyzer and returns
    the result. For more control, use Analyzer directly.

    Args:
        data: Price/value series
        dates: Optional date series
        **kwargs: Additional arguments passed to Analyzer

    Returns:
        AnalysisResult with hurst, fractal_dim, volatility, and regime

    Examples:
        >>> result = ft.analyze(prices)
        >>> print(result.hurst)
        >>> print(result.regime)
    """
    return Analyzer(data, dates, **kwargs).result
