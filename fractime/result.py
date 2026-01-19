"""
Result types for fractal analysis and forecasting.

This module provides the core data classes that hold analysis and forecast results.
All computations are lazy - values are computed on first access and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import numpy as np
import polars as pl


class Metric:
    """
    A metric that can be viewed as point estimate, rolling series, or distribution.

    All views are computed lazily on first access and cached for subsequent calls.

    Examples:
        >>> m = analyzer.hurst
        >>> m.value              # Point estimate (fast)
        0.67
        >>> m.rolling            # Rolling values (computed on demand)
        shape: (200, 2)
        >>> m.ci(0.95)           # Bootstrap CI (computed on demand)
        (0.61, 0.73)
    """

    def __init__(
        self,
        name: str,
        compute_point_fn: Callable[[], float],
        compute_rolling_fn: Optional[Callable[[], pl.DataFrame]] = None,
        compute_bootstrap_fn: Optional[Callable[[], np.ndarray]] = None,
    ):
        """
        Initialize a lazy metric.

        Args:
            name: Metric name (e.g., 'hurst', 'fractal_dim')
            compute_point_fn: Function to compute point estimate
            compute_rolling_fn: Function to compute rolling values
            compute_bootstrap_fn: Function to compute bootstrap distribution
        """
        self._name = name
        self._compute_point_fn = compute_point_fn
        self._compute_rolling_fn = compute_rolling_fn
        self._compute_bootstrap_fn = compute_bootstrap_fn

        # Lazy caches
        self._point: Optional[float] = None
        self._rolling: Optional[pl.DataFrame] = None
        self._distribution: Optional[np.ndarray] = None

    @property
    def value(self) -> float:
        """Point estimate of the metric."""
        if self._point is None:
            self._point = self._compute_point_fn()
        return self._point

    @property
    def rolling(self) -> pl.DataFrame:
        """
        Rolling values as Polars DataFrame.

        Returns DataFrame with columns:
            - 'index' or 'date': Position or timestamp
            - 'value': Metric value at that point
        """
        if self._rolling is None:
            if self._compute_rolling_fn is None:
                raise ValueError(
                    f"Rolling computation not available for {self._name}. "
                    "Ensure dates were provided to the Analyzer."
                )
            self._rolling = self._compute_rolling_fn()
        return self._rolling

    @property
    def rolling_values(self) -> np.ndarray:
        """Just the rolling values as numpy array."""
        return self.rolling['value'].to_numpy()

    @property
    def distribution(self) -> np.ndarray:
        """Bootstrap distribution samples."""
        if self._distribution is None:
            if self._compute_bootstrap_fn is None:
                raise ValueError(
                    f"Bootstrap computation not available for {self._name}."
                )
            self._distribution = self._compute_bootstrap_fn()
        return self._distribution

    @property
    def std(self) -> float:
        """Standard error from bootstrap distribution."""
        return float(np.std(self.distribution))

    @property
    def median(self) -> float:
        """Median from bootstrap distribution."""
        return float(np.median(self.distribution))

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """
        Confidence interval from bootstrap distribution.

        Args:
            level: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower, upper) bounds
        """
        alpha = (1 - level) / 2
        dist = self.distribution
        return (
            float(np.percentile(dist, alpha * 100)),
            float(np.percentile(dist, (1 - alpha) * 100))
        )

    def quantile(self, q: float) -> float:
        """
        Get quantile from bootstrap distribution.

        Args:
            q: Quantile (0-1)

        Returns:
            Value at the specified quantile
        """
        return float(np.percentile(self.distribution, q * 100))

    def __float__(self) -> float:
        """Allow using metric as a float."""
        return self.value

    def __repr__(self) -> str:
        return f"{self._name}={self.value:.4f}"

    def __str__(self) -> str:
        return f"{self.value:.4f}"


@dataclass
class AnalysisResult:
    """
    Complete fractal analysis result.

    Access individual metrics as properties. Each metric supports
    point estimates, rolling values, and bootstrap distributions.

    Examples:
        >>> result = analyzer.result
        >>> result.hurst.value          # Point estimate
        >>> result.hurst.rolling        # Time series
        >>> result.hurst.ci(0.95)       # Confidence interval
    """

    hurst: Metric
    fractal_dim: Metric
    volatility: Metric
    regime: str
    regime_probabilities: dict[str, float]

    def summary(self) -> str:
        """Text summary of analysis results."""
        lines = [
            "Fractal Analysis Summary",
            "=" * 40,
            f"Hurst Exponent:    {self.hurst}",
            f"Fractal Dimension: {self.fractal_dim}",
            f"Volatility:        {self.volatility}",
            f"Regime:            {self.regime}",
        ]

        # Add uncertainty if available
        try:
            lines[2] = f"Hurst Exponent:    {self.hurst} ± {self.hurst.std:.4f}"
        except ValueError:
            pass

        try:
            lines[3] = f"Fractal Dimension: {self.fractal_dim} ± {self.fractal_dim.std:.4f}"
        except ValueError:
            pass

        # Add regime probabilities
        lines.append("-" * 40)
        lines.append("Regime Probabilities:")
        for regime, prob in sorted(self.regime_probabilities.items(), key=lambda x: -x[1]):
            lines.append(f"  {regime}: {prob:.1%}")

        return "\n".join(lines)

    def to_frame(self) -> pl.DataFrame:
        """
        Export all rolling metrics as a single Polars DataFrame.

        Returns:
            DataFrame with columns for each metric's rolling values
        """
        try:
            hurst_df = self.hurst.rolling.rename({'value': 'hurst'})
            fractal_df = self.fractal_dim.rolling.rename({'value': 'fractal_dim'})
            vol_df = self.volatility.rolling.rename({'value': 'volatility'})

            # Determine join column
            join_col = 'date' if 'date' in hurst_df.columns else 'index'

            return (
                hurst_df
                .join(fractal_df, on=join_col)
                .join(vol_df, on=join_col)
            )
        except ValueError as e:
            raise ValueError(
                "Rolling values not available. Ensure dates were provided to Analyzer."
            ) from e

    def __repr__(self) -> str:
        return (
            f"AnalysisResult("
            f"hurst={self.hurst}, "
            f"fractal_dim={self.fractal_dim}, "
            f"regime='{self.regime}')"
        )


@dataclass
class ForecastResult:
    """
    Forecast result with uncertainty quantification.

    Contains the primary forecast, confidence intervals, and all
    Monte Carlo paths used to generate the forecast.

    Examples:
        >>> result = model.predict(steps=30)
        >>> result.forecast              # Primary forecast (median)
        >>> result.ci(0.90)              # 90% confidence interval
        >>> result.paths                 # All simulated paths
    """

    # Primary forecasts
    _paths: np.ndarray                    # Shape: (n_paths, n_steps)
    _probabilities: np.ndarray            # Shape: (n_paths,)

    # Optional metadata
    dates: Optional[np.ndarray] = None    # Forecast dates
    metadata: dict = field(default_factory=dict)

    # Cached computations
    _forecast: Optional[np.ndarray] = field(default=None, repr=False)
    _mean: Optional[np.ndarray] = field(default=None, repr=False)
    _percentiles: dict = field(default_factory=dict, repr=False)

    @property
    def paths(self) -> np.ndarray:
        """All simulated paths. Shape: (n_paths, n_steps)"""
        return self._paths

    @property
    def probabilities(self) -> np.ndarray:
        """Probability weight for each path."""
        return self._probabilities

    @property
    def n_paths(self) -> int:
        """Number of simulated paths."""
        return self._paths.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of forecast steps."""
        return self._paths.shape[1]

    @property
    def forecast(self) -> np.ndarray:
        """Primary forecast (probability-weighted median)."""
        if self._forecast is None:
            self._forecast = self._weighted_quantile(0.5)
        return self._forecast

    @property
    def mean(self) -> np.ndarray:
        """Mean forecast across all paths."""
        if self._mean is None:
            self._mean = np.average(self._paths, axis=0, weights=self._probabilities)
        return self._mean

    @property
    def lower(self) -> np.ndarray:
        """Lower bound (5th percentile)."""
        return self.quantile(0.05)

    @property
    def upper(self) -> np.ndarray:
        """Upper bound (95th percentile)."""
        return self.quantile(0.95)

    @property
    def std(self) -> np.ndarray:
        """Standard deviation at each step."""
        return np.sqrt(
            np.average((self._paths - self.mean) ** 2, axis=0, weights=self._probabilities)
        )

    def quantile(self, q: float) -> np.ndarray:
        """
        Get quantile forecast.

        Args:
            q: Quantile (0-1)

        Returns:
            Forecast at the specified quantile
        """
        if q not in self._percentiles:
            self._percentiles[q] = self._weighted_quantile(q)
        return self._percentiles[q]

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """
        Confidence interval.

        Args:
            level: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower, upper) arrays
        """
        alpha = (1 - level) / 2
        return (self.quantile(alpha), self.quantile(1 - alpha))

    def _weighted_quantile(self, q: float) -> np.ndarray:
        """Compute weighted quantile across paths."""
        result = np.zeros(self.n_steps)
        for t in range(self.n_steps):
            sorted_idx = np.argsort(self._paths[:, t])
            sorted_values = self._paths[sorted_idx, t]
            sorted_weights = self._probabilities[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            idx = np.searchsorted(cumsum, q * cumsum[-1])
            idx = min(idx, len(sorted_values) - 1)
            result[t] = sorted_values[idx]
        return result

    def to_frame(self) -> pl.DataFrame:
        """
        Export forecast summary as Polars DataFrame.

        Returns:
            DataFrame with forecast, lower, upper, and optional date columns
        """
        data = {
            'step': list(range(1, self.n_steps + 1)),
            'forecast': self.forecast.tolist(),
            'lower': self.lower.tolist(),
            'upper': self.upper.tolist(),
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        }

        if self.dates is not None:
            data['date'] = self.dates.tolist()

        return pl.DataFrame(data)

    def __repr__(self) -> str:
        return (
            f"ForecastResult("
            f"steps={self.n_steps}, "
            f"paths={self.n_paths}, "
            f"final={self.forecast[-1]:.2f})"
        )
