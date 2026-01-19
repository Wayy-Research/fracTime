"""
Fractal-based time series forecasting.

The Forecaster class generates probabilistic forecasts using fractal patterns
and Monte Carlo simulation.

Examples:
    >>> import fractime as ft
    >>> model = ft.Forecaster(prices)
    >>> result = model.predict(steps=30)
    >>> result.forecast            # Primary forecast
    >>> result.ci(0.95)            # 95% confidence interval
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import polars as pl

from .result import ForecastResult
from .analyzer import Analyzer, _ensure_numpy, _ensure_dates
from ._numba import (
    compute_hurst_rs,
    compute_rolling_volatility,
    find_similar_patterns,
    generate_paths_from_pattern,
    generate_fbm,
)


class Forecaster:
    """
    Fractal-based probabilistic forecaster.

    Generates forecasts by:
    1. Analyzing fractal properties of historical data
    2. Finding similar historical patterns
    3. Simulating future paths based on pattern outcomes
    4. Weighting paths by pattern similarity and fractal consistency

    Args:
        data: Historical price series
        dates: Optional date series
        exogenous: Optional dict of exogenous variables {'name': series}
        analyzer: Pre-computed Analyzer (optional, for reuse)
        lookback: Historical window for pattern matching
        method: Hurst calculation method ('rs' or 'dfa')
        time_warp: Whether to use Mandelbrot's trading time concept
        path_weights: Custom weights for path probability calculation

    Examples:
        Basic usage:
            >>> model = Forecaster(prices)
            >>> result = model.predict(steps=30)
            >>> result.forecast

        With exogenous variables:
            >>> model = Forecaster(prices, exogenous={'VIX': vix})
            >>> result = model.predict(steps=30)

        With custom analyzer:
            >>> analyzer = Analyzer(prices, method='dfa')
            >>> model = Forecaster(prices, analyzer=analyzer)
            >>> result = model.predict(steps=30)

        Access internals:
            >>> model.hurst              # Shortcut to analyzer.hurst
            >>> model.analyzer           # The analyzer used
    """

    def __init__(
        self,
        data: Union[np.ndarray, pl.Series],
        dates: Optional[Union[np.ndarray, pl.Series]] = None,
        exogenous: Optional[dict] = None,
        analyzer: Optional[Analyzer] = None,
        lookback: int = 252,
        method: str = 'rs',
        time_warp: bool = False,
        path_weights: Optional[dict] = None,
    ):
        self._data = _ensure_numpy(data)
        self._dates = _ensure_dates(dates)
        self._lookback = lookback
        self._method = method
        self._time_warp = time_warp

        # Path probability weights
        self._path_weights = path_weights or {
            'hurst': 0.3,
            'volatility': 0.3,
            'pattern': 0.4,
        }

        # Exogenous variables
        self._exogenous = None
        if exogenous is not None:
            self._exogenous = {k: _ensure_numpy(v) for k, v in exogenous.items()}

        # Create or use provided analyzer
        if analyzer is not None:
            self._analyzer = analyzer
        else:
            self._analyzer = Analyzer(
                self._data,
                dates=self._dates,
                method=method,
            )

        # Precompute returns
        self._returns = np.diff(np.log(self._data))

        # Lazy caches
        self._volatility_history: Optional[np.ndarray] = None
        self._pattern_matches: Optional[np.ndarray] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def analyzer(self) -> Analyzer:
        """The analyzer used for fractal analysis."""
        return self._analyzer

    @property
    def hurst(self):
        """Shortcut to analyzer.hurst."""
        return self._analyzer.hurst

    @property
    def fractal_dim(self):
        """Shortcut to analyzer.fractal_dim."""
        return self._analyzer.fractal_dim

    @property
    def regime(self) -> str:
        """Shortcut to analyzer.regime."""
        return self._analyzer.regime

    # =========================================================================
    # Forecasting
    # =========================================================================

    def predict(
        self,
        steps: int = 30,
        n_paths: int = 1000,
        confidence: float = 0.95,
    ) -> ForecastResult:
        """
        Generate probabilistic forecast.

        Args:
            steps: Number of steps to forecast
            n_paths: Number of Monte Carlo paths to simulate
            confidence: Confidence level for intervals (used in result)

        Returns:
            ForecastResult with forecast, paths, and uncertainty measures

        Examples:
            >>> result = model.predict(steps=30)
            >>> result.forecast              # Median forecast
            >>> result.mean                  # Mean forecast
            >>> result.ci(0.90)              # 90% CI
            >>> result.paths                 # All simulated paths
        """
        # Get current fractal properties
        hurst = self.hurst.value
        volatility = self._compute_current_volatility()

        # Find similar historical patterns
        pattern_len = min(21, len(self._returns) // 10)
        current_pattern = self._returns[-pattern_len:]

        matches = find_similar_patterns(
            self._returns[:-pattern_len],
            current_pattern,
            min_similarity=0.3
        )

        # Generate paths
        if len(matches) > 0:
            paths = self._generate_pattern_paths(
                matches, pattern_len, steps, n_paths, hurst, volatility
            )
        else:
            paths = self._generate_fbm_paths(steps, n_paths, hurst, volatility)

        # Calculate path probabilities
        probabilities = self._calculate_path_probabilities(
            paths, hurst, volatility, steps
        )

        # Prepare forecast dates if available
        forecast_dates = None
        if self._dates is not None:
            forecast_dates = self._generate_forecast_dates(steps)

        return ForecastResult(
            _paths=paths,
            _probabilities=probabilities,
            dates=forecast_dates,
            metadata={
                'hurst': hurst,
                'fractal_dim': self.fractal_dim.value,
                'volatility': volatility,
                'regime': self.regime,
                'n_patterns': len(matches),
            }
        )

    def _compute_current_volatility(self) -> float:
        """Compute current volatility estimate."""
        window = min(21, len(self._returns))
        recent_returns = self._returns[-window:]
        return float(np.std(recent_returns) * np.sqrt(252))

    def _generate_pattern_paths(
        self,
        matches: np.ndarray,
        pattern_len: int,
        steps: int,
        n_paths: int,
        hurst: float,
        volatility: float,
    ) -> np.ndarray:
        """Generate paths based on historical pattern matches."""
        # Normalize match similarities to probabilities
        similarities = matches[:, 1]
        weights = similarities / np.sum(similarities)

        # Generate paths
        last_price = self._data[-1]
        daily_vol = volatility / np.sqrt(252)

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

    def _generate_fbm_paths(
        self,
        steps: int,
        n_paths: int,
        hurst: float,
        volatility: float,
    ) -> np.ndarray:
        """Generate paths using fractional Brownian motion."""
        last_price = self._data[-1]
        daily_vol = volatility / np.sqrt(252)

        paths = np.zeros((n_paths, steps))

        for i in range(n_paths):
            # Generate fBm increments
            fbm = generate_fbm(steps, hurst)
            increments = np.diff(np.concatenate([[0], fbm]))

            # Scale by volatility
            returns = increments * daily_vol

            # Convert to prices
            price = last_price
            for t in range(steps):
                price = price * np.exp(returns[t])
                paths[i, t] = price

        return paths

    def _calculate_path_probabilities(
        self,
        paths: np.ndarray,
        target_hurst: float,
        target_vol: float,
        steps: int,
    ) -> np.ndarray:
        """Calculate probability weights for each path."""
        n_paths = paths.shape[0]
        scores = np.zeros(n_paths)

        # Get weight configuration
        w_hurst = self._path_weights.get('hurst', 0.3)
        w_vol = self._path_weights.get('volatility', 0.3)
        w_pattern = self._path_weights.get('pattern', 0.4)

        for i in range(n_paths):
            path = paths[i]

            # Convert to returns
            path_with_start = np.concatenate([[self._data[-1]], path])
            path_returns = np.diff(np.log(path_with_start))

            if len(path_returns) < 10:
                scores[i] = 1.0
                continue

            # Score based on Hurst similarity
            try:
                path_hurst = compute_hurst_rs(path, 5, min(20, steps // 2))
                hurst_score = 1.0 / (1.0 + abs(path_hurst - target_hurst) * 5)
            except:
                hurst_score = 0.5

            # Score based on volatility similarity
            path_vol = np.std(path_returns) * np.sqrt(252)
            vol_ratio = path_vol / target_vol if target_vol > 0 else 1.0
            vol_score = 1.0 / (1.0 + abs(np.log(vol_ratio + 1e-10)) * 2)

            # Pattern continuity score (smoothness)
            if len(path_returns) > 1:
                second_diff = np.diff(path_returns)
                pattern_score = 1.0 / (1.0 + np.std(second_diff) * 10)
            else:
                pattern_score = 0.5

            # Combine scores
            scores[i] = (
                w_hurst * hurst_score +
                w_vol * vol_score +
                w_pattern * pattern_score
            )

        # Normalize to probabilities
        scores = np.clip(scores, 1e-10, None)
        probabilities = scores / np.sum(scores)

        return probabilities

    def _generate_forecast_dates(self, steps: int) -> np.ndarray:
        """Generate forecast dates based on historical frequency."""
        if self._dates is None or len(self._dates) < 2:
            return None

        # Infer frequency from last dates
        last_dates = self._dates[-10:]
        if len(last_dates) < 2:
            return None

        # Calculate median time delta
        try:
            # Convert to datetime64 if needed
            if not np.issubdtype(last_dates.dtype, np.datetime64):
                last_dates = last_dates.astype('datetime64[ns]')

            deltas = np.diff(last_dates)
            median_delta = np.median(deltas)

            # Generate future dates
            last_date = self._dates[-1]
            forecast_dates = np.array([
                last_date + (i + 1) * median_delta
                for i in range(steps)
            ])
            return forecast_dates
        except:
            return None

    # =========================================================================
    # Exogenous Support
    # =========================================================================

    def _apply_exogenous_adjustment(
        self,
        paths: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Adjust path probabilities based on exogenous variables."""
        if self._exogenous is None:
            return probabilities

        # Simple correlation-based adjustment
        # TODO: More sophisticated exogenous handling
        adjusted = probabilities.copy()

        for name, exog in self._exogenous.items():
            if len(exog) < len(self._data):
                continue

            # Compute correlation with target
            aligned_exog = exog[-len(self._returns):]
            if len(aligned_exog) != len(self._returns):
                continue

            corr = np.corrcoef(self._returns, aligned_exog)[0, 1]
            if np.isnan(corr):
                continue

            # Adjust based on recent exogenous trend
            exog_trend = np.mean(np.diff(aligned_exog[-21:])) if len(aligned_exog) > 21 else 0

            # Favor paths aligned with exogenous signal
            for i in range(len(paths)):
                path_trend = (paths[i, -1] - paths[i, 0]) / paths[i, 0]
                alignment = np.sign(path_trend) == np.sign(exog_trend * corr)
                if alignment:
                    adjusted[i] *= 1.2
                else:
                    adjusted[i] *= 0.8

        # Renormalize
        adjusted = adjusted / np.sum(adjusted)
        return adjusted

    def __repr__(self) -> str:
        return (
            f"Forecaster("
            f"n={len(self._data)}, "
            f"hurst={self.hurst.value:.3f}, "
            f"regime='{self.regime}')"
        )


# =============================================================================
# Convenience Function
# =============================================================================

def forecast(
    data: Union[np.ndarray, pl.Series],
    steps: int = 30,
    n_paths: int = 1000,
    **kwargs
) -> ForecastResult:
    """
    Quick forecast of a time series.

    This is a convenience function that creates a Forecaster and returns
    the prediction. For more control, use Forecaster directly.

    Args:
        data: Historical price series
        steps: Number of steps to forecast
        n_paths: Number of Monte Carlo paths
        **kwargs: Additional arguments passed to Forecaster

    Returns:
        ForecastResult with forecast and uncertainty measures

    Examples:
        >>> result = ft.forecast(prices, steps=30)
        >>> print(result.forecast[-1])  # Final forecasted price
        >>> print(result.ci(0.95))      # 95% confidence interval
    """
    model = Forecaster(data, **kwargs)
    return model.predict(steps=steps, n_paths=n_paths)
