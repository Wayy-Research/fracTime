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
        scoring: str = 'v1',
    ):
        self._data = _ensure_numpy(data)
        self._dates = _ensure_dates(dates)
        self._lookback = lookback
        self._method = method
        self._time_warp = time_warp
        self._scoring = scoring

        # Path probability weights — defaults differ by scoring version
        if path_weights is not None:
            self._path_weights = path_weights
        elif scoring in ('v2', 'v3'):
            self._path_weights = {
                'direction': 0.4,
                'hurst': 0.2,
                'volatility': 0.2,
                'pattern': 0.2,
            }
        else:
            self._path_weights = {
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
        if self._scoring == 'v2':
            probabilities = self._calculate_path_probabilities_v2(
                paths, hurst, volatility, steps
            )
        elif self._scoring == 'v3':
            probabilities = self._calculate_path_probabilities_v2(
                paths, hurst, volatility, steps
            )
        else:
            probabilities = self._calculate_path_probabilities(
                paths, hurst, volatility, steps
            )

        # For v3: compute H-conditioned point forecast directly
        point_override = None
        if self._scoring == 'v3':
            point_override = self._compute_h_conditioned_forecast(
                steps, hurst, volatility
            )

        # Prepare forecast dates if available
        forecast_dates = None
        if self._dates is not None:
            forecast_dates = self._generate_forecast_dates(steps)

        return ForecastResult(
            _paths=paths,
            _probabilities=probabilities,
            _point_override=point_override,
            dates=forecast_dates,
            metadata={
                'hurst': hurst,
                'fractal_dim': self.fractal_dim.value,
                'volatility': volatility,
                'regime': self.regime,
                'n_patterns': len(matches),
                'scoring': self._scoring,
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

    def _calculate_path_probabilities_v2(
        self,
        paths: np.ndarray,
        target_hurst: float,
        target_vol: float,
        steps: int,
    ) -> np.ndarray:
        """Calculate path probabilities using directional tilt + fractal matching.

        V2 scoring uses the Hurst exponent to act on its measurement:
        - H > 0.5 + upward momentum → weight upward paths (persistence)
        - H < 0.5 + upward momentum → weight downward paths (reversion)
        - H ≈ 0.5 → stay neutral (random walk, no tilt)

        Fractal pattern matching is retained: paths whose fractal structure
        matches the observed Hurst get weighted up.
        """
        n_paths = paths.shape[0]
        scores = np.zeros(n_paths)

        # --- Directional signal from H + momentum ---
        lookback = min(21, len(self._returns))
        recent_returns = self._returns[-lookback:]
        momentum = float(np.mean(recent_returns))
        vol = float(np.std(recent_returns))
        # Normalize momentum by volatility → z-score-like quantity
        momentum_z = momentum / vol if vol > 0 else 0.0

        # persistence ∈ [-1, 1]: positive = trending, negative = reverting
        persistence = 2.0 * target_hurst - 1.0
        # directional_signal: sign = predicted direction, magnitude = confidence
        directional_signal = persistence * momentum_z
        signal_strength = float(np.tanh(abs(directional_signal)))

        # --- Weights ---
        w_dir = self._path_weights.get('direction', 0.4)
        w_hurst = self._path_weights.get('hurst', 0.2)
        w_vol = self._path_weights.get('volatility', 0.2)
        w_pattern = self._path_weights.get('pattern', 0.2)

        last_price = self._data[-1]

        for i in range(n_paths):
            path = paths[i]
            path_with_start = np.concatenate([[last_price], path])
            path_returns = np.diff(np.log(path_with_start))

            if len(path_returns) < 2:
                scores[i] = 1.0
                continue

            # 1. Directional tilt — the key new component
            path_return = (path[-1] - last_price) / last_price
            if abs(directional_signal) > 0.01:
                aligned = np.sign(path_return) == np.sign(directional_signal)
                dir_score = (1.0 + signal_strength) if aligned else (1.0 - signal_strength * 0.5)
            else:
                dir_score = 1.0

            # 2. Fractal pattern match — keep paths that look like observed H
            try:
                path_hurst = compute_hurst_rs(path, 5, min(20, max(steps // 2, 6)))
                hurst_score = 1.0 / (1.0 + abs(path_hurst - target_hurst) * 5)
            except Exception:
                hurst_score = 0.5

            # 3. Volatility match
            path_vol = np.std(path_returns) * np.sqrt(252)
            vol_ratio = path_vol / target_vol if target_vol > 0 else 1.0
            vol_score = 1.0 / (1.0 + abs(np.log(vol_ratio + 1e-10)) * 2)

            # 4. Pattern continuity (smoothness)
            if len(path_returns) > 1:
                second_diff = np.diff(path_returns)
                pattern_score = 1.0 / (1.0 + np.std(second_diff) * 10)
            else:
                pattern_score = 0.5

            scores[i] = (
                w_dir * dir_score
                + w_hurst * hurst_score
                + w_vol * vol_score
                + w_pattern * pattern_score
            )

        scores = np.clip(scores, 1e-10, None)
        return scores / np.sum(scores)

    def _compute_h_conditioned_forecast(
        self,
        steps: int,
        hurst: float,
        volatility: float,
    ) -> np.ndarray:
        """Compute point forecast using Wiener-Hopf optimal linear prediction.

        Uses the fractional Gaussian noise (fGn) autocovariance structure
        to find the optimal linear combination of past returns for predicting
        each future return. This is the best linear unbiased predictor (BLUP)
        for fGn.

        For a lookback window of p past returns r_{t}, ..., r_{t-p+1}:
            r̂_{t+h} = a_1*r_t + a_2*r_{t-1} + ... + a_p*r_{t-p+1}
        where a = Gamma^{-1} * gamma_h, with:
            Gamma[i,j] = rho(|i-j|)      (autocovariance matrix)
            gamma_h[j]  = rho(h+j)        (cross-covariance at horizon h)

        Lookback is set adaptively by finding where the autocovariance
        drops below a threshold, ensuring anti-persistent series use few
        lags (signal at lag 1) and persistent series use many.
        """
        last_price = self._data[-1]
        h2 = 2.0 * hurst

        # --- fGn autocovariance function ---
        def fgn_acov(k: int) -> float:
            """Exact fGn autocovariance at lag k (unit variance)."""
            if k == 0:
                return 1.0
            return 0.5 * (abs(k + 1) ** h2 - 2.0 * abs(k) ** h2 + abs(k - 1) ** h2)

        # --- Adaptive lookback from autocovariance decay ---
        # Include lags where |rho(k)| > threshold
        acov_threshold = 0.01
        max_lookback = min(63, len(self._returns))
        lookback = 1
        for k in range(1, max_lookback + 1):
            if abs(fgn_acov(k)) >= acov_threshold:
                lookback = k
            else:
                break
        # Minimum 3 lags for stability, cap at available data
        lookback = max(3, min(lookback + 1, len(self._returns)))

        # Recent returns in reverse chronological order: r_t, r_{t-1}, ...
        recent = self._returns[-lookback:][::-1]
        p = len(recent)

        # --- Build autocovariance matrix Gamma[i,j] = rho(|i-j|) ---
        gamma_matrix = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                gamma_matrix[i, j] = fgn_acov(abs(i - j))

        # Regularize for numerical stability
        gamma_matrix += np.eye(p) * 1e-8

        # --- Predict each future step ---
        forecast = np.zeros(steps)
        cum_return = 0.0

        for t in range(1, steps + 1):
            # Cross-covariance vector: gamma_h[j] = rho(t + j) for j=0..p-1
            # j=0 corresponds to most recent return r_t
            gamma_vec = np.array([fgn_acov(t + j) for j in range(p)])

            # Optimal weights: a = Gamma^{-1} * gamma_h
            try:
                weights = np.linalg.solve(gamma_matrix, gamma_vec)
            except np.linalg.LinAlgError:
                weights = np.zeros(p)

            # Predicted return = weighted sum of recent returns
            predicted_return = float(np.dot(weights, recent))
            cum_return += predicted_return
            forecast[t - 1] = last_price * np.exp(cum_return)

        return forecast

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
