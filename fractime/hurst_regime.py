"""
Hurst Regime Persistence Analysis.

Fits an HMM to the rolling Hurst exponent series H(t) to detect regime
transitions, estimate regime duration, and recommend forecast horizons.

Unlike :class:`RegimeDetector` (which fits HMM to raw returns and identifies
bull/bear), this module models the *memory structure* of the process itself.
A practitioner can use it to answer: "How long will the current Hurst regime
last?" and set forecast horizons accordingly.

Examples:
    >>> import fractime as ft
    >>> import numpy as np
    >>> prices = np.cumprod(1 + np.random.normal(0.0003, 0.01, 600)) * 100
    >>> hra = ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
    >>> hra.current_regime
    'trending'
    >>> hra.recommended_horizon
    30
    >>> print(hra.summary())
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._numba import compute_hurst_rs, compute_hurst_dfa

try:
    from hmmlearn.hmm import GaussianHMM

    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    GaussianHMM = None


# ---------------------------------------------------------------------------
# Helpers (re-used from analyzer.py — keep local to avoid circular imports)
# ---------------------------------------------------------------------------

def _ensure_numpy(data: object) -> np.ndarray:
    """Convert input to numpy array and validate."""
    if data is None:
        raise ValueError("Data cannot be None")
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float64, copy=False)
    elif isinstance(data, pl.Series):
        arr = data.to_numpy().astype(np.float64)
    elif isinstance(data, pl.DataFrame):
        raise ValueError("Expected Series, got DataFrame. Select a column first.")
    else:
        arr = np.asarray(data, dtype=np.float64)

    if np.any(~np.isfinite(arr)):
        raise ValueError("Input data contains NaN or Inf values.")
    return arr


def _ensure_dates(dates: object) -> np.ndarray | None:
    """Convert dates to numpy array if provided."""
    if dates is None:
        return None
    if isinstance(dates, np.ndarray):
        return dates
    if isinstance(dates, pl.Series):
        return dates.to_numpy()
    return np.asarray(dates)


# ---------------------------------------------------------------------------
# HurstRegimeAnalyzer
# ---------------------------------------------------------------------------

_VALID_METHODS = ("rs", "dfa", "wavelet")


class HurstRegimeAnalyzer:
    """Hurst-regime persistence analyzer.

    Computes a rolling Hurst exponent series ``H(t)`` from *prices*, then
    fits a Gaussian HMM to that series.  HMM states are mapped to regime
    labels (``trending``, ``random``, ``mean_reverting``) by their emission
    mean and the configured thresholds.  Properties expose the current
    regime, its age, stability, expected remaining duration, and a
    recommended forecast horizon.

    This is an analysis class: it eagerly computes on construction and
    exposes results via properties.  For train/test workflows, construct
    separate instances on each split.

    Args:
        prices: Price series (numpy array, Polars Series, or list-like).
        dates: Optional date array aligned with *prices*.
        method: Hurst estimation method (``'rs'``, ``'dfa'``, ``'wavelet'``).
        window: Rolling window for Hurst estimation (default 252).
        n_regimes: Number of HMM states (2 or 3, default 3).
        random_state: Seed for reproducibility.
        trending_threshold: H above this is labelled ``trending``.
        mean_reverting_threshold: H below this is labelled ``mean_reverting``.
    """

    def __init__(
        self,
        prices: np.ndarray | pl.Series | list,
        dates: np.ndarray | pl.Series | list | None = None,
        method: str = "dfa",
        window: int = 252,
        n_regimes: int = 3,
        random_state: int = 42,
        trending_threshold: float = 0.55,
        mean_reverting_threshold: float = 0.45,
    ) -> None:
        if not HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HurstRegimeAnalyzer. "
                "Install with: uv pip install hmmlearn"
            )

        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got '{method}'"
            )

        if n_regimes not in (2, 3):
            raise ValueError(f"n_regimes must be 2 or 3, got {n_regimes}")

        self._prices = _ensure_numpy(prices)
        self._dates = _ensure_dates(dates)
        self._method = method
        self._window = window
        self._n_regimes = n_regimes
        self._random_state = random_state
        self._trending_threshold = trending_threshold
        self._mean_reverting_threshold = mean_reverting_threshold

        min_obs = window + 50
        if len(self._prices) < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations "
                f"(window={window} + 50), got {len(self._prices)}"
            )

        # Eagerly compute
        self._hurst_values: np.ndarray = np.array([])
        self._hurst_indices: np.ndarray = np.array([], dtype=int)
        self._model: GaussianHMM | None = None  # type: ignore[assignment]
        self._state_sequence: np.ndarray = np.array([], dtype=int)
        self._posteriors: np.ndarray = np.array([])
        self._state_to_regime: dict[int, str] = {}
        self._regime_labels: np.ndarray = np.array([])

        self._fit()

    # ------------------------------------------------------------------
    # Internal: rolling Hurst
    # ------------------------------------------------------------------

    def _compute_rolling_hurst(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute rolling Hurst exponent over *prices*.

        Returns:
            ``(hurst_values, indices)`` — arrays of Hurst estimates and the
            corresponding indices into the original price array.
        """
        n = len(self._prices)
        values: list[float] = []
        indices: list[int] = []
        max_scale = min(100, self._window // 2)

        for i in range(self._window, n):
            segment = self._prices[i - self._window : i]
            h = self._compute_single_hurst(segment, max_scale)
            values.append(h)
            indices.append(i)

        return np.array(values, dtype=np.float64), np.array(indices, dtype=int)

    def _compute_single_hurst(self, segment: np.ndarray, max_scale: int) -> float:
        """Compute a single Hurst estimate on *segment*."""
        min_scale = 10
        if self._method == "wavelet":
            from .advanced import compute_wavelet_hurst

            return compute_wavelet_hurst(segment)
        elif self._method == "dfa":
            return float(compute_hurst_dfa(segment, min_scale, max_scale))
        else:
            return float(compute_hurst_rs(segment, min_scale, max_scale))

    # ------------------------------------------------------------------
    # Internal: HMM fitting
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if model has not been fitted."""
        if self._model is None:
            raise RuntimeError("HurstRegimeAnalyzer model not fitted.")

    def _fit(self) -> None:
        """Compute rolling H(t), filter invalid values, and fit HMM."""
        self._hurst_values, self._hurst_indices = self._compute_rolling_hurst()

        # Filter NaN/Inf from rolling Hurst series
        valid_mask = np.isfinite(self._hurst_values)
        if valid_mask.sum() < 50:
            raise ValueError(
                f"Only {valid_mask.sum()} valid Hurst values computed "
                f"(need at least 50). Check input data for constant segments."
            )
        if not valid_mask.all():
            self._hurst_values = self._hurst_values[valid_mask]
            self._hurst_indices = self._hurst_indices[valid_mask]

        X = self._hurst_values.reshape(-1, 1)
        self._model = GaussianHMM(
            n_components=self._n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self._random_state,
        )
        self._model.fit(X)

        self._state_sequence = self._model.predict(X)
        self._posteriors = self._model.predict_proba(X)

        # Map states to regime names using thresholds on emission means
        self._state_to_regime = self._map_states_to_regimes()

        self._regime_labels = np.array(
            [self._state_to_regime[s] for s in self._state_sequence]
        )

    def _map_states_to_regimes(self) -> dict[int, str]:
        """Map HMM states to regime labels using emission means and thresholds.

        For 3-regime mode: tries threshold-based assignment first.  If that
        produces duplicate labels (e.g. all means above the trending threshold),
        falls back to ordinal ranking by emission mean.

        For 2-regime mode: always uses ordinal ranking (higher mean = trending,
        lower = mean_reverting) since there is no "random" category.
        """
        self._check_fitted()
        means = self._model.means_.flatten()
        sorted_states = np.argsort(means)

        # 2-regime mode: always ordinal (no "random" category)
        if self._n_regimes == 2:
            return {
                int(sorted_states[1]): "trending",
                int(sorted_states[0]): "mean_reverting",
            }

        # 3-regime mode: attempt threshold-based mapping
        mapping: dict[int, str] = {}
        for state in range(self._n_regimes):
            m = means[state]
            if m > self._trending_threshold:
                mapping[state] = "trending"
            elif m < self._mean_reverting_threshold:
                mapping[state] = "mean_reverting"
            else:
                mapping[state] = "random"

        # If labels are all unique, use threshold-based mapping
        labels = list(mapping.values())
        if len(set(labels)) == self._n_regimes:
            return mapping

        # Ordinal fallback: sort by emission mean
        return {
            int(sorted_states[2]): "trending",
            int(sorted_states[1]): "random",
            int(sorted_states[0]): "mean_reverting",
        }

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> str:
        """Current Hurst regime: ``'trending'``, ``'mean_reverting'``, or ``'random'``."""
        return str(self._regime_labels[-1])

    @property
    def current_hurst(self) -> float:
        """Latest rolling Hurst estimate."""
        return float(self._hurst_values[-1])

    @property
    def regime_age(self) -> int:
        """Number of observations in the current regime."""
        current = self._state_sequence[-1]
        age = 0
        for s in reversed(self._state_sequence):
            if s == current:
                age += 1
            else:
                break
        return max(age, 1)

    @property
    def regime_stability(self) -> float:
        """Confidence in the current regime (0-1).

        Weighted combination of:
        - HMM posterior probability of current state (weight 0.4)
        - Inverse Hurst volatility over recent quarter-window (weight 0.3)
        - Distance from nearest threshold boundary (weight 0.3)
        """
        return float(np.clip(self._compute_stability(), 0.0, 1.0))

    @property
    def expected_duration(self) -> float:
        """Expected total duration of the current regime (from HMM + empirical)."""
        return float(max(self._compute_expected_duration(), 1.0))

    @property
    def expected_remaining(self) -> float:
        """Expected remaining days in the current regime."""
        return float(max(self._compute_expected_remaining(), 1.0))

    @property
    def recommended_horizon(self) -> int:
        """Suggested forecast horizon (days)."""
        return self._compute_recommended_horizon()

    # ------------------------------------------------------------------
    # History / diagnostics
    # ------------------------------------------------------------------

    @property
    def regime_history(self) -> pl.DataFrame:
        """Full history DataFrame with columns:
        ``date`` (or ``index``), ``hurst``, ``regime``, ``regime_age``, ``state_probability``.
        """
        # Compute per-row regime age
        ages = self._compute_regime_ages()

        # State probability = posterior of the assigned state
        state_probs = np.array([
            self._posteriors[i, self._state_sequence[i]]
            for i in range(len(self._state_sequence))
        ])

        data: dict[str, object] = {}
        if self._dates is not None:
            data["date"] = self._dates[self._hurst_indices]
        else:
            data["index"] = self._hurst_indices

        data["hurst"] = self._hurst_values
        data["regime"] = self._regime_labels.tolist()
        data["regime_age"] = ages
        data["state_probability"] = state_probs

        return pl.DataFrame(data)

    @property
    def change_points(self) -> list[int]:
        """Indices (into the Hurst series) where the regime changed."""
        return self._detect_change_points()

    @property
    def transition_matrix(self) -> np.ndarray:
        """HMM transition matrix ``P[i,j]`` between Hurst regimes."""
        self._check_fitted()
        return self._model.transmat_

    # ------------------------------------------------------------------
    # Internal: stability, durations, horizon
    # ------------------------------------------------------------------

    def _compute_stability(self) -> float:
        """Weighted stability score."""
        # 1. HMM posterior of current state
        current_state = self._state_sequence[-1]
        posterior_prob = float(self._posteriors[-1, current_state])

        # 2. Inverse Hurst volatility (recent quarter-window)
        #    Exponential decay — half-life at h_std = 0.05
        quarter = max(self._window // 4, 5)
        recent_h = self._hurst_values[-quarter:]
        h_std = float(np.std(recent_h)) if len(recent_h) > 1 else 0.0
        inv_vol = float(np.exp(-h_std / 0.05))

        # 3. Distance from nearest threshold boundary
        h = self.current_hurst
        dist_trend = abs(h - self._trending_threshold)
        dist_mr = abs(h - self._mean_reverting_threshold)
        min_dist = min(dist_trend, dist_mr)
        boundary_score = min(min_dist / 0.05, 1.0)

        return 0.4 * posterior_prob + 0.3 * inv_vol + 0.3 * boundary_score

    def _compute_expected_duration(self) -> float:
        """HMM-based expected total duration of the current regime."""
        self._check_fitted()
        current_state = int(self._state_sequence[-1])
        p_self = float(self._model.transmat_[current_state, current_state])
        hmm_duration = 1.0 / (1.0 - p_self + 1e-10)

        # Blend with empirical if enough observations
        emp = self._empirical_durations()
        regime_name = self._state_to_regime[current_state]
        if regime_name in emp and len(emp[regime_name]) >= 5:
            emp_median = float(np.median(emp[regime_name]))
            return 0.3 * hmm_duration + 0.7 * emp_median

        return hmm_duration

    def _compute_expected_remaining(self) -> float:
        """Blended remaining duration estimate."""
        self._check_fitted()
        current_state = int(self._state_sequence[-1])
        p_self = float(self._model.transmat_[current_state, current_state])
        age = self.regime_age

        # HMM geometric (memoryless)
        hmm_remaining = 1.0 / (1.0 - p_self + 1e-10)

        # Empirical: median_duration - current_age
        emp = self._empirical_durations()
        regime_name = self._state_to_regime[current_state]
        if regime_name in emp and len(emp[regime_name]) >= 5:
            emp_median = float(np.median(emp[regime_name]))
            emp_remaining = max(emp_median - age, 1.0)
            return 0.3 * hmm_remaining + 0.7 * emp_remaining

        return hmm_remaining

    def _compute_recommended_horizon(self) -> int:
        """Forecast horizon recommendation."""
        remaining = self.expected_remaining
        stability = self.regime_stability
        max_horizon = self._window // 4

        horizon = min(max_horizon, remaining / 2.0) * stability
        return int(np.clip(horizon, 5, max_horizon))

    # ------------------------------------------------------------------
    # Internal: change points, runs, empirical durations
    # ------------------------------------------------------------------

    def _detect_change_points(self) -> list[int]:
        """Indices where HMM Viterbi state sequence changes."""
        cps: list[int] = []
        for i in range(1, len(self._state_sequence)):
            if self._state_sequence[i] != self._state_sequence[i - 1]:
                cps.append(i)
        return cps

    def _compute_regime_runs(self) -> list[tuple[str, int, int]]:
        """Run-length encoding: ``[(regime, start, length), ...]``."""
        runs: list[tuple[str, int, int]] = []
        if len(self._regime_labels) == 0:
            return runs

        current = self._regime_labels[0]
        start = 0
        length = 1
        for i in range(1, len(self._regime_labels)):
            if self._regime_labels[i] == current:
                length += 1
            else:
                runs.append((str(current), start, length))
                current = self._regime_labels[i]
                start = i
                length = 1
        runs.append((str(current), start, length))
        return runs

    def _empirical_durations(self) -> dict[str, np.ndarray]:
        """Dict mapping regime name to array of observed run lengths."""
        runs = self._compute_regime_runs()
        durations: dict[str, list[int]] = {}
        for regime, _, length in runs:
            durations.setdefault(regime, []).append(length)
        return {k: np.array(v) for k, v in durations.items()}

    def _compute_regime_ages(self) -> np.ndarray:
        """Per-observation age within its regime run."""
        ages = np.ones(len(self._state_sequence), dtype=int)
        for i in range(1, len(self._state_sequence)):
            if self._state_sequence[i] == self._state_sequence[i - 1]:
                ages[i] = ages[i - 1] + 1
            else:
                ages[i] = 1
        return ages

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Text report of Hurst regime analysis."""
        self._check_fitted()
        lines = [
            "Hurst Regime Analysis",
            "=" * 50,
            "",
            f"Method: {self._method}  |  Window: {self._window}  |  Regimes: {self._n_regimes}",
            "",
            "Current State",
            "-" * 50,
            f"  Regime:              {self.current_regime}",
            f"  Hurst:               {self.current_hurst:.4f}",
            f"  Regime age:          {self.regime_age} periods",
            f"  Stability:           {self.regime_stability:.2f}",
            f"  Expected duration:   {self.expected_duration:.1f} periods",
            f"  Expected remaining:  {self.expected_remaining:.1f} periods",
            f"  Recommended horizon: {self.recommended_horizon} periods",
            "",
            "HMM Regime Parameters",
            "-" * 50,
        ]

        means = self._model.means_.flatten()
        covars = self._model.covars_
        if covars.ndim == 3:
            stds = np.sqrt(covars[:, 0, 0])
        else:
            stds = np.sqrt(covars.flatten())

        durations_hmm = 1.0 / (1.0 - np.diag(self._model.transmat_) + 1e-10)

        for state in range(self._n_regimes):
            regime_name = self._state_to_regime[state]
            lines.append(
                f"  {regime_name.upper()} (state {state}):  "
                f"mean H = {means[state]:.4f}, "
                f"std H = {stds[state]:.4f}, "
                f"HMM duration = {durations_hmm[state]:.1f}"
            )

        emp = self._empirical_durations()
        lines.extend(["", "Empirical Durations", "-" * 50])
        for regime_name in sorted(emp.keys()):
            arr = emp[regime_name]
            lines.append(
                f"  {regime_name}: n={len(arr)}, "
                f"median={np.median(arr):.0f}, "
                f"mean={np.mean(arr):.1f}, "
                f"max={np.max(arr)}"
            )

        lines.extend(["", "Transition Matrix", "-" * 50])
        header = "         " + "".join(
            f"{self._state_to_regime[j]:>16}" for j in range(self._n_regimes)
        )
        lines.append(header)
        for i in range(self._n_regimes):
            row = f"  {self._state_to_regime[i]:>6}"
            for j in range(self._n_regimes):
                row += f"{self._model.transmat_[i, j]:>16.4f}"
            lines.append(row)

        cps = self.change_points
        lines.extend([
            "",
            f"Change points: {len(cps)}",
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HurstRegimeAnalyzer("
            f"n={len(self._prices)}, "
            f"method='{self._method}', "
            f"window={self._window}, "
            f"regime='{self.current_regime}')"
        )
