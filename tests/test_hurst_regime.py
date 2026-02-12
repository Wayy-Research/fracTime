"""
Tests for HurstRegimeAnalyzer.

Covers construction, current state, durations, history, regime switching,
summary output, and top-level import.
"""

import numpy as np
import polars as pl
import pytest

from fractime.hurst_regime import HurstRegimeAnalyzer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trending_prices() -> np.ndarray:
    """Strong uptrend — H >> 0.5 (600 points)."""
    np.random.seed(10)
    drift = 0.002
    noise = np.random.normal(0, 0.005, 600)
    returns = drift + noise
    return 100.0 * np.cumprod(1 + returns)


@pytest.fixture
def random_walk_prices() -> np.ndarray:
    """Random walk — H ~ 0.5 (600 points)."""
    np.random.seed(20)
    returns = np.random.normal(0, 0.01, 600)
    return 100.0 * np.cumprod(1 + returns)


@pytest.fixture
def regime_switching_prices() -> np.ndarray:
    """300 trending + 300 mean-reverting (600 points)."""
    np.random.seed(30)
    # Trending phase: strong positive drift
    trending = np.random.normal(0.003, 0.005, 300)
    # Mean-reverting phase: alternate sign with small noise
    mr_base = np.tile([0.01, -0.01], 150)
    mr_noise = np.random.normal(0, 0.002, 300)
    mean_reverting = mr_base + mr_noise
    returns = np.concatenate([trending, mean_reverting])
    return 100.0 * np.cumprod(1 + returns)


@pytest.fixture
def sample_dates() -> np.ndarray:
    """Datetime array with 600 points."""
    return np.array(
        [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(600)]
    )


# =============================================================================
# TestConstruction
# =============================================================================


class TestConstruction:
    """Basic construction and parameter validation."""

    def test_basic_creation(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert hra is not None

    def test_method_rs(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="rs", window=252)
        assert hra.current_regime in ("trending", "random", "mean_reverting")

    def test_method_dfa(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert hra.current_regime in ("trending", "random", "mean_reverting")

    def test_method_wavelet(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="wavelet", window=252)
        assert hra.current_regime in ("trending", "random", "mean_reverting")

    def test_invalid_method_raises(self, trending_prices: np.ndarray) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            HurstRegimeAnalyzer(trending_prices, method="bogus")

    def test_insufficient_data_raises(self) -> None:
        short = np.cumprod(1 + np.random.normal(0, 0.01, 100)) * 100
        with pytest.raises(ValueError, match="Need at least"):
            HurstRegimeAnalyzer(short, window=252)

    def test_with_dates(
        self, trending_prices: np.ndarray, sample_dates: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(
            trending_prices, dates=sample_dates, method="dfa", window=252
        )
        assert "date" in hra.regime_history.columns

    def test_two_regimes(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(
            trending_prices, method="dfa", window=252, n_regimes=2
        )
        assert hra.current_regime in ("trending", "mean_reverting")

    def test_invalid_n_regimes_raises(self, trending_prices: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n_regimes must be 2 or 3"):
            HurstRegimeAnalyzer(trending_prices, n_regimes=5)


# =============================================================================
# TestCurrentState
# =============================================================================


class TestCurrentState:
    """Properties describing the current regime."""

    def test_regime_is_valid_string(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert hra.current_regime in ("trending", "random", "mean_reverting")

    def test_current_hurst_in_range(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert 0.0 < hra.current_hurst < 1.0

    def test_age_at_least_one(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert hra.regime_age >= 1

    def test_stability_in_range(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert 0.0 <= hra.regime_stability <= 1.0


# =============================================================================
# TestDurations
# =============================================================================


class TestDurations:
    """Duration and horizon estimates."""

    def test_expected_duration_positive(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert hra.expected_duration > 0

    def test_expected_remaining_at_least_one(
        self, trending_prices: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert hra.expected_remaining >= 1.0

    def test_horizon_in_valid_range(self, trending_prices: np.ndarray) -> None:
        window = 252
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=window)
        assert 5 <= hra.recommended_horizon <= window // 4


# =============================================================================
# TestHistory
# =============================================================================


class TestHistory:
    """Regime history DataFrame and diagnostics."""

    def test_history_is_dataframe(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert isinstance(hra.regime_history, pl.DataFrame)

    def test_history_columns_without_dates(
        self, trending_prices: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        df = hra.regime_history
        assert "index" in df.columns
        assert "hurst" in df.columns
        assert "regime" in df.columns
        assert "regime_age" in df.columns
        assert "state_probability" in df.columns

    def test_history_columns_with_dates(
        self, trending_prices: np.ndarray, sample_dates: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(
            trending_prices, dates=sample_dates, method="dfa", window=252
        )
        df = hra.regime_history
        assert "date" in df.columns
        assert "index" not in df.columns

    def test_change_points_is_list(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        assert isinstance(hra.change_points, list)

    def test_transition_matrix_rows_sum_to_one(
        self, trending_prices: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        row_sums = hra.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# =============================================================================
# TestRegimeSwitching
# =============================================================================


class TestRegimeSwitching:
    """Detecting regime changes in switching data."""

    def test_detects_changes_in_switching_data(
        self, regime_switching_prices: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(
            regime_switching_prices, method="dfa", window=252
        )
        # Should detect at least one change point
        assert len(hra.change_points) >= 1

    def test_two_regime_mode_works(
        self, regime_switching_prices: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(
            regime_switching_prices, method="dfa", window=252, n_regimes=2
        )
        regimes = set(hra.regime_history["regime"].to_list())
        # Must only contain the 2-regime labels
        assert regimes.issubset({"trending", "mean_reverting"})


# =============================================================================
# TestSummary
# =============================================================================


class TestSummary:
    """Summary and repr output."""

    def test_summary_returns_string(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        s = hra.summary()
        assert isinstance(s, str)

    def test_summary_has_expected_sections(
        self, trending_prices: np.ndarray
    ) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        s = hra.summary()
        assert "Hurst Regime Analysis" in s
        assert "Current State" in s
        assert "Transition Matrix" in s
        assert "Change points" in s

    def test_repr_format(self, trending_prices: np.ndarray) -> None:
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=252)
        r = repr(hra)
        assert "HurstRegimeAnalyzer(" in r
        assert "method='dfa'" in r


# =============================================================================
# TestTopLevelImport
# =============================================================================


class TestTopLevelImport:
    """Verify the class is accessible from ``import fractime``."""

    def test_import_from_fractime(self) -> None:
        import fractime as ft

        assert hasattr(ft, "HurstRegimeAnalyzer")

    def test_in_all(self) -> None:
        import fractime as ft

        assert "HurstRegimeAnalyzer" in ft.__all__


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge cases and input validation."""

    def test_none_data_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            HurstRegimeAnalyzer(None)

    def test_nan_data_raises(self) -> None:
        prices = np.ones(600) * 100.0
        prices[300] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            HurstRegimeAnalyzer(prices, window=252)

    def test_inf_data_raises(self) -> None:
        prices = np.cumprod(1 + np.random.normal(0, 0.01, 600)) * 100
        prices[400] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            HurstRegimeAnalyzer(prices, window=252)

    def test_polars_series_input(self, trending_prices: np.ndarray) -> None:
        series = pl.Series("price", trending_prices)
        hra = HurstRegimeAnalyzer(series, method="dfa", window=252)
        assert hra.current_regime in ("trending", "random", "mean_reverting")

    def test_dataframe_input_raises(self) -> None:
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Expected Series, got DataFrame"):
            HurstRegimeAnalyzer(df)

    def test_history_row_count(self, trending_prices: np.ndarray) -> None:
        window = 252
        hra = HurstRegimeAnalyzer(trending_prices, method="dfa", window=window)
        df = hra.regime_history
        expected_rows = len(trending_prices) - window
        assert len(df) == expected_rows

    def test_boundary_minimum_data(self) -> None:
        """Exactly window + 50 observations."""
        rng = np.random.default_rng(99)
        n = 302  # 252 + 50
        prices = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        hra = HurstRegimeAnalyzer(prices, method="dfa", window=252)
        assert hra.current_regime in ("trending", "random", "mean_reverting")
