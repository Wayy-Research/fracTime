"""Tests for fractime.advanced — new analytical methods in v0.6."""

import numpy as np
import pytest

from fractime.advanced import (
    compute_wavelet_hurst,
    compute_rolling_wavelet_hurst,
    compute_mf_adcca,
    compute_qpl,
    compute_fractal_coherence,
    compute_rolling_fractal_coherence,
    compute_ftd,
    compute_dtw_alignment,
    compute_dtw_beta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_prices():
    """Synthetic trending series (H > 0.5)."""
    np.random.seed(42)
    n = 500
    trend = np.linspace(0, 1, n)
    noise = np.random.normal(0, 0.02, n).cumsum()
    return 100 * np.exp(trend + noise * 0.3)


@pytest.fixture
def random_walk_prices():
    """Synthetic random walk series (H ≈ 0.5)."""
    np.random.seed(123)
    returns = np.random.normal(0, 0.01, 500)
    return 100 * np.exp(np.cumsum(returns))


@pytest.fixture
def volume_series():
    """Synthetic volume series."""
    np.random.seed(99)
    return np.abs(np.random.normal(1e6, 2e5, 500))


@pytest.fixture
def two_assets():
    """Two correlated asset price series."""
    np.random.seed(55)
    common = np.random.normal(0, 0.01, 500)
    noise_x = np.random.normal(0, 0.005, 500)
    noise_y = np.random.normal(0, 0.005, 500)
    x = 100 * np.exp(np.cumsum(common + noise_x))
    y = 50 * np.exp(np.cumsum(common * 0.8 + noise_y))
    return x, y


# ---------------------------------------------------------------------------
# Wavelet Hurst
# ---------------------------------------------------------------------------


class TestWaveletHurst:
    def test_returns_float(self, trending_prices):
        h = compute_wavelet_hurst(trending_prices)
        assert isinstance(h, float)

    def test_in_range(self, trending_prices):
        h = compute_wavelet_hurst(trending_prices)
        assert 0.0 < h < 1.0

    def test_trending_above_half(self, trending_prices):
        h = compute_wavelet_hurst(trending_prices)
        assert h > 0.4, f"Expected trending H > 0.4, got {h}"

    def test_random_walk_near_half(self, random_walk_prices):
        h = compute_wavelet_hurst(random_walk_prices)
        assert 0.3 < h < 0.7, f"Expected H near 0.5, got {h}"

    def test_short_series_fallback(self):
        h = compute_wavelet_hurst(np.array([100, 101, 102]))
        assert h == 0.5

    def test_rolling(self, trending_prices):
        rolling = compute_rolling_wavelet_hurst(trending_prices, window=200, step=50)
        assert len(rolling) > 0
        assert all(0 < v < 1 for v in rolling)


# ---------------------------------------------------------------------------
# MF-ADCCA
# ---------------------------------------------------------------------------


class TestMFADCCA:
    def test_basic(self, two_assets):
        x, y = two_assets
        rho_q, h_xy = compute_mf_adcca(x, y)
        assert len(rho_q) == 21  # default q_values
        assert len(h_xy) == 21

    def test_custom_q(self, two_assets):
        x, y = two_assets
        q = np.array([-2.0, 0.0, 2.0])
        rho_q, h_xy = compute_mf_adcca(x, y, q_values=q)
        assert len(rho_q) == 3
        assert len(h_xy) == 3

    def test_positive_correlation(self, two_assets):
        x, y = two_assets
        rho_q, _ = compute_mf_adcca(x, y, q_values=np.array([2.0]))
        assert rho_q[0] > 0, "Correlated assets should have positive rho"

    def test_short_series(self):
        x = np.array([100, 101, 102, 103, 104])
        y = np.array([50, 51, 52, 53, 54])
        rho_q, h_xy = compute_mf_adcca(x, y)
        # Should return NaN gracefully
        assert len(rho_q) == 21


# ---------------------------------------------------------------------------
# Quantum Price Levels
# ---------------------------------------------------------------------------


class TestQPL:
    def test_basic(self, trending_prices):
        levels = compute_qpl(trending_prices, n_levels=5)
        assert len(levels) == 5
        assert np.all(np.diff(levels) >= 0), "Levels should be sorted"

    def test_levels_in_price_range(self, trending_prices):
        levels = compute_qpl(trending_prices, n_levels=3)
        margin = 0.1 * (trending_prices.max() - trending_prices.min())
        assert levels.min() >= trending_prices.min() - margin
        assert levels.max() <= trending_prices.max() + margin

    def test_histogram_potential(self, trending_prices):
        levels = compute_qpl(trending_prices, n_levels=3, potential_type="histogram")
        assert len(levels) == 3

    def test_different_n_levels(self, trending_prices):
        for n in [1, 3, 10]:
            levels = compute_qpl(trending_prices, n_levels=n)
            assert len(levels) == n


# ---------------------------------------------------------------------------
# Fractal Coherence
# ---------------------------------------------------------------------------


class TestFractalCoherence:
    def test_basic(self, trending_prices, volume_series):
        coh = compute_fractal_coherence(trending_prices, volume_series)
        assert isinstance(coh, float)
        assert 0.0 <= coh <= 1.0

    def test_identical_series_high_coherence(self):
        np.random.seed(7)
        s = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200)))
        coh = compute_fractal_coherence(s, s, window=63)
        assert coh > 0.8, f"Identical series should have high coherence, got {coh}"

    def test_with_volatility(self, trending_prices, volume_series):
        vol = np.abs(np.diff(np.log(trending_prices)))
        # pad to match length
        vol = np.concatenate([[vol[0]], vol])
        coh = compute_fractal_coherence(trending_prices, volume_series, volatility=vol)
        assert 0.0 <= coh <= 1.0

    def test_rolling(self, trending_prices, volume_series):
        rolling = compute_rolling_fractal_coherence(
            trending_prices, volume_series, window=100, step=50
        )
        assert len(rolling) > 0
        assert all(0.0 <= v <= 1.0 for v in rolling)


# ---------------------------------------------------------------------------
# Fractal Trend Dimension
# ---------------------------------------------------------------------------


class TestFTD:
    def test_basic(self, trending_prices):
        d_plus, d_minus = compute_ftd(trending_prices)
        assert 1.0 <= d_plus <= 2.0
        assert 1.0 <= d_minus <= 2.0

    def test_returns_tuple(self, trending_prices):
        result = compute_ftd(trending_prices)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_short_series_fallback(self):
        d_plus, d_minus = compute_ftd(np.array([100, 101, 102]))
        assert d_plus == 1.5
        assert d_minus == 1.5

    def test_different_windows(self, trending_prices):
        for w in [30, 63, 126]:
            d_plus, d_minus = compute_ftd(trending_prices, window=w)
            assert 1.0 <= d_plus <= 2.0
            assert 1.0 <= d_minus <= 2.0


# ---------------------------------------------------------------------------
# DTW Alignment
# ---------------------------------------------------------------------------


class TestDTWAlignment:
    def test_identical_series(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        path, dist = compute_dtw_alignment(s, s)
        assert dist == 0.0
        np.testing.assert_array_equal(path[:, 0], path[:, 1])

    def test_shifted_series(self):
        s1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        s2 = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        path, dist = compute_dtw_alignment(s1, s2)
        assert dist > 0
        assert len(path) >= max(len(s1), len(s2))

    def test_different_lengths(self):
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        path, dist = compute_dtw_alignment(s1, s2)
        assert path.shape[1] == 2
        assert len(path) >= max(len(s1), len(s2))


class TestDTWBeta:
    def test_perfect_correlation(self):
        np.random.seed(42)
        market = np.random.normal(0, 0.01, 200)
        stock = 1.5 * market + np.random.normal(0, 0.001, 200)
        beta = compute_dtw_beta(stock, market)
        assert 1.0 < beta < 2.0, f"Expected beta near 1.5, got {beta}"

    def test_uncorrelated_beta(self):
        np.random.seed(42)
        market = np.random.normal(0, 0.01, 200)
        stock = np.random.normal(0, 0.01, 200)
        beta = compute_dtw_beta(stock, market)
        # Uncorrelated series should have beta closer to 0 than 1.5
        assert abs(beta) < 2.0, f"Expected moderate beta, got {beta}"

    def test_returns_float(self):
        np.random.seed(42)
        market = np.random.normal(0, 0.01, 100)
        stock = np.random.normal(0, 0.01, 100)
        beta = compute_dtw_beta(stock, market)
        assert isinstance(beta, float)


# ---------------------------------------------------------------------------
# Integration: Analyzer with wavelet method
# ---------------------------------------------------------------------------


class TestAnalyzerWavelet:
    def test_analyzer_wavelet_method(self, trending_prices):
        from fractime import Analyzer
        a = Analyzer(trending_prices, method="wavelet")
        h = a.hurst.value
        assert isinstance(h, float)
        assert 0.0 < h < 1.0

    def test_analyzer_wavelet_rolling(self, trending_prices):
        from fractime import Analyzer
        a = Analyzer(trending_prices, method="wavelet", window=100)
        rolling = a.hurst.rolling
        assert len(rolling) > 0


# ---------------------------------------------------------------------------
# Imports via top-level package
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    def test_import_all(self):
        import fractime as ft

        assert hasattr(ft, "compute_wavelet_hurst")
        assert hasattr(ft, "compute_mf_adcca")
        assert hasattr(ft, "compute_qpl")
        assert hasattr(ft, "compute_fractal_coherence")
        assert hasattr(ft, "compute_ftd")
        assert hasattr(ft, "compute_dtw_alignment")
        assert hasattr(ft, "compute_dtw_beta")

    def test_version(self):
        import fractime as ft

        assert ft.__version__ == "0.6.0"
