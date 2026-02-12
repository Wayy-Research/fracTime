"""Tests for fractime.synthetic data generators."""

import numpy as np
import pytest
from scipy import stats

from fractime.synthetic import (
    create_synthetic_asset,
    generate_arima_prices,
    generate_fat_tail_prices,
    generate_fbm_prices,
    generate_regime_switching_prices,
    generate_volatility_shift_prices,
)


class TestFBMPrices:
    """Tests for generate_fbm_prices."""

    def test_shape(self) -> None:
        prices = generate_fbm_prices(500, hurst=0.5, seed=42)
        assert len(prices) == 500

    def test_start_price(self) -> None:
        prices = generate_fbm_prices(500, hurst=0.5, start_price=200.0, seed=42)
        assert prices[0] == pytest.approx(200.0, rel=1e-6)

    def test_positive_prices(self) -> None:
        prices = generate_fbm_prices(1000, hurst=0.5, volatility=0.20, seed=42)
        assert np.all(prices > 0)

    def test_seed_reproducibility(self) -> None:
        p1 = generate_fbm_prices(500, hurst=0.7, seed=123)
        p2 = generate_fbm_prices(500, hurst=0.7, seed=123)
        np.testing.assert_array_equal(p1, p2)

    def test_hurst_recovery(self) -> None:
        """Generated fBm should have Hurst exponent close to target."""
        from fractime._numba import compute_hurst_rs

        prices = generate_fbm_prices(2000, hurst=0.7, seed=42)
        estimated_h = compute_hurst_rs(prices, min_lag=10, max_lag=200)
        # Hurst estimation is noisy; allow +-0.15
        assert abs(estimated_h - 0.7) < 0.15, f"Expected ~0.7, got {estimated_h:.3f}"


class TestARIMAPrices:
    """Tests for generate_arima_prices."""

    def test_shape(self) -> None:
        prices = generate_arima_prices(500, order=(1, 1, 0), seed=42)
        assert len(prices) == 500

    def test_positive_prices(self) -> None:
        prices = generate_arima_prices(500, order=(1, 1, 0), volatility=0.005, seed=42)
        assert np.all(prices > 0)

    def test_start_price(self) -> None:
        prices = generate_arima_prices(500, order=(1, 1, 0), start_price=50.0, seed=42)
        assert prices[0] == pytest.approx(50.0, rel=1e-6)

    def test_custom_params(self) -> None:
        prices = generate_arima_prices(
            500, order=(1, 1, 1), ar_params=[0.5], ma_params=[0.3], seed=42
        )
        assert len(prices) == 500
        assert np.all(np.isfinite(prices))


class TestRegimeSwitching:
    """Tests for generate_regime_switching_prices."""

    def test_shape(self) -> None:
        prices, labels = generate_regime_switching_prices(500, seed=42)
        assert len(prices) == 500
        assert len(labels) == 500

    def test_regime_labels(self) -> None:
        prices, labels = generate_regime_switching_prices(
            500,
            regime_hursts=(0.3, 0.7),
            regime_durations=(126, 126),
            seed=42,
        )
        unique_labels = np.unique(labels)
        assert 0 in unique_labels
        assert 1 in unique_labels

    def test_label_durations(self) -> None:
        """Regime durations should match specification (approximately)."""
        _, labels = generate_regime_switching_prices(
            505,  # 504 returns -> 4 segments of 126
            regime_hursts=(0.3, 0.7),
            regime_durations=(126, 126),
            seed=42,
        )
        # First regime block (indices 1..126 in labels) should be regime 0
        block0 = labels[1:127]
        assert np.all(block0 == 0)


class TestVolatilityShift:
    """Tests for generate_volatility_shift_prices."""

    def test_shape(self) -> None:
        prices, vol_regime = generate_volatility_shift_prices(1000, seed=42)
        assert len(prices) == 1000
        assert len(vol_regime) == 1000

    def test_crisis_higher_vol(self) -> None:
        """Crisis period should have detectably higher realized vol."""
        prices, vol_regime = generate_volatility_shift_prices(
            1000,
            volatility_sequence=[(333, 0.15), (334, 0.60), (333, 0.15)],
            seed=42,
        )
        returns = np.diff(np.log(prices))

        calm_returns = returns[:333]
        crisis_returns = returns[333:667]

        calm_vol = np.std(calm_returns) * np.sqrt(252)
        crisis_vol = np.std(crisis_returns) * np.sqrt(252)

        assert crisis_vol > calm_vol * 1.5, (
            f"Crisis vol ({crisis_vol:.3f}) should be much higher than "
            f"calm vol ({calm_vol:.3f})"
        )


class TestFatTails:
    """Tests for generate_fat_tail_prices."""

    def test_shape(self) -> None:
        prices = generate_fat_tail_prices(1000, seed=42)
        assert len(prices) == 1000

    def test_positive_prices(self) -> None:
        prices = generate_fat_tail_prices(1000, seed=42)
        assert np.all(prices > 0)

    def test_higher_kurtosis(self) -> None:
        """Fat-tail series should have higher excess kurtosis than Gaussian."""
        np.random.seed(42)

        # Fat-tail prices (df=5)
        fat_prices = generate_fat_tail_prices(5000, hurst=0.5, df=5.0, seed=42)
        fat_returns = np.diff(np.log(fat_prices))

        # Gaussian prices (same Hurst, no fat tails)
        gauss_prices = generate_fbm_prices(5000, hurst=0.5, seed=42)
        gauss_returns = np.diff(np.log(gauss_prices))

        fat_kurt = stats.kurtosis(fat_returns, fisher=True)
        gauss_kurt = stats.kurtosis(gauss_returns, fisher=True)

        assert fat_kurt > gauss_kurt, (
            f"Fat-tail kurtosis ({fat_kurt:.2f}) should exceed "
            f"Gaussian kurtosis ({gauss_kurt:.2f})"
        )


class TestCreateSyntheticAsset:
    """Tests for create_synthetic_asset."""

    def test_basic(self) -> None:
        prices = generate_fbm_prices(500, hurst=0.5, seed=42)
        asset = create_synthetic_asset(prices, "SYN_TEST", "Test Asset")

        assert asset.ticker == "SYN_TEST"
        assert asset.name == "Test Asset"
        assert len(asset.prices) == 500
        assert len(asset.dates) == 500
        assert len(asset.returns) == 500
        assert asset.returns[0] == 0.0
        assert asset.category == "synthetic"

    def test_dates_are_business_days(self) -> None:
        prices = generate_fbm_prices(100, hurst=0.5, seed=42)
        asset = create_synthetic_asset(prices, "SYN", "Synth")

        for d in asset.dates:
            assert d.weekday() < 5, f"Date {d} is a weekend"

    def test_returns_match_prices(self) -> None:
        prices = generate_fbm_prices(200, hurst=0.5, seed=42)
        asset = create_synthetic_asset(prices, "SYN", "Synth")

        expected_returns = np.diff(np.log(prices))
        np.testing.assert_allclose(asset.returns[1:], expected_returns, rtol=1e-10)
