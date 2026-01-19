"""
Tests for the new FracTime API.

This file tests the simplified, composable API introduced in v0.3.0.
"""

import numpy as np
import pytest
import fractime as ft


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    returns = np.random.randn(500) * 0.02
    prices = 100 * np.cumprod(1 + returns)
    return prices


@pytest.fixture
def sample_volumes():
    """Generate sample volume data."""
    np.random.seed(42)
    return np.random.randint(1000, 10000, 500).astype(float)


@pytest.fixture
def sample_dates():
    """Generate sample dates."""
    import datetime
    start = datetime.datetime(2023, 1, 1)
    return np.array([start + datetime.timedelta(days=i) for i in range(500)])


# =============================================================================
# Analyzer Tests
# =============================================================================

class TestAnalyzer:
    """Tests for the Analyzer class."""

    def test_basic_creation(self, sample_prices):
        """Test basic Analyzer creation."""
        analyzer = ft.Analyzer(sample_prices)
        assert analyzer is not None

    def test_hurst_point(self, sample_prices):
        """Test Hurst exponent point estimate."""
        analyzer = ft.Analyzer(sample_prices)
        hurst = analyzer.hurst.value
        assert 0 <= hurst <= 1

    def test_hurst_as_float(self, sample_prices):
        """Test Hurst exponent can be used as float."""
        analyzer = ft.Analyzer(sample_prices)
        hurst = float(analyzer.hurst)
        assert 0 <= hurst <= 1

    def test_hurst_repr(self, sample_prices):
        """Test Hurst exponent string representation."""
        analyzer = ft.Analyzer(sample_prices)
        assert 'hurst=' in repr(analyzer.hurst)

    def test_fractal_dim_point(self, sample_prices):
        """Test fractal dimension point estimate."""
        analyzer = ft.Analyzer(sample_prices)
        fd = analyzer.fractal_dim.value
        assert 1 <= fd <= 2

    def test_volatility_point(self, sample_prices):
        """Test volatility point estimate."""
        analyzer = ft.Analyzer(sample_prices)
        vol = analyzer.volatility.value
        assert vol >= 0

    def test_regime_detection(self, sample_prices):
        """Test regime detection."""
        analyzer = ft.Analyzer(sample_prices)
        regime = analyzer.regime
        assert regime in ['trending', 'mean_reverting', 'random']

    def test_regime_probabilities(self, sample_prices):
        """Test regime probabilities."""
        analyzer = ft.Analyzer(sample_prices)
        probs = analyzer.regime_probabilities
        assert 'trending' in probs
        assert 'mean_reverting' in probs
        assert 'random' in probs
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_rolling_hurst(self, sample_prices):
        """Test rolling Hurst exponent."""
        analyzer = ft.Analyzer(sample_prices)
        rolling = analyzer.hurst.rolling
        assert rolling is not None
        assert 'value' in rolling.columns
        assert len(rolling) > 0

    def test_bootstrap_ci(self, sample_prices):
        """Test bootstrap confidence interval."""
        analyzer = ft.Analyzer(sample_prices, n_samples=100)  # Reduce for speed
        ci = analyzer.hurst.ci(0.95)
        assert len(ci) == 2
        assert ci[0] < ci[1]
        # CI should be reasonable (within 0-1 for Hurst)
        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1

    def test_bootstrap_std(self, sample_prices):
        """Test bootstrap standard error."""
        analyzer = ft.Analyzer(sample_prices, n_samples=100)
        std = analyzer.hurst.std
        assert std >= 0

    def test_result_property(self, sample_prices):
        """Test result property returns AnalysisResult."""
        analyzer = ft.Analyzer(sample_prices)
        result = analyzer.result
        assert isinstance(result, ft.AnalysisResult)

    def test_summary(self, sample_prices):
        """Test summary generation."""
        analyzer = ft.Analyzer(sample_prices)
        summary = analyzer.summary()
        assert 'Hurst' in summary
        assert 'Fractal' in summary

    def test_multi_dimensional(self, sample_prices, sample_volumes):
        """Test multi-dimensional analysis."""
        analyzer = ft.Analyzer({
            'price': sample_prices,
            'volume': sample_volumes,
        })
        assert 'price' in analyzer.dimensions
        assert 'volume' in analyzer.dimensions
        assert analyzer['price'].hurst.value >= 0
        assert analyzer['volume'].hurst.value >= 0

    def test_coherence(self, sample_prices, sample_volumes):
        """Test cross-dimensional coherence."""
        analyzer = ft.Analyzer({
            'price': sample_prices,
            'volume': sample_volumes,
        })
        coherence = analyzer.coherence.value
        assert 0 <= coherence <= 1


# =============================================================================
# Forecaster Tests
# =============================================================================

class TestForecaster:
    """Tests for the Forecaster class."""

    def test_basic_creation(self, sample_prices):
        """Test basic Forecaster creation."""
        model = ft.Forecaster(sample_prices)
        assert model is not None

    def test_predict(self, sample_prices):
        """Test basic prediction."""
        model = ft.Forecaster(sample_prices)
        result = model.predict(steps=30, n_paths=100)
        assert result is not None
        assert isinstance(result, ft.ForecastResult)

    def test_forecast_shape(self, sample_prices):
        """Test forecast shape."""
        model = ft.Forecaster(sample_prices)
        result = model.predict(steps=30, n_paths=100)
        assert result.forecast.shape == (30,)
        assert result.paths.shape == (100, 30)
        assert result.probabilities.shape == (100,)

    def test_confidence_interval(self, sample_prices):
        """Test confidence interval."""
        model = ft.Forecaster(sample_prices)
        result = model.predict(steps=30, n_paths=100)
        lower, upper = result.ci(0.95)
        assert lower.shape == (30,)
        assert upper.shape == (30,)
        assert all(lower <= result.forecast)
        assert all(result.forecast <= upper)

    def test_hurst_shortcut(self, sample_prices):
        """Test hurst property shortcut."""
        model = ft.Forecaster(sample_prices)
        assert model.hurst == model.analyzer.hurst

    def test_regime_shortcut(self, sample_prices):
        """Test regime property shortcut."""
        model = ft.Forecaster(sample_prices)
        assert model.regime == model.analyzer.regime

    def test_with_exogenous(self, sample_prices, sample_volumes):
        """Test forecaster with exogenous variables."""
        model = ft.Forecaster(sample_prices, exogenous={'volume': sample_volumes})
        result = model.predict(steps=30, n_paths=100)
        assert result is not None

    def test_custom_path_weights(self, sample_prices):
        """Test custom path weights."""
        model = ft.Forecaster(
            sample_prices,
            path_weights={'hurst': 0.5, 'volatility': 0.3, 'pattern': 0.2}
        )
        result = model.predict(steps=30, n_paths=100)
        assert result is not None


# =============================================================================
# Simulator Tests
# =============================================================================

class TestSimulator:
    """Tests for the Simulator class."""

    def test_basic_creation(self, sample_prices):
        """Test basic Simulator creation."""
        sim = ft.Simulator(sample_prices)
        assert sim is not None

    def test_generate_paths(self, sample_prices):
        """Test path generation."""
        sim = ft.Simulator(sample_prices)
        paths = sim.generate(n_paths=100, steps=30)
        assert paths.shape == (100, 30)

    def test_fbm_method(self, sample_prices):
        """Test fBm generation method."""
        sim = ft.Simulator(sample_prices)
        paths = sim.generate(n_paths=100, steps=30, method='fbm')
        assert paths.shape == (100, 30)

    def test_bootstrap_method(self, sample_prices):
        """Test bootstrap generation method."""
        sim = ft.Simulator(sample_prices)
        paths = sim.generate(n_paths=100, steps=30, method='bootstrap')
        assert paths.shape == (100, 30)

    def test_time_warp(self, sample_prices):
        """Test time warping."""
        sim = ft.Simulator(sample_prices, time_warp=True)
        assert sim._time_warp is True
        paths = sim.generate(n_paths=100, steps=30)
        assert paths.shape == (100, 30)


# =============================================================================
# Ensemble Tests
# =============================================================================

class TestEnsemble:
    """Tests for the Ensemble class."""

    def test_basic_creation(self, sample_prices):
        """Test basic Ensemble creation."""
        ensemble = ft.Ensemble(sample_prices)
        assert ensemble is not None

    def test_default_models(self, sample_prices):
        """Test default model creation."""
        ensemble = ft.Ensemble(sample_prices)
        assert ensemble.n_models >= 2

    def test_predict(self, sample_prices):
        """Test ensemble prediction."""
        ensemble = ft.Ensemble(sample_prices)
        result = ensemble.predict(steps=30, n_paths=50)
        assert isinstance(result, ft.ForecastResult)

    def test_custom_models(self, sample_prices):
        """Test custom models."""
        models = [
            ft.Forecaster(sample_prices, method='rs'),
            ft.Forecaster(sample_prices, method='dfa'),
        ]
        ensemble = ft.Ensemble(sample_prices, models=models)
        assert ensemble.n_models == 2

    def test_stacking_strategy(self, sample_prices):
        """Test stacking strategy."""
        ensemble = ft.Ensemble(sample_prices, strategy='stacking')
        result = ensemble.predict(steps=30, n_paths=50)
        assert result is not None

    def test_boosting_strategy(self, sample_prices):
        """Test boosting strategy."""
        ensemble = ft.Ensemble(sample_prices, strategy='boosting')
        result = ensemble.predict(steps=30, n_paths=50)
        assert result is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_function(self, sample_prices):
        """Test analyze() convenience function."""
        result = ft.analyze(sample_prices)
        assert isinstance(result, ft.AnalysisResult)

    def test_forecast_function(self, sample_prices):
        """Test forecast() convenience function."""
        result = ft.forecast(sample_prices, steps=20, n_paths=100)
        assert isinstance(result, ft.ForecastResult)


# =============================================================================
# Result Object Tests
# =============================================================================

class TestForecastResult:
    """Tests for ForecastResult."""

    def test_to_frame(self, sample_prices):
        """Test DataFrame export."""
        model = ft.Forecaster(sample_prices)
        result = model.predict(steps=30, n_paths=100)
        df = result.to_frame()
        assert 'forecast' in df.columns
        assert 'lower' in df.columns
        assert 'upper' in df.columns

    def test_metadata(self, sample_prices):
        """Test metadata."""
        model = ft.Forecaster(sample_prices)
        result = model.predict(steps=30, n_paths=100)
        assert 'hurst' in result.metadata
        assert 'regime' in result.metadata


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_summary(self, sample_prices):
        """Test summary generation."""
        analyzer = ft.Analyzer(sample_prices)
        summary = analyzer.result.summary()
        assert len(summary) > 0

    def test_repr(self, sample_prices):
        """Test string representation."""
        analyzer = ft.Analyzer(sample_prices)
        repr_str = repr(analyzer.result)
        assert 'hurst=' in repr_str
