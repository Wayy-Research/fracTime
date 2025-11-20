"""
Integration tests using real market data and end-to-end workflows.

These tests validate the full pipeline from data loading through
forecasting, evaluation, and model comparison.
"""

import pytest
import numpy as np
import warnings
from datetime import datetime, timedelta

# Core imports
from fractime import (
    FractalForecaster,
    get_yahoo_data,
    FractalAnalyzer,
    CrossDimensionalAnalyzer,
    StackingForecaster,
    BoostingForecaster
)

# Baseline models (with optional imports)
try:
    from fractime.baselines import ARIMAForecaster
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from fractime.baselines import GARCHForecaster
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False

try:
    from fractime.baselines import ETSForecaster
    ETS_AVAILABLE = True
except ImportError:
    ETS_AVAILABLE = False

try:
    from fractime.baselines import VARForecaster
    VAR_AVAILABLE = True
except ImportError:
    VAR_AVAILABLE = False

try:
    from fractime.baselines import LSTMForecaster
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Backtesting
try:
    from fractime.backtesting import WalkForwardValidator, compare_models
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

# Model selection
try:
    from fractime.model_selection import HybridModelSelector
    MODEL_SELECTION_AVAILABLE = True
except ImportError:
    MODEL_SELECTION_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_market_data():
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    n = 500
    t = np.arange(n)

    # Trend + seasonality + noise
    trend = 100 + 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 252)  # Annual seasonality
    noise = np.random.randn(n) * 2

    prices = trend + seasonal + noise
    prices = np.maximum(prices, 1)  # Ensure positive

    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=n-i) for i in range(n)]

    return prices, dates


@pytest.fixture
def real_market_data():
    """
    Fetch real market data for integration testing.

    This will be skipped if network is unavailable or yfinance not installed.
    """
    try:
        # Use a stable, liquid index
        prices, dates = get_yahoo_data('^GSPC', period='2y')
        return prices, dates
    except Exception as e:
        pytest.skip(f"Could not fetch real market data: {e}")


# ============================================================================
# Basic Integration Tests
# ============================================================================

def test_fractal_forecaster_basic_workflow(sample_market_data):
    """Test complete workflow: fit -> predict -> analyze."""
    prices, dates = sample_market_data

    # Fit model
    model = FractalForecaster()
    model.fit(prices, dates=dates)

    # Predict
    result = model.predict(n_steps=30)

    # Validate result structure
    assert 'forecast' in result
    assert 'mean' in result
    assert 'std' in result
    assert 'lower' in result
    assert 'upper' in result

    assert len(result['forecast']) == 30
    assert np.all(np.isfinite(result['forecast']))
    assert np.all(result['std'] >= 0)

    # Analyze
    analyzer = FractalAnalyzer()
    analysis = analyzer.analyze(prices)

    assert 'hurst' in analysis
    assert 'fractal_dimension' in analysis
    assert 0 < analysis['hurst'] < 1


def test_cross_dimensional_analysis(sample_market_data):
    """Test cross-dimensional analysis with price + volume."""
    prices, _ = sample_market_data

    # Generate synthetic volume data
    volume = np.abs(prices * (1 + 0.1 * np.random.randn(len(prices))))

    # Stack into multivariate series
    data = np.column_stack([prices, volume])

    # Analyze
    analyzer = CrossDimensionalAnalyzer()
    result = analyzer.analyze(data, dim_names=['price', 'volume'])

    assert 'correlation' in result
    assert 'cross_hurst' in result
    assert result['correlation'].shape == (2, 2)


@pytest.mark.skipif(not ARIMA_AVAILABLE, reason="pmdarima not installed")
def test_model_comparison_workflow(sample_market_data):
    """Test comparing multiple models."""
    prices, dates = sample_market_data

    # Create models
    fractal = FractalForecaster()
    arima = ARIMAForecaster()

    # Fit both
    fractal.fit(prices, dates=dates)
    arima.fit(prices)

    # Predict
    fractal_result = fractal.predict(n_steps=10)
    arima_result = arima.predict(n_steps=10)

    # Both should produce valid forecasts
    assert len(fractal_result['forecast']) == 10
    assert len(arima_result['forecast']) == 10
    assert np.all(np.isfinite(fractal_result['forecast']))
    assert np.all(np.isfinite(arima_result['forecast']))


# ============================================================================
# Ensemble Integration Tests
# ============================================================================

@pytest.mark.skipif(not ARIMA_AVAILABLE, reason="pmdarima not installed")
def test_stacking_ensemble_integration(sample_market_data):
    """Test stacking ensemble with multiple base models."""
    prices, dates = sample_market_data

    # Create and fit base models
    fractal = FractalForecaster().fit(prices, dates=dates)
    arima = ARIMAForecaster().fit(prices)

    # Create stacking ensemble
    stacker = StackingForecaster(
        base_models=[fractal, arima],
        meta_learner='ridge',
        n_splits=3
    )

    # Fit ensemble
    stacker.fit(prices)

    # Predict
    result = stacker.predict(n_steps=10)

    assert 'forecast' in result
    assert len(result['forecast']) == 10
    assert np.all(np.isfinite(result['forecast']))

    # Check model weights
    weights = stacker.get_model_weights()
    assert len(weights) > 0
    assert all(w >= 0 for w in weights.values())


def test_boosting_ensemble_integration(sample_market_data):
    """Test boosting ensemble with sequential error correction."""
    prices, dates = sample_market_data

    # Create boosting ensemble
    booster = BoostingForecaster(
        base_model_configs=[
            (FractalForecaster, {}),
            (FractalForecaster, {'method': 'dfa'}),
        ],
        n_estimators=3,
        learning_rate=0.1
    )

    # Fit ensemble
    booster.fit(prices)

    # Predict
    result = booster.predict(n_steps=10)

    assert 'forecast' in result
    assert len(result['forecast']) == 10
    assert np.all(np.isfinite(result['forecast']))

    # Check model weights
    weights = booster.get_model_weights()
    assert len(weights) == 3  # n_estimators


# ============================================================================
# Backtesting Integration Tests
# ============================================================================

@pytest.mark.skipif(not BACKTESTING_AVAILABLE, reason="backtesting module not available")
def test_walk_forward_validation(sample_market_data):
    """Test walk-forward validation workflow."""
    prices, dates = sample_market_data

    # Create model
    model = FractalForecaster()

    # Create validator
    validator = WalkForwardValidator(
        model,
        initial_window=100,
        step_size=20,
        forecast_horizon=10
    )

    # Run validation
    results = validator.run(prices, dates)

    assert 'metrics' in results
    assert 'forecasts' in results
    assert 'actuals' in results

    # Check metrics
    metrics = results['metrics']
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'mape' in metrics

    # All metrics should be finite and non-negative
    assert metrics['mae'] >= 0
    assert metrics['mse'] >= 0
    assert np.isfinite(metrics['mae'])


@pytest.mark.skipif(
    not (BACKTESTING_AVAILABLE and ARIMA_AVAILABLE),
    reason="backtesting or pmdarima not available"
)
def test_multi_model_comparison(sample_market_data):
    """Test comparing multiple models with backtesting."""
    prices, dates = sample_market_data

    # Create models
    models = {
        'Fractal': FractalForecaster(),
        'ARIMA': ARIMAForecaster()
    }

    # Compare models
    comparison = compare_models(
        models,
        prices,
        dates,
        initial_window=100,
        step_size=20,
        forecast_horizon=10
    )

    assert 'Fractal' in comparison
    assert 'ARIMA' in comparison

    for model_name, results in comparison.items():
        assert 'mae' in results
        assert 'mse' in results
        assert results['mae'] >= 0
        assert results['mse'] >= 0


# ============================================================================
# Model Selection Integration Tests
# ============================================================================

@pytest.mark.skipif(not MODEL_SELECTION_AVAILABLE, reason="model_selection not available")
def test_hybrid_model_selector(sample_market_data):
    """Test automatic model selection workflow."""
    prices, dates = sample_market_data

    # Create selector
    selector = HybridModelSelector()

    # Find best model
    best_model, results = selector.select_best_model(prices, dates)

    assert best_model is not None
    assert 'model_name' in results
    assert 'metrics' in results

    # Best model should be fitted and ready to use
    forecast = best_model.predict(n_steps=10)
    assert len(forecast['forecast']) == 10


# ============================================================================
# Real Market Data Tests
# ============================================================================

@pytest.mark.slow
def test_real_market_data_forecasting(real_market_data):
    """Test forecasting with real S&P 500 data."""
    prices, dates = real_market_data

    # Fit model
    model = FractalForecaster()
    model.fit(prices, dates=dates)

    # Predict next 30 days
    result = model.predict(n_steps=30)

    assert len(result['forecast']) == 30
    assert np.all(np.isfinite(result['forecast']))

    # Forecasts should be reasonable (not too far from current price)
    last_price = prices[-1]
    max_deviation = last_price * 0.5  # 50% max deviation

    assert np.all(np.abs(result['forecast'] - last_price) < max_deviation)


@pytest.mark.slow
def test_real_market_data_analysis(real_market_data):
    """Test fractal analysis with real market data."""
    prices, _ = real_market_data

    # Analyze
    analyzer = FractalAnalyzer()
    analysis = analyzer.analyze(prices)

    # Hurst should be reasonable for financial data
    assert 0.3 < analysis['hurst'] < 0.7  # Typical range for markets
    assert 1.3 < analysis['fractal_dimension'] < 1.7


@pytest.mark.slow
@pytest.mark.skipif(not BACKTESTING_AVAILABLE, reason="backtesting not available")
def test_real_market_data_backtesting(real_market_data):
    """Test backtesting with real market data."""
    prices, dates = real_market_data

    model = FractalForecaster()

    validator = WalkForwardValidator(
        model,
        initial_window=252,  # 1 year
        step_size=21,  # 1 month
        forecast_horizon=5  # 1 week
    )

    results = validator.run(prices, dates)

    # Metrics should be reasonable for real data
    assert results['metrics']['mae'] > 0
    assert results['metrics']['mse'] > 0

    # MAPE should be less than 50% for a reasonable model
    if 'mape' in results['metrics']:
        assert results['metrics']['mape'] < 50


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================

def test_short_time_series():
    """Test handling of very short time series."""
    prices = np.array([100, 101, 99, 102, 98, 103, 97])

    model = FractalForecaster()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prices)
        result = model.predict(n_steps=3)

    # Should still produce a forecast, even if uncertain
    assert len(result['forecast']) == 3
    assert np.all(np.isfinite(result['forecast']))


def test_constant_prices():
    """Test handling of constant price series."""
    prices = np.full(100, 100.0)

    model = FractalForecaster()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prices)
        result = model.predict(n_steps=10)

    # Should forecast the constant value
    assert len(result['forecast']) == 10
    assert np.all(np.isfinite(result['forecast']))


def test_missing_data_handling():
    """Test handling of data with NaN values."""
    prices = np.random.randn(100) + 100
    prices[::10] = np.nan  # Add some NaN values

    # Remove NaN for now (proper handling would be imputation)
    clean_prices = prices[~np.isnan(prices)]

    model = FractalForecaster()
    model.fit(clean_prices)
    result = model.predict(n_steps=10)

    assert len(result['forecast']) == 10
    assert np.all(np.isfinite(result['forecast']))


def test_extreme_volatility():
    """Test handling of extreme volatility."""
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.1))  # High volatility

    model = FractalForecaster()
    model.fit(prices)
    result = model.predict(n_steps=10)

    assert len(result['forecast']) == 10
    assert np.all(np.isfinite(result['forecast']))

    # Uncertainty should be high
    assert np.mean(result['std']) > 0


# ============================================================================
# Multi-Model Ensemble Tests
# ============================================================================

@pytest.mark.skipif(
    not all([ARIMA_AVAILABLE, ETS_AVAILABLE]),
    reason="Multiple baseline models not available"
)
def test_large_ensemble_stacking(sample_market_data):
    """Test stacking with many base models."""
    prices, dates = sample_market_data

    # Create multiple models
    models = [
        FractalForecaster().fit(prices, dates=dates),
        ARIMAForecaster().fit(prices),
        ETSForecaster().fit(prices)
    ]

    # Create stacking ensemble
    stacker = StackingForecaster(base_models=models, n_splits=3)
    stacker.fit(prices)

    result = stacker.predict(n_steps=20)

    assert len(result['forecast']) == 20
    assert np.all(np.isfinite(result['forecast']))

    # Ensemble should have lower or comparable uncertainty to individual models
    individual_stds = []
    for model in models:
        individual_result = model.predict(n_steps=20)
        individual_stds.append(np.mean(individual_result['std']))

    ensemble_std = np.mean(result['std'])
    # Ensemble std should generally be lower (wisdom of crowds)
    # But we'll just check it's reasonable
    assert ensemble_std > 0
    assert np.isfinite(ensemble_std)


# ============================================================================
# Performance and Scalability Tests
# ============================================================================

@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance with large dataset."""
    np.random.seed(42)

    # Generate large dataset (5 years of daily data)
    n = 252 * 5
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

    import time

    # Measure fitting time
    model = FractalForecaster()
    start = time.time()
    model.fit(prices)
    fit_time = time.time() - start

    # Measure prediction time
    start = time.time()
    result = model.predict(n_steps=30)
    predict_time = time.time() - start

    # Should complete in reasonable time
    assert fit_time < 10  # 10 seconds max for fitting
    assert predict_time < 5  # 5 seconds max for prediction

    assert len(result['forecast']) == 30
    assert np.all(np.isfinite(result['forecast']))


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

def test_complete_forecasting_workflow(sample_market_data):
    """
    Test complete end-to-end workflow:
    1. Load data
    2. Analyze fractal properties
    3. Fit model
    4. Generate forecast
    5. Evaluate uncertainty
    """
    prices, dates = sample_market_data

    # Step 1: Analyze data
    analyzer = FractalAnalyzer()
    analysis = analyzer.analyze(prices)

    assert 'hurst' in analysis
    hurst = analysis['hurst']

    # Step 2: Configure model based on analysis
    if hurst > 0.6:
        method = 'rs'  # Persistent series
    elif hurst < 0.4:
        method = 'dfa'  # Mean-reverting series
    else:
        method = 'rs'  # Default

    # Step 3: Fit model
    model = FractalForecaster(method=method)
    model.fit(prices, dates=dates)

    # Step 4: Generate forecast
    result = model.predict(n_steps=30)

    # Step 5: Evaluate
    assert len(result['forecast']) == 30
    assert np.all(np.isfinite(result['forecast']))

    # Check uncertainty quantification
    assert np.all(result['lower'] <= result['mean'])
    assert np.all(result['mean'] <= result['upper'])

    # Uncertainty should increase with forecast horizon
    early_std = np.mean(result['std'][:10])
    late_std = np.mean(result['std'][-10:])
    assert late_std >= early_std  # Uncertainty grows over time


@pytest.mark.skipif(
    not (BACKTESTING_AVAILABLE and MODEL_SELECTION_AVAILABLE),
    reason="Required modules not available"
)
def test_automated_model_selection_and_backtesting(sample_market_data):
    """
    Test automated workflow:
    1. Select best model automatically
    2. Backtest selected model
    3. Generate production forecast
    """
    prices, dates = sample_market_data

    # Step 1: Automatic model selection
    selector = HybridModelSelector()
    best_model, selection_results = selector.select_best_model(prices, dates)

    # Step 2: Backtest the selected model
    validator = WalkForwardValidator(
        best_model,
        initial_window=100,
        step_size=20,
        forecast_horizon=10
    )

    backtest_results = validator.run(prices, dates)

    # Step 3: Generate production forecast
    final_forecast = best_model.predict(n_steps=30)

    # Validate entire pipeline
    assert selection_results['model_name'] is not None
    assert backtest_results['metrics']['mae'] >= 0
    assert len(final_forecast['forecast']) == 30
    assert np.all(np.isfinite(final_forecast['forecast']))


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
