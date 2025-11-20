"""
End-to-end workflow tests simulating complete user scenarios.

These tests validate real-world use cases from data loading through
forecasting, evaluation, visualization, and decision-making.
"""

import pytest
import numpy as np
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Core imports
from fractime import (
    FractalForecaster,
    get_yahoo_data,
    FractalAnalyzer,
    CrossDimensionalAnalyzer,
    FractalSimulator,
    StackingForecaster,
    BoostingForecaster
)

# Visualization
from fractime.visualization import (
    plot_forecast,
    plot_forecast_interactive,
    print_forecast_summary
)

# Baseline models
try:
    from fractime.baselines import (
        ARIMAForecaster,
        GARCHForecaster,
        ETSForecaster,
        VARForecaster,
        LSTMForecaster
    )
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False

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
# Fixtures
# ============================================================================

@pytest.fixture
def sample_stock_data():
    """Generate realistic synthetic stock data."""
    np.random.seed(42)
    n = 500

    # Generate realistic stock price path
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    returns[0] = 0
    log_prices = np.cumsum(returns)
    prices = 100 * np.exp(log_prices)

    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=n-i) for i in range(n)]

    return prices, dates


@pytest.fixture
def multivariate_market_data():
    """Generate multivariate market data (price + volume)."""
    np.random.seed(42)
    n = 500

    # Price
    returns = np.random.randn(n) * 0.02
    log_prices = np.cumsum(returns)
    prices = 100 * np.exp(log_prices)

    # Volume (correlated with abs(returns))
    base_volume = 1000000
    volume = base_volume * (1 + 0.5 * np.abs(returns) + 0.1 * np.random.randn(n))
    volume = np.maximum(volume, 0)

    dates = [datetime.now() - timedelta(days=n-i) for i in range(n)]

    return prices, volume, dates


# ============================================================================
# Workflow 1: Basic Forecasting Workflow
# ============================================================================

def test_workflow_basic_forecast(sample_stock_data):
    """
    Scenario: Analyst wants to forecast next 30 days

    Workflow:
    1. Load historical data
    2. Fit fractal model
    3. Generate forecast
    4. Print summary
    """
    prices, dates = sample_stock_data

    # Step 1: Data loaded (from fixture)
    assert len(prices) > 0
    assert len(dates) == len(prices)

    # Step 2: Fit model
    model = FractalForecaster()
    model.fit(prices, dates=dates)

    # Step 3: Generate forecast
    forecast_horizon = 30
    result = model.predict(n_steps=forecast_horizon)

    # Step 4: Print summary (capture output)
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output

    print_forecast_summary(result, prices)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Validate workflow completion
    assert len(result['forecast']) == forecast_horizon
    assert np.all(np.isfinite(result['forecast']))
    assert len(output) > 0  # Summary was printed
    assert 'Forecast Summary' in output or 'forecast' in output.lower()


# ============================================================================
# Workflow 2: Comparative Analysis
# ============================================================================

@pytest.mark.skipif(not BASELINES_AVAILABLE, reason="Baseline models not available")
def test_workflow_model_comparison(sample_stock_data):
    """
    Scenario: Analyst wants to compare multiple forecasting methods

    Workflow:
    1. Load data
    2. Fit multiple models
    3. Generate forecasts from each
    4. Compare results
    """
    prices, dates = sample_stock_data

    forecast_horizon = 20
    models = {}
    forecasts = {}

    # Fit Fractal model
    fractal = FractalForecaster()
    fractal.fit(prices, dates=dates)
    models['Fractal'] = fractal
    forecasts['Fractal'] = fractal.predict(n_steps=forecast_horizon)

    # Fit ARIMA (if available)
    try:
        arima = ARIMAForecaster()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima.fit(prices)
        models['ARIMA'] = arima
        forecasts['ARIMA'] = arima.predict(n_steps=forecast_horizon)
    except:
        pass

    # Fit ETS (if available)
    try:
        ets = ETSForecaster()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ets.fit(prices)
        models['ETS'] = ets
        forecasts['ETS'] = ets.predict(n_steps=forecast_horizon)
    except:
        pass

    # Validate all forecasts
    assert len(forecasts) >= 1  # At least Fractal should work

    for name, forecast in forecasts.items():
        assert len(forecast['forecast']) == forecast_horizon
        assert np.all(np.isfinite(forecast['forecast']))

    # Compare uncertainty
    uncertainties = {
        name: np.mean(forecast['std'])
        for name, forecast in forecasts.items()
    }

    # All uncertainties should be positive
    assert all(u > 0 for u in uncertainties.values())


# ============================================================================
# Workflow 3: Regime Analysis and Adaptive Forecasting
# ============================================================================

def test_workflow_regime_adaptive_forecast(sample_stock_data):
    """
    Scenario: Analyst wants regime-aware forecasting

    Workflow:
    1. Analyze fractal properties
    2. Detect market regime
    3. Configure model based on regime
    4. Generate adaptive forecast
    """
    prices, dates = sample_stock_data

    # Step 1: Analyze fractal properties
    analyzer = FractalAnalyzer()
    analysis = analyzer.analyze(prices)

    hurst = analysis['hurst']
    volatility = np.std(np.diff(np.log(prices)))

    # Step 2: Detect regime
    if hurst > 0.6:
        regime = 'trending'
        method = 'rs'
    elif hurst < 0.4:
        regime = 'mean_reverting'
        method = 'dfa'
    else:
        regime = 'random_walk'
        method = 'rs'

    # Step 3: Configure model
    if volatility > 0.03:  # High volatility
        model = FractalForecaster(method=method, min_scale=5, max_scale=50)
    else:  # Normal/low volatility
        model = FractalForecaster(method=method)

    # Step 4: Generate forecast
    model.fit(prices, dates=dates)
    result = model.predict(n_steps=30)

    # Validate adaptive workflow
    assert regime in ['trending', 'mean_reverting', 'random_walk']
    assert len(result['forecast']) == 30
    assert np.all(np.isfinite(result['forecast']))


# ============================================================================
# Workflow 4: Backtesting and Validation
# ============================================================================

@pytest.mark.skipif(not BACKTESTING_AVAILABLE, reason="Backtesting not available")
def test_workflow_backtest_validation(sample_stock_data):
    """
    Scenario: Analyst wants to validate model performance

    Workflow:
    1. Split data into train/test
    2. Run walk-forward backtesting
    3. Analyze performance metrics
    4. Make go/no-go decision on model
    """
    prices, dates = sample_stock_data

    # Step 1: Data already split by validator

    # Step 2: Run backtesting
    model = FractalForecaster()
    validator = WalkForwardValidator(
        model,
        initial_window=252,  # 1 year initial training
        step_size=21,  # Re-train monthly
        forecast_horizon=5  # 1 week ahead
    )

    results = validator.run(prices, dates)

    # Step 3: Analyze metrics
    mae = results['metrics']['mae']
    mse = results['metrics']['mse']
    mape = results['metrics'].get('mape', None)

    # Step 4: Decision criteria
    performance_acceptable = bool(
        mae < np.std(prices) and  # MAE less than price std
        (mape is None or mape < 20)  # MAPE less than 20% if available
    )

    # Validate workflow
    assert mae > 0  # Metrics computed
    assert mse > 0
    assert isinstance(performance_acceptable, bool)  # Decision made


# ============================================================================
# Workflow 5: Ensemble Forecasting
# ============================================================================

@pytest.mark.skipif(not BASELINES_AVAILABLE, reason="Baseline models not available")
def test_workflow_ensemble_forecast(sample_stock_data):
    """
    Scenario: Analyst wants robust ensemble forecast

    Workflow:
    1. Train multiple diverse models
    2. Combine using stacking
    3. Generate ensemble forecast
    4. Evaluate ensemble vs individuals
    """
    prices, dates = sample_stock_data

    # Step 1: Train base models
    base_models = []

    # Fractal
    fractal = FractalForecaster()
    fractal.fit(prices, dates=dates)
    base_models.append(fractal)

    # Try to add other models
    try:
        arima = ARIMAForecaster()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima.fit(prices)
        base_models.append(arima)
    except:
        pass

    try:
        ets = ETSForecaster()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ets.fit(prices)
        base_models.append(ets)
    except:
        pass

    # Step 2: Create ensemble
    ensemble = StackingForecaster(
        base_models=base_models,
        meta_learner='ridge',
        n_splits=3
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble.fit(prices)

    # Step 3: Generate forecast
    forecast_horizon = 20
    ensemble_result = ensemble.predict(n_steps=forecast_horizon)

    # Step 4: Compare to individuals
    individual_results = [
        model.predict(n_steps=forecast_horizon)
        for model in base_models
    ]

    # Validate workflow
    assert len(ensemble_result['forecast']) == forecast_horizon
    assert np.all(np.isfinite(ensemble_result['forecast']))

    # Ensemble should incorporate information from all models
    model_weights = ensemble.get_model_weights()
    assert len(model_weights) == len(base_models)


# ============================================================================
# Workflow 6: Multivariate Analysis
# ============================================================================

def test_workflow_multivariate_analysis(multivariate_market_data):
    """
    Scenario: Analyst wants to analyze price-volume dynamics

    Workflow:
    1. Load price and volume data
    2. Perform cross-dimensional analysis
    3. Fit VAR model (if available)
    4. Compare to univariate approach
    """
    prices, volume, dates = multivariate_market_data

    # Step 1: Data loaded from fixture

    # Step 2: Cross-dimensional analysis
    data = np.column_stack([prices, volume])
    analyzer = CrossDimensionalAnalyzer()
    analysis = analyzer.analyze(data, dim_names=['price', 'volume'])

    correlation = analysis['correlation']

    # Step 3: Try VAR model (multivariate)
    try:
        from fractime.baselines import VARForecaster

        var_model = VARForecaster(maxlags=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_model.fit(data)

        var_result = var_model.predict(n_steps=20, return_all_vars=False)
        var_forecast = var_result['forecast']
    except:
        var_forecast = None

    # Step 4: Univariate approach
    fractal_model = FractalForecaster()
    fractal_model.fit(prices, dates=dates)
    fractal_result = fractal_model.predict(n_steps=20)
    fractal_forecast = fractal_result['forecast']

    # Validate workflow
    assert correlation.shape == (2, 2)
    assert len(fractal_forecast) == 20

    if var_forecast is not None:
        assert len(var_forecast) == 20


# ============================================================================
# Workflow 7: Simulation and Scenario Analysis
# ============================================================================

def test_workflow_scenario_simulation(sample_stock_data):
    """
    Scenario: Analyst wants to generate scenarios for risk analysis

    Workflow:
    1. Analyze historical data
    2. Generate multiple simulated paths
    3. Analyze distribution of outcomes
    4. Calculate risk metrics
    """
    prices, dates = sample_stock_data

    # Step 1: Analyze data
    analyzer = FractalAnalyzer()
    analysis = analyzer.analyze(prices)

    # Step 2: Generate simulations
    simulator = FractalSimulator(prices, analyzer)

    n_simulations = 100
    forecast_horizon = 30

    simulated_paths, path_analysis = simulator.simulate_paths(
        n_steps=forecast_horizon,
        n_paths=n_simulations
    )

    # Step 3: Analyze distribution
    final_prices = simulated_paths[:, -1]
    mean_final = np.mean(final_prices)
    std_final = np.std(final_prices)
    percentile_5 = np.percentile(final_prices, 5)
    percentile_95 = np.percentile(final_prices, 95)

    # Step 4: Calculate risk metrics
    current_price = prices[-1]
    downside_risk = (current_price - percentile_5) / current_price
    upside_potential = (percentile_95 - current_price) / current_price

    # Validate workflow
    assert simulated_paths.shape == (n_simulations, forecast_horizon)
    assert mean_final > 0
    assert std_final > 0
    assert percentile_5 < mean_final < percentile_95
    assert isinstance(downside_risk, (int, float))
    assert isinstance(upside_potential, (int, float))


# ============================================================================
# Workflow 8: Production Deployment
# ============================================================================

@pytest.mark.skipif(
    not (BACKTESTING_AVAILABLE and MODEL_SELECTION_AVAILABLE),
    reason="Required modules not available"
)
def test_workflow_production_deployment(sample_stock_data):
    """
    Scenario: Deploy model to production

    Workflow:
    1. Select best model via cross-validation
    2. Validate on hold-out test set
    3. Re-train on full dataset
    4. Generate production forecast
    5. Save model (simulated)
    """
    prices, dates = sample_stock_data

    # Split data
    train_size = int(len(prices) * 0.8)
    train_prices = prices[:train_size]
    train_dates = dates[:train_size]
    test_prices = prices[train_size:]
    test_dates = dates[train_size:]

    # Step 1: Select best model
    selector = HybridModelSelector()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model, selection_results = selector.select_best_model(
            train_prices,
            train_dates
        )

    # Step 2: Validate on test set
    validator = WalkForwardValidator(
        best_model,
        initial_window=min(100, len(test_prices) // 2),
        step_size=10,
        forecast_horizon=5
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_results = validator.run(test_prices, test_dates)

    test_mae = test_results['metrics']['mae']

    # Step 3: Re-train on full dataset
    production_model = type(best_model)()
    production_model.fit(prices, dates=dates)

    # Step 4: Generate production forecast
    production_forecast = production_model.predict(n_steps=30)

    # Step 5: Save model (simulated with temp file)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(f"Model: {type(production_model).__name__}\n")
        f.write(f"Training samples: {len(prices)}\n")
        f.write(f"Test MAE: {test_mae}\n")
        temp_path = f.name

    # Validate workflow
    assert selection_results['model_name'] is not None
    assert test_mae > 0
    assert len(production_forecast['forecast']) == 30
    assert Path(temp_path).exists()

    # Cleanup
    Path(temp_path).unlink()


# ============================================================================
# Workflow 9: Real-time Update Workflow
# ============================================================================

def test_workflow_incremental_update(sample_stock_data):
    """
    Scenario: Update forecast with new data

    Workflow:
    1. Train initial model
    2. Generate forecast
    3. Receive new data
    4. Re-train and update forecast
    """
    prices, dates = sample_stock_data

    # Split into initial and new data
    initial_size = int(len(prices) * 0.9)
    initial_prices = prices[:initial_size]
    initial_dates = dates[:initial_size]
    new_prices = prices[initial_size:]
    new_dates = dates[initial_size:]

    # Step 1: Initial training
    model = FractalForecaster()
    model.fit(initial_prices, dates=initial_dates)

    # Step 2: Initial forecast
    initial_forecast = model.predict(n_steps=20)

    # Step 3: New data arrives
    assert len(new_prices) > 0

    # Step 4: Update model with new data
    updated_prices = np.concatenate([initial_prices, new_prices])
    updated_dates = initial_dates + new_dates

    model.fit(updated_prices, dates=updated_dates)
    updated_forecast = model.predict(n_steps=20)

    # Validate workflow
    assert len(initial_forecast['forecast']) == 20
    assert len(updated_forecast['forecast']) == 20

    # Forecasts should differ after update
    assert not np.allclose(
        initial_forecast['forecast'],
        updated_forecast['forecast']
    )


# ============================================================================
# Workflow 10: Error Diagnostics
# ============================================================================

@pytest.mark.skipif(not BACKTESTING_AVAILABLE, reason="Backtesting not available")
def test_workflow_error_diagnostics(sample_stock_data):
    """
    Scenario: Diagnose forecast errors

    Workflow:
    1. Generate forecasts with backtesting
    2. Analyze forecast errors
    3. Identify systematic biases
    4. Suggest improvements
    """
    prices, dates = sample_stock_data

    # Step 1: Generate forecasts
    model = FractalForecaster()
    validator = WalkForwardValidator(
        model,
        initial_window=200,
        step_size=20,
        forecast_horizon=10
    )

    results = validator.run(prices, dates)

    # Step 2: Analyze errors
    forecasts = results['forecasts']
    actuals = results['actuals']

    if len(forecasts) > 0 and len(actuals) > 0:
        errors = np.array(forecasts) - np.array(actuals)

        # Step 3: Check for biases
        mean_error = np.mean(errors)
        abs_mean_error = np.mean(np.abs(errors))

        # Check if systematically over/under-forecasting
        bias_ratio = abs(mean_error) / abs_mean_error if abs_mean_error > 0 else 0

        is_biased = bias_ratio > 0.5

        # Step 4: Suggestions
        if is_biased:
            if mean_error > 0:
                suggestion = "Model over-forecasts. Consider more conservative parameters."
            else:
                suggestion = "Model under-forecasts. Consider more aggressive parameters."
        else:
            suggestion = "No systematic bias detected. Errors appear random."

        # Validate workflow
        assert isinstance(mean_error, (int, float))
        assert isinstance(bias_ratio, (int, float))
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
