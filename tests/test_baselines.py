"""
Quick tests for baseline forecasting models.

Tests that all baseline models can be imported, fitted, and used with
the backtesting framework.
"""

import numpy as np
import sys

def test_imports():
    """Test that all baseline models can be imported."""
    print("Testing baseline model imports...")

    try:
        from fractime.baselines import ARIMAForecaster, GARCHForecaster, ProphetForecaster
        print("  ✓ All baseline models imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("  Install with: uv pip install -e '.[baselines]'")
        return False


def test_arima_basic():
    """Test ARIMA basic functionality."""
    print("\nTesting ARIMA model...")
    from fractime.baselines import ARIMAForecaster

    # Generate simple test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(100)])

    # Fit and predict
    model = ARIMAForecaster(max_p=2, max_q=2, stepwise=True)
    model.fit(prices, dates)
    result = model.predict(n_steps=5)

    # Check result structure
    assert 'forecast' in result, "Result should contain 'forecast'"
    assert 'lower' in result, "Result should contain 'lower'"
    assert 'upper' in result, "Result should contain 'upper'"
    assert len(result['forecast']) == 5, "Should forecast 5 steps"

    print(f"  Forecast: {result['forecast'][:2]}...")
    print(f"  Model: {model}")
    print("  ✓ ARIMA working correctly")
    return True


def test_garch_basic():
    """Test GARCH basic functionality."""
    print("\nTesting GARCH model...")
    from fractime.baselines import GARCHForecaster

    # Generate test data with volatility clustering
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(100)])

    # Fit and predict
    model = GARCHForecaster(p=1, q=1)
    model.fit(prices, dates)
    result = model.predict(n_steps=5, n_paths=100)

    # Check result structure
    assert 'forecast' in result, "Result should contain 'forecast'"
    assert 'lower' in result, "Result should contain 'lower'"
    assert 'upper' in result, "Result should contain 'upper'"
    assert len(result['forecast']) == 5, "Should forecast 5 steps"

    print(f"  Forecast: {result['forecast'][:2]}...")
    print(f"  Model: {model}")
    print("  ✓ GARCH working correctly")
    return True


def test_prophet_basic():
    """Test Prophet basic functionality."""
    print("\nTesting Prophet model...")
    from fractime.baselines import ProphetForecaster

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(100)])

    # Fit and predict
    model = ProphetForecaster(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prices, dates)
    result = model.predict(n_steps=5)

    # Check result structure
    assert 'forecast' in result, "Result should contain 'forecast'"
    assert 'lower' in result, "Result should contain 'lower'"
    assert 'upper' in result, "Result should contain 'upper'"
    assert len(result['forecast']) == 5, "Should forecast 5 steps"

    print(f"  Forecast: {result['forecast'][:2]}...")
    print(f"  Model: {model}")
    print("  ✓ Prophet working correctly")
    return True


def test_with_backtesting():
    """Test baseline models with backtesting framework."""
    print("\nTesting baselines with WalkForwardValidator...")
    from fractime.baselines import ARIMAForecaster
    from fractime.backtesting import WalkForwardValidator

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(200)])

    # Run validation
    model = ARIMAForecaster(max_p=2, max_q=2, stepwise=True)
    validator = WalkForwardValidator(
        model=model,
        initial_window=150,
        step_size=10,
        forecast_horizon=1,
        verbose=False
    )

    results = validator.run(prices, dates)

    # Check results
    assert 'forecasts' in results, "Results should contain 'forecasts'"
    assert 'metrics' in results, "Results should contain 'metrics'"
    assert len(results['forecasts']) > 0, "Should have some forecasts"

    print(f"  Generated {len(results['forecasts'])} forecasts")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  Direction Accuracy: {results['metrics']['direction_accuracy']:.2%}")
    print("  ✓ Backtesting integration working")
    return True


def test_model_comparison():
    """Test comparing baseline models."""
    print("\nTesting model comparison...")
    from fractime.baselines import ARIMAForecaster, GARCHForecaster
    from fractime.backtesting import WalkForwardValidator, compare_models
    import fractime as ft

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(200)])

    # Run validation for multiple models
    models = {
        'Fractal': ft.FractalForecaster(),
        'ARIMA': ARIMAForecaster(max_p=2, max_q=2, stepwise=True),
        'GARCH': GARCHForecaster(p=1, q=1),
    }

    results = {}
    for name, model in models.items():
        validator = WalkForwardValidator(
            model=model,
            initial_window=150,
            step_size=25,
            forecast_horizon=1,
            verbose=False
        )
        results[name] = validator.run(prices, dates)

    # Compare models
    comparison = compare_models(results)

    assert 'best_model' in comparison, "Comparison should have best_model"
    assert comparison['best_model'] in models.keys(), "Best model should be one of the tested models"

    print(f"  Best model: {comparison['best_model']}")
    print(f"  Number of models compared: {len(results)}")
    print("  ✓ Model comparison working")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("BASELINE MODELS - UNIT TESTS")
    print("=" * 80)

    tests = [
        test_imports,
        test_arima_basic,
        test_garch_basic,
        test_prophet_basic,
        test_with_backtesting,
        test_model_comparison,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
