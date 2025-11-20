"""
Quick unit tests for backtesting framework.

Tests basic functionality of metrics, validator, and scorer.
"""

import numpy as np
from datetime import datetime
import sys

def test_metrics_import():
    """Test that all metrics can be imported."""
    print("Testing metrics import...")
    from fractime.backtesting import (
        ForecastMetrics,
        compute_rmse,
        compute_mae,
        compute_mape,
        compute_direction_accuracy,
        compute_coverage,
        compute_crps
    )
    print("  ✓ All metrics imported successfully")
    return True


def test_metrics_computation():
    """Test basic metrics computation."""
    print("\nTesting metrics computation...")
    from fractime.backtesting import compute_rmse, compute_mae, compute_direction_accuracy

    # Simple test data
    forecasts = np.array([100, 101, 102, 103, 104])
    actuals = np.array([100.5, 101.2, 101.8, 103.1, 104.3])
    current_prices = np.array([99, 100, 101, 102, 103])

    # Compute metrics
    rmse = compute_rmse(forecasts, actuals)
    mae = compute_mae(forecasts, actuals)
    direction_acc = compute_direction_accuracy(forecasts, actuals, current_prices)

    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Direction Accuracy: {direction_acc:.2%}")

    # Basic sanity checks
    assert rmse > 0, "RMSE should be positive"
    assert mae > 0, "MAE should be positive"
    assert 0 <= direction_acc <= 1, "Direction accuracy should be between 0 and 1"

    print("  ✓ Metrics computation working correctly")
    return True


def test_validator_import():
    """Test that validator can be imported."""
    print("\nTesting validator import...")
    from fractime.backtesting import WalkForwardValidator
    print("  ✓ WalkForwardValidator imported successfully")
    return True


def test_validator_basic():
    """Test basic validator functionality."""
    print("\nTesting validator basic functionality...")
    from fractime.backtesting import WalkForwardValidator
    import fractime as ft

    # Generate simple test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(300)])

    # Create model and validator
    model = ft.FractalForecaster()
    validator = WalkForwardValidator(
        model=model,
        initial_window=200,
        step_size=20,
        forecast_horizon=1,
        window_type='expanding',
        verbose=False
    )

    # Run validation
    results = validator.run(prices, dates)

    # Check results structure
    assert 'forecasts' in results, "Results should contain 'forecasts'"
    assert 'actuals' in results, "Results should contain 'actuals'"
    assert 'metrics' in results, "Results should contain 'metrics'"
    assert len(results['forecasts']) > 0, "Should have some forecasts"

    print(f"  Generated {len(results['forecasts'])} forecasts")
    print(f"  Metrics: {list(results['metrics'].keys())}")
    print("  ✓ Validator working correctly")
    return True


def test_scorer_import():
    """Test that scorer can be imported."""
    print("\nTesting scorer import...")
    from fractime.backtesting import DualPenaltyScorer, compare_models
    print("  ✓ DualPenaltyScorer and compare_models imported successfully")
    return True


def test_scorer_basic():
    """Test basic scorer functionality."""
    print("\nTesting scorer basic functionality...")
    from fractime.backtesting import DualPenaltyScorer, WalkForwardValidator
    import fractime as ft

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(300)])

    # Run validation
    model = ft.FractalForecaster()
    validator = WalkForwardValidator(
        model=model,
        initial_window=200,
        step_size=20,
        forecast_horizon=1,
        verbose=False
    )
    results = validator.run(prices, dates)

    # Score results
    scorer = DualPenaltyScorer(alpha=1.0, beta=0.5)
    score = scorer.score(results)

    # Check score structure
    assert 'total_score' in score, "Score should contain 'total_score'"
    assert 'accuracy_score' in score, "Score should contain 'accuracy_score'"
    assert 'overfitting_penalty' in score, "Score should contain 'overfitting_penalty'"
    assert 'components' in score, "Score should contain 'components'"

    print(f"  Total Score: {score['total_score']:.4f}")
    print(f"  Accuracy: {score['accuracy_score']:.4f}")
    print(f"  Overfitting: {score['overfitting_penalty']:.4f}")
    print("  ✓ Scorer working correctly")
    return True


def test_model_comparison():
    """Test model comparison functionality."""
    print("\nTesting model comparison...")
    from fractime.backtesting import compare_models, WalkForwardValidator, DualPenaltyScorer
    import fractime as ft

    # Generate test data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)
    dates = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(300)])

    # Run validation for two "models" (same model, different configs)
    results = {}
    for name in ['model1', 'model2']:
        model = ft.FractalForecaster()
        validator = WalkForwardValidator(
            model=model,
            initial_window=200,
            step_size=20,
            forecast_horizon=1,
            verbose=False
        )
        results[name] = validator.run(prices, dates)

    # Compare models
    comparison = compare_models(results)

    # Check comparison structure
    assert 'rankings' in comparison, "Comparison should contain 'rankings'"
    assert 'scores' in comparison, "Comparison should contain 'scores'"
    assert 'best_model' in comparison, "Comparison should contain 'best_model'"
    assert 'comparison_table' in comparison, "Comparison should contain 'comparison_table'"

    print(f"  Best model: {comparison['best_model']}")
    print("  ✓ Model comparison working correctly")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("BACKTESTING FRAMEWORK - UNIT TESTS")
    print("=" * 80)

    tests = [
        test_metrics_import,
        test_metrics_computation,
        test_validator_import,
        test_validator_basic,
        test_scorer_import,
        test_scorer_basic,
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
