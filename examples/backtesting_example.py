"""
Walk-Forward Backtesting Example

Demonstrates the core principle: Fitting = Backtesting

This example shows how to:
1. Run walk-forward validation with classical and Bayesian models
2. Compare models using dual penalty scoring
3. Interpret accuracy vs overfitting tradeoffs
4. Visualize results

The key insight: Every fit is a backtest step, providing continuous validation.
"""

import numpy as np
import polars as pl
from datetime import datetime, timedelta

# Core fractime imports
import fractime as ft
from fractime.backtesting import WalkForwardValidator, DualPenaltyScorer, compare_models

# Check if Bayesian is available
try:
    from fractime import BayesianFractalForecaster
    BAYESIAN_AVAILABLE = BayesianFractalForecaster is not None
    if not BAYESIAN_AVAILABLE:
        print("Warning: PyMC not installed, Bayesian examples will be skipped")
        print("Install with: uv pip install -e '.[bayesian]'")
except ImportError:
    BAYESIAN_AVAILABLE = False
    BayesianFractalForecaster = None
    print("Warning: PyMC not installed, Bayesian examples will be skipped")
    print("Install with: uv pip install -e '.[bayesian]'")


def generate_sample_data(n_points: int = 500, hurst: float = 0.6) -> tuple:
    """
    Generate synthetic fractal time series for testing.

    Args:
        n_points: Number of data points
        hurst: True Hurst exponent (for validation)

    Returns:
        (prices, dates) tuple
    """
    # Generate simple fractional Brownian motion for testing
    # Using basic fBm formula: dS = μ dt + σ dW^H
    np.random.seed(42)

    initial_price = 100.0
    dt = 1.0
    sigma = 0.02  # 2% daily volatility

    # Generate increments with long-range dependence
    increments = np.random.randn(n_points)

    # Apply autocorrelation structure based on Hurst
    # For simplicity, use a basic AR(1) approximation
    phi = 2 * hurst - 1  # Correlation coefficient
    for i in range(1, n_points):
        increments[i] = phi * increments[i-1] + np.sqrt(1 - phi**2) * increments[i]

    # Convert to price series
    log_returns = sigma * increments * np.sqrt(dt)
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    # Generate dates as numpy datetime64 array (compatible with Polars)
    start_date = np.datetime64('2020-01-01')
    dates = np.array([start_date + np.timedelta64(i, 'D') for i in range(n_points)])

    return prices, dates


def example_1_basic_validation():
    """
    Example 1: Basic walk-forward validation with classical model.

    This demonstrates the simplest use case: validate a single model.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Walk-Forward Validation")
    print("=" * 80)

    # Generate data
    prices, dates = generate_sample_data(n_points=500, hurst=0.6)

    # Create classical forecaster
    model = ft.FractalForecaster()

    # Create validator
    validator = WalkForwardValidator(
        model=model,
        initial_window=252,      # 1 year of daily data
        step_size=5,             # Refit every 5 steps (faster for demo)
        forecast_horizon=1,      # 1-step-ahead forecast
        window_type='expanding', # Use all historical data
        verbose=True
    )

    # Run validation
    results = validator.run(prices, dates)

    # Results contain:
    print("\nValidation Results:")
    print(f"  Number of forecasts: {len(results['forecasts'])}")
    print(f"  Metrics available: {list(results['metrics'].keys())}")

    # Plot results (skip in headless mode)
    # validator.plot_results()

    return results


def example_2_compare_models():
    """
    Example 2: Compare multiple models using dual penalty scoring.

    This demonstrates model selection: which model balances accuracy vs overfitting?
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Model Comparison with Dual Penalty Scoring")
    print("=" * 80)

    # Generate data
    prices, dates = generate_sample_data(n_points=500, hurst=0.6)

    # Models to compare
    models = {
        'classical_fractal': ft.FractalForecaster(),
    }

    # Add Bayesian if available
    if BAYESIAN_AVAILABLE:
        models['bayesian_fast'] = BayesianFractalForecaster(
            mode='fast',
            n_samples=500
        )
        models['bayesian_hybrid'] = BayesianFractalForecaster(
            mode='hybrid',
            n_samples=500
        )

    # Run validation for each model
    results = {}
    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Validating: {model_name}")
        print(f"{'=' * 60}")

        validator = WalkForwardValidator(
            model=model,
            initial_window=252,
            step_size=10,  # Faster for demo
            forecast_horizon=1,
            window_type='expanding',
            verbose=False  # Quiet mode for cleaner output
        )

        results[model_name] = validator.run(prices, dates)

        # Print metrics for this model
        metrics = results[model_name]['metrics']
        print(f"\nMetrics for {model_name}:")
        print(f"  RMSE:                 {metrics['rmse']:.4f}")
        print(f"  Direction Accuracy:   {metrics['direction_accuracy']:.2%}")
        if 'coverage' in metrics:
            print(f"  95% CI Coverage:      {metrics['coverage']:.2%}")

    # Compare using dual penalty scoring
    print("\n" + "=" * 80)
    print("DUAL PENALTY COMPARISON")
    print("=" * 80)

    scorer = DualPenaltyScorer(alpha=1.0, beta=0.5)
    comparison = compare_models(results, scorer)

    # Print comparison table
    print(comparison['comparison_table'])

    # Print best model details
    best_model = comparison['best_model']
    best_score = comparison['scores'][best_model]

    print("\nBest Model Analysis:")
    print(f"  Winner: {best_model}")
    print(f"  Total Score: {best_score['total_score']:.4f}")
    print(f"\n  Accuracy Components:")
    for component, value in best_score['components']['accuracy'].items():
        print(f"    {component}: {value:.4f}")
    print(f"\n  Overfitting Components:")
    for component, value in best_score['components']['overfitting'].items():
        print(f"    {component}: {value:.4f}")

    return comparison


def example_3_rolling_window():
    """
    Example 3: Rolling window validation for non-stationary series.

    When data regime changes, rolling window can be more appropriate than expanding.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Rolling Window Validation")
    print("=" * 80)

    # Generate data with regime change
    prices1, dates1 = generate_sample_data(n_points=250, hurst=0.4)  # Mean-reverting
    prices2, dates2 = generate_sample_data(n_points=250, hurst=0.7)  # Trending

    # Concatenate (simulates regime shift)
    prices = np.concatenate([prices1, prices2[1:]]) # Avoid duplicate at junction
    dates = dates1 + dates2[1:]

    print(f"\nData characteristics:")
    print(f"  Total points: {len(prices)}")
    print(f"  Regime 1 (0-250): H=0.4 (mean-reverting)")
    print(f"  Regime 2 (250-500): H=0.7 (trending)")

    # Compare expanding vs rolling
    results = {}

    for window_type in ['expanding', 'rolling']:
        print(f"\n{'=' * 60}")
        print(f"Validating with {window_type} window")
        print(f"{'=' * 60}")

        model = ft.FractalForecaster()
        validator = WalkForwardValidator(
            model=model,
            initial_window=200,
            step_size=10,
            forecast_horizon=1,
            window_type=window_type,
            rolling_window_size=150,  # Used only if window_type='rolling'
            verbose=False
        )

        results[window_type] = validator.run(prices, dates)

        metrics = results[window_type]['metrics']
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")

    # Compare using dual penalty
    scorer = DualPenaltyScorer(alpha=1.0, beta=0.5)
    comparison = compare_models(results, scorer)
    print("\n" + comparison['comparison_table'])

    print("\nInsight:")
    print("  For non-stationary data with regime changes, rolling windows")
    print("  often perform better by adapting to recent regime.")

    return comparison


def example_4_parameter_tracking():
    """
    Example 4: Track parameter evolution for overfitting detection.

    Stable parameters = good generalization.
    Volatile parameters = overfitting to noise.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Parameter Stability Analysis")
    print("=" * 80)

    # Generate data
    prices, dates = generate_sample_data(n_points=500, hurst=0.55)

    # Run validation
    model = ft.FractalForecaster()
    validator = WalkForwardValidator(
        model=model,
        initial_window=252,
        step_size=5,
        forecast_horizon=1,
        window_type='expanding',
        verbose=False
    )

    results = validator.run(prices, dates)

    # Analyze parameter stability
    param_history = results['parameter_history']

    if param_history:
        hurst_values = [p['hurst'] for p in param_history]
        times = [p['time'] for p in param_history]

        print(f"\nParameter Evolution:")
        print(f"  Number of estimates: {len(hurst_values)}")
        print(f"  Mean Hurst: {np.mean(hurst_values):.4f}")
        print(f"  Std Dev: {np.std(hurst_values):.4f}")
        print(f"  Range: [{np.min(hurst_values):.4f}, {np.max(hurst_values):.4f}]")

        # Compute stability metric
        stability = validator.get_parameter_stability(window=20)
        print(f"\nParameter Stability (lower is better): {stability:.4f}")

        if stability < 0.01:
            assessment = "Excellent - parameters very stable"
        elif stability < 0.05:
            assessment = "Good - parameters reasonably stable"
        elif stability < 0.10:
            assessment = "Moderate - some parameter drift"
        else:
            assessment = "Poor - parameters highly volatile (possible overfitting)"

        print(f"  Assessment: {assessment}")

    # Plot to visualize parameter evolution (skip in headless mode)
    # validator.plot_results()

    return results


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "=" * 80)
    print("FRACTIME BACKTESTING FRAMEWORK - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the core principle:")
    print("  FITTING = BACKTESTING")
    print("\nEvery model fit is a validation step in an expanding framework.")
    print("This provides continuous, rigorous evaluation.")

    # Example 1: Basic validation
    results_1 = example_1_basic_validation()

    # Example 2: Model comparison
    results_2 = example_2_compare_models()

    # Example 3: Rolling windows
    results_3 = example_3_rolling_window()

    # Example 4: Parameter stability
    results_4 = example_4_parameter_tracking()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Walk-forward validation provides rigorous out-of-sample testing")
    print("  2. Dual penalty scoring balances accuracy vs overfitting")
    print("  3. Parameter stability is a key indicator of generalization")
    print("  4. Rolling windows adapt better to non-stationary regimes")
    print("  5. Bayesian models provide uncertainty quantification")

    return {
        'basic': results_1,
        'comparison': results_2,
        'rolling': results_3,
        'stability': results_4
    }


if __name__ == '__main__':
    # Run individual examples or all
    import sys

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_basic_validation()
        elif example_num == '2':
            example_2_compare_models()
        elif example_num == '3':
            example_3_rolling_window()
        elif example_num == '4':
            example_4_parameter_tracking()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python backtesting_example.py [1|2|3|4]")
    else:
        # Run all examples
        run_all_examples()
