"""
Automated Model Selection Example

Demonstrates the model selection framework:
1. Model Registry - discover available models
2. Auto-Selection - automatically pick the best model
3. Statistical Testing - test significance of differences
4. Ensemble Methods - combine multiple models

This answers the question: "Which model should I use for my data?"
"""

import numpy as np
import sys

# Core imports
import fractime as ft
from fractime.selection import (
    ModelRegistry,
    AutoSelector,
    diebold_mariano_test,
    model_confidence_set,
    create_ensemble
)


def generate_test_data(n_points: int = 400, regime: str = 'trending') -> tuple:
    """Generate synthetic data for testing."""
    np.random.seed(42)

    initial_price = 100.0
    sigma = 0.02

    if regime == 'trending':
        hurst = 0.6
    elif regime == 'mean_reverting':
        hurst = 0.4
    else:
        hurst = 0.5

    phi = 2 * hurst - 1

    increments = np.random.randn(n_points)
    for i in range(1, n_points):
        increments[i] = phi * increments[i-1] + np.sqrt(1 - phi**2) * increments[i]

    log_returns = sigma * increments
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    start_date = np.datetime64('2020-01-01')
    dates = np.array([start_date + np.timedelta64(i, 'D') for i in range(n_points)])

    return prices, dates


def example_1_model_registry():
    """Example 1: Explore the model registry."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: MODEL REGISTRY")
    print("=" * 80)

    registry = ModelRegistry()
    registry.discover_models()

    # Print all available models
    registry.print_registry()

    # Get models by category
    print("\n" + "=" * 80)
    print("MODELS BY CATEGORY")
    print("=" * 80)

    for category in ['fractal', 'baseline', 'bayesian']:
        models = registry.get_by_category(category)
        if models:
            print(f"\n{category.upper()}: {', '.join(models)}")

    # Get models by characteristics
    print("\n" + "=" * 80)
    print("MODELS BY CHARACTERISTICS")
    print("=" * 80)

    probabilistic = registry.get_by_characteristics(['probabilistic'])
    print(f"\nProbabilistic models: {', '.join(probabilistic)}")

    fast = registry.get_by_characteristics(['fast'])
    print(f"Fast models: {', '.join(fast)}")

    return registry


def example_2_auto_selection():
    """Example 2: Automatically select best model."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: AUTOMATED MODEL SELECTION")
    print("=" * 80)

    # Generate test data
    prices, dates = generate_test_data(n_points=400, regime='trending')

    # Create auto-selector
    selector = AutoSelector(verbose=True)

    # Select best model
    result = selector.select_best(prices, dates)

    # Print results
    print("\n" + "=" * 80)
    print("SELECTION RESULTS")
    print("=" * 80)
    print(f"\nBest Model: {result.best_model_name}")
    print(f"Score: {result.best_score:.4f}")

    print("\nAll Model Scores:")
    for name, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {score:.4f}")

    # Use the best model for forecasting
    forecast = result.best_model.predict(n_steps=10)
    print(f"\n10-step forecast: {forecast['forecast'][:3]}...")

    return result


def example_3_statistical_testing():
    """Example 3: Statistical significance testing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)

    # Generate test data
    prices, dates = generate_test_data(n_points=400)

    # Get forecasts from multiple models
    print("\nGenerating forecasts from different models...")

    from fractime.backtesting import WalkForwardValidator

    models = {
        'Fractal': ft.FractalForecaster(),
    }

    # Add baseline models if available
    try:
        from fractime.baselines import ARIMAForecaster, GARCHForecaster
        models['ARIMA'] = ARIMAForecaster(max_p=3, max_q=3, stepwise=True)
        models['GARCH'] = GARCHForecaster(p=1, q=1)
    except ImportError:
        print("  (Baseline models not available)")

    # Run validation
    validation_results = {}
    for name, model in models.items():
        validator = WalkForwardValidator(
            model=model,
            initial_window=300,
            step_size=20,
            forecast_horizon=1,
            verbose=False
        )
        validation_results[name] = validator.run(prices, dates)

    # Extract forecasts and actuals
    forecasts = {}
    for name, results in validation_results.items():
        forecasts[name] = np.array([f[0] for f in results['forecasts']])

    actuals = np.array([a[0] for a in validation_results[list(models.keys())[0]]['actuals']])

    # Pairwise Diebold-Mariano tests
    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISONS (Diebold-Mariano Test)")
    print("=" * 80)

    model_names = list(forecasts.keys())
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            dm_result = diebold_mariano_test(
                forecasts[model1],
                forecasts[model2],
                actuals
            )

            print(f"\n{model1} vs {model2}:")
            print(f"  {dm_result['interpretation']}")

    # Model Confidence Set
    if len(forecasts) > 2:
        print("\n" + "=" * 80)
        print("MODEL CONFIDENCE SET")
        print("=" * 80)

        mcs = model_confidence_set(forecasts, actuals)
        print(f"\nSuperior Set (models not significantly worse than best):")
        print(f"  {', '.join(mcs['superior_set'])}")

        if mcs['eliminated']:
            print(f"\nEliminated (in order):")
            for i, name in enumerate(mcs['eliminated']):
                print(f"  {i+1}. {name} (p={mcs['p_values'][i]:.4f})")

    return validation_results


def example_4_ensemble():
    """Example 4: Ensemble forecasting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: ENSEMBLE FORECASTING")
    print("=" * 80)

    # Generate test data
    prices, dates = generate_test_data(n_points=400)

    # Create ensemble
    print("\nCreating ensemble of all available models...")

    try:
        ensemble = create_ensemble(
            ['Fractal', 'ARIMA', 'GARCH'],
            weighting='equal'
        )

        # Fit ensemble
        ensemble.fit(prices, dates)

        # Generate forecast
        result = ensemble.predict(n_steps=10)

        print("\nEnsemble Forecast:")
        print(f"  Forecast: {result['forecast'][:3]}...")
        print(f"\nModel Weights:")
        for name, weight in result['weights'].items():
            print(f"  {name}: {weight:.3f}")

        print(f"\nIndividual Forecasts (first step):")
        for name, forecast in result['individual_forecasts'].items():
            print(f"  {name}: {forecast[0]:.2f}")

    except Exception as e:
        print(f"  ✗ Could not create ensemble: {e}")
        print("  (Baseline models may not be installed)")

    return None


def example_5_select_by_characteristics():
    """Example 5: Select best model with specific characteristics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: SELECT BY CHARACTERISTICS")
    print("=" * 80)

    # Generate test data
    prices, dates = generate_test_data(n_points=400)

    # Select best probabilistic model
    print("\nSelecting best PROBABILISTIC model...")

    selector = AutoSelector(verbose=True)

    try:
        result = selector.select_by_characteristics(
            prices, dates,
            required_characteristics=['probabilistic']
        )

        print(f"\nBest probabilistic model: {result.best_model_name}")
        print(f"Score: {result.best_score:.4f}")

    except Exception as e:
        print(f"  ✗ Selection failed: {e}")

    # Select best fast model
    print("\n" + "-" * 80)
    print("Selecting best FAST model...")

    try:
        result = selector.select_by_characteristics(
            prices, dates,
            required_characteristics=['fast']
        )

        print(f"\nBest fast model: {result.best_model_name}")
        print(f"Score: {result.best_score:.4f}")

    except Exception as e:
        print(f"  ✗ Selection failed: {e}")

    return None


def example_6_compare_categories():
    """Example 6: Compare best model from each category."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: COMPARE MODEL CATEGORIES")
    print("=" * 80)

    # Generate test data
    prices, dates = generate_test_data(n_points=400)

    # Compare categories
    selector = AutoSelector(verbose=False)

    category_results = selector.compare_categories(prices, dates)

    # Print results
    print("\n" + "=" * 80)
    print("CATEGORY WINNERS")
    print("=" * 80)

    for category, result in sorted(category_results.items(), key=lambda x: -x[1].best_score):
        print(f"\n{category.upper()}:")
        print(f"  Winner: {result.best_model_name}")
        print(f"  Score: {result.best_score:.4f}")

    return category_results


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "=" * 80)
    print("MODEL SELECTION FRAMEWORK - COMPREHENSIVE EXAMPLES")
    print("=" * 80)

    # Example 1: Registry
    registry = example_1_model_registry()

    # Example 2: Auto-selection
    selection_result = example_2_auto_selection()

    # Example 3: Statistical testing
    stat_results = example_3_statistical_testing()

    # Example 4: Ensemble
    ensemble_result = example_4_ensemble()

    # Example 5: Select by characteristics
    char_result = example_5_select_by_characteristics()

    # Example 6: Compare categories
    category_results = example_6_compare_categories()

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe model selection framework provides:")
    print("  1. Model Registry - discover and catalog all available models")
    print("  2. Auto-Selection - automatically pick the best model for your data")
    print("  3. Statistical Testing - verify significance of performance differences")
    print("  4. Ensemble Methods - combine models for robust forecasts")
    print("  5. Flexible Selection - filter by characteristics, categories")
    print("\nThis enables data-driven model selection rather than guesswork!")

    return {
        'registry': registry,
        'selection': selection_result,
        'statistical': stat_results,
        'ensemble': ensemble_result,
        'characteristics': char_result,
        'categories': category_results
    }


if __name__ == '__main__':
    # Run individual examples or all
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_model_registry()
        elif example_num == '2':
            example_2_auto_selection()
        elif example_num == '3':
            example_3_statistical_testing()
        elif example_num == '4':
            example_4_ensemble()
        elif example_num == '5':
            example_5_select_by_characteristics()
        elif example_num == '6':
            example_6_compare_categories()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python model_selection_example.py [1|2|3|4|5|6]")
    else:
        # Run all examples
        run_all_examples()
