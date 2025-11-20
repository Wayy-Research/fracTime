"""
Fractal vs Baseline Models - Comprehensive Comparison

Compares fractal forecasting models against classical baselines:
- ARIMA (Auto-regressive Integrated Moving Average)
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- Prophet (Facebook's forecasting tool)

Uses walk-forward validation and dual penalty scoring to rigorously
evaluate which model generalizes best.

The key question: Do fractal models beat classical baselines?
"""

import numpy as np
from datetime import datetime
import sys

# Core fractime imports
import fractime as ft
from fractime.backtesting import WalkForwardValidator, DualPenaltyScorer, compare_models

# Check baselines availability
try:
    from fractime.baselines import ARIMAForecaster, GARCHForecaster, ProphetForecaster
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("Warning: Baseline models not installed")
    print("Install with: uv pip install -e '.[baselines]'")
    sys.exit(1)

# Check Bayesian availability
try:
    from fractime import BayesianFractalForecaster
    BAYESIAN_AVAILABLE = BayesianFractalForecaster is not None
except (ImportError, AttributeError):
    BAYESIAN_AVAILABLE = False
    BayesianFractalForecaster = None


def generate_test_data(n_points: int = 500, regime: str = 'trending') -> tuple:
    """
    Generate synthetic data with different characteristics.

    Args:
        n_points: Number of data points
        regime: 'trending' (H>0.5), 'mean_reverting' (H<0.5), or 'mixed'

    Returns:
        (prices, dates) tuple
    """
    np.random.seed(42)

    initial_price = 100.0
    sigma = 0.02  # 2% daily volatility

    if regime == 'trending':
        # Trending series (H ≈ 0.6)
        hurst = 0.6
        phi = 2 * hurst - 1
    elif regime == 'mean_reverting':
        # Mean-reverting series (H ≈ 0.4)
        hurst = 0.4
        phi = 2 * hurst - 1
    elif regime == 'mixed':
        # Regime switch halfway through
        n_half = n_points // 2
        prices1, dates1 = generate_test_data(n_half, 'trending')
        prices2, dates2 = generate_test_data(n_half, 'mean_reverting')
        prices = np.concatenate([prices1, prices2 * (prices1[-1] / prices2[0])])
        dates = np.concatenate([dates1, dates2 + (dates1[-1] - dates2[0]) + np.timedelta64(1, 'D')])
        return prices, dates
    else:
        hurst = 0.5
        phi = 0

    # Generate correlated increments
    increments = np.random.randn(n_points)
    for i in range(1, n_points):
        increments[i] = phi * increments[i-1] + np.sqrt(1 - phi**2) * increments[i]

    # Convert to prices
    log_returns = sigma * increments
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    # Generate dates
    start_date = np.datetime64('2020-01-01')
    dates = np.array([start_date + np.timedelta64(i, 'D') for i in range(n_points)])

    return prices, dates


def compare_all_models(prices, dates, verbose=True):
    """
    Compare all available models on the given data.

    Args:
        prices: Price time series
        dates: Corresponding dates
        verbose: Print progress

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 80)
    print("FRACTAL VS BASELINE MODELS - COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print(f"Data: {len(prices)} points from {dates[0]} to {dates[-1]}")

    # Define all models to compare
    models = {
        'Fractal': ft.FractalForecaster(),
        'ARIMA': ARIMAForecaster(max_p=3, max_q=3, stepwise=True),
        'GARCH': GARCHForecaster(p=1, q=1),
        'Prophet': ProphetForecaster(),
    }

    # Add Bayesian if available
    if BAYESIAN_AVAILABLE:
        models['Bayesian (Fast)'] = BayesianFractalForecaster(mode='fast', n_samples=500)
        print("\n  Bayesian models included")
    else:
        print("\n  Bayesian models not available (install PyMC)")

    # Run validation for each model
    results = {}
    for model_name, model in models.items():
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Validating: {model_name}")
            print(f"{'=' * 70}")

        try:
            validator = WalkForwardValidator(
                model=model,
                initial_window=min(200, len(prices) // 2),
                step_size=10,
                forecast_horizon=1,
                window_type='expanding',
                verbose=False
            )

            results[model_name] = validator.run(prices, dates)

            # Print quick metrics
            metrics = results[model_name]['metrics']
            if verbose:
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
                if 'coverage' in metrics:
                    print(f"  95% CI Coverage: {metrics['coverage']:.2%}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    # Compare using dual penalty scoring
    print("\n" + "=" * 80)
    print("DUAL PENALTY SCORING")
    print("=" * 80)

    scorer = DualPenaltyScorer(alpha=1.0, beta=0.5)
    comparison = compare_models(results, scorer)

    # Print comparison table
    print(comparison['comparison_table'])

    # Detailed analysis of top 3 models
    print("\n" + "=" * 80)
    print("TOP 3 MODELS - DETAILED ANALYSIS")
    print("=" * 80)

    top_3 = comparison['rankings'][:3]
    for rank, (model_name, score_dict) in enumerate(top_3, 1):
        print(f"\n{rank}. {model_name}")
        print(f"   Total Score: {score_dict['total_score']:.4f}")
        print(f"   Accuracy: {score_dict['accuracy_score']:.4f}")
        print(f"   Overfitting Penalty: {score_dict['overfitting_penalty']:.4f}")

        # Breakdown
        print(f"\n   Accuracy Components:")
        for component, value in score_dict['components']['accuracy'].items():
            print(f"     {component}: {value:.4f}")

        print(f"\n   Overfitting Components:")
        for component, value in score_dict['components']['overfitting'].items():
            print(f"     {component}: {value:.4f}")

        # Raw metrics
        metrics = results[model_name]['metrics']
        print(f"\n   Raw Metrics:")
        print(f"     RMSE: {metrics['rmse']:.4f}")
        print(f"     MAE: {metrics['mae']:.4f}")
        print(f"     Direction Accuracy: {metrics['direction_accuracy']:.2%}")
        if 'coverage' in metrics:
            print(f"     Coverage: {metrics['coverage']:.2%}")
            print(f"     Calibration Error: {metrics['calibration_error']:.4f}")

    return comparison


def example_1_trending_series():
    """Example 1: Trending time series (H > 0.5)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: TRENDING TIME SERIES (H ≈ 0.6)")
    print("=" * 80)
    print("Expected: Fractal models should perform well on trending data")

    prices, dates = generate_test_data(n_points=500, regime='trending')
    comparison = compare_all_models(prices, dates, verbose=True)

    print("\nConclusion:")
    print(f"  Winner: {comparison['best_model']}")
    print("  Fractal models exploit long-range dependence in trending series")

    return comparison


def example_2_mean_reverting_series():
    """Example 2: Mean-reverting time series (H < 0.5)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MEAN-REVERTING TIME SERIES (H ≈ 0.4)")
    print("=" * 80)
    print("Expected: ARIMA/GARCH may compete better with fractal models")

    prices, dates = generate_test_data(n_points=500, regime='mean_reverting')
    comparison = compare_all_models(prices, dates, verbose=True)

    print("\nConclusion:")
    print(f"  Winner: {comparison['best_model']}")
    print("  Mean-reversion can be captured by both fractal and classical models")

    return comparison


def example_3_mixed_regime():
    """Example 3: Mixed regime (trend → mean-reversion)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: MIXED REGIME (Trend → Mean-Reversion)")
    print("=" * 80)
    print("Expected: Models with adaptive windows should perform better")

    prices, dates = generate_test_data(n_points=500, regime='mixed')
    comparison = compare_all_models(prices, dates, verbose=True)

    print("\nConclusion:")
    print(f"  Winner: {comparison['best_model']}")
    print("  Regime changes test model adaptability")

    return comparison


def example_4_real_world_data():
    """Example 4: Real market data (if available)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: REAL MARKET DATA")
    print("=" * 80)

    try:
        import yfinance as yf

        print("Downloading S&P 500 data...")
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(period="2y")

        prices = df['Close'].values
        dates = df.index.values

        print(f"Data: {len(prices)} days of S&P 500 prices")

        comparison = compare_all_models(prices, dates, verbose=True)

        print("\nConclusion:")
        print(f"  Winner on real data: {comparison['best_model']}")
        print("  This is the ultimate test: how models perform on actual markets")

        return comparison

    except ImportError:
        print("  ✗ yfinance not installed, skipping real data example")
        return None
    except Exception as e:
        print(f"  ✗ Failed to download data: {e}")
        return None


def run_all_examples():
    """Run all comparison examples."""
    print("\n" + "=" * 80)
    print("FRACTAL VS BASELINES - ALL EXAMPLES")
    print("=" * 80)

    results = {}

    # Example 1: Trending
    results['trending'] = example_1_trending_series()

    # Example 2: Mean-reverting
    results['mean_reverting'] = example_2_mean_reverting_series()

    # Example 3: Mixed regime
    results['mixed'] = example_3_mixed_regime()

    # Example 4: Real data
    results['real'] = example_4_real_world_data()

    # Summary across all examples
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL EXAMPLES")
    print("=" * 80)

    winner_counts = {}
    for example_name, comparison in results.items():
        if comparison is not None:
            winner = comparison['best_model']
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
            print(f"{example_name:20s} → {winner}")

    print("\n" + "=" * 80)
    print("OVERALL WINNER COUNTS")
    print("=" * 80)
    for model, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"  {model:20s} {count} wins")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("  1. Fractal models excel at capturing long-range dependence")
    print("  2. ARIMA is strong for short-term autocorrelation")
    print("  3. GARCH captures volatility clustering")
    print("  4. Prophet handles seasonality and trends")
    print("  5. Model selection should be data-driven using backtesting")

    return results


if __name__ == '__main__':
    if not BASELINES_AVAILABLE:
        print("\n" + "=" * 80)
        print("ERROR: Baseline models not installed")
        print("=" * 80)
        print("Install with: uv pip install -e '.[baselines]'")
        print("\nThis will install:")
        print("  - pmdarima (for ARIMA)")
        print("  - arch (for GARCH)")
        print("  - prophet (for Prophet)")
        sys.exit(1)

    # Run individual examples or all
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_trending_series()
        elif example_num == '2':
            example_2_mean_reverting_series()
        elif example_num == '3':
            example_3_mixed_regime()
        elif example_num == '4':
            example_4_real_world_data()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python fractal_vs_baselines.py [1|2|3|4]")
    else:
        # Run all examples
        run_all_examples()
