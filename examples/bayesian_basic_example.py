#!/usr/bin/env python3
"""
Basic Bayesian Fractal Forecasting Example

Demonstrates the three Bayesian inference modes:
1. Fast (ADVI) - 15-20 seconds
2. Hybrid (ADVI + Monte Carlo) - ~1 minute, RECOMMENDED
3. Pure Bayesian (MCMC) - 1-2 minutes, most rigorous

NOTE: Requires PyMC installation:
    pip install -e ".[bayesian]"
or:
    uv pip install -e ".[bayesian]"
"""

import numpy as np
import polars as pl
from datetime import datetime, timedelta

try:
    from fractime.bayesian import BayesianFractalForecaster
    import fractime as ft
    BAYESIAN_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")
    print("\nBayesian dependencies not installed.")
    print("Install with: uv pip install -e '.[bayesian]'")
    BAYESIAN_AVAILABLE = False
    exit(1)

print("=" * 70)
print("BAYESIAN FRACTAL FORECASTING - BASIC EXAMPLE")
print("=" * 70)

# 1. Generate sample data
print("\n1. GENERATING SAMPLE DATA")
np.random.seed(42)
prices = 100 + np.random.randn(200).cumsum()
end_date = datetime(2025, 11, 19)
start_date = end_date - timedelta(days=199)
dates = pl.datetime_range(start=start_date, end=end_date, interval='1d', eager=True).to_numpy()

print(f"   Period: {dates[0]} to {dates[-1]}")
print(f"   Length: {len(prices)} days")
print(f"   Current price: ${prices[-1]:.2f}")

# 2. Fast Mode (ADVI) - Recommended for production
print("\n" + "=" * 70)
print("2. FAST MODE (ADVI - Variational Inference)")
print("=" * 70)
print("   Best for: Production, real-time forecasting")
print("   Speed: 15-20 seconds")

forecaster_fast = BayesianFractalForecaster(
    mode='fast',
    model_type='neutral',
    n_samples=1000
)

print("\n   Fitting model...")
forecaster_fast.fit(prices, dates=dates, verbose=True)

print("\n   Generating forecast...")
result_fast = forecaster_fast.predict(end_date='2025-11-27', n_paths=500)

print(f"\n   Fast Mode Results:")
print(f"   Median forecast: ${result_fast['forecast'][-1]:.2f}")
print(f"   Weighted forecast: ${result_fast['weighted_forecast'][-1]:.2f}")
print(f"   95% CI: [${result_fast['lower'][-1]:.2f}, ${result_fast['upper'][-1]:.2f}]")

# Get parameter posteriors
params = forecaster_fast.get_parameter_posterior_summary()
print(f"\n   Parameter Posteriors:")
print(f"   Hurst: {params['hurst']['mean']:.3f} ± {params['hurst']['sd']:.3f}")
print(f"   Fractal Dim: {params['fractal_dim']['mean']:.3f} ± {params['fractal_dim']['sd']:.3f}")

# 3. Hybrid Mode - Recommended for research
print("\n" + "=" * 70)
print("3. HYBRID MODE (ADVI + Monte Carlo)")
print("=" * 70)
print("   Best for: Research, weekly forecasts")
print("   Speed: ~1 minute")

forecaster_hybrid = BayesianFractalForecaster(
    mode='hybrid',
    model_type='neutral',
    n_samples=1000
)

print("\n   Fitting model...")
forecaster_hybrid.fit(prices, dates=dates, verbose=True)

print("\n   Generating forecast...")
result_hybrid = forecaster_hybrid.predict(end_date='2025-11-27', n_paths=1000)

print(f"\n   Hybrid Mode Results:")
print(f"   Median forecast: ${result_hybrid['forecast'][-1]:.2f}")
print(f"   Weighted forecast: ${result_hybrid['weighted_forecast'][-1]:.2f}")
print(f"   95% CI: [${result_hybrid['lower'][-1]:.2f}, ${result_hybrid['upper'][-1]:.2f}]")

# Compare with classical approach
print("\n" + "=" * 70)
print("4. COMPARISON WITH CLASSICAL FORECASTER")
print("=" * 70)

classical = ft.FractalForecaster()
classical.fit(prices, dates=dates)
result_classical = classical.predict(end_date='2025-11-27', n_paths=1000)

print(f"\n   Classical Fractal:")
print(f"   Forecast: ${result_classical['weighted_forecast'][-1]:.2f}")
print(f"   95% CI: [${result_classical['lower'][-1]:.2f}, ${result_classical['upper'][-1]:.2f}]")

print(f"\n   Bayesian Fast:")
print(f"   Forecast: ${result_fast['weighted_forecast'][-1]:.2f}")
print(f"   95% CI: [${result_fast['lower'][-1]:.2f}, ${result_fast['upper'][-1]:.2f}]")

print(f"\n   Bayesian Hybrid:")
print(f"   Forecast: ${result_hybrid['weighted_forecast'][-1]:.2f}")
print(f"   95% CI: [${result_hybrid['lower'][-1]:.2f}, ${result_hybrid['upper'][-1]:.2f}]")

diff_fast = abs(result_fast['weighted_forecast'][-1] - result_classical['weighted_forecast'][-1])
diff_hybrid = abs(result_hybrid['weighted_forecast'][-1] - result_classical['weighted_forecast'][-1])

print(f"\n   Difference from classical:")
print(f"   Fast: ${diff_fast:.2f}")
print(f"   Hybrid: ${diff_hybrid:.2f}")

# 5. Visualize
print("\n" + "=" * 70)
print("5. VISUALIZATION")
print("=" * 70)

print("\n   Creating comparison plot...")
chart = ft.plot_forecast_interactive(
    prices=prices[-60:],
    result=result_hybrid,
    dates=dates[-60:],
    title="Bayesian Hybrid Forecast - All Features",
    top_n_paths=20,
    use_weighted_ci=True
)

output_file = 'bayesian_forecast_example.html'
chart.write_html(output_file)
print(f"   ✓ Saved to: {output_file}")

# 6. Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Bayesian Benefits:
1. Proper uncertainty quantification for parameters
2. Parameter posteriors show regime information
3. Multiple inference modes for speed/accuracy tradeoff
4. Can incorporate prior knowledge (e.g., for equities vs FX)

Mode Selection:
- Fast (ADVI): Daily production forecasts (15-20 sec)
- Hybrid: Weekly research/analysis (~1 min) ✓ RECOMMENDED
- Pure Bayesian: Model validation (1-2 min)

Next Steps:
- Try different model types: 'equities', 'fx', 'crypto'
- Use real data via wrdata
- Implement backtesting to compare accuracy
- Track parameter evolution over time (expanding windows)
""")

print("=" * 70)
