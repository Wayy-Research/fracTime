#!/usr/bin/env python3
"""
Complete Workflow Demo - All Features
Shows: Date-based forecasting, pretty print, weighted CI, responsive charts
"""

import numpy as np
import polars as pl
import fractime as ft
from datetime import datetime, timedelta

print("=" * 70)
print("FRACTIME COMPLETE WORKFLOW DEMO")
print("=" * 70)

# 1. Generate sample data with dates
np.random.seed(42)
prices = 100 + np.random.randn(200).cumsum()
end_date = datetime(2025, 11, 19)
start_date = end_date - timedelta(days=199)
dates = pl.datetime_range(start=start_date, end=end_date, interval='1d', eager=True).to_numpy()

print(f"\n1. DATA")
print(f"   Period: {dates[0]} to {dates[-1]}")
print(f"   Length: {len(prices)} days")
print(f"   Current price: ${prices[-1]:.2f}")

# 2. Fit forecaster
print(f"\n2. FITTING FORECASTER")
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)
print("   ✓ Fitted with dates")

# 3. Generate forecast using date-based API
print(f"\n3. GENERATING FORECAST")
print("   Using date-based API: end_date='2025-11-27'")
result = forecaster.predict(end_date='2025-11-27', n_paths=500)
print(f"   ✓ Generated {len(result['probabilities'])} paths")

# 4. Pretty print the results
print(f"\n4. FORECAST RESULTS (using print_forecast_summary)")
ft.print_forecast_summary(result, current_price=prices[-1], show_paths=5)

# 5. Create interactive visualization
print("\n5. CREATING INTERACTIVE VISUALIZATION")
print("   Features:")
print("   ✓ Responsive width (fits Jupyter)")
print("   ✓ Probability-weighted CI (default)")
print("   ✓ Date formatting on x-axis")
print("   ✓ Visual continuity (paths connect)")

chart = ft.plot_forecast_interactive(
    prices=prices[-60:],  # Show last 60 days
    result=result,
    dates=dates[-60:],
    title="Complete Demo: All Features",
    top_n_paths=20,
    use_weighted_ci=True  # Default, but explicit here
)

output = 'complete_workflow_demo.html'
chart.write_html(output)
print(f"\n   ✓ Saved to: {output}")

# 6. Show how to access data programmatically
print("\n" + "=" * 70)
print("6. PROGRAMMATIC ACCESS")
print("=" * 70)

print(f"""
# Point forecasts
median = result['forecast'][-1]              # ${result['forecast'][-1]:.2f}
weighted = result['weighted_forecast'][-1]   # ${result['weighted_forecast'][-1]:.2f}

# Confidence intervals
std_ci = (result['lower'][-1], result['upper'][-1])
weighted_ci = (result['weighted_lower'][-1], result['weighted_upper'][-1])

# All paths and probabilities
paths = result['paths']           # shape: {result['paths'].shape}
probabilities = result['probabilities']  # shape: {result['probabilities'].shape}

# Forecast dates
forecast_dates = result['dates']  # shape: {result['dates'].shape}
""")

print("=" * 70)
print("WORKFLOW COMPLETE!")
print("=" * 70)

print("""
Summary of Features Demonstrated:
1. ✓ Date-based forecasting (no manual step calculation)
2. ✓ Pretty print summary (readable output)
3. ✓ Probability-weighted CI (more accurate uncertainty)
4. ✓ Responsive charts (fits Jupyter notebooks)
5. ✓ Interactive Plotly visualization
6. ✓ Programmatic access to all results

Next Steps:
- Open the HTML file in a browser
- Try with your own data
- Experiment with different forecast periods
- Compare weighted vs standard CI
""")

print("=" * 70)
