#!/usr/bin/env python3
"""
BNB Forecast Example - Corrected Version

This example shows how to properly create a forecast visualization
with dates displayed correctly on the x-axis.
"""

from wrdata import DataStream
import fractime as ft

# Get data
stream = DataStream()
df = stream.get('BNB-USD', start='2020-01-01', end='2025-11-19')

dates = df['timestamp']
prices = df['close']

print(f"Loaded {len(prices)} days of BNB price data")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Price range: ${prices.min():.2f} to ${prices.max():.2f}")

# Fit forecaster
forecaster = ft.FractalForecaster()
forecaster.fit(prices.to_numpy(), dates=dates.to_numpy())

# Forecast to specific date
result = forecaster.predict(end_date='2025-11-27', n_paths=500)

print(f"\nForecast to: {result['dates'][-1]}")
print(f"Weighted forecast: ${result['weighted_forecast'][-1]:.2f}")
print(f"95% CI: [${result['lower'][-1]:.2f}, ${result['upper'][-1]:.2f}]")

# Create interactive chart
# IMPORTANT: Must pass dates parameter!
chart = ft.plot_forecast_interactive(
    prices=prices.to_numpy(),
    result=result,
    dates=dates.to_numpy(),  # ← THIS IS REQUIRED for proper date display!
    title="Prob-weighted Forecast Paths - BNB",
    top_n_paths=50
)

# Save and display
output = 'bnb_forecast.html'
chart.write_html(output)
print(f"\n✓ Chart saved to: {output}")

# Optionally show in browser
chart.show()
