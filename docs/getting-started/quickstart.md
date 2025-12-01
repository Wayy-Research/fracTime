# Quick Start

This guide walks through the basic FracTime workflow.

## Basic Workflow

### 1. Load Data

```python
import fractime as ft
import numpy as np

# Option 1: Use the built-in Yahoo Finance loader
prices, dates = ft.get_yahoo_data('AAPL', period='2y')

# Option 2: Your own data
prices = np.array([100, 102, 101, 105, 103, 108, ...])
```

### 2. Analyze Fractal Properties

```python
# Understand your data's characteristics
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)

print(f"Hurst exponent: {hurst:.3f}")
if hurst > 0.5:
    print("Series is trending (persistent)")
elif hurst < 0.5:
    print("Series is mean-reverting (anti-persistent)")
else:
    print("Series is a random walk")
```

### 3. Fit and Forecast

```python
# Create forecaster
forecaster = ft.FractalForecaster()
forecaster.fit(prices)

# Generate forecast with uncertainty
result = forecaster.predict(n_steps=30, n_paths=1000)

# Access results
print(f"Point forecast: {result['forecast'][-1]:.2f}")
print(f"95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")
```

### 4. Visualize

```python
# Interactive plot
chart = ft.plot_forecast_interactive(
    prices=prices,
    result=result,
    title="30-Day Forecast"
)
chart.show()

# Static plot
fig = ft.plot_forecast(
    prices=prices,
    forecast=result['forecast'],
    paths=result['paths']
)
fig.show()
```

## Date-Based Forecasting

If you have date information:

```python
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)

# Forecast to a specific date
result = forecaster.predict(end_date='2025-12-31')

# Or by period
result = forecaster.predict(period='2w')  # 2 weeks
result = forecaster.predict(period='1M')  # 1 month
```

## Compare Models

```python
from fractime.baselines import ARIMAForecaster, ETSForecaster
from fractime.backtesting import compare_models

comparison = compare_models(
    models={
        'Fractal': ft.FractalForecaster(),
        'ARIMA': ARIMAForecaster(),
        'ETS': ETSForecaster()
    },
    prices=prices,
    dates=dates,
    initial_window=252,
    step_size=20,
    forecast_horizon=10
)

for name, metrics in comparison.items():
    print(f"{name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
```

## Next Steps

- [Core Concepts](../guide/concepts.md) - Understanding fractal analysis
- [Backtesting Guide](../guide/backtesting.md) - Rigorous model validation
- [API Reference](../api/core.md) - Complete API documentation
