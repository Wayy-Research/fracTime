# Basic Forecasting Example

A complete walkthrough of fractal-based forecasting.

## Setup

```python
import fractime as ft
import numpy as np
import matplotlib.pyplot as plt
```

## Generate Sample Data

```python
# Simulate a time series with fractal properties
np.random.seed(42)
n = 500

# Create trending data with some noise
trend = np.linspace(0, 20, n)
noise = np.random.randn(n).cumsum() * 0.5
prices = 100 + trend + noise
```

## Analyze Fractal Properties

```python
# Create analyzer
analyzer = ft.FractalAnalyzer()

# Compute Hurst exponent
hurst = analyzer.compute_hurst(prices)
print(f"Hurst Exponent: {hurst:.3f}")

# Interpret
if hurst > 0.5:
    print("Series is persistent (trending)")
elif hurst < 0.5:
    print("Series is anti-persistent (mean-reverting)")
else:
    print("Series is a random walk")

# Full analysis
analysis = analyzer.analyze(prices)
print(f"Fractal Dimension: {analysis['fractal_dim']:.3f}")
```

## Fit Forecaster

```python
# Create and fit
forecaster = ft.FractalForecaster(lookback=252)
forecaster.fit(prices)

# Check fitted parameters
print(f"Fitted Hurst: {forecaster.hurst:.3f}")
print(f"Fitted Fractal Dim: {forecaster.fractal_dim:.3f}")
```

## Generate Forecast

```python
# Forecast 30 steps ahead
result = forecaster.predict(
    n_steps=30,
    n_paths=1000,
    confidence=0.95
)

# Access results
forecast = result['forecast']
lower = result['lower']
upper = result['upper']
paths = result['paths']

print(f"Final forecast: {forecast[-1]:.2f}")
print(f"95% CI: [{lower[-1]:.2f}, {upper[-1]:.2f}]")
```

## Visualize

```python
# Interactive plot (Plotly)
chart = ft.plot_forecast_interactive(
    prices=prices,
    result=result,
    title="30-Step Fractal Forecast",
    top_n_paths=20
)
chart.show()

# Static plot (Matplotlib)
fig = ft.plot_forecast(
    prices=prices[-100:],  # Last 100 points
    forecast=forecast,
    paths=paths,
    confidence_intervals=result,
    title="30-Step Forecast"
)
plt.show()
```

## Print Summary

```python
ft.print_forecast_summary(
    result,
    current_price=prices[-1],
    show_paths=10
)
```

## Analyze Probabilities

```python
# Get path probabilities
probs = result['probabilities']
final_values = paths[:, -1]
current = prices[-1]

# Probability of increase
prob_up = np.sum(probs[final_values > current])
print(f"Probability of increase: {prob_up:.1%}")

# Most likely outcome
most_likely_idx = np.argmax(probs)
most_likely = final_values[most_likely_idx]
print(f"Most likely outcome: {most_likely:.2f}")

# Probability-weighted VaR
sorted_idx = np.argsort(final_values)
cumsum = np.cumsum(probs[sorted_idx])
var_5 = final_values[sorted_idx[np.searchsorted(cumsum, 0.05)]]
print(f"5% VaR: {var_5:.2f}")
```
