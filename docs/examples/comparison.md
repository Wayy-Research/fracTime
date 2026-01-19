# Forecasting Example

Complete walkthrough of probabilistic forecasting with FracTime.

---

## Setup

```python
import fractime as ft
import numpy as np
```

---

## Basic Forecast

```python
# Generate sample data
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Create forecaster
model = ft.Forecaster(prices)

# Generate forecast
result = model.predict(steps=30, n_paths=1000)

# View results
print(f"Current price: {prices[-1]:.2f}")
print(f"30-day forecast: {result.forecast[-1]:.2f}")
print(f"Expected return: {(result.forecast[-1]/prices[-1] - 1)*100:.1f}%")
```

---

## Confidence Intervals

```python
# 95% CI
lower_95, upper_95 = result.ci(0.95)
print(f"95% CI at day 30: ({lower_95[-1]:.2f}, {upper_95[-1]:.2f})")

# 90% CI
lower_90, upper_90 = result.ci(0.90)
print(f"90% CI at day 30: ({lower_90[-1]:.2f}, {upper_90[-1]:.2f})")

# Custom quantiles
q10 = result.quantile(0.10)
q90 = result.quantile(0.90)
print(f"10th-90th percentile: ({q10[-1]:.2f}, {q90[-1]:.2f})")
```

---

## Probability Analysis

```python
current = prices[-1]
final_values = result.paths[:, -1]

# Probability of increase
prob_up = np.mean(final_values > current)
print(f"P(increase): {prob_up:.1%}")

# Probability of >5% gain
prob_5_gain = np.mean(final_values > current * 1.05)
print(f"P(>5% gain): {prob_5_gain:.1%}")

# Probability of >10% loss
prob_10_loss = np.mean(final_values < current * 0.90)
print(f"P(>10% loss): {prob_10_loss:.1%}")

# Value at Risk (5%)
var_95 = np.percentile(final_values, 5)
print(f"95% VaR: {var_95:.2f} ({(var_95/current - 1)*100:.1f}%)")
```

---

## Compare Methods

```python
# R/S method
model_rs = ft.Forecaster(prices, method='rs')
result_rs = model_rs.predict(steps=30, n_paths=1000)

# DFA method
model_dfa = ft.Forecaster(prices, method='dfa')
result_dfa = model_dfa.predict(steps=30, n_paths=1000)

# With time warping
model_tw = ft.Forecaster(prices, time_warp=True)
result_tw = model_tw.predict(steps=30, n_paths=1000)

# Compare
print("Method Comparison (30-day forecast):")
print(f"  R/S:         {result_rs.forecast[-1]:.2f}")
print(f"  DFA:         {result_dfa.forecast[-1]:.2f}")
print(f"  Time Warp:   {result_tw.forecast[-1]:.2f}")
```

---

## Ensemble Forecasting

```python
# Create ensemble
ensemble = ft.Ensemble(prices)
result_ens = ensemble.predict(steps=30, n_paths=500)

print(f"Ensemble forecast: {result_ens.forecast[-1]:.2f}")
print(f"Number of models: {ensemble.n_models}")
print(f"Total paths: {result_ens.n_paths}")

# Compare strategies
strategies = ['average', 'weighted', 'stacking', 'boosting']
for strategy in strategies:
    ens = ft.Ensemble(prices, strategy=strategy)
    res = ens.predict(steps=30, n_paths=300)
    print(f"{strategy:10}: {res.forecast[-1]:.2f}")
```

---

## Custom Ensemble

```python
# Build diverse ensemble
models = [
    ft.Forecaster(prices, method='rs'),
    ft.Forecaster(prices, method='dfa'),
    ft.Forecaster(prices, time_warp=True),
    ft.Forecaster(prices, path_weights={
        'hurst': 0.7,
        'volatility': 0.2,
        'pattern': 0.1,
    }),
]

ensemble = ft.Ensemble(prices, models=models, strategy='weighted')
result = ensemble.predict(steps=30, n_paths=500)

print(f"Custom ensemble forecast: {result.forecast[-1]:.2f}")
if 'model_weights' in result.metadata:
    print(f"Model weights: {result.metadata['model_weights']}")
```

---

## With Dates

```python
import datetime

# Create dates
dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i)
         for i in range(len(prices))]

# Forecaster with dates
model = ft.Forecaster(prices, dates=dates)
result = model.predict(steps=30)

# Forecast includes future dates
print(f"Last historical date: {dates[-1]}")
print(f"First forecast date: {result.dates[0]}")
print(f"Last forecast date: {result.dates[-1]}")
```

---

## Backtest Simulation

```python
# Split data
train = prices[:-30]
test = prices[-30:]

# Fit on training data
model = ft.Forecaster(train)
result = model.predict(steps=30)

# Compare to actual
forecast = result.forecast
actual = test

# Metrics
mae = np.mean(np.abs(forecast - actual))
rmse = np.sqrt(np.mean((forecast - actual)**2))
mape = np.mean(np.abs((forecast - actual) / actual)) * 100

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# Coverage (what % of actual values fall within CI)
lower, upper = result.ci(0.95)
coverage = np.mean((actual >= lower) & (actual <= upper))
print(f"95% CI Coverage: {coverage:.1%}")
```

---

## Visualization

```python
# Plot forecast
ft.plot(result, title="30-Day Forecast")

# Save to file
fig = ft.plot(result, show=False)
fig.write_html("forecast.html")
```

---

## Export Results

```python
# To DataFrame
df = result.to_frame()
print(df.head())

# Save
df.write_csv("forecast.csv")
```

---

## Complete Comparison Script

```python
import fractime as ft
import numpy as np

# Data
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Split for testing
train = prices[:-30]
test = prices[-30:]

# Models to compare
configs = {
    'RS': {'method': 'rs'},
    'DFA': {'method': 'dfa'},
    'TimeWarp': {'time_warp': True},
    'Ensemble': None,  # Special case
}

results = {}

for name, config in configs.items():
    if name == 'Ensemble':
        model = ft.Ensemble(train)
        result = model.predict(steps=30, n_paths=500)
    else:
        model = ft.Forecaster(train, **config)
        result = model.predict(steps=30, n_paths=1000)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((result.forecast - test)**2))
    results[name] = {'forecast': result.forecast[-1], 'rmse': rmse}

# Print comparison
print("=" * 50)
print("Model Comparison")
print("=" * 50)
print(f"{'Model':<15} {'Forecast':>12} {'RMSE':>12}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['forecast']:>12.2f} {metrics['rmse']:>12.4f}")
print("-" * 50)
print(f"Actual value: {test[-1]:.2f}")
```

---

## Next Steps

- [Advanced Usage](real-world.md) - Real-world applications
- [Ensemble Guide](../guide/ensemble.md) - Ensemble methods
- [API Reference](../api/forecaster.md) - Complete documentation
