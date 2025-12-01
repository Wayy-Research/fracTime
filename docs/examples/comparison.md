# Model Comparison Example

Comparing FracTime against baseline models.

## Setup

```python
import fractime as ft
from fractime.baselines import ARIMAForecaster, ETSForecaster, GARCHForecaster
from fractime.backtesting import WalkForwardValidator, compare_models
import numpy as np
```

## Load Data

```python
# Use Yahoo Finance data
prices, dates = ft.get_yahoo_data('SPY', period='5y')
print(f"Loaded {len(prices)} data points")
```

## Single Model Backtesting

```python
# Test fractal forecaster
validator = WalkForwardValidator(
    model=ft.FractalForecaster(),
    initial_window=252,    # 1 year
    step_size=20,          # Refit every 20 days
    forecast_horizon=10    # 10-day forecast
)

results = validator.run(prices, dates)

print("Fractal Forecaster Results:")
print(f"  RMSE: {results['metrics']['rmse']:.4f}")
print(f"  MAE: {results['metrics']['mae']:.4f}")
print(f"  Direction Accuracy: {results['metrics']['direction_accuracy']:.2%}")
print(f"  Coverage: {results['metrics']['coverage']:.2%}")
```

## Multi-Model Comparison

```python
# Define models to compare
models = {
    'Fractal': ft.FractalForecaster(),
    'ARIMA': ARIMAForecaster(seasonal=False),
    'ETS': ETSForecaster(trend='add'),
}

# Run comparison
comparison = compare_models(
    models=models,
    prices=prices,
    dates=dates,
    initial_window=252,
    step_size=20,
    forecast_horizon=10
)

# Print results
print("\nModel Comparison:")
print("-" * 60)
print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'Direction':>12}")
print("-" * 60)

for name, metrics in comparison.items():
    print(f"{name:<15} {metrics['rmse']:>10.4f} {metrics['mae']:>10.4f} {metrics['direction_accuracy']:>11.1%}")
```

## Statistical Significance

```python
from fractime.selection import diebold_mariano_test

# Get errors from comparison
fractal_errors = comparison['Fractal']['errors']
arima_errors = comparison['ARIMA']['errors']

# Diebold-Mariano test
dm_stat, p_value = diebold_mariano_test(fractal_errors, arima_errors)
print(f"\nDiebold-Mariano Test (Fractal vs ARIMA):")
print(f"  DM Statistic: {dm_stat:.4f}")
print(f"  P-value: {p_value:.4f}")

if p_value < 0.05:
    if dm_stat < 0:
        print("  Result: Fractal significantly outperforms ARIMA")
    else:
        print("  Result: ARIMA significantly outperforms Fractal")
else:
    print("  Result: No significant difference")
```

## Model Confidence Set

```python
from fractime.selection import model_confidence_set

# Build error dict
model_errors = {
    name: metrics['errors']
    for name, metrics in comparison.items()
}

# Find models that cannot be distinguished
mcs = model_confidence_set(model_errors, alpha=0.1)
print(f"\nModel Confidence Set (90% level):")
print(f"  {mcs}")
```

## Visualize Results

```python
import matplotlib.pyplot as plt

# Bar chart of RMSE
models = list(comparison.keys())
rmse_values = [comparison[m]['rmse'] for m in models]

plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('RMSE')
plt.title('Model Comparison - RMSE')
plt.show()
```
