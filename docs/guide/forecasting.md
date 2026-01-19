# Forecasting Guide

Complete guide to probabilistic forecasting with FracTime.

---

## Basic Forecasting

```python
import fractime as ft
import numpy as np

# Your data
prices = ...  # numpy array or Polars Series

# Create forecaster
model = ft.Forecaster(prices)

# Generate forecast
result = model.predict(steps=30, n_paths=1000)

# Access results
print(f"Final value: {result.forecast[-1]:.2f}")
print(f"95% CI: {result.ci(0.95)}")
```

---

## Forecaster Parameters

```python
model = ft.Forecaster(
    data,                    # Price series (required)
    dates=None,              # Optional date array
    method='rs',             # Hurst method: 'rs' or 'dfa'
    time_warp=False,         # Enable trading time
    exogenous=None,          # Dict of exogenous variables
    path_weights=None,       # Custom weighting scheme
)
```

### Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array | required | Historical price series |
| `dates` | array | None | Corresponding dates |
| `method` | str | 'rs' | Hurst estimation method |
| `time_warp` | bool | False | Use Mandelbrot's trading time |
| `exogenous` | dict | None | External variables |
| `path_weights` | dict | None | Custom path weighting |

---

## Prediction Options

```python
result = model.predict(
    steps=30,           # Forecast horizon
    n_paths=1000,       # Number of Monte Carlo paths
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | int | 30 | Number of steps to forecast |
| `n_paths` | int | 1000 | Number of simulation paths |

---

## ForecastResult Object

The `predict()` method returns a `ForecastResult` object:

### Properties

```python
result.forecast        # Primary forecast (median)
result.mean            # Mean forecast
result.lower           # Lower bound (2.5th percentile)
result.upper           # Upper bound (97.5th percentile)
result.std             # Standard deviation at each step
result.paths           # All paths: shape (n_paths, steps)
result.probabilities   # Path weights: shape (n_paths,)
result.n_paths         # Number of paths
result.n_steps         # Forecast horizon
result.dates           # Forecast dates (if provided)
result.metadata        # Additional info dict
```

### Methods

```python
# Confidence interval at any level
lower, upper = result.ci(0.90)  # 90% CI
lower, upper = result.ci(0.99)  # 99% CI

# Any quantile
q10 = result.quantile(0.10)
q90 = result.quantile(0.90)

# Export to DataFrame
df = result.to_frame()
```

---

## With Dates

```python
import datetime

# Create date array
dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i)
         for i in range(len(prices))]

# Forecaster with dates
model = ft.Forecaster(prices, dates=dates)
result = model.predict(steps=30)

# Forecast includes future dates
print(result.dates)
```

---

## Exogenous Variables

Include external predictors in your forecast:

```python
# Exogenous variables must align with prices
model = ft.Forecaster(
    prices,
    exogenous={
        'VIX': vix_values,
        'volume': volume_values,
    }
)

result = model.predict(steps=30)
```

The forecaster analyzes how exogenous variables relate to the target series and incorporates this into path generation.

---

## Custom Path Weights

Control how paths are weighted:

```python
model = ft.Forecaster(
    prices,
    path_weights={
        'hurst': 0.5,       # Weight for Hurst consistency
        'volatility': 0.3,  # Weight for volatility matching
        'pattern': 0.2,     # Weight for pattern similarity
    }
)
```

Default weights give equal importance to all factors.

---

## Time Warping

Enable Mandelbrot's trading time concept:

```python
model = ft.Forecaster(prices, time_warp=True)
result = model.predict(steps=30)
```

Time warping:
- Compresses time during high-volatility periods
- Expands time during calm periods
- Produces more realistic volatility clustering

---

## Accessing Internal State

```python
# Get the underlying analyzer
analyzer = model.analyzer

# Hurst exponent
print(f"Hurst: {model.hurst}")  # Shortcut

# Regime
print(f"Regime: {model.regime}")  # Shortcut

# Full analysis
print(analyzer.summary())
```

---

## Forecast Visualization

```python
# Plot forecast
ft.plot(result)

# Custom title
ft.plot(result, title="30-Day Price Forecast")

# Don't show immediately
fig = ft.plot(result, show=False)
fig.write_html("forecast.html")
```

---

## Example: Complete Workflow

```python
import fractime as ft
import numpy as np

# 1. Load data
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# 2. Analyze first
analyzer = ft.Analyzer(prices)
print(f"Hurst: {analyzer.hurst}")
print(f"Regime: {analyzer.regime}")

# 3. Create forecaster
model = ft.Forecaster(prices)

# 4. Generate forecast
result = model.predict(steps=30, n_paths=1000)

# 5. Examine results
print(f"Current price: {prices[-1]:.2f}")
print(f"30-day forecast: {result.forecast[-1]:.2f}")
print(f"Expected return: {(result.forecast[-1]/prices[-1] - 1)*100:.1f}%")

# 6. Uncertainty
lower, upper = result.ci(0.95)
print(f"95% CI: ({lower[-1]:.2f}, {upper[-1]:.2f})")

# 7. Probability of increase
prob_up = np.mean(result.paths[:, -1] > prices[-1])
print(f"P(increase): {prob_up:.1%}")

# 8. Visualize
ft.plot(result)
```

---

## Tips for Better Forecasts

### 1. Use Sufficient History

```python
# Minimum 100 points, ideally 250+
model = ft.Forecaster(prices[-252:])  # Last year
```

### 2. Match Horizon to Data Frequency

| Data Frequency | Reasonable Horizon |
|----------------|-------------------|
| Daily | 5-30 days |
| Weekly | 4-12 weeks |
| Monthly | 3-12 months |

### 3. Increase Paths for Stability

```python
# More paths = more stable estimates
result = model.predict(steps=30, n_paths=5000)
```

### 4. Consider Time Warping for Volatile Series

```python
# If volatility clusters, enable time warping
model = ft.Forecaster(prices, time_warp=True)
```

---

## Next Steps

- [Ensemble Methods](ensemble.md) - Combine multiple models
- [Simulator API](../api/simulator.md) - Direct path generation
- [Examples](../examples/basic.md) - Real-world usage
