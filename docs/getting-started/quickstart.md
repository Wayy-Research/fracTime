# Quick Start

Get started with FracTime in 5 minutes.

---

## Basic Analysis

```python
import fractime as ft
import numpy as np

# Create sample data (or use your own prices)
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Create analyzer
analyzer = ft.Analyzer(prices)

# Get fractal properties
print(f"Hurst exponent: {analyzer.hurst}")
print(f"Fractal dimension: {analyzer.fractal_dim}")
print(f"Volatility: {analyzer.volatility}")
print(f"Regime: {analyzer.regime}")
```

Output:

```
Hurst exponent: hurst=0.5723
Fractal dimension: fractal_dim=1.4277
Volatility: volatility=0.1892
Regime: trending
```

---

## Understanding Results

### Hurst Exponent

The Hurst exponent (H) measures long-term memory:

| Value | Interpretation | Market Behavior |
|-------|----------------|-----------------|
| H < 0.5 | Anti-persistent | Mean-reverting |
| H = 0.5 | Random walk | No memory |
| H > 0.5 | Persistent | Trending |

### Regime Detection

FracTime classifies the current regime:

- **trending**: H > 0.55 - momentum strategies work
- **mean_reverting**: H < 0.45 - reversion strategies work
- **random**: 0.45 ≤ H ≤ 0.55 - no clear pattern

---

## Three Views

Every metric supports three views:

### 1. Point Estimate

```python
# Single value
hurst_value = analyzer.hurst.value
print(f"Hurst: {hurst_value:.4f}")
```

### 2. Rolling Series

```python
# Values over time (Polars DataFrame)
rolling = analyzer.hurst.rolling
print(rolling)
# ┌─────────────────────┬──────────┐
# │ index               │ value    │
# ├─────────────────────┼──────────┤
# │ 63                  │ 0.5823   │
# │ 64                  │ 0.5891   │
# │ ...                 │ ...      │
# └─────────────────────┴──────────┘
```

### 3. Bootstrap Distribution

```python
# Uncertainty quantification
ci = analyzer.hurst.ci(0.95)
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

std = analyzer.hurst.std
print(f"Standard error: {std:.4f}")
```

---

## Forecasting

```python
# Create forecaster
model = ft.Forecaster(prices)

# Generate forecast
result = model.predict(steps=30, n_paths=1000)

# Access results
print(f"Forecast: {result.forecast[-1]:.2f}")
print(f"Mean: {result.mean[-1]:.2f}")

# Confidence interval
lower, upper = result.ci(0.95)
print(f"95% CI: ({lower[-1]:.2f}, {upper[-1]:.2f})")
```

---

## Visualization

```python
# Plot forecast
ft.plot(result)

# Plot analysis dashboard
ft.plot(analyzer)

# Plot single metric
ft.plot(analyzer.hurst)

# Plot rolling values
ft.plot(analyzer.hurst, view='rolling')

# Plot bootstrap distribution
ft.plot(analyzer.hurst, view='distribution')
```

---

## Convenience Functions

For quick one-liners:

```python
# Quick analysis
result = ft.analyze(prices)
print(result.summary())

# Quick forecast
forecast = ft.forecast(prices, steps=30)
print(forecast.forecast)
```

---

## Next Steps

- [Core Concepts](../guide/concepts.md) - Understand fractal analysis
- [Forecasting Guide](../guide/forecasting.md) - Advanced forecasting
- [API Reference](../api/analyzer.md) - Complete documentation
- [Examples](../examples/basic.md) - Real-world usage
