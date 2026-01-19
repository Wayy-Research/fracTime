# Results API Reference

FracTime uses specialized result objects to provide rich access to analysis and forecast outputs.

---

## Metric

Represents a single metric with three views: point estimate, rolling series, and bootstrap distribution.

### Overview

```python
import fractime as ft

analyzer = ft.Analyzer(prices)
metric = analyzer.hurst

# Three views
metric.value         # Point estimate
metric.rolling       # Rolling series
metric.distribution  # Bootstrap samples
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `value` | float | Point estimate |
| `rolling` | DataFrame | Rolling values (Polars) |
| `rolling_values` | array | Rolling values (numpy) |
| `distribution` | array | Bootstrap samples |
| `std` | float | Standard error |
| `median` | float | Median of distribution |

### Methods

#### ci(level)

Get confidence interval at specified level.

```python
lower, upper = metric.ci(0.95)  # 95% CI
lower, upper = metric.ci(0.90)  # 90% CI
```

#### quantile(q)

Get quantile from bootstrap distribution.

```python
q10 = metric.quantile(0.10)  # 10th percentile
q90 = metric.quantile(0.90)  # 90th percentile
```

### Type Conversion

```python
# Use as float
hurst_float = float(analyzer.hurst)

# String representation
print(str(analyzer.hurst))    # "0.5723"
print(repr(analyzer.hurst))   # "hurst=0.5723"
```

### Example

```python
import fractime as ft

analyzer = ft.Analyzer(prices, n_samples=1000)

# Point estimate
print(f"Hurst: {analyzer.hurst.value:.4f}")

# With uncertainty
ci = analyzer.hurst.ci(0.95)
print(f"95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
print(f"Std Error: {analyzer.hurst.std:.4f}")

# Rolling analysis
rolling = analyzer.hurst.rolling
print(rolling.head())

# Full distribution
dist = analyzer.hurst.distribution
print(f"Bootstrap samples: {len(dist)}")
```

---

## AnalysisResult

Contains all metrics from an analysis.

### Overview

```python
import fractime as ft

analyzer = ft.Analyzer(prices)
result = analyzer.result
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `hurst` | Metric | Hurst exponent |
| `fractal_dim` | Metric | Fractal dimension |
| `volatility` | Metric | Annualized volatility |
| `regime` | str | Market regime |
| `regime_probabilities` | dict | Regime probabilities |

### Methods

#### summary()

Generate text summary.

```python
summary = result.summary()
print(summary)
```

#### to_frame()

Export to Polars DataFrame.

```python
df = result.to_frame()
print(df)
```

### String Representation

```python
print(repr(result))
# AnalysisResult(hurst=0.5723, fractal_dim=1.4277, volatility=0.1892, regime='trending')
```

### Example

```python
import fractime as ft

analyzer = ft.Analyzer(prices)
result = analyzer.result

print(f"Hurst: {result.hurst.value:.3f}")
print(f"Fractal Dim: {result.fractal_dim.value:.3f}")
print(f"Volatility: {result.volatility.value:.1%}")
print(f"Regime: {result.regime}")
print(f"Regime Probs: {result.regime_probabilities}")

print(result.summary())
```

---

## ForecastResult

Contains forecast paths and derived statistics.

### Overview

```python
import fractime as ft

model = ft.Forecaster(prices)
result = model.predict(steps=30, n_paths=1000)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `forecast` | array | Primary forecast (median) |
| `mean` | array | Mean forecast |
| `lower` | array | Lower bound (2.5th percentile) |
| `upper` | array | Upper bound (97.5th percentile) |
| `std` | array | Standard deviation |
| `paths` | array | All paths (n_paths × steps) |
| `probabilities` | array | Path weights (n_paths,) |
| `n_paths` | int | Number of paths |
| `n_steps` | int | Forecast horizon |
| `dates` | array | Forecast dates (if provided) |
| `metadata` | dict | Additional information |

### Methods

#### ci(level)

Get confidence interval at specified level.

```python
lower, upper = result.ci(0.95)  # Arrays
lower, upper = result.ci(0.90)

# Access final step
print(f"95% CI at step 30: ({lower[-1]:.2f}, {upper[-1]:.2f})")
```

#### quantile(q)

Get quantile forecast.

```python
q10 = result.quantile(0.10)  # 10th percentile path
q50 = result.quantile(0.50)  # Median (same as forecast)
q90 = result.quantile(0.90)  # 90th percentile path
```

#### to_frame()

Export to Polars DataFrame.

```python
df = result.to_frame()
print(df)
# ┌──────┬──────────┬─────────┬─────────┐
# │ step │ forecast │ lower   │ upper   │
# ├──────┼──────────┼─────────┼─────────┤
# │ 1    │ 101.23   │ 98.45   │ 104.01  │
# │ 2    │ 101.89   │ 97.21   │ 106.57  │
# │ ...  │ ...      │ ...     │ ...     │
# └──────┴──────────┴─────────┴─────────┘
```

### Metadata

```python
# Access metadata
print(result.metadata)
# {'hurst': 0.5723, 'regime': 'trending', ...}

# Common keys
print(result.metadata['hurst'])
print(result.metadata['regime'])
```

### Example

```python
import fractime as ft
import numpy as np

model = ft.Forecaster(prices)
result = model.predict(steps=30, n_paths=1000)

# Primary forecast
print(f"30-day forecast: {result.forecast[-1]:.2f}")

# Uncertainty
lower, upper = result.ci(0.95)
print(f"95% CI: ({lower[-1]:.2f}, {upper[-1]:.2f})")

# Standard deviation over time
print(f"Day 1 std: {result.std[0]:.2f}")
print(f"Day 30 std: {result.std[-1]:.2f}")

# Probability of increase
current = prices[-1]
prob_up = np.mean(result.paths[:, -1] > current)
print(f"P(increase): {prob_up:.1%}")

# Specific quantiles
q5 = result.quantile(0.05)
q95 = result.quantile(0.95)
print(f"5th percentile: {q5[-1]:.2f}")
print(f"95th percentile: {q95[-1]:.2f}")

# Export
df = result.to_frame()
df.write_csv("forecast.csv")
```

---

## Working with Results

### Combining with Polars

```python
import polars as pl
import fractime as ft

# Get rolling Hurst
analyzer = ft.Analyzer(prices, dates=dates, window=63)
rolling = analyzer.hurst.rolling

# Filter and transform with Polars
result = (
    rolling
    .filter(pl.col('value') > 0.5)
    .with_columns(
        (pl.col('value') - 0.5).alias('excess_hurst')
    )
)
```

### Custom Statistics from Paths

```python
result = model.predict(steps=30, n_paths=5000)

# Weighted statistics (using path probabilities)
weights = result.probabilities
paths = result.paths

# Weighted mean (already available as result.mean)
weighted_mean = np.average(paths, axis=0, weights=weights)

# Custom percentiles
p1 = np.percentile(paths[:, -1], 1)   # 1st percentile
p99 = np.percentile(paths[:, -1], 99)  # 99th percentile

# Value at Risk
var_95 = np.percentile(paths[:, -1], 5)

# Expected Shortfall
es_95 = np.mean(paths[:, -1][paths[:, -1] <= var_95])
```

---

## See Also

- [Analyzer](analyzer.md) - Creates AnalysisResult
- [Forecaster](forecaster.md) - Creates ForecastResult
- [Visualization](visualization.md) - Plotting results
