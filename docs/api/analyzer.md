# Analyzer API Reference

The `Analyzer` class computes fractal properties of time series data.

---

## Overview

```python
import fractime as ft

analyzer = ft.Analyzer(prices)
print(analyzer.hurst)        # Hurst exponent
print(analyzer.fractal_dim)  # Fractal dimension
print(analyzer.volatility)   # Annualized volatility
print(analyzer.regime)       # Market regime
```

---

## Constructor

```python
ft.Analyzer(
    data,
    dates=None,
    method='rs',
    window=63,
    n_samples=1000,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array or dict | required | Price series or dict of series |
| `dates` | array | None | Corresponding dates |
| `method` | str | 'rs' | Hurst estimation method |
| `window` | int | 63 | Rolling window size |
| `n_samples` | int | 1000 | Bootstrap samples for CI |

### Data Types

The `data` parameter accepts:

- **numpy array**: Single time series
- **Polars Series**: Single time series
- **dict**: Multiple time series for multi-dimensional analysis

```python
# Single series
analyzer = ft.Analyzer(prices)

# Multiple series
analyzer = ft.Analyzer({
    'price': prices,
    'volume': volumes,
})
```

---

## Properties

### hurst

The Hurst exponent as a `Metric` object.

```python
analyzer.hurst.value       # Point estimate (float)
analyzer.hurst.rolling     # Rolling values (Polars DataFrame)
analyzer.hurst.distribution  # Bootstrap samples (numpy array)
analyzer.hurst.std         # Standard error
analyzer.hurst.ci(0.95)    # 95% confidence interval
```

### fractal_dim

Fractal dimension as a `Metric` object.

```python
analyzer.fractal_dim.value  # Point estimate
analyzer.fractal_dim.rolling
analyzer.fractal_dim.ci(0.95)
```

### volatility

Annualized volatility as a `Metric` object.

```python
analyzer.volatility.value   # Point estimate
analyzer.volatility.rolling
analyzer.volatility.ci(0.95)
```

### regime

Current market regime classification.

```python
analyzer.regime  # 'trending', 'mean_reverting', or 'random'
```

### regime_probabilities

Bootstrap-based regime probabilities.

```python
analyzer.regime_probabilities
# {'trending': 0.72, 'random': 0.21, 'mean_reverting': 0.07}
```

### result

Returns an `AnalysisResult` object containing all metrics.

```python
result = analyzer.result
print(result.summary())
```

### dimensions

For multi-dimensional analysis, list of dimension names.

```python
analyzer = ft.Analyzer({'price': prices, 'volume': volumes})
print(analyzer.dimensions)  # ['price', 'volume']
```

### coherence

For multi-dimensional analysis, cross-dimensional coherence.

```python
analyzer.coherence.value  # 0.0 to 1.0
```

---

## Methods

### summary()

Generate a text summary of the analysis.

```python
summary = analyzer.summary()
print(summary)
```

### __getitem__()

Access individual dimensions in multi-dimensional analysis.

```python
analyzer = ft.Analyzer({'price': prices, 'volume': volumes})
price_analyzer = analyzer['price']
print(price_analyzer.hurst)
```

---

## Multi-Dimensional Analysis

Analyze multiple related series together:

```python
analyzer = ft.Analyzer({
    'price': prices,
    'volume': volumes,
    'volatility': vol_index,
})

# Individual metrics
for dim in analyzer.dimensions:
    print(f"{dim}: H={analyzer[dim].hurst.value:.3f}")

# Cross-dimensional coherence
print(f"Coherence: {analyzer.coherence.value:.3f}")
```

---

## Estimation Methods

### Rescaled Range (R/S)

```python
analyzer = ft.Analyzer(prices, method='rs')
```

- Classic Hurst estimator
- More robust to outliers
- Works well for most series

### Detrended Fluctuation Analysis (DFA)

```python
analyzer = ft.Analyzer(prices, method='dfa')
```

- Better for non-stationary series
- Removes polynomial trends
- Preferred for financial data with trends

---

## Rolling Analysis

```python
# Set window size
analyzer = ft.Analyzer(prices, dates=dates, window=63)

# Get rolling Hurst
rolling_hurst = analyzer.hurst.rolling
print(rolling_hurst)
# ┌─────────────────────┬──────────┐
# │ date                │ value    │
# ├─────────────────────┼──────────┤
# │ 2023-04-01 00:00:00 │ 0.6234   │
# │ 2023-04-02 00:00:00 │ 0.6312   │
# │ ...                 │ ...      │
# └─────────────────────┴──────────┘

# Work with Polars
import polars as pl
high_hurst = rolling_hurst.filter(pl.col('value') > 0.6)
```

---

## Bootstrap Confidence Intervals

```python
# More samples = tighter estimates
analyzer = ft.Analyzer(prices, n_samples=2000)

# 95% CI
ci = analyzer.hurst.ci(0.95)
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# Access full distribution
dist = analyzer.hurst.distribution
print(f"Samples: {len(dist)}")
```

---

## Lazy Computation

The Analyzer uses lazy computation - metrics are only calculated when accessed:

```python
analyzer = ft.Analyzer(prices)  # Fast - no computation yet

# First access triggers computation (cached)
h = analyzer.hurst.value  # Computes Hurst

# Subsequent access uses cache
h = analyzer.hurst.value  # Instant - uses cached value
```

---

## Examples

### Basic Analysis

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

analyzer = ft.Analyzer(prices)
print(f"Hurst: {analyzer.hurst.value:.3f}")
print(f"Fractal Dim: {analyzer.fractal_dim.value:.3f}")
print(f"Volatility: {analyzer.volatility.value:.1%}")
print(f"Regime: {analyzer.regime}")
```

### With Uncertainty

```python
analyzer = ft.Analyzer(prices, n_samples=1000)

h_ci = analyzer.hurst.ci(0.95)
print(f"Hurst: {analyzer.hurst.value:.3f} ({h_ci[0]:.3f}, {h_ci[1]:.3f})")
```

### Rolling Regime Detection

```python
import polars as pl

analyzer = ft.Analyzer(prices, dates=dates, window=63)
rolling = analyzer.hurst.rolling

result = rolling.with_columns(
    pl.when(pl.col('value') > 0.55)
      .then(pl.lit('trending'))
      .when(pl.col('value') < 0.45)
      .then(pl.lit('mean_reverting'))
      .otherwise(pl.lit('random'))
      .alias('regime')
)

print(result.group_by('regime').count())
```

---

## See Also

- [Metric](results.md#metric) - Understanding metric objects
- [AnalysisResult](results.md#analysisresult) - Result container
- [Core Concepts](../guide/concepts.md) - Fractal analysis background
