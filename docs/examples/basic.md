# Basic Analysis Example

A complete walkthrough of fractal analysis with FracTime.

---

## Setup

```python
import fractime as ft
import numpy as np
```

---

## Generate Sample Data

```python
# Simulate a time series with fractal properties
np.random.seed(42)
n = 500

# Create trending data with some noise
trend = np.linspace(0, 20, n)
noise = np.random.randn(n).cumsum() * 0.5
prices = 100 + trend + noise

print(f"Data points: {len(prices)}")
print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
```

---

## Analyze Fractal Properties

```python
# Create analyzer
analyzer = ft.Analyzer(prices)

# Get point estimates
print(f"Hurst Exponent: {analyzer.hurst.value:.4f}")
print(f"Fractal Dimension: {analyzer.fractal_dim.value:.4f}")
print(f"Volatility: {analyzer.volatility.value:.2%}")
print(f"Regime: {analyzer.regime}")
```

### Interpret Results

```python
h = analyzer.hurst.value

if h > 0.55:
    print("Series is TRENDING (persistent)")
    print("→ Past trends tend to continue")
    print("→ Momentum strategies may work")
elif h < 0.45:
    print("Series is MEAN-REVERTING (anti-persistent)")
    print("→ Movements tend to reverse")
    print("→ Reversion strategies may work")
else:
    print("Series is RANDOM (no memory)")
    print("→ No clear pattern")
    print("→ Difficult to predict")
```

---

## Uncertainty Quantification

```python
# Get confidence intervals
h_ci = analyzer.hurst.ci(0.95)
fd_ci = analyzer.fractal_dim.ci(0.95)

print(f"\nHurst: {analyzer.hurst.value:.4f}")
print(f"  95% CI: ({h_ci[0]:.4f}, {h_ci[1]:.4f})")
print(f"  Std Error: {analyzer.hurst.std:.4f}")

print(f"\nFractal Dim: {analyzer.fractal_dim.value:.4f}")
print(f"  95% CI: ({fd_ci[0]:.4f}, {fd_ci[1]:.4f})")
```

---

## Rolling Analysis

```python
import datetime

# Create dates
dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i)
         for i in range(len(prices))]

# Analyzer with rolling window
analyzer = ft.Analyzer(prices, dates=dates, window=63)

# Get rolling Hurst
rolling_hurst = analyzer.hurst.rolling
print(rolling_hurst.head())
```

### Classify Periods

```python
import polars as pl

# Add regime classification
result = rolling_hurst.with_columns(
    pl.when(pl.col('value') > 0.55)
      .then(pl.lit('trending'))
      .when(pl.col('value') < 0.45)
      .then(pl.lit('mean_reverting'))
      .otherwise(pl.lit('random'))
      .alias('regime')
)

# Count regimes
regime_counts = result.group_by('regime').count()
print(regime_counts)
```

---

## Regime Probabilities

```python
# Bootstrap-based regime probabilities
probs = analyzer.regime_probabilities
print("\nRegime Probabilities:")
for regime, prob in probs.items():
    print(f"  {regime}: {prob:.1%}")
```

---

## Multi-Dimensional Analysis

```python
# Generate volume data
np.random.seed(43)
volumes = np.random.randint(1000, 10000, n).astype(float)

# Analyze multiple series together
multi_analyzer = ft.Analyzer({
    'price': prices,
    'volume': volumes,
})

print(f"\nDimensions: {multi_analyzer.dimensions}")

# Individual analysis
for dim in multi_analyzer.dimensions:
    h = multi_analyzer[dim].hurst.value
    print(f"{dim}: H={h:.4f}")

# Cross-dimensional coherence
print(f"\nCoherence: {multi_analyzer.coherence.value:.4f}")
```

---

## Visualization

```python
# Analysis dashboard
ft.plot(analyzer, title="Fractal Analysis Dashboard")

# Rolling Hurst
ft.plot(analyzer.hurst, view='rolling', title="Rolling Hurst Exponent")

# Bootstrap distribution
ft.plot(analyzer.hurst, view='distribution', title="Hurst Distribution")
```

---

## Export Results

```python
# Get AnalysisResult
result = analyzer.result

# Print summary
print(result.summary())

# Export to DataFrame
df = result.to_frame()
print(df)

# Save to file
df.write_csv("analysis_results.csv")
```

---

## Complete Script

```python
import fractime as ft
import numpy as np
import datetime

# 1. Generate data
np.random.seed(42)
n = 500
prices = 100 + np.linspace(0, 20, n) + np.random.randn(n).cumsum() * 0.5
dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i)
         for i in range(n)]

# 2. Create analyzer
analyzer = ft.Analyzer(prices, dates=dates, window=63, n_samples=1000)

# 3. Get results
print("=== Fractal Analysis ===")
print(f"Hurst: {analyzer.hurst.value:.4f} (CI: {analyzer.hurst.ci(0.95)})")
print(f"Fractal Dim: {analyzer.fractal_dim.value:.4f}")
print(f"Volatility: {analyzer.volatility.value:.2%}")
print(f"Regime: {analyzer.regime}")

# 4. Regime probabilities
print("\nRegime Probabilities:")
for regime, prob in analyzer.regime_probabilities.items():
    print(f"  {regime}: {prob:.1%}")

# 5. Visualize
ft.plot(analyzer, title="Analysis Dashboard")
ft.plot(analyzer.hurst, view='rolling', title="Rolling Hurst")
```

---

## Next Steps

- [Forecasting Example](comparison.md) - Generate forecasts
- [Advanced Usage](real-world.md) - Real-world applications
- [API Reference](../api/analyzer.md) - Complete documentation
