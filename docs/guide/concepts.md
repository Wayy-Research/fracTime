# Core Concepts

Understanding fractal analysis and how FracTime uses it for forecasting.

---

## What is Fractal Analysis?

Fractal geometry, pioneered by Benoit Mandelbrot, describes patterns that exhibit **self-similarity** across scales. Financial time series often display fractal characteristics:

- Price movements look similar whether viewed hourly, daily, or monthly
- Volatility clusters persist across time scales
- Extreme events occur more frequently than Gaussian models predict

Traditional methods assume:

- Normal distributions
- Statistical independence
- Short-term memory only

FracTime captures what traditional methods miss.

---

## The Hurst Exponent

The Hurst exponent (H) is the fundamental measure of **long-term memory** in a time series.

### Interpretation

| H Value | Name | Behavior | Trading Implication |
|---------|------|----------|---------------------|
| H < 0.5 | Anti-persistent | Mean-reverting | Reversals likely |
| H = 0.5 | Random walk | No memory | Unpredictable |
| H > 0.5 | Persistent | Trending | Trends continue |

### Example

```python
import fractime as ft

analyzer = ft.Analyzer(prices)

# Point estimate
print(f"Hurst: {analyzer.hurst.value:.3f}")

# With confidence interval
ci = analyzer.hurst.ci(0.95)
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

# Over time
print(analyzer.hurst.rolling)
```

### Estimation Methods

FracTime supports two methods:

| Method | Name | Best For |
|--------|------|----------|
| `'rs'` | Rescaled Range | General use, more robust |
| `'dfa'` | Detrended Fluctuation Analysis | Non-stationary series |

```python
# Using DFA method
analyzer = ft.Analyzer(prices, method='dfa')
```

---

## Fractal Dimension

The fractal dimension measures the **complexity** or "roughness" of a time series.

### Interpretation

| Dimension | Interpretation |
|-----------|----------------|
| D ≈ 1.0 | Smooth, trending |
| D ≈ 1.5 | Moderate complexity |
| D ≈ 2.0 | Very rough, noisy |

### Relationship with Hurst

For self-affine series:

```
D = 2 - H
```

Where:
- Higher H (trending) → Lower D (smoother)
- Lower H (mean-reverting) → Higher D (rougher)

```python
analyzer = ft.Analyzer(prices)
print(f"Fractal dimension: {analyzer.fractal_dim.value:.3f}")
```

---

## Market Regimes

FracTime classifies the current market regime based on the Hurst exponent:

| Regime | Hurst Range | Characteristics |
|--------|-------------|-----------------|
| **trending** | H > 0.55 | Momentum persists |
| **random** | 0.45 ≤ H ≤ 0.55 | No clear pattern |
| **mean_reverting** | H < 0.45 | Prices revert to mean |

### Usage

```python
analyzer = ft.Analyzer(prices)

# Current regime
print(f"Regime: {analyzer.regime}")

# Regime probabilities (from bootstrap)
print(analyzer.regime_probabilities)
# {'trending': 0.72, 'random': 0.21, 'mean_reverting': 0.07}
```

### Rolling Regime Detection

```python
import polars as pl

analyzer = ft.Analyzer(prices, dates=dates, window=63)
rolling = analyzer.hurst.rolling

# Classify each period
result = rolling.with_columns(
    pl.when(pl.col('value') > 0.55)
      .then(pl.lit('trending'))
      .when(pl.col('value') < 0.45)
      .then(pl.lit('mean_reverting'))
      .otherwise(pl.lit('random'))
      .alias('regime')
)
```

---

## How Forecasting Works

FracTime's forecasting process:

### 1. Fractal Analysis

First, analyze the historical data:

- Compute Hurst exponent
- Estimate fractal dimension
- Measure volatility structure
- Detect current regime

### 2. Path Generation

Generate future scenarios using one of:

| Method | Description | When to Use |
|--------|-------------|-------------|
| **pattern** | Match historical patterns | Long history available |
| **fbm** | Fractional Brownian Motion | Shorter history |
| **bootstrap** | Resample historical returns | Preserve distribution |

### 3. Probability Weighting

Each simulated path receives a probability weight based on:

- Hurst consistency with historical data
- Volatility pattern matching
- Multi-scale trend alignment

### 4. Forecast Generation

The final forecast is the probability-weighted median:

```python
model = ft.Forecaster(prices)
result = model.predict(steps=30, n_paths=1000)

# Primary forecast (probability-weighted median)
result.forecast

# Mean forecast
result.mean

# All paths with their probabilities
result.paths        # Shape: (1000, 30)
result.probabilities  # Shape: (1000,)
```

---

## Fractional Brownian Motion

Fractional Brownian Motion (fBm) extends standard Brownian motion to incorporate long-term memory:

- Standard Brownian motion assumes H = 0.5 (random walk)
- fBm uses the estimated Hurst exponent to preserve memory characteristics

### Properties

| Property | Standard BM | Fractional BM |
|----------|-------------|---------------|
| Memory | None | Long-term |
| Increments | Independent | Correlated |
| Hurst | 0.5 | Variable (0-1) |

```python
sim = ft.Simulator(prices)
paths = sim.generate(n_paths=1000, steps=30, method='fbm')
```

---

## Time Warping

Mandelbrot observed that markets have their own sense of "trading time" that differs from clock time:

- Volatile periods: Time moves faster (more information per unit time)
- Calm periods: Time moves slower

FracTime can incorporate this:

```python
# Enable time warping
model = ft.Forecaster(prices, time_warp=True)
result = model.predict(steps=30)
```

This produces more realistic paths during high-volatility regimes.

---

## Multi-Dimensional Analysis

Analyze relationships between multiple related series:

```python
analyzer = ft.Analyzer({
    'price': prices,
    'volume': volumes,
})

# Individual analysis
print(analyzer['price'].hurst)
print(analyzer['volume'].hurst)

# Cross-dimensional coherence
print(f"Coherence: {analyzer.coherence.value:.3f}")
```

The coherence measure indicates how aligned the fractal properties are across dimensions. High coherence suggests the series move together at a fundamental level.

---

## Next Steps

- [Forecasting Guide](forecasting.md) - Detailed forecasting usage
- [Ensemble Methods](ensemble.md) - Combining multiple models
- [API Reference](../api/analyzer.md) - Complete documentation
