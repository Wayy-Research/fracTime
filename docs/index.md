# FracTime

**Fractal-based time series analysis and forecasting.**

FracTime uses fractal geometry to analyze and forecast time series data. It computes properties like the Hurst exponent (which measures long-term memory) and uses Monte Carlo simulation to generate probabilistic forecasts.

---

## Why FracTime?

Traditional forecasting methods assume:

- **Normal distributions** - but real data has fat tails
- **Statistical independence** - but markets have memory
- **Short-term dependencies** - but past events affect the distant future

FracTime recognizes that time series data often exhibits:

| Property | What It Means | How FracTime Uses It |
|----------|---------------|---------------------|
| **Long-term memory** | Past events influence the distant future | Hurst exponent modeling |
| **Self-similarity** | Patterns repeat across time scales | Fractal dimension analysis |
| **Regime changes** | Markets shift between trending and mean-reverting | Regime detection |
| **Fat tails** | Extreme events occur more than expected | Monte Carlo with realistic paths |

---

## Features

### Simple, Composable API

```python
import fractime as ft

# Analyze
analyzer = ft.Analyzer(prices)
print(analyzer.hurst)           # Point estimate
print(analyzer.hurst.rolling)   # Rolling values
print(analyzer.hurst.ci(0.95))  # Confidence interval

# Forecast
model = ft.Forecaster(prices)
result = model.predict(steps=30)

# Plot
ft.plot(result)
```

### Three Views for Every Metric

Every analysis metric supports three views:

1. **Point estimate** - Single value (`analyzer.hurst.value`)
2. **Rolling series** - Time-varying values (`analyzer.hurst.rolling`)
3. **Distribution** - Bootstrap samples (`analyzer.hurst.distribution`)

### High Performance

- **Numba JIT compilation** for core algorithms
- **Lazy computation** - only computes what you access
- **Automatic caching** - repeated access is instant
- **Polars DataFrames** for fast data manipulation

---

## Quick Example

```python
import fractime as ft
import numpy as np

# Sample data (or use your own)
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Analyze fractal properties
analyzer = ft.Analyzer(prices)
print(f"Hurst exponent: {analyzer.hurst}")
print(f"Fractal dimension: {analyzer.fractal_dim}")
print(f"Market regime: {analyzer.regime}")

# Generate probabilistic forecast
model = ft.Forecaster(prices)
result = model.predict(steps=30, n_paths=1000)

print(f"30-day forecast: {result.forecast[-1]:.2f}")
print(f"95% CI: {result.ci(0.95)}")

# Visualize
ft.plot(result)
```

---

## Core Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **Analyzer** | Compute fractal properties | `ft.Analyzer(prices).hurst` |
| **Forecaster** | Probabilistic forecasting | `ft.Forecaster(prices).predict(30)` |
| **Simulator** | Monte Carlo path generation | `ft.Simulator(prices).generate(1000)` |
| **Ensemble** | Combine multiple models | `ft.Ensemble(prices).predict(30)` |

---

## What's New in v0.3.0

- **Unified API** - One composable interface for everything
- **Three-view metrics** - Point, rolling, and distribution views
- **Lazy computation** - Fast by default, detailed on demand
- **Polars integration** - Fast DataFrames for rolling analysis
- **Simplified forecasting** - Clean `Forecaster` class
- **Better visualization** - Single `plot()` function handles everything

---

## Installation

```bash
pip install fractime
```

Or from source:

```bash
git clone https://github.com/wayy-research/fracTime.git
cd fracTime
pip install -e .
```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Quick Start**

    ---

    Get up and running in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Core Concepts**

    ---

    Understand fractal analysis fundamentals

    [:octicons-arrow-right-24: Concepts](guide/concepts.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation

    [:octicons-arrow-right-24: Analyzer](api/analyzer.md)

-   :material-test-tube:{ .lg .middle } **Examples**

    ---

    Real-world usage examples

    [:octicons-arrow-right-24: Examples](examples/basic.md)

</div>
