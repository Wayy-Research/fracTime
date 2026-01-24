# FracTime

**Fractal-based time series analysis and forecasting.**

FracTime uses fractal geometry to analyze and forecast time series data. It computes properties like the Hurst exponent (which tells you if a series is trending or mean-reverting) and uses Monte Carlo simulation to generate probabilistic forecasts.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [What is Fractal Analysis?](#what-is-fractal-analysis)
  - [The Hurst Exponent](#the-hurst-exponent)
  - [Fractal Dimension](#fractal-dimension)
  - [Market Regimes](#market-regimes)
- [API Reference](#api-reference)
  - [Analyzer](#analyzer)
  - [Forecaster](#forecaster)
  - [Simulator](#simulator)
  - [Ensemble](#ensemble)
  - [Result Objects](#result-objects)
  - [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
  - [Rolling Analysis](#rolling-analysis)
  - [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
  - [Multi-Dimensional Analysis](#multi-dimensional-analysis)
  - [Exogenous Variables](#exogenous-variables)
  - [Custom Path Weights](#custom-path-weights)
  - [Time Warping](#time-warping)
- [Examples](#examples)
- [Performance](#performance)
- [License](#license)

---

## Installation

```bash
pip install fractime
```

Or install from source:

```bash
git clone https://github.com/wayy-research/fracTime.git
cd fracTime
pip install -e .
```

### Dependencies

FracTime uses:
- **NumPy** and **Polars** for fast data manipulation
- **Numba** for JIT-compiled performance-critical code
- **wrchart** for interactive visualizations
- **scikit-learn** for clustering and preprocessing

---

## Quick Start

### 30-Second Example

```python
import fractime as ft
import numpy as np

# Create sample data (or use your own prices)
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Analyze
analyzer = ft.Analyzer(prices)
print(f"Hurst: {analyzer.hurst}")        # hurst=0.5723
print(f"Regime: {analyzer.regime}")      # 'trending'

# Forecast
model = ft.Forecaster(prices)
result = model.predict(steps=30)
print(f"30-day forecast: {result.forecast[-1]:.2f}")

# Visualize
ft.plot(result)
```

### What Just Happened?

1. **Analyzer** computed the Hurst exponent (0.57 > 0.5 means the series tends to trend)
2. **Forecaster** generated 1000 Monte Carlo paths and weighted them by fractal similarity
3. **plot()** created an interactive chart showing the forecast with confidence intervals

---

## Core Concepts

### What is Fractal Analysis?

Fractal analysis examines the self-similar patterns in data across different time scales. Financial markets exhibit fractal properties - patterns that repeat at daily, weekly, monthly, and yearly scales.

FracTime uses these properties to:
1. **Understand market behavior** - Is the market trending or mean-reverting?
2. **Generate realistic forecasts** - Paths that preserve historical fractal characteristics
3. **Quantify uncertainty** - Probability distributions over future outcomes

### The Hurst Exponent

The Hurst exponent (H) is a number between 0 and 1 that describes the long-term memory of a time series:

| Hurst Value | Meaning | Market Behavior |
|-------------|---------|-----------------|
| H < 0.5 | Anti-persistent | Mean-reverting: moves tend to reverse |
| H = 0.5 | Random walk | No memory: coin flip behavior |
| H > 0.5 | Persistent | Trending: moves tend to continue |

**Example interpretation:**
- H = 0.7: Strong trending behavior. An upward move is likely followed by more upward moves.
- H = 0.3: Strong mean-reversion. An upward move is likely followed by a downward move.
- H = 0.5: Random. Past moves don't predict future moves.

```python
analyzer = ft.Analyzer(prices)
h = analyzer.hurst.value

if h > 0.55:
    print("Market is trending - momentum strategies may work")
elif h < 0.45:
    print("Market is mean-reverting - contrarian strategies may work")
else:
    print("Market is random - no clear edge from persistence")
```

### Fractal Dimension

The fractal dimension (D) measures the complexity or "roughness" of a time series:

| Dimension | Meaning |
|-----------|---------|
| D ≈ 1.0 | Smooth, simple patterns |
| D ≈ 1.5 | Moderate complexity (typical for markets) |
| D ≈ 2.0 | Highly complex, space-filling patterns |

```python
analyzer = ft.Analyzer(prices)
d = analyzer.fractal_dim.value
print(f"Fractal dimension: {d:.2f}")
```

### Market Regimes

FracTime classifies markets into three regimes based on fractal properties:

1. **Trending** (H > 0.55): Momentum-driven markets
2. **Mean-reverting** (H < 0.45): Range-bound markets
3. **Random** (0.45 ≤ H ≤ 0.55): Efficient markets with no clear pattern

```python
analyzer = ft.Analyzer(prices)
print(f"Current regime: {analyzer.regime}")
print(f"Regime probabilities: {analyzer.regime_probabilities}")
# {'trending': 0.72, 'random': 0.20, 'mean_reverting': 0.08}
```

---

## API Reference

### Analyzer

The `Analyzer` class computes fractal properties of time series data. All computations are **lazy** - they only run when you access them, and results are cached.

#### Creating an Analyzer

```python
import fractime as ft

# Basic usage
analyzer = ft.Analyzer(prices)

# With dates (enables time-indexed rolling analysis)
analyzer = ft.Analyzer(prices, dates=dates)

# With configuration
analyzer = ft.Analyzer(
    prices,
    dates=dates,
    method='rs',        # 'rs' (Rescaled Range) or 'dfa' (Detrended Fluctuation Analysis)
    window=63,          # Rolling window size (default: 63 trading days ≈ 3 months)
    n_samples=1000,     # Bootstrap samples for confidence intervals
    min_scale=10,       # Minimum scale for fractal analysis
    max_scale=100,      # Maximum scale for fractal analysis
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `hurst` | `Metric` | Hurst exponent (0-1) |
| `fractal_dim` | `Metric` | Fractal dimension (1-2) |
| `volatility` | `Metric` | Annualized volatility |
| `regime` | `str` | Current regime: 'trending', 'mean_reverting', or 'random' |
| `regime_probabilities` | `dict` | Probability distribution over regimes |
| `result` | `AnalysisResult` | All metrics as a structured object |

#### Metric Object

Each metric (hurst, fractal_dim, volatility) is a `Metric` object with three views:

```python
analyzer = ft.Analyzer(prices)

# 1. POINT ESTIMATE - Single value (always fast)
analyzer.hurst.value      # 0.67
float(analyzer.hurst)     # 0.67 (can use as float)
str(analyzer.hurst)       # "0.6700"

# 2. ROLLING - Time series (computed on first access)
analyzer.hurst.rolling    # Polars DataFrame with 'index'/'date' and 'value' columns
analyzer.hurst.rolling_values  # Just the values as numpy array

# 3. DISTRIBUTION - Bootstrap uncertainty (computed on first access)
analyzer.hurst.distribution   # Array of 1000 bootstrap samples
analyzer.hurst.std            # Standard error
analyzer.hurst.ci(0.95)       # 95% confidence interval: (0.61, 0.73)
analyzer.hurst.median         # Median of distribution
analyzer.hurst.quantile(0.75) # 75th percentile
```

#### Methods

```python
# Get text summary
print(analyzer.summary())
# Fractal Analysis Summary
# ========================================
# Hurst Exponent:    0.6700 ± 0.0400
# Fractal Dimension: 1.4300 ± 0.0200
# Volatility:        0.3200
# Regime:            trending
# ----------------------------------------
# Regime Probabilities:
#   trending: 72.0%
#   random: 20.0%
#   mean_reverting: 8.0%

# Export all rolling metrics as Polars DataFrame
df = analyzer.to_frame()
# shape: (437, 4)
# ┌───────┬─────────┬─────────────┬────────────┐
# │ index │ hurst   │ fractal_dim │ volatility │
# ├───────┼─────────┼─────────────┼────────────┤
# │ 63    │ 0.9107  │ 1.3200      │ 0.2800     │
# │ 64    │ 0.8923  │ 1.3400      │ 0.2750     │
# │ ...   │ ...     │ ...         │ ...        │
# └───────┴─────────┴─────────────┴────────────┘
```

#### Convenience Function

```python
# Quick analysis without creating analyzer instance
result = ft.analyze(prices)
print(result.hurst)   # hurst=0.6700
print(result.regime)  # 'trending'
```

---

### Forecaster

The `Forecaster` class generates probabilistic forecasts using Monte Carlo simulation weighted by fractal similarity.

#### Creating a Forecaster

```python
import fractime as ft

# Basic usage
model = ft.Forecaster(prices)

# With dates
model = ft.Forecaster(prices, dates=dates)

# With exogenous variables
model = ft.Forecaster(
    prices,
    exogenous={'VIX': vix_prices, 'bonds': bond_prices}
)

# With configuration
model = ft.Forecaster(
    prices,
    dates=dates,
    exogenous=None,           # Dict of exogenous variables
    analyzer=None,            # Pre-computed Analyzer (optional)
    lookback=252,             # Historical window for pattern matching
    method='rs',              # Hurst calculation method
    time_warp=False,          # Use Mandelbrot's trading time
    path_weights={            # Custom path probability weights
        'hurst': 0.3,
        'volatility': 0.3,
        'pattern': 0.4,
    },
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `analyzer` | `Analyzer` | The underlying analyzer |
| `hurst` | `Metric` | Shortcut to `analyzer.hurst` |
| `fractal_dim` | `Metric` | Shortcut to `analyzer.fractal_dim` |
| `regime` | `str` | Shortcut to `analyzer.regime` |

#### Predicting

```python
model = ft.Forecaster(prices)

# Basic prediction
result = model.predict(steps=30)

# With more paths (more accurate but slower)
result = model.predict(steps=30, n_paths=5000)

# With custom confidence level
result = model.predict(steps=30, confidence=0.90)
```

#### ForecastResult Object

The `predict()` method returns a `ForecastResult` object:

```python
result = model.predict(steps=30)

# Primary forecasts
result.forecast       # Median forecast (probability-weighted)
result.mean           # Mean forecast
result.lower          # 5th percentile (lower bound)
result.upper          # 95th percentile (upper bound)
result.std            # Standard deviation at each step

# Custom quantiles and intervals
result.quantile(0.25)      # 25th percentile
result.quantile(0.75)      # 75th percentile
result.ci(0.90)            # 90% confidence interval: (lower, upper)

# All paths
result.paths               # Array of shape (n_paths, n_steps)
result.probabilities       # Probability weight for each path (sums to 1)
result.n_paths             # Number of paths
result.n_steps             # Number of forecast steps

# Dates (if provided to forecaster)
result.dates               # Forecast dates

# Metadata
result.metadata            # {'hurst': 0.67, 'fractal_dim': 1.43, 'regime': 'trending', ...}

# Export to DataFrame
df = result.to_frame()
# ┌──────┬──────────┬─────────┬─────────┬─────────┬─────────┐
# │ step │ forecast │ lower   │ upper   │ mean    │ std     │
# ├──────┼──────────┼─────────┼─────────┼─────────┼─────────┤
# │ 1    │ 101.23   │ 98.50   │ 104.10  │ 101.15  │ 1.42    │
# │ 2    │ 101.89   │ 97.80   │ 106.20  │ 101.75  │ 2.15    │
# │ ...  │ ...      │ ...     │ ...     │ ...     │ ...     │
# └──────┴──────────┴─────────┴─────────┴─────────┴─────────┘
```

#### Convenience Function

```python
# Quick forecast without creating forecaster instance
result = ft.forecast(prices, steps=30, n_paths=1000)
```

---

### Simulator

The `Simulator` class generates Monte Carlo price paths directly, without the forecasting wrapper.

#### Creating a Simulator

```python
import fractime as ft

# Basic usage
sim = ft.Simulator(prices)

# With time warping (Mandelbrot's trading time)
sim = ft.Simulator(prices, time_warp=True)

# With custom configuration
sim = ft.Simulator(
    prices,
    dates=None,
    analyzer=None,        # Pre-computed Analyzer (optional)
    time_warp=False,      # Use trading time transformation
    weights={             # Path generation weights
        'hurst': 0.4,
        'volatility': 0.3,
        'pattern': 0.3,
    },
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `analyzer` | `Analyzer` | The underlying analyzer |
| `hurst` | `float` | Hurst exponent |
| `volatility` | `float` | Annualized volatility |

#### Generating Paths

```python
sim = ft.Simulator(prices)

# Generate paths
paths = sim.generate(n_paths=1000, steps=30)
# Returns: numpy array of shape (1000, 30)

# Specify generation method
paths = sim.generate(n_paths=1000, steps=30, method='auto')     # Best method based on data
paths = sim.generate(n_paths=1000, steps=30, method='fbm')      # Fractional Brownian motion
paths = sim.generate(n_paths=1000, steps=30, method='pattern')  # Pattern-based
paths = sim.generate(n_paths=1000, steps=30, method='bootstrap')# Block bootstrap
```

#### Generation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `auto` | Automatically chooses best method | General use |
| `fbm` | Fractional Brownian motion | Short histories, strong fractal properties |
| `pattern` | Historical pattern matching | Long histories with repeating patterns |
| `bootstrap` | Block bootstrap of returns | Preserving exact historical distribution |

---

### Ensemble

The `Ensemble` class combines multiple forecasters for improved predictions.

#### Creating an Ensemble

```python
import fractime as ft

# Basic usage (creates default models)
ensemble = ft.Ensemble(prices)

# With custom models
ensemble = ft.Ensemble(
    prices,
    models=[
        ft.Forecaster(prices, method='rs'),
        ft.Forecaster(prices, method='dfa'),
        ft.Forecaster(prices, path_weights={'hurst': 0.6, 'volatility': 0.2, 'pattern': 0.2}),
    ]
)

# With strategy selection
ensemble = ft.Ensemble(
    prices,
    strategy='weighted',  # 'average', 'weighted', 'stacking', 'boosting'
    meta_learner='ridge', # For stacking: 'ridge', 'linear', 'rf'
)
```

#### Combination Strategies

| Strategy | Description |
|----------|-------------|
| `average` | Simple average of all forecasts |
| `weighted` | Weights based on model diversity (default) |
| `stacking` | Meta-learner combines forecasts |
| `boosting` | Sequential error correction |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `models` | `list` | List of forecaster instances |
| `n_models` | `int` | Number of models in ensemble |

#### Predicting

```python
ensemble = ft.Ensemble(prices)

# Generate ensemble forecast
result = ensemble.predict(steps=30, n_paths=200)

# Result includes combined paths from all models
print(f"Total paths: {result.n_paths}")  # n_paths × n_models
print(f"Strategy: {result.metadata['strategy']}")
```

---

### Result Objects

#### Metric

Represents a single metric with three views: point, rolling, and distribution.

```python
# Properties
metric.value           # Point estimate (float)
metric.rolling         # Rolling values (Polars DataFrame)
metric.rolling_values  # Rolling values (numpy array)
metric.distribution    # Bootstrap samples (numpy array)
metric.std             # Standard error (float)
metric.median          # Median of distribution (float)

# Methods
metric.ci(level)       # Confidence interval at given level
metric.quantile(q)     # Quantile from distribution

# Usage as number
float(metric)          # Convert to float
str(metric)            # String of value
repr(metric)           # "name=value"
```

#### AnalysisResult

Contains all analysis metrics.

```python
# Properties
result.hurst           # Metric
result.fractal_dim     # Metric
result.volatility      # Metric
result.regime          # str
result.regime_probabilities  # dict

# Methods
result.summary()       # Text summary
result.to_frame()      # Export to Polars DataFrame
```

#### ForecastResult

Contains forecast paths and statistics.

```python
# Properties
result.forecast        # Primary forecast (numpy array)
result.mean            # Mean forecast (numpy array)
result.lower           # Lower bound (numpy array)
result.upper           # Upper bound (numpy array)
result.std             # Standard deviation (numpy array)
result.paths           # All paths (numpy array)
result.probabilities   # Path weights (numpy array)
result.n_paths         # Number of paths (int)
result.n_steps         # Number of steps (int)
result.dates           # Forecast dates (numpy array or None)
result.metadata        # Additional info (dict)

# Methods
result.quantile(q)     # Get quantile forecast
result.ci(level)       # Get confidence interval
result.to_frame()      # Export to Polars DataFrame
```

---

### Visualization

The `plot()` function creates interactive visualizations for any FracTime object.

#### Basic Usage

```python
import fractime as ft

# Plot forecast result
result = ft.Forecaster(prices).predict(steps=30)
ft.plot(result)

# Plot analysis
analyzer = ft.Analyzer(prices)
ft.plot(analyzer)

# Plot single metric
ft.plot(analyzer.hurst)

# Don't show immediately (get chart object)
chart = ft.plot(result, show=False)
chart.to_html("forecast.html")
```

#### Metric Views

```python
analyzer = ft.Analyzer(prices, dates=dates)

# Auto-detect best view
ft.plot(analyzer.hurst)

# Specify view
ft.plot(analyzer.hurst, view='point')        # Gauge chart
ft.plot(analyzer.hurst, view='rolling')      # Time series
ft.plot(analyzer.hurst, view='distribution') # Histogram
```

#### Customization

```python
ft.plot(
    result,
    title="My Custom Title",
    show=True,           # Display immediately
    height=600,
    width=1000,
    theme='dark',        # or 'light'
)
```

---

## Advanced Usage

### Rolling Analysis

See how fractal properties change over time:

```python
import fractime as ft

analyzer = ft.Analyzer(prices, dates=dates, window=63)

# Access rolling values
rolling_hurst = analyzer.hurst.rolling
print(rolling_hurst)
# ┌─────────────────────┬──────────┐
# │ date                │ value    │
# ├─────────────────────┼──────────┤
# │ 2023-04-01 00:00:00 │ 0.6234   │
# │ 2023-04-02 00:00:00 │ 0.6312   │
# │ ...                 │ ...      │
# └─────────────────────┴──────────┘

# Filter with Polars expressions
import polars as pl

# Find trending periods
trending = rolling_hurst.filter(pl.col('value') > 0.6)

# Get recent values
recent = rolling_hurst.filter(pl.col('date') > '2024-01-01')

# Plot
ft.plot(analyzer.hurst, view='rolling')
```

### Bootstrap Confidence Intervals

Quantify uncertainty in your estimates:

```python
import fractime as ft

# Configure bootstrap samples
analyzer = ft.Analyzer(prices, n_samples=2000)

# Get confidence interval
ci = analyzer.hurst.ci(0.95)
print(f"Hurst: {analyzer.hurst.value:.3f}")
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
print(f"Std Error: {analyzer.hurst.std:.4f}")

# Access full distribution
dist = analyzer.hurst.distribution
print(f"Distribution shape: {dist.shape}")  # (2000,)

# Custom quantiles
print(f"10th percentile: {analyzer.hurst.quantile(0.10):.3f}")
print(f"90th percentile: {analyzer.hurst.quantile(0.90):.3f}")

# Plot distribution
ft.plot(analyzer.hurst, view='distribution')
```

### Multi-Dimensional Analysis

Analyze relationships between multiple time series:

```python
import fractime as ft

# Create multi-dimensional analyzer
analyzer = ft.Analyzer({
    'price': prices,
    'volume': volumes,
    'volatility': realized_vol,
})

# Access individual dimensions
print(f"Dimensions: {analyzer.dimensions}")  # ['price', 'volume', 'volatility']

print(f"Price Hurst: {analyzer['price'].hurst}")
print(f"Volume Hurst: {analyzer['volume'].hurst}")
print(f"Volatility Hurst: {analyzer['volatility'].hurst}")

# Cross-dimensional coherence
# Measures how consistently fractal properties behave across dimensions
coherence = analyzer.coherence.value
print(f"Coherence: {coherence:.3f}")

# Rolling coherence
ft.plot(analyzer.coherence, view='rolling')
```

### Exogenous Variables

Incorporate external predictors into forecasts:

```python
import fractime as ft

# Prepare exogenous data (must align with prices)
exogenous = {
    'VIX': vix_prices,
    'bonds': bond_prices,
    'gold': gold_prices,
}

# Create forecaster with exogenous
model = ft.Forecaster(prices, exogenous=exogenous)

# Forecast (automatically uses exogenous adjustments)
result = model.predict(steps=30)

# View what was used
print(f"Hurst: {model.hurst}")
print(f"Regime: {model.regime}")
```

### Custom Path Weights

Control how paths are weighted in the forecast:

```python
import fractime as ft

# Default weights
# hurst: 0.3 (similarity of Hurst exponent)
# volatility: 0.3 (similarity of volatility)
# pattern: 0.4 (similarity of recent pattern)

# Custom weights - emphasize Hurst similarity
model = ft.Forecaster(
    prices,
    path_weights={
        'hurst': 0.6,
        'volatility': 0.2,
        'pattern': 0.2,
    }
)

# Custom weights - emphasize pattern matching
model = ft.Forecaster(
    prices,
    path_weights={
        'hurst': 0.2,
        'volatility': 0.2,
        'pattern': 0.6,
    }
)
```

### Time Warping

Use Mandelbrot's concept of trading time (time flows faster during high volatility):

```python
import fractime as ft

# Enable time warping in simulator
sim = ft.Simulator(prices, time_warp=True)
paths = sim.generate(n_paths=1000, steps=30)

# Enable time warping in forecaster
model = ft.Forecaster(prices, time_warp=True)
result = model.predict(steps=30)
```

### Reusing Analysis

For efficiency, you can reuse an analyzer across multiple forecasters:

```python
import fractime as ft

# Analyze once
analyzer = ft.Analyzer(prices, method='dfa', n_samples=2000)

# Reuse in multiple forecasters
model1 = ft.Forecaster(prices, analyzer=analyzer)
model2 = ft.Forecaster(prices, analyzer=analyzer, time_warp=True)

# Both use the same analysis (no recomputation)
result1 = model1.predict(steps=30)
result2 = model2.predict(steps=30)

# Reuse in simulator
sim = ft.Simulator(prices, analyzer=analyzer)
```

---

## Examples

### Example 1: Basic Workflow

```python
import fractime as ft
import numpy as np

# Generate or load data
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Step 1: Understand the data
analyzer = ft.Analyzer(prices)
print(analyzer.summary())

# Step 2: Forecast
model = ft.Forecaster(prices)
result = model.predict(steps=30)

# Step 3: Visualize
ft.plot(result)

# Step 4: Export results
df = result.to_frame()
print(df)
```

### Example 2: Regime-Based Strategy

```python
import fractime as ft

def get_position(prices):
    """Determine position based on fractal regime."""
    analyzer = ft.Analyzer(prices[-252:])  # Last year

    regime = analyzer.regime
    hurst = analyzer.hurst.value

    if regime == 'trending' and hurst > 0.6:
        return 'follow trend'
    elif regime == 'mean_reverting' and hurst < 0.4:
        return 'fade moves'
    else:
        return 'neutral'

position = get_position(prices)
print(f"Recommended position: {position}")
```

### Example 3: Multi-Asset Analysis

```python
import fractime as ft

# Analyze multiple assets
assets = {
    'SPY': spy_prices,
    'TLT': bond_prices,
    'GLD': gold_prices,
}

for name, prices in assets.items():
    analyzer = ft.Analyzer(prices)
    print(f"{name}: Hurst={analyzer.hurst.value:.2f}, Regime={analyzer.regime}")
```

### Example 4: Ensemble Forecast with Confidence

```python
import fractime as ft

# Create ensemble
ensemble = ft.Ensemble(
    prices,
    models=[
        ft.Forecaster(prices, method='rs'),
        ft.Forecaster(prices, method='dfa'),
        ft.Forecaster(prices, time_warp=True),
    ],
    strategy='weighted'
)

# Generate forecast
result = ensemble.predict(steps=30, n_paths=500)

# Get confidence intervals
ci_90 = result.ci(0.90)
ci_95 = result.ci(0.95)

print(f"Forecast in 30 days: {result.forecast[-1]:.2f}")
print(f"90% CI: ({ci_90[0][-1]:.2f}, {ci_90[1][-1]:.2f})")
print(f"95% CI: ({ci_95[0][-1]:.2f}, {ci_95[1][-1]:.2f})")
```

### Example 5: Rolling Regime Detection

```python
import fractime as ft
import polars as pl

# Analyze with rolling window
analyzer = ft.Analyzer(prices, dates=dates, window=63)

# Get rolling Hurst
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

# Count regime distribution
regime_counts = result.group_by('regime').count()
print(regime_counts)
```

### Example 6: Forecast Comparison

```python
import fractime as ft
import numpy as np

# Split data
train = prices[:-30]
test = prices[-30:]

# Method 1: Basic forecaster
model1 = ft.Forecaster(train)
result1 = model1.predict(steps=30)

# Method 2: With time warping
model2 = ft.Forecaster(train, time_warp=True)
result2 = model2.predict(steps=30)

# Method 3: Ensemble
model3 = ft.Ensemble(train)
result3 = model3.predict(steps=30)

# Compare RMSE
from sklearn.metrics import mean_squared_error

rmse1 = np.sqrt(mean_squared_error(test, result1.forecast))
rmse2 = np.sqrt(mean_squared_error(test, result2.forecast))
rmse3 = np.sqrt(mean_squared_error(test, result3.forecast))

print(f"Basic RMSE:     {rmse1:.4f}")
print(f"Time Warp RMSE: {rmse2:.4f}")
print(f"Ensemble RMSE:  {rmse3:.4f}")
```

---

## Performance

FracTime is optimized for speed:

- **Numba JIT compilation** for Hurst exponent and fractal dimension calculations
- **Lazy computation** - only computes what you access
- **Caching** - repeated access is instant
- **Polars** for fast DataFrame operations

### Benchmarks

| Operation | Time (500 data points) |
|-----------|------------------------|
| Analyzer creation | ~1ms |
| Hurst (point) | ~5ms (first), <1ms (cached) |
| Hurst (rolling) | ~50ms |
| Hurst (bootstrap, 1000 samples) | ~500ms |
| Forecast (1000 paths, 30 steps) | ~200ms |

### Tips for Speed

1. **Reuse analyzers** across forecasters
2. **Reduce n_samples** if bootstrap CI precision isn't critical
3. **Use method='fbm'** for shorter histories (faster than pattern matching)
4. **Limit n_paths** if you don't need high precision

---

## API Summary

### Top-Level Exports

```python
import fractime as ft

# Classes
ft.Analyzer           # Fractal analysis
ft.Forecaster         # Probabilistic forecasting
ft.Simulator          # Path generation
ft.Ensemble           # Model combination

# Convenience functions
ft.analyze(prices)    # Quick analysis
ft.forecast(prices)   # Quick forecast
ft.plot(obj)          # Plot anything

# Result types
ft.Metric             # Single metric with 3 views
ft.AnalysisResult     # Complete analysis
ft.ForecastResult     # Forecast with paths
```

### Quick Reference

```python
# Analyze
analyzer = ft.Analyzer(prices)
analyzer.hurst.value          # Point estimate
analyzer.hurst.rolling        # Rolling series
analyzer.hurst.ci(0.95)       # Confidence interval
analyzer.regime               # 'trending' / 'mean_reverting' / 'random'

# Forecast
model = ft.Forecaster(prices)
result = model.predict(steps=30)
result.forecast               # Median forecast
result.ci(0.95)               # Confidence interval
result.paths                  # All Monte Carlo paths

# Simulate
sim = ft.Simulator(prices)
paths = sim.generate(n_paths=1000, steps=30)

# Ensemble
ensemble = ft.Ensemble(prices)
result = ensemble.predict(steps=30)

# Plot
ft.plot(result)
ft.plot(analyzer)
ft.plot(analyzer.hurst, view='rolling')
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please see our [contributing guide](CONTRIBUTING.md) for details.

## Citation

If you use FracTime in research, please cite:

```bibtex
@software{fractime,
  title = {FracTime: Fractal-based Time Series Analysis and Forecasting},
  author = {Wayy Research},
  year = {2024},
  url = {https://github.com/wayy-research/fracTime}
}
```
