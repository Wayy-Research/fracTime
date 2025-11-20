# FracTime

**Fractal-based time series forecasting with ensemble methods and rigorous backtesting.**

FracTime uses fractal geometry and chaos theory to create accurate forecasts. Unlike traditional methods that assume normal distributions and independence, FracTime captures long-term memory, self-similarity, and regime changes in time series data.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ What's New

**Baseline Models**: Compare fractal forecasts against ARIMA, ETS, and LSTM (PyTorch)

**Ensemble Methods**: Combine models with Stacking (meta-learning) and Boosting (sequential error correction)

**Backtesting Framework**: Rigorous walk-forward validation with comprehensive metrics

**Cross-Dimensional Analysis**: Analyze fractal properties across multiple time series

## 🎯 Features

### Core Forecasting
- 🔮 **Fractal Forecasting**: Hurst exponent, fractal dimension, long-term memory
- 📊 **Baseline Models**: ARIMA (auto), ETS, LSTM with Monte Carlo dropout
- 🎲 **Uncertainty Quantification**: Confidence intervals, probability distributions
- 📅 **Date-Based Forecasting**: Forecast to specific dates or periods ('7d', '2w', '1M')

### Advanced Methods
- 🧠 **Stacking Ensemble**: Meta-learning with Ridge/Linear/RandomForest
- ⚡ **Boosting Ensemble**: Sequential error correction
- 🌐 **Cross-Dimensional**: Multi-variate fractal analysis
- 🔄 **Regime-Adaptive**: Automatic adjustment to market regimes

### Validation & Metrics
- ✅ **Walk-Forward Validation**: Rigorous backtesting framework
- 📈 **Comprehensive Metrics**: RMSE, MAE, MAPE, directional accuracy, coverage
- 🎯 **Model Comparison**: Compare multiple models with one function
- 📊 **Dual Penalty Scoring**: Balance accuracy vs overfitting

### Visualization
- 📉 **Interactive Plots**: Plotly charts with probability weighting
- 🎨 **Static Plots**: Publication-ready matplotlib figures
- 📝 **Pretty Summaries**: Formatted forecast summaries for terminal/notebook

---

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

# Optional: For ARIMA support
pip install pmdarima
```

---

## Quick Start

```python
import fractime as ft
import numpy as np

# 1. Create time series data
prices = np.random.randn(500).cumsum() + 100

# 2. Fit forecaster
forecaster = ft.FractalForecaster()
forecaster.fit(prices)

# 3. Generate forecast
result = forecaster.predict(n_steps=30)

# 4. Plot
fig = ft.plot_forecast(
    prices=prices,
    forecast=result['forecast'],
    paths=result['paths'],
    confidence_intervals=result,
    title="30-Step Forecast"
)
fig.show()

# 5. View results
print(f"Forecast: {result['forecast'][-1]:.2f}")
print(f"95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")
```

That's it! One method call gives you everything: forecast, confidence intervals, simulation paths, **and probability weights** for each path based on fractal similarity.

---

## Date-Based Forecasting

**No more manual step calculations!** Just provide your target date or forecast period.

```python
import fractime as ft
import polars as pl
import numpy as np

# Your daily price data with dates
dates = pl.date_range(end=pl.datetime(2025, 11, 19), interval='1d', eager=True).tail(500)
prices = np.random.randn(500).cumsum() + 100

# Fit with dates
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates.to_numpy())

# Method 1: Forecast to specific date (steps computed automatically!)
result = forecaster.predict(end_date='2025-11-27')
print(f"Forecast to: {result['dates'][-1]}")
print(f"Price: ${result['weighted_forecast'][-1]:.2f}")

# Method 2: Forecast by period (even simpler!)
result = forecaster.predict(period='7d')   # 7 days
result = forecaster.predict(period='2w')   # 2 weeks
result = forecaster.predict(period='1M')   # 1 month

# Method 3: Traditional (still works)
result = forecaster.predict(n_steps=30)

# Dates automatically included in results!
# Visualizations use dates automatically!
```

**Supported frequencies:** Daily, hourly, minute (auto-detected)
**Supported periods:** `'7d'`, `'2w'`, `'1M'`, `'12h'`, `'30m'`, etc.

---

## Interactive Probability-Weighted Visualization

**Visualize high-probability forecast paths with interactive Plotly charts!**

```python
import fractime as ft
import polars as pl
import numpy as np

# Generate data with dates
np.random.seed(42)
prices = 100 + np.random.randn(500).cumsum()
dates = pl.date_range(end=pl.datetime(2025, 11, 19), interval='1d', eager=True).tail(500)

forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates.to_numpy())
result = forecaster.predict(end_date='2025-12-19', n_paths=500)

# Create interactive visualization
# Paths are colored and sized by probability based on:
# - Hurst exponent similarity
# - Volatility consistency
# - Multi-scale pattern matching
chart = ft.plot_forecast_interactive(
    prices=prices,
    result=result,
    dates=dates.to_numpy(),  # ← IMPORTANT: Include dates for proper x-axis!
    title="Probability-Weighted Forecast Paths",
    top_n_paths=50  # Show top 50 most likely paths
)

# In Jupyter notebook
chart.show()

# Or save to HTML
chart.write_html('forecast.html')
```

**Important:** When working with datetime data, always pass the `dates` parameter to `plot_forecast_interactive()`. This ensures:
- ✓ Proper date formatting on x-axis (YYYY-MM-DD)
- ✓ Visual continuity between historical and forecast data
- ✓ Correct time alignment of all traces

Without `dates`, the historical data will use integer indices while forecast uses dates, causing rendering issues.

The interactive chart shows:
- **Historical data** (black line)
- **Probability cloud** (light blue, all paths with opacity by probability)
- **High-probability paths** (orange-red gradient, top N paths with clean lines)
- **Probability-weighted forecast** (red dashed line)
- **95% probability-weighted confidence interval** (green band - based on path likelihoods, not just quantiles)
- **Interactive tooltips** with values and probabilities
- **Probability labels** for top 3 paths

**Clean visualization:** High-probability paths use distinct orange-red colors with gradient based on rank, making them easy to distinguish from the background cloud. No markers - just smooth lines!

**Probability-Weighted CI:** Unlike traditional confidence intervals that treat all paths equally, our weighted CI accounts for path probabilities. More likely paths contribute more to the confidence bounds, giving you a more accurate uncertainty range.

```python
# Forecast results include both standard and weighted CI
result = forecaster.predict(end_date='2025-11-27')

print(f"Standard 95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")
print(f"Weighted 95% CI: [{result['weighted_lower'][-1]:.2f}, {result['weighted_upper'][-1]:.2f}]")

# Visualization uses weighted CI by default
chart = ft.plot_forecast_interactive(prices, result, dates=dates, use_weighted_ci=True)
```

---

## Pretty Print Forecast Summary

Get a nicely formatted summary of your forecast results:

```python
import fractime as ft
import polars as pl
import numpy as np

# Your data
prices = np.random.randn(200).cumsum() + 100
dates = pl.date_range(end=pl.datetime(2025, 11, 19), interval='1d', eager=True).tail(200)

# Forecast
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates.to_numpy())
result = forecaster.predict(end_date='2025-11-27', n_paths=500)

# Pretty print summary
ft.print_forecast_summary(result, current_price=prices[-1], show_paths=10)
```

**Output:**
```
======================================================================
FORECAST SUMMARY
======================================================================

Period: 2025-11-20 to 2025-11-27 (8 steps)
Current Price: $91.85

----------------------------------------------------------------------
POINT FORECASTS (at final step)
----------------------------------------------------------------------
  Median Forecast:           $91.20
  Probability-Weighted:      $91.13  ← Recommended
  Mean:                      $91.15

  Expected Change:           ↓ 0.78%

----------------------------------------------------------------------
95% CONFIDENCE INTERVALS (at final step)
----------------------------------------------------------------------
  Standard CI:      [$87.00, $95.78]  (width: $8.78)
  Weighted CI:      [$87.02, $95.79]  (width: $8.77)  ← Recommended

----------------------------------------------------------------------
STATISTICS
----------------------------------------------------------------------
  Standard Deviation:        $2.18
  Number of Paths:           500

----------------------------------------------------------------------
TOP 10 MOST LIKELY PATHS
----------------------------------------------------------------------
  Rank   Probability     Final Value     Change
  ------------------------------------------------------------
  #1     0.002494 (0.249%)  $   93.90     +2.24%  ██
  #2     0.002492 (0.249%)  $   93.09     +1.36%  ██
  ...

----------------------------------------------------------------------
FORECAST RANGE VISUALIZATION
----------------------------------------------------------------------

  $87.02  [                           ●                               ]  $95.79
  Weighted 95% CI:                             ↑
                                               Forecast: $91.13
```

Perfect for quick inspection in Jupyter notebooks or terminal output!

---

## Complete Example

```python
import fractime as ft
import numpy as np

# Generate data
np.random.seed(42)
prices = 100 + np.random.randn(500).cumsum()

# Understand the data's fractal properties
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
print(f"Hurst: {hurst:.3f} ({'Trending' if hurst > 0.5 else 'Mean-reverting'})")

# Fit and predict
forecaster = ft.FractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=30, n_paths=1000)

# Analyze results
current = prices[-1]
forecast = result['forecast'][-1]
lower = result['lower'][-1]
upper = result['upper'][-1]

print(f"\n30-Step Forecast:")
print(f"  Current: {current:.2f}")
print(f"  Forecast: {forecast:.2f} ({(forecast/current-1)*100:+.1f}%)")
print(f"  95% CI: [{lower:.2f}, {upper:.2f}]")

# Calculate probabilities from paths
paths = result['paths']
final_values = paths[:, -1]
prob_increase = np.mean(final_values > current)
percentile_5 = np.percentile(final_values, 5)
percentile_95 = np.percentile(final_values, 95)

print(f"\nProbabilities:")
print(f"  Any increase: {prob_increase:.1%}")
print(f"  Range (5th-95th): [{percentile_5:.2f}, {percentile_95:.2f}]")

# Visualize
fig = ft.plot_forecast(
    prices=prices[-150:],
    forecast=result['forecast'],
    paths=result['paths'],
    confidence_intervals=result,
    title="30-Step Fractal Forecast"
)
fig.show()
```

**Output:**
```
Hurst: 0.548 (Trending)

30-Step Forecast:
  Current: 103.42
  Forecast: 104.16 (+0.7%)
  95% CI: [94.26, 113.23]

Probabilities:
  Any increase: 55.7%
  Range (5th-95th): [95.05, 112.14]
```

---

## API Reference

### Core Classes

#### FractalForecaster

**Main forecasting class** - Fractal-based time series forecasting.

```python
forecaster = ft.FractalForecaster(lookback=252, method='rs')
```

**Methods:**
- `fit(prices, dates=None)` → self - Fit model to data
- `predict(n_steps=None, end_date=None, period=None, n_paths=1000, confidence=0.95)` → dict

**Attributes:** `hurst`, `fractal_dim`

#### FractalAnalyzer

**Analyze fractal properties** of time series.

```python
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
```

**Methods:**
- `compute_hurst(prices)` → float - Hurst exponent (H>0.5=trending, H<0.5=mean-reverting)
- `compute_fractal_dimension(prices)` → float - Fractal dimension
- `analyze_patterns(prices)` → dict - Full pattern analysis
- `analyze(prices)` → dict - Complete fractal analysis

---

### Baseline Models

#### ARIMAForecaster

Auto-ARIMA with automatic parameter selection via pmdarima.

```python
from fractime.baselines import ARIMAForecaster

arima = ARIMAForecaster(seasonal=False, max_p=5, max_q=5)
arima.fit(prices)
result = arima.predict(n_steps=30)
```

**Parameters:** `seasonal`, `m`, `max_p`, `max_q`, `max_d`, `stepwise`

#### ETSForecaster

Exponential smoothing state space model.

```python
from fractime.baselines import ETSForecaster

ets = ETSForecaster(trend='add', seasonal=None)
ets.fit(prices)
result = ets.predict(n_steps=30)
```

**Parameters:** `trend` ('add', 'mul', None), `seasonal` ('add', 'mul', None), `damped`

#### LSTMForecaster

Deep learning with PyTorch and Monte Carlo dropout for uncertainty.

```python
from fractime.baselines import LSTMForecaster

lstm = LSTMForecaster(hidden_size=50, num_layers=2, dropout=0.2)
lstm.fit(prices, epochs=100)
result = lstm.predict(n_steps=30, n_simulations=100)
```

**Parameters:** `hidden_size`, `num_layers`, `dropout`, `learning_rate`, `batch_size`

**All baseline models** share the same API: `fit(prices)` → `predict(n_steps)` → dict with `forecast`, `mean`, `std`, `lower`, `upper`

---

### Ensemble Methods

#### StackingForecaster

Meta-learning ensemble using cross-validation.

```python
from fractime.ensemble import StackingForecaster

stacker = StackingForecaster(
    base_models=[model1, model2, model3],
    meta_learner='ridge',  # or 'linear', 'rf'
    n_splits=5
)
stacker.fit(prices)
result = stacker.predict(n_steps=30)
weights = stacker.get_model_weights()
```

**Methods:**
- `fit(prices)` → self
- `predict(n_steps)` → dict
- `get_model_weights()` → dict - Model importance scores
- `add_model(model, name=None)` - Add a base model

#### BoostingForecaster

Sequential error correction ensemble.

```python
from fractime.ensemble import BoostingForecaster

booster = BoostingForecaster(
    base_model_configs=[(ModelClass, params), ...],
    n_estimators=5,
    learning_rate=0.1
)
booster.fit(prices)
result = booster.predict(n_steps=30)
```

**Methods:**
- `fit(prices)` → self
- `predict(n_steps)` → dict
- `get_model_weights()` → list - Model weights
- `add_model_config(model_class, params)` - Add model configuration

---

### Backtesting

#### WalkForwardValidator

Rigorous walk-forward validation framework.

```python
from fractime.backtesting import WalkForwardValidator

validator = WalkForwardValidator(
    model=FractalForecaster(),
    initial_window=252,
    step_size=20,
    forecast_horizon=10
)
results = validator.run(prices, dates)
```

**Returns dict with:**
- `metrics` - Comprehensive accuracy metrics
- `forecasts` - All forecast values
- `actuals` - Actual observed values
- `parameter_history` - Model parameter evolution

#### compare_models()

Compare multiple models with walk-forward validation.

```python
from fractime.backtesting import compare_models

comparison = compare_models(
    models={'Fractal': FractalForecaster(), 'ARIMA': ARIMAForecaster()},
    prices=prices,
    dates=dates,
    initial_window=100,
    step_size=20,
    forecast_horizon=10
)
```

**Returns:** Dict mapping model names to metrics (MAE, RMSE, MAPE, MSE, etc.)

#### ForecastMetrics

Comprehensive forecast evaluation metrics.

```python
from fractime.backtesting import ForecastMetrics

metrics = ForecastMetrics.compute_all(
    forecasts=predictions,
    actuals=actual_values,
    current_prices=current,
    lower=lower_bound,
    upper=upper_bound
)
```

**Metrics computed:** RMSE, MAE, MAPE, MSE, directional accuracy, coverage, calibration error, CRPS

---

### Analysis

#### CrossDimensionalAnalyzer

Analyze fractal properties across multiple time series.

```python
from fractime.analysis import CrossDimensionalAnalyzer

analyzer = CrossDimensionalAnalyzer()
analyzer.add_dimension('Stock A', prices_a)
analyzer.add_dimension('Stock B', prices_b)

correlation = analyzer.compute_cross_correlation()
hurst_exp = analyzer.compute_hurst_exponents()
```

**Methods:**
- `add_dimension(name, prices)` - Add a dimension
- `compute_cross_correlation()` → ndarray - Correlation matrix
- `compute_hurst_exponents()` → dict - Hurst per dimension
- `analyze(data, dim_names)` → dict - Complete analysis

---

### Visualization

#### plot_forecast_interactive()

Interactive Plotly visualization with probability weighting.

```python
chart = ft.plot_forecast_interactive(
    prices,
    result,
    dates=None,
    title="Forecast",
    top_n_paths=20
)
chart.show()
```

**Features:** Probability cloud, high-probability path highlighting, weighted CI, interactive tooltips

---

#### plot_forecast()

Static matplotlib visualization.

```python
fig = ft.plot_forecast(
    prices,
    forecast=None,
    paths=None,
    confidence_intervals=None,
    title="Forecast"
)
fig.show()
```

#### print_forecast_summary()

Pretty-print forecast summary to terminal/notebook.

```python
ft.print_forecast_summary(result, current_price=prices[-1], show_paths=10)
```

---

## Usage Examples

### Basic Forecasting

```python
import fractime as ft
import numpy as np

# Your data
prices = np.array([100, 102, 101, 105, 103, 108, ...])

# Fit and predict
forecaster = ft.FractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=10)

print(f"10-step forecast: {result['forecast'][-1]:.2f}")
```

### With Real Data

```python
import polars as pl
import fractime as ft

# Load from CSV
df = pl.read_csv('data.csv')
prices = df['close'].to_numpy()
dates = df['date'].str.to_datetime().to_numpy()

# Fit with dates (enables date-based forecasting)
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)

# Forecast to specific date
result = forecaster.predict(end_date='2025-12-31')

# Plot (dates handled automatically from result)
fig = ft.plot_forecast_interactive(
    prices=prices,
    result=result,
    dates=dates,
    title="Forecast to 2025-12-31"
)
fig.show()
```

### Date-Based Forecasting Examples

```python
import polars as pl
import fractime as ft

# Your daily price data
df = pl.read_csv('data.csv')
prices = df['close'].to_numpy()
dates = df['date'].str.to_datetime().to_numpy()

# Fit with dates
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)

# Example 1: Forecast to specific date (no step calculation!)
result = forecaster.predict(end_date='2025-11-27')
print(f"Forecast to: {result['dates'][-1]}")
print(f"Price: ${result['weighted_forecast'][-1]:.2f}")

# Example 2: Forecast by period
result_1w = forecaster.predict(period='1w')   # 1 week
result_2w = forecaster.predict(period='2w')   # 2 weeks
result_1M = forecaster.predict(period='1M')   # 1 month

# Example 3: Compare forecasts
print(f"1-week forecast:  ${result_1w['weighted_forecast'][-1]:.2f}")
print(f"2-week forecast:  ${result_2w['weighted_forecast'][-1]:.2f}")
print(f"1-month forecast: ${result_1M['weighted_forecast'][-1]:.2f}")

# Dates automatically included in all results!
print(f"1-week ends: {result_1w['dates'][-1]}")
```

### Probability-Weighted Risk Analysis

```python
# Generate forecast with probabilities
result = forecaster.predict(n_steps=30, n_paths=2000)

# Get paths and their probabilities
paths = result['paths']
probs = result['probabilities']
final_values = paths[:, -1]
current = prices[-1]

# Probability-weighted VaR (more accurate than percentile!)
sorted_idx = np.argsort(final_values)
cumsum_prob = np.cumsum(probs[sorted_idx])
var_95_idx = sorted_idx[np.searchsorted(cumsum_prob, 0.05)]
var_95_weighted = final_values[var_95_idx]

# Traditional VaR for comparison
var_95_traditional = np.percentile(final_values, 5)

print(f"Current: {current:.2f}")
print(f"Probability-weighted VaR: {var_95_weighted:.2f}")
print(f"Traditional VaR: {var_95_traditional:.2f}")

# Most likely outcome (highest probability path)
most_likely_idx = np.argmax(probs)
most_likely_outcome = final_values[most_likely_idx]
print(f"Most likely outcome: {most_likely_outcome:.2f} (prob: {probs[most_likely_idx]:.4f})")
```

### Backtesting Example

```python
from fractime.backtesting import WalkForwardValidator

# Automated walk-forward validation
validator = WalkForwardValidator(
    model=ft.FractalForecaster(),
    initial_window=250,
    step_size=30,
    forecast_horizon=30
)

results = validator.run(prices, dates)

# Comprehensive metrics automatically computed
print(f"RMSE: {results['metrics']['rmse']:.2f}")
print(f"MAE: {results['metrics']['mae']:.2f}")
print(f"Directional Accuracy: {results['metrics']['direction_accuracy']:.2%}")
print(f"Coverage: {results['metrics']['coverage']:.2%}")
```

---

## Why Fractal-Based Forecasting?

**Traditional methods** (ARIMA, exponential smoothing) assume:
- Normal distributions
- Statistical independence
- Short-term memory only

**FracTime recognizes** that real time series have:
- **Long-term memory** - Past affects distant future (Hurst exponent)
- **Self-similarity** - Patterns repeat across time scales
- **Regime changes** - Trending ↔ mean-reverting shifts
- **Fat tails** - Extreme events are more common

This leads to better forecasts for:
- Financial markets
- Energy systems
- Economic indicators
- Any complex temporal data

---

## How It Works

### 1. Fractal Analysis

Computes Hurst exponent and fractal dimension to characterize the series:

```python
H = 0.65  # Trending
H = 0.35  # Mean-reverting
H = 0.50  # Random walk
```

### 2. Pattern Recognition

Finds self-similar patterns across different time scales.

### 3. Path Simulation

Generates multiple future scenarios using Fractional Brownian Motion with the measured Hurst exponent.

### 4. Probability Weighting

**Each forecast path gets a probability weight based on:**
- **Hurst consistency**: How well the path matches historical long-term memory
- **Volatility similarity**: Matching historical volatility patterns
- **Multi-scale pattern matching**: Short, medium, and long-term trend consistency

This means high-probability paths are those that are most consistent with the historical fractal structure!

### 5. Uncertainty Quantification

Provides:
- Confidence intervals
- Full probability distributions
- Probability-weighted forecasts
- Interactive visualizations

---

## Baseline Models

Compare fractal forecasts against classical and deep learning baselines:

```python
from fractime.baselines import ARIMAForecaster, ETSForecaster, LSTMForecaster

# ARIMA - Auto-ARIMA with automatic parameter selection
arima = ARIMAForecaster()
arima.fit(prices)
arima_result = arima.predict(n_steps=30)

# ETS - Exponential smoothing state space model
ets = ETSForecaster(trend='add', seasonal=None)
ets.fit(prices)
ets_result = ets.predict(n_steps=30)

# LSTM - Deep learning with PyTorch and Monte Carlo dropout
lstm = LSTMForecaster(hidden_size=50, num_layers=2)
lstm.fit(prices)
lstm_result = lstm.predict(n_steps=30, n_simulations=100)

# Fractal
fractal = ft.FractalForecaster()
fractal.fit(prices)
fractal_result = fractal.predict(n_steps=30)

# Compare
print(f"Fractal: {fractal_result['forecast'][-1]:.2f}")
print(f"ARIMA:   {arima_result['forecast'][-1]:.2f}")
print(f"ETS:     {ets_result['forecast'][-1]:.2f}")
print(f"LSTM:    {lstm_result['forecast'][-1]:.2f}")
```

**All models** share the same simple API: `fit(prices)` → `predict(n_steps)` → returns dict with `forecast`, `mean`, `std`, `lower`, `upper`.

---

## Ensemble Methods

Combine multiple models for more robust predictions:

### Stacking Ensemble

Uses meta-learning to optimally weight model predictions via cross-validation:

```python
from fractime import FractalForecaster
from fractime.baselines import ARIMAForecaster, ETSForecaster
from fractime.ensemble import StackingForecaster

# Fit base models
models = [
    FractalForecaster().fit(prices),
    ARIMAForecaster().fit(prices),
    ETSForecaster().fit(prices)
]

# Create stacking ensemble with Ridge meta-learner
stacker = StackingForecaster(base_models=models, meta_learner='ridge')
stacker.fit(prices)

# Generate ensemble forecast
result = stacker.predict(n_steps=30)
print(f"Ensemble forecast: {result['forecast'][-1]:.2f}")

# Check model weights
weights = stacker.get_model_weights()
print("Model contributions:", weights)
```

**Meta-learners**: `'ridge'` (default), `'linear'`, or `'rf'` (Random Forest)

### Boosting Ensemble

Sequential error correction where each model focuses on previous mistakes:

```python
from fractime.ensemble import BoostingForecaster

# Define model configurations
configs = [
    (FractalForecaster, {}),
    (ARIMAForecaster, {}),
    (ETSForecaster, {'trend': 'add'})
]

# Create boosting ensemble
booster = BoostingForecaster(
    base_model_configs=configs,
    n_estimators=5,
    learning_rate=0.1
)
booster.fit(prices)

result = booster.predict(n_steps=30)
print(f"Boosted forecast: {result['forecast'][-1]:.2f}")
```

---

## Backtesting Framework

Rigorous walk-forward validation for any model:

```python
from fractime.backtesting import WalkForwardValidator, ForecastMetrics

# Create validator
validator = WalkForwardValidator(
    model=FractalForecaster(),
    initial_window=252,      # 1 year of training data
    step_size=20,            # Refit every 20 steps
    forecast_horizon=10      # 10-step-ahead forecasts
)

# Run validation
results = validator.run(prices, dates)

# Comprehensive metrics
metrics = results['metrics']
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE:  {metrics['mae']:.4f}")
print(f"Directional Accuracy: {metrics['direction_accuracy']:.2%}")
print(f"Coverage (95% CI): {metrics['coverage']:.2%}")

# Compare multiple models
from fractime.backtesting import compare_models

comparison = compare_models(
    models={
        'Fractal': FractalForecaster(),
        'ARIMA': ARIMAForecaster(),
        'ETS': ETSForecaster()
    },
    prices=prices,
    dates=dates,
    initial_window=100,
    step_size=20,
    forecast_horizon=10
)

for name, metrics in comparison.items():
    print(f"{name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
```

**Metrics computed**: RMSE, MAE, MAPE, MSE, directional accuracy, coverage, calibration error

---

## Cross-Dimensional Analysis

Analyze fractal properties across multiple related time series:

```python
from fractime.analysis import CrossDimensionalAnalyzer

# Multiple time series (e.g., different stocks, assets, or markets)
data = np.column_stack([prices_stock1, prices_stock2, prices_stock3])
dimension_names = ['Stock A', 'Stock B', 'Stock C']

# Analyze cross-dimensional fractal structure
analyzer = CrossDimensionalAnalyzer()
for i, name in enumerate(dimension_names):
    analyzer.add_dimension(name, data[:, i])

# Compute correlations and cross-Hurst exponents
correlation = analyzer.compute_cross_correlation()
hurst_exp = analyzer.compute_hurst_exponents()

print("Cross-correlation matrix:")
print(correlation)
print("\nHurst exponents:")
for name, h in hurst_exp.items():
    print(f"{name}: {h:.3f}")
```

---

## Advanced

### Direct Simulation

```python
analyzer = ft.FractalAnalyzer()
simulator = ft.FractalSimulator(prices, analyzer)
paths, metadata = simulator.simulate_paths(n_steps=30, n_paths=1000)
```

### Custom Confidence Levels

```python
result = forecaster.predict(n_steps=30, confidence=0.90)  # 90% CI
result = forecaster.predict(n_steps=30, confidence=0.99)  # 99% CI
```

### Regime-Adaptive Forecasting

Automatically adjust forecasts based on detected regime changes:

```python
# Fit with regime detection
forecaster = ft.FractalForecaster()
forecaster.fit(prices)

# Predict with regime adaptation
result = forecaster.predict(n_steps=30, n_paths=1000)

# Analyze regime characteristics
if hasattr(forecaster, 'current_regime'):
    print(f"Current regime: {forecaster.current_regime}")
    print(f"Hurst: {forecaster.hurst:.3f}")
```

---

## Development

```bash
# Run tests
pytest

# Format
black fractime/ tests/

# Lint
ruff check fractime/

# Type check
mypy fractime/
```

---

## License

MIT - see [LICENSE](LICENSE)

---

## Citation

```bibtex
@software{fractime2024,
  title = {FracTime: Fractal-Based Time Series Forecasting},
  year = {2024},
  url = {https://github.com/Wayy-Research/fractime}
}
```

---

**Built with Python | Powered by Fractal Geometry | Inspired by Mandelbrot**
