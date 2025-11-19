# FracTime

**Fractal-based time series forecasting in Python.**

FracTime uses fractal geometry and chaos theory to create accurate forecasts. Unlike traditional methods that assume normal distributions and independence, FracTime captures long-term memory, self-similarity, and regime changes in time series data.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
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
import pandas as pd
import numpy as np

# Your daily price data with dates
dates = pd.date_range(end='2025-11-19', periods=500, freq='D')
prices = np.random.randn(500).cumsum() + 100

# Fit with dates
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)

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
import numpy as np

# Generate data and forecast
np.random.seed(42)
prices = 100 + np.random.randn(500).cumsum()

forecaster = ft.FractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=30, n_paths=500)

# Create interactive visualization
# Paths are colored and sized by probability based on:
# - Hurst exponent similarity
# - Volatility consistency
# - Multi-scale pattern matching
chart = ft.plot_forecast_interactive(
    prices=prices,
    result=result,
    title="Probability-Weighted Forecast Paths",
    top_n_paths=50  # Show top 50 most likely paths
)

# In Jupyter notebook
chart.show()

# Or save to HTML
chart.write_html('forecast.html')
```

The interactive chart shows:
- **Historical data** (black line)
- **Probability cloud** (light blue, all paths with opacity by probability)
- **High-probability paths** (orange-red gradient, top N paths with clean lines)
- **Probability-weighted forecast** (red dashed line)
- **95% confidence interval** (green band)
- **Interactive tooltips** with values and probabilities
- **Probability labels** for top 3 paths

**Clean visualization:** High-probability paths use distinct orange-red colors with gradient based on rank, making them easy to distinguish from the background cloud. No markers - just smooth lines!

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

### FractalForecaster

**Main class** - Use this for forecasting.

```python
forecaster = ft.FractalForecaster(lookback=252)
```

#### Methods

**`fit(prices, dates=None)`**

Fit the model to historical data.

```python
forecaster.fit(prices)                    # Without dates
forecaster.fit(prices, dates=date_array)  # With dates (enables date-based forecasting)
```

**`predict(n_steps=None, end_date=None, period=None, n_paths=1000, confidence=0.95)`**

Generate forecast with uncertainty quantification.

**Provide exactly ONE of:** `n_steps`, `end_date`, or `period`

```python
# Method 1: Traditional (number of steps)
result = forecaster.predict(n_steps=30)

# Method 2: Forecast to specific date (requires dates in fit())
result = forecaster.predict(end_date='2025-11-27')

# Method 3: Forecast for period (requires dates in fit())
result = forecaster.predict(period='7d')
```

**Parameters:**
- `n_steps` - Number of steps ahead to forecast
- `end_date` - Target date string (e.g., '2025-11-27') - requires dates in fit()
- `period` - Period string (e.g., '7d', '2w', '1M') - requires dates in fit()
- `n_paths` - Number of Monte Carlo paths (default 1000)
- `confidence` - Confidence level for intervals (default 0.95)

**Returns dict with:**
- `forecast` - Median forecast
- `weighted_forecast` - Probability-weighted forecast (recommended)
- `mean` - Mean forecast
- `lower` - Lower confidence bound
- `upper` - Upper confidence bound
- `std` - Standard deviation
- `paths` - All simulated paths (n_paths x n_steps)
- `probabilities` - Probability weight for each path based on fractal similarity
- `dates` - Forecast dates (only if dates provided to fit())

#### Attributes (after fitting)

- `hurst` - Hurst exponent
- `fractal_dim` - Fractal dimension

---

### FractalAnalyzer

**Analyze fractal properties** of time series.

```python
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
fractal_dim = analyzer.compute_fractal_dimension(prices)
```

**Methods:**
- `compute_hurst(prices)` → float
- `compute_fractal_dimension(prices)` → float
- `analyze_patterns(prices)` → dict

**Interpretation:**
- H > 0.5: Trending (persistent)
- H < 0.5: Mean-reverting (anti-persistent)
- H ≈ 0.5: Random walk

---

### plot_forecast()

**Static matplotlib visualization.**

```python
fig = ft.plot_forecast(
    prices,                    # Historical data
    forecast=None,             # Forecast line
    paths=None,                # Simulated paths
    confidence_intervals=None, # Dict with 'lower', 'upper'
    title="Forecast",
    dates=None,                # Optional date array
    show_patterns=False        # Show individual paths
)
fig.show()
fig.savefig('forecast.png', dpi=300)
```

### plot_forecast_interactive()

**Interactive Plotly visualization with probability weighting.**

```python
chart = ft.plot_forecast_interactive(
    prices,                      # Historical data
    result,                      # Full result dict from predict()
    dates=None,                  # Optional historical dates (auto-extracted from result if available)
    title="Forecast",
    top_n_paths=20,              # Number of high-probability paths to highlight
    show_probability_cloud=True  # Show probability cloud of all paths
)
chart.show()                    # Display in Jupyter
chart.write_html('forecast.html')  # Save to HTML file
```

**Features:**
- Probability cloud showing all possible paths (light blue)
- High-probability paths highlighted with orange-red gradient
- Clean lines only (no markers) for easy viewing
- Line thickness based on probability rank
- Probability labels for top 3 paths
- Interactive hover tooltips
- Zoom and pan
- Dates handled automatically from result

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
import pandas as pd
import fractime as ft

# Load from CSV
df = pd.read_csv('data.csv')
prices = df['close'].values
dates = pd.to_datetime(df['date'].values)

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
import pandas as pd
import fractime as ft

# Your daily price data
df = pd.read_csv('data.csv')
prices = df['close'].values
dates = pd.to_datetime(df['date'].values)

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

### Backtesting

```python
# Walk-forward validation
train_size = 250
test_size = 30
errors = []

for i in range(0, len(prices) - train_size - test_size, test_size):
    # Train
    train_data = prices[i:i+train_size]
    forecaster = ft.FractalForecaster()
    forecaster.fit(train_data)

    # Predict
    result = forecaster.predict(n_steps=test_size)

    # Evaluate
    actual = prices[i+train_size:i+train_size+test_size]
    rmse = np.sqrt(np.mean((result['forecast'] - actual)**2))
    errors.append(rmse)

print(f"Average RMSE: {np.mean(errors):.2f}")
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

### Compare with ARIMA

```python
from fractime.forecasting import ARIMAForecaster

# Fractal
fractal = ft.FractalForecaster()
fractal.fit(prices)
fractal_result = fractal.predict(n_steps=30)

# ARIMA
arima = ARIMAForecaster(p=1, d=1, q=1)
arima.fit(X=prices[:-30].reshape(-1, 1), y=prices[:-30])
arima_forecast = arima.predict(X=np.zeros((30, 1)))

print(f"Fractal: {fractal_result['forecast'][-1]:.2f}")
print(f"ARIMA: {arima_forecast[-1]:.2f}")
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
