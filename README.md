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

# 1. Load data
data = ft.get_yahoo_data("AAPL", start_date="2023-01-01")
prices = data['Close'].values

# 2. Create and fit forecaster
forecaster = ft.FractalForecaster()
forecaster.fit(prices)

# 3. Generate 30-day forecast with confidence intervals
result = forecaster.forecast(n_steps=30, confidence=0.95)

# 4. Plot it
forecast, paths, _ = forecaster.predict(n_steps=30, return_paths=True)
fig = ft.plot_forecast(
    prices=prices,
    forecast=result['forecast'],
    paths=paths,
    confidence_intervals=result,
    title="30-Day Forecast"
)
fig.show()
```

---

## The Forecasting Workflow

### 1. Load Your Data

```python
import fractime as ft

# Use built-in Yahoo Finance loader
data = ft.get_yahoo_data("SPY", start_date="2020-01-01")
prices = data['Close'].values

# Or use your own data
import numpy as np
prices = np.array([100, 102, 101, 105, ...])  # Any time series
```

### 2. Understand Fractal Properties

```python
# Analyze fractal characteristics
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
fractal_dim = analyzer.compute_fractal_dimension(prices)

print(f"Hurst exponent: {hurst:.3f}")
print(f"Fractal dimension: {fractal_dim:.3f}")

# Interpretation:
# H > 0.5: Trending (persistent)
# H < 0.5: Mean-reverting (anti-persistent)
# H ≈ 0.5: Random walk
```

### 3. Create Forecasts

**Simple forecast:**
```python
# Unified forecaster combines fractal analysis, pattern recognition,
# and regime detection automatically
forecaster = ft.FractalForecaster(lookback=252)  # 1 year
forecaster.fit(prices)

# Get point forecast
forecast = forecaster.predict(n_steps=30)
```

**Forecast with uncertainty:**
```python
# Get forecast with confidence intervals
result = forecaster.forecast(n_steps=30, confidence=0.95)

print(f"Forecast: ${result['forecast'][-1]:.2f}")
print(f"95% CI: ${result['lower'][-1]:.2f} - ${result['upper'][-1]:.2f}")
```

**Get all scenario paths:**
```python
# Generate multiple paths to see range of outcomes
forecast, paths, metadata = forecaster.predict(
    n_steps=30,
    n_paths=1000,
    return_paths=True
)

# Analyze outcomes
final_prices = paths[:, -1]
prob_gain = np.mean(final_prices > prices[-1])
var_95 = np.percentile(final_prices, 5)

print(f"Probability of gain: {prob_gain:.1%}")
print(f"Value at Risk (95%): ${var_95:.2f}")
```

### 4. Visualize Results

**Basic forecast plot:**
```python
import fractime as ft

result = forecaster.forecast(n_steps=30)

fig = ft.plot_forecast(
    prices=prices,
    forecast=result['forecast'],
    confidence_intervals=result,
    title="30-Day Forecast"
)
fig.show()
```

**Show probability paths:**
```python
forecast, paths, _ = forecaster.predict(n_steps=30, n_paths=500, return_paths=True)

fig = ft.plot_forecast(
    prices=prices[-60:],  # Last 60 days
    forecast=forecast,
    paths=paths,
    title="Forecast with Probability Paths"
)
fig.show()
```

**With dates:**
```python
fig = ft.plot_forecast(
    prices=prices,
    forecast=result['forecast'],
    confidence_intervals=result,
    dates=data['Date'].values,
    title="AAPL Forecast"
)
fig.savefig('forecast.png', dpi=300, bbox_inches='tight')
```

---

## Complete Example

```python
import fractime as ft
import numpy as np

# Load data
data = ft.get_yahoo_data("TSLA", start_date="2023-01-01")
prices = data['Close'].values
dates = data['Date'].values

# Analyze fractal properties
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
print(f"Hurst: {hurst:.3f} ({'Trending' if hurst > 0.5 else 'Mean-reverting'})")

# Create forecaster
forecaster = ft.FractalForecaster(lookback=252)
forecaster.fit(prices)

# Generate forecast with confidence intervals
result = forecaster.forecast(n_steps=30, confidence=0.95)
forecast, paths, _ = forecaster.predict(n_steps=30, n_paths=1000, return_paths=True)

# Calculate risk metrics
final_prices = paths[:, -1]
current_price = prices[-1]

expected = np.mean(final_prices)
prob_gain = np.mean(final_prices > current_price)
prob_10pct_gain = np.mean(final_prices > current_price * 1.10)
var_95 = np.percentile(final_prices, 5)
cvar_95 = np.mean(final_prices[final_prices <= var_95])

print(f"\n30-Day Forecast:")
print(f"  Current: ${current_price:.2f}")
print(f"  Expected: ${expected:.2f} ({(expected/current_price-1):.1%})")
print(f"  95% CI: ${result['lower'][-1]:.2f} - ${result['upper'][-1]:.2f}")
print(f"\nProbabilities:")
print(f"  Any gain: {prob_gain:.1%}")
print(f"  >10% gain: {prob_10pct_gain:.1%}")
print(f"\nRisk:")
print(f"  VaR (95%): ${var_95:.2f} ({(var_95/current_price-1):.1%})")
print(f"  CVaR (95%): ${cvar_95:.2f} ({(cvar_95/current_price-1):.1%})")

# Plot
fig = ft.plot_forecast(
    prices=prices[-90:],
    forecast=result['forecast'],
    paths=paths,
    confidence_intervals=result,
    dates=dates[-90:],
    title="TSLA 30-Day Fractal Forecast"
)
fig.show()
```

---

## Why Fractal-Based Forecasting?

Traditional methods (ARIMA, exponential smoothing) assume:
- Normal distributions
- Statistical independence
- Short-term memory only

**FracTime recognizes that time series have:**
- **Long-term memory**: Past events influence the distant future (Hurst exponent)
- **Self-similarity**: Patterns repeat across time scales
- **Regime changes**: Markets shift between trending and mean-reverting states
- **Fat tails**: Extreme events are more common than normal distributions predict

This leads to more accurate forecasts, especially for:
- Financial markets
- Energy consumption
- Economic indicators
- Any series with complex temporal patterns

---

## API Reference

### FractalForecaster

**Main forecasting class** - Use this for most applications.

```python
forecaster = ft.FractalForecaster(lookback=252)
```

**Methods:**
- `fit(prices)` - Fit to historical data
- `predict(n_steps, n_paths=1000, return_paths=False)` - Generate forecast
- `forecast(n_steps, confidence=0.95)` - Forecast with confidence intervals

**Attributes:**
- `hurst` - Hurst exponent (after fitting)
- `fractal_dim` - Fractal dimension (after fitting)

### FractalAnalyzer

**Understand fractal properties** of your time series.

```python
analyzer = ft.FractalAnalyzer()
```

**Methods:**
- `compute_hurst(prices)` → float - Calculate Hurst exponent
- `compute_fractal_dimension(prices)` → float - Calculate fractal dimension
- `analyze_patterns(prices)` → dict - Comprehensive analysis

### plot_forecast()

**Visualize forecasts** with one function call.

```python
ft.plot_forecast(
    prices,                    # Historical data
    forecast=None,             # Point forecast
    paths=None,                # Simulated paths
    confidence_intervals=None, # CI dict with 'lower', 'upper'
    title="Forecast",
    dates=None,                # Date array for x-axis
    show_patterns=False        # Show individual paths
)
```

### Utility Functions

```python
# Load data from Yahoo Finance
data = ft.get_yahoo_data(symbol, start_date, end_date=None)

# Advanced: Direct path simulation
simulator = ft.FractalSimulator(prices, analyzer)
paths, metadata = simulator.simulate_paths(n_steps=30, n_paths=1000)
```

---

## Advanced: Custom Workflows

### Compare with Traditional Methods

```python
from fractime.forecasting import ARIMAForecaster, ExponentialSmoothingForecaster

# Fractal forecast
fractal = ft.FractalForecaster()
fractal.fit(prices)
fractal_forecast = fractal.predict(n_steps=30)

# ARIMA baseline
arima = ARIMAForecaster(p=1, d=1, q=1)
arima.fit(X=prices[:-30].reshape(-1, 1), y=prices[:-30])
arima_forecast = arima.predict(X=np.zeros((30, 1)))

# Compare
print(f"Fractal: ${fractal_forecast[-1]:.2f}")
print(f"ARIMA: ${arima_forecast[-1]:.2f}")
```

### Backtesting

```python
# Simple walk-forward backtest
train_size = 252
test_size = 30
results = []

for i in range(0, len(prices) - train_size - test_size, test_size):
    # Train
    train_data = prices[i:i+train_size]
    forecaster = ft.FractalForecaster()
    forecaster.fit(train_data)

    # Test
    forecast = forecaster.predict(n_steps=test_size)
    actual = prices[i+train_size:i+train_size+test_size]

    # Evaluate
    rmse = np.sqrt(np.mean((forecast - actual)**2))
    results.append(rmse)

print(f"Average RMSE: {np.mean(results):.2f}")
```

---

## Mathematical Background

### Hurst Exponent (H)

Measures **long-term memory**:
- **H > 0.5**: Persistent (trending) - increases likely followed by increases
- **H = 0.5**: Random walk - no memory
- **H < 0.5**: Anti-persistent (mean-reverting) - increases likely followed by decreases

Calculated using Rescaled Range (R/S) analysis.

### Fractal Dimension (D)

Measures **complexity**:
- D = 2 - H for time series
- Higher D → more jagged, complex movements
- Lower D → smoother, more persistent trends

Calculated using box-counting method.

### Fractional Brownian Motion

FracTime uses Fractional Brownian Motion (FBM) to generate forecasts. FBM generalizes standard Brownian motion with a memory parameter (H), creating realistic paths that capture the fractal nature of time series.

---

## Development

```bash
# Run tests
pytest

# Format code
black fractime/ tests/

# Lint
ruff check fractime/ tests/

# Type check
mypy fractime/
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Citation

```bibtex
@software{fractime2024,
  title = {FracTime: Fractal-Based Time Series Forecasting},
  year = {2024},
  url = {https://github.com/Wayy-Research/fractime},
  version = {0.1.0}
}
```

---

## Disclaimer

**For research and educational purposes only.**

- Past performance does not guarantee future results
- No warranty of any kind
- Consult financial professionals before investing

---

**Built with Python | Powered by Fractal Geometry | Inspired by Mandelbrot**
