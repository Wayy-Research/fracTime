# Forecaster API Reference

The `Forecaster` class generates probabilistic forecasts using fractal analysis.

---

## Overview

```python
import fractime as ft

model = ft.Forecaster(prices)
result = model.predict(steps=30)
print(result.forecast)
```

---

## Constructor

```python
ft.Forecaster(
    data,
    dates=None,
    method='rs',
    time_warp=False,
    exogenous=None,
    path_weights=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array | required | Historical price series |
| `dates` | array | None | Corresponding dates |
| `method` | str | 'rs' | Hurst estimation method |
| `time_warp` | bool | False | Use trading time |
| `exogenous` | dict | None | External variables |
| `path_weights` | dict | None | Custom path weighting |

---

## Methods

### predict()

Generate probabilistic forecast.

```python
result = model.predict(
    steps=30,       # Forecast horizon
    n_paths=1000,   # Number of Monte Carlo paths
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | int | 30 | Number of steps to forecast |
| `n_paths` | int | 1000 | Number of simulation paths |

**Returns:** `ForecastResult` object

---

## Properties

### analyzer

Access the underlying `Analyzer` instance.

```python
analyzer = model.analyzer
print(analyzer.hurst)
```

### hurst

Shortcut to the analyzer's Hurst exponent.

```python
print(model.hurst)  # Same as model.analyzer.hurst
```

### regime

Shortcut to the analyzer's regime classification.

```python
print(model.regime)  # Same as model.analyzer.regime
```

---

## Exogenous Variables

Include external predictors in your forecast:

```python
model = ft.Forecaster(
    prices,
    exogenous={
        'VIX': vix_values,       # Must align with prices
        'volume': volume_values,
    }
)

result = model.predict(steps=30)
```

The forecaster:

1. Analyzes correlations between exogenous variables and returns
2. Identifies lag relationships
3. Incorporates this information into path generation

---

## Path Weights

Control how simulated paths are weighted:

```python
model = ft.Forecaster(
    prices,
    path_weights={
        'hurst': 0.5,       # Hurst consistency weight
        'volatility': 0.3,  # Volatility matching weight
        'pattern': 0.2,     # Pattern similarity weight
    }
)
```

Default (when `path_weights=None`) uses equal weights.

| Weight | Description |
|--------|-------------|
| `hurst` | Favor paths with similar Hurst exponent |
| `volatility` | Favor paths with similar volatility |
| `pattern` | Favor paths matching historical patterns |

---

## Time Warping

Enable Mandelbrot's trading time concept:

```python
model = ft.Forecaster(prices, time_warp=True)
result = model.predict(steps=30)
```

Effects:

- High volatility periods have compressed time
- Low volatility periods have expanded time
- Produces more realistic volatility clustering

---

## Hurst Methods

### Rescaled Range (R/S)

```python
model = ft.Forecaster(prices, method='rs')
```

- Classic method
- More robust to outliers

### Detrended Fluctuation Analysis (DFA)

```python
model = ft.Forecaster(prices, method='dfa')
```

- Better for non-stationary series
- Removes polynomial trends

---

## Examples

### Basic Forecast

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

model = ft.Forecaster(prices)
result = model.predict(steps=30, n_paths=1000)

print(f"Current: {prices[-1]:.2f}")
print(f"Forecast: {result.forecast[-1]:.2f}")
print(f"95% CI: {result.ci(0.95)}")
```

### With Dates

```python
import datetime

dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i)
         for i in range(len(prices))]

model = ft.Forecaster(prices, dates=dates)
result = model.predict(steps=30)

print(f"Forecast dates: {result.dates}")
```

### With Exogenous Variables

```python
model = ft.Forecaster(
    prices,
    exogenous={'VIX': vix_values}
)

result = model.predict(steps=30)
```

### Custom Path Weights

```python
# Emphasize Hurst consistency
model = ft.Forecaster(
    prices,
    path_weights={
        'hurst': 0.8,
        'volatility': 0.1,
        'pattern': 0.1,
    }
)
```

### Full Example

```python
import fractime as ft
import numpy as np

# Data
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Create forecaster
model = ft.Forecaster(
    prices,
    method='dfa',
    time_warp=True,
)

# Generate forecast
result = model.predict(steps=30, n_paths=2000)

# Analysis
print(f"Hurst: {model.hurst}")
print(f"Regime: {model.regime}")

# Forecast statistics
print(f"Current price: {prices[-1]:.2f}")
print(f"30-day forecast: {result.forecast[-1]:.2f}")
print(f"Expected return: {(result.forecast[-1]/prices[-1] - 1)*100:.1f}%")

# Uncertainty
lower, upper = result.ci(0.95)
print(f"95% CI: ({lower[-1]:.2f}, {upper[-1]:.2f})")

# Probability of increase
prob_up = np.mean(result.paths[:, -1] > prices[-1])
print(f"P(increase): {prob_up:.1%}")

# Visualize
ft.plot(result)
```

---

## See Also

- [ForecastResult](results.md#forecastresult) - Result object
- [Simulator](simulator.md) - Direct path generation
- [Ensemble](ensemble.md) - Multiple model combination
- [Forecasting Guide](../guide/forecasting.md) - Usage guide
