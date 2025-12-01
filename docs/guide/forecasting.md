# Forecasting Guide

Detailed guide to forecasting with FracTime.

## FractalForecaster

The main forecasting class.

### Basic Usage

```python
import fractime as ft

forecaster = ft.FractalForecaster(lookback=252, method='rs')
forecaster.fit(prices)
result = forecaster.predict(n_steps=30)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 252 | Historical window for fractal analysis |
| `method` | 'rs' | Hurst estimation method ('rs', 'dfa') |

### Prediction Options

```python
# By number of steps
result = forecaster.predict(n_steps=30)

# By end date (requires dates in fit)
result = forecaster.predict(end_date='2025-12-31')

# By period
result = forecaster.predict(period='2w')  # 2 weeks

# Simulation parameters
result = forecaster.predict(
    n_steps=30,
    n_paths=1000,      # Number of simulation paths
    confidence=0.95    # Confidence level for intervals
)
```

### Result Dictionary

```python
result = {
    'forecast': np.array,      # Point forecast
    'mean': np.array,          # Mean of paths
    'median': np.array,        # Median of paths
    'std': np.array,           # Standard deviation
    'lower': np.array,         # Lower confidence bound
    'upper': np.array,         # Upper confidence bound
    'paths': np.array,         # All simulated paths
    'probabilities': np.array, # Path probability weights
    'dates': np.array,         # Forecast dates (if dates provided)
}
```

## Baseline Models

All baseline models share the same API.

### ARIMAForecaster

```python
from fractime.baselines import ARIMAForecaster

arima = ARIMAForecaster(
    seasonal=False,
    max_p=5,
    max_q=5,
    max_d=2
)
arima.fit(prices)
result = arima.predict(n_steps=30)
```

### GARCHForecaster

For volatility modeling:

```python
from fractime.baselines import GARCHForecaster

garch = GARCHForecaster(p=1, q=1)
garch.fit(prices)
result = garch.predict(n_steps=30)
```

### ProphetForecaster

Facebook's forecasting tool:

```python
from fractime.baselines import ProphetForecaster

prophet = ProphetForecaster()
prophet.fit(prices, dates=dates)
result = prophet.predict(n_steps=30)
```

### ETSForecaster

Exponential smoothing:

```python
from fractime.baselines import ETSForecaster

ets = ETSForecaster(
    trend='add',     # 'add', 'mul', or None
    seasonal=None,   # 'add', 'mul', or None
    damped=False
)
ets.fit(prices)
result = ets.predict(n_steps=30)
```

### LSTMForecaster

Deep learning with PyTorch:

```python
from fractime.baselines import LSTMForecaster

lstm = LSTMForecaster(
    hidden_size=50,
    num_layers=2,
    dropout=0.2
)
lstm.fit(prices, epochs=100)
result = lstm.predict(n_steps=30, n_simulations=100)
```

## Bayesian Forecasting

Full Bayesian inference with PyMC:

```python
from fractime import BayesianFractalForecaster

forecaster = BayesianFractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=30)

# Access posterior samples
posterior = forecaster.get_posterior()
```

Requires: `pip install fractime[bayesian]`
