# FracTime

Fractal-based time series forecasting with ensemble methods and rigorous backtesting.

FracTime uses fractal geometry and chaos theory to create accurate forecasts. Unlike traditional methods that assume normal distributions and independence, FracTime captures long-term memory, self-similarity, and regime changes in time series data.

## Why FracTime?

Traditional forecasting methods (ARIMA, exponential smoothing) assume:

- Normal distributions
- Statistical independence
- Short-term memory only

Real-world time series often violate these assumptions. FracTime recognizes that data has:

- **Long-term memory**: Past events affect the distant future (captured via Hurst exponent)
- **Self-similarity**: Patterns repeat across different time scales
- **Regime changes**: Markets shift between trending and mean-reverting behavior
- **Fat tails**: Extreme events occur more frequently than normal distributions predict

## Features

- **Fractal Forecasting**: Hurst exponent, fractal dimension, long-term memory modeling
- **Baseline Models**: ARIMA, ETS, GARCH, Prophet, VAR, LSTM
- **Ensemble Methods**: Stacking and boosting for robust predictions
- **Backtesting**: Walk-forward validation with comprehensive metrics
- **Model Selection**: Automatic selection with statistical significance testing
- **Bayesian Inference**: Full posterior distributions with PyMC (optional)

## Quick Example

```python
import fractime as ft
import numpy as np

# Your time series data
prices = np.random.randn(500).cumsum() + 100

# Fit and forecast
forecaster = ft.FractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=30)

# Visualize
fig = ft.plot_forecast(prices, result['forecast'], result['paths'])
fig.show()

print(f"Forecast: {result['forecast'][-1]:.2f}")
print(f"95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")
```

## Installation

```bash
pip install fractime
```

For additional features:

```bash
pip install fractime[baselines]  # ARIMA, GARCH, Prophet
pip install fractime[bayesian]   # Bayesian forecasting with PyMC
pip install fractime[all]        # Everything
```

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Core Concepts](guide/concepts.md)
