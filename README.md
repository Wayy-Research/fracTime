# FracTime

Fractal-based time series forecasting with ensemble methods and rigorous backtesting.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://wayy-research.github.io/fractime)

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

## Quick Start

```python
import fractime as ft
import numpy as np

# Load or create data
prices = np.random.randn(500).cumsum() + 100

# Analyze fractal properties
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
print(f"Hurst: {hurst:.3f} ({'trending' if hurst > 0.5 else 'mean-reverting'})")

# Forecast
forecaster = ft.FractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=30)

print(f"Forecast: {result['forecast'][-1]:.2f}")
print(f"95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")

# Visualize
fig = ft.plot_forecast(prices, result['forecast'], result['paths'])
fig.show()
```

## Features

- **Fractal Forecasting**: Hurst exponent, fractal dimension, long-term memory
- **Baseline Models**: ARIMA, ETS, GARCH, Prophet, VAR, LSTM
- **Ensemble Methods**: Stacking and boosting
- **Backtesting**: Walk-forward validation with comprehensive metrics
- **Model Selection**: Automatic selection with statistical testing

## Documentation

Full documentation: [https://wayy-research.github.io/fractime](https://wayy-research.github.io/fractime)

- [Installation Guide](https://wayy-research.github.io/fractime/getting-started/installation/)
- [Quick Start Tutorial](https://wayy-research.github.io/fractime/getting-started/quickstart/)
- [API Reference](https://wayy-research.github.io/fractime/api/core/)
- [Examples](https://wayy-research.github.io/fractime/examples/basic/)

## License

MIT
