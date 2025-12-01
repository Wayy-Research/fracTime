# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Installation

Install FracTime from PyPI:

```bash
pip install fractime
```

This installs the complete package including:

- Fractal forecasting (Hurst exponent, fractal dimension)
- Baseline models (ARIMA, GARCH, Prophet)
- Bayesian forecasting (PyMC)
- Machine learning (XGBoost)
- Ensemble methods and backtesting

## Development Installation

For contributing or development:

```bash
git clone https://github.com/Wayy-Research/fracTime.git
cd fracTime
pip install -e ".[dev,docs]"
```

## Verifying Installation

```python
import fractime as ft
print(ft.__version__)

# Quick test
import numpy as np
prices = np.random.randn(100).cumsum() + 100
forecaster = ft.FractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=10)
print(f"Forecast generated: {len(result['forecast'])} steps")
```

## Troubleshooting

### Numba Compilation

On first import, Numba compiles optimized functions. This may take a few seconds but only happens once.

### PyMC on Apple Silicon

On Apple Silicon Macs, if you encounter PyMC issues:

```bash
conda install -c conda-forge pymc
```
