# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Basic Installation

Install FracTime from PyPI:

```bash
pip install fractime
```

This installs the core package with fractal forecasting capabilities.

## Optional Dependencies

FracTime has optional dependency groups for extended functionality:

### Baseline Models

Includes ARIMA (pmdarima), GARCH (arch), and Prophet:

```bash
pip install fractime[baselines]
```

### Bayesian Forecasting

Includes PyMC for Bayesian inference:

```bash
pip install fractime[bayesian]
```

### Machine Learning Extras

Includes XGBoost:

```bash
pip install fractime[ml-extra]
```

### Everything

Install all optional dependencies:

```bash
pip install fractime[all]
```

## Development Installation

For contributing or development:

```bash
git clone https://github.com/Wayy-Research/fractime.git
cd fractime
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

### PyMC Installation Issues

If you encounter issues with PyMC:

```bash
pip install pymc arviz pytensor
```

On Apple Silicon Macs, you may need:

```bash
conda install -c conda-forge pymc
```
