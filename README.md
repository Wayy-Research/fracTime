# FracTime

FracTime is an advanced time series forecasting tool that leverages fractal geometry and chaos theory principles for market analysis and prediction.

## Features

- **Pattern Recognition**
  - Hurst exponent calculation (R/S analysis)
  - Fractal dimension estimation
  - Self-similarity detection
  - Multi-scale price movement analysis

- **Advanced Simulation**
  - Fractional Brownian motion via spectral synthesis
  - Pattern-weighted path generation
  - Non-parametric bootstrapping with recency bias
  - Regime detection using HMMs

- **Path Analysis**
  - K-means clustering of trajectories
  - Probability density estimation
  - Representative path selection
  - Empirical confidence intervals

## Installation

FracTime uses `uv` for dependency management. To install:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```python
from fractime import analyze_timeseries, simulate_paths

# Load and analyze time series
results = analyze_timeseries(data)

# Generate simulated paths
paths = simulate_paths(data, n_paths=1000)
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## Requirements

- Python >=3.11
- Modern web browser for interactive visualizations
- Internet connection for real-time market data

## License

MIT
