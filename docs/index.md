# FracTime

Fractal-based time series forecasting with ensemble methods, exogenous predictors, and interactive visualizations.

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

### Core Forecasting
- **Fractal Analysis**: Hurst exponent, fractal dimension, long-term memory modeling
- **Monte Carlo Simulation**: Generate thousands of potential future paths
- **Probability Weighting**: Paths weighted by fractal similarity to historical patterns
- **Trading Time Warping**: Mandelbrot's concept of market time dilation

### Interactive Visualization (New in v0.2.0)
- **Path Density Plots**: See clusters of high-probability paths
- **Color-coded Density**: Purple (low probability) to yellow (high probability)
- **Percentile Overlays**: 5th, 25th, 50th, 75th, 95th percentile lines
- **Fully Interactive**: Zoom, pan, hover for details (Plotly-based)

### Exogenous Predictors (New in v0.2.0)
- **External Variables**: Condition forecasts on market indicators, economic data
- **Automatic Lag Selection**: Finds optimal lag for each predictor
- **Regime Detection**: Identifies how exogenous states affect returns
- **Fractal Coherence**: Analyze alignment between target and predictors

### Model Comparison
- **Baseline Models**: ARIMA, ETS, GARCH, Prophet, VAR, LSTM
- **ML Models**: Random Forest, XGBoost, SVR, KNN
- **Ensemble Methods**: Stacking and boosting for robust predictions
- **Backtesting**: Walk-forward validation with comprehensive metrics

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

# Interactive visualization with path density
fig = ft.plot_forecast(prices, result, colorscale='Viridis')
fig.show()

print(f"Forecast: {result['weighted_forecast'][-1]:.2f}")
print(f"95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")
```

## With Exogenous Predictors

```python
import fractime as ft
import pandas as pd

# Target and exogenous data
target = spy_prices
exogenous = pd.DataFrame({
    'VIX': vix_prices,
    'TLT': bond_prices
})

# Fit with exogenous support
forecaster = ft.FractalForecaster(use_exogenous=True)
forecaster.fit(target, dates=dates, exogenous=exogenous)

# View exogenous analysis
print(forecaster.get_exogenous_summary())

# Predict
result = forecaster.predict(n_steps=30)
```

## Installation

```bash
pip install fractime
```

All dependencies are included by default.

## What's New in v0.2.0

- **Interactive Visualizations**: All plots now use Plotly for full interactivity
- **Path Density Coloring**: See probability clusters in forecast paths
- **Exogenous Predictors**: Condition forecasts on external variables
- **Fractal Coherence Analysis**: Analyze alignment between series

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Core Concepts](guide/concepts.md)
- [Exogenous Predictors](api/exogenous.md)
- [Visualization Guide](api/visualization.md)
