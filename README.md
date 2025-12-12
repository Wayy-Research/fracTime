# FracTime

Fractal-based time series forecasting with ensemble methods, exogenous predictors, and interactive visualizations.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/fractime.svg)](https://pypi.org/project/fractime/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://wayy-research.github.io/fracTime)

## Installation

```bash
pip install fractime
```

All dependencies (ARIMA, GARCH, Prophet, PyMC, XGBoost) are included.

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

print(f"Forecast: {result['weighted_forecast'][-1]:.2f}")
print(f"95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")

# Interactive visualization with path density
fig = ft.plot_forecast(prices, result, colorscale='Viridis')
fig.show()
```

## Features

### Core Forecasting
- **Fractal Analysis**: Hurst exponent, fractal dimension, long-term memory detection
- **Monte Carlo Simulation**: Generate thousands of potential future paths
- **Probability Weighting**: Paths weighted by fractal similarity to historical patterns
- **Trading Time Warping**: Mandelbrot's concept of market time dilation

### Interactive Visualization
- **Path Density Plots**: See clusters of high-probability paths
- **Color-coded Density**: Purple (low probability) to yellow (high probability)
- **Percentile Overlays**: 5th, 25th, 50th, 75th, 95th percentile lines
- **Fully Interactive**: Zoom, pan, hover for details (Plotly-based)

### Exogenous Predictors
- **External Variables**: Condition forecasts on market indicators, economic data
- **Automatic Lag Selection**: Finds optimal lag for each predictor
- **Regime Detection**: Identifies how exogenous states affect returns
- **Probability Adjustment**: Shifts path probabilities based on current conditions

### Model Comparison
- **Baseline Models**: ARIMA, ETS, GARCH, Prophet, VAR, LSTM
- **ML Models**: Random Forest, XGBoost, SVR, KNN
- **Ensemble Methods**: Stacking and boosting
- **Backtesting**: Walk-forward validation with comprehensive metrics

## Interactive Path Density Visualization

The new visualization system shows all simulated paths colored by probability density:

```python
import fractime as ft

# Fit and predict
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)
result = forecaster.predict(n_steps=30, n_paths=1000)

# Create interactive density plot
fig = ft.plot_forecast(
    prices[-100:],           # Show last 100 days of history
    result,
    dates=dates[-100:],
    title="Fractal Forecast with Path Density",
    colorscale='Viridis',    # Options: 'Viridis', 'Plasma', 'Inferno', 'Hot'
    show_percentiles=True,   # Show percentile lines
    max_paths=500            # Limit paths for performance
)
fig.show()  # Interactive in Jupyter
fig.write_html("forecast.html")  # Save as HTML
```

**Colorscales:**
- `Viridis`: Purple → Teal → Yellow (default, perceptually uniform)
- `Plasma`: Blue → Pink → Yellow
- `Inferno`: Black → Red → Yellow
- `Hot`: Blue → Red → Yellow

## Exogenous Predictors

Incorporate external variables to improve forecasts:

```python
import fractime as ft
import pandas as pd

# Load target and exogenous data
# Example: SPY with VIX, TLT (bonds), GLD (gold) as predictors
target = spy_prices
exogenous = pd.DataFrame({
    'VIX': vix_prices,
    'TLT': bond_prices,
    'GLD': gold_prices
})

# Create forecaster with exogenous support
forecaster = ft.FractalForecaster(
    use_exogenous=True,
    exog_max_lags=10,              # Search up to 10 lags
    exog_min_correlation=0.05,     # Include vars with |corr| > 0.05
    exog_adjustment_strength=0.4   # How strongly exog affects probabilities
)

# Fit with exogenous data
forecaster.fit(target, dates=dates, exogenous=exogenous)

# View exogenous analysis
summary = forecaster.get_exogenous_summary()
for name, info in summary['variables'].items():
    status = "INCLUDED" if info['included'] else "excluded"
    print(f"{name}: lag={info['best_lag']}, corr={info['correlation']:.3f} [{status}]")

# Predict (automatically uses exogenous adjustments)
result = forecaster.predict(n_steps=30)
```

### Exogenous Classes

```python
from fractime import (
    ExogenousHandler,           # Preprocessing and lag selection
    ExogenousRegimeModifier,    # Regime-based probability adjustment
    ExogenousForecastAdjuster,  # Incorporate external forecasts
    compute_exogenous_fractal_coherence  # Analyze fractal alignment
)

# Analyze fractal coherence between series
coherence = compute_exogenous_fractal_coherence(
    target=spy_prices,
    exogenous=vix_prices,
    window_sizes=[21, 63, 126]
)
print(f"Fractal coherence: {coherence['overall_coherence']:.3f}")
print(f"Hurst correlation: {coherence['hurst_correlation']:.3f}")
```

## Model Comparison Example

```python
import fractime as ft
from fractime.baselines.arima import ARIMAForecaster
from fractime.baselines.ets import ETSForecaster

# Prepare data
train, test = prices[:-30], prices[-30:]

# Fractal with exogenous
fc_exog = ft.FractalForecaster(use_exogenous=True)
fc_exog.fit(train, exogenous=exog_train)
pred_exog = fc_exog.predict(n_steps=30)

# ARIMA baseline
arima = ARIMAForecaster()
arima.fit(train)
pred_arima = arima.predict(n_steps=30)

# Compare
from sklearn.metrics import mean_squared_error
print(f"Fractal RMSE: {mean_squared_error(test, pred_exog['weighted_forecast'], squared=False):.4f}")
print(f"ARIMA RMSE: {mean_squared_error(test, pred_arima['forecast'], squared=False):.4f}")
```

## Documentation

Full documentation: [https://wayy-research.github.io/fracTime](https://wayy-research.github.io/fracTime)

- [Installation Guide](https://wayy-research.github.io/fracTime/getting-started/installation/)
- [Quick Start Tutorial](https://wayy-research.github.io/fracTime/getting-started/quickstart/)
- [API Reference](https://wayy-research.github.io/fracTime/api/core/)
- [Examples](https://wayy-research.github.io/fracTime/examples/basic/)

## What's New in v0.2.0

### Interactive Path Density Visualization
- All plots now use Plotly for interactivity
- Path density coloring shows probability clusters
- Multiple colorscale options
- Dark theme optimized for density visualization

### Exogenous Predictors
- New `exogenous` parameter in `FractalForecaster.fit()`
- Automatic lag selection and correlation analysis
- Regime-based probability adjustment
- Fractal coherence analysis between series

### Improved API
- `get_exogenous_summary()` for analysis results
- Enhanced `plot_forecast()` with density options
- New utility functions for exogenous analysis

## License

MIT
