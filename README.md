# FracTime

Fractal-based time series forecasting with ensemble methods and rigorous backtesting.

FracTime uses fractal geometry and chaos theory to create accurate forecasts. Unlike traditional methods that assume normal distributions and independence, FracTime captures long-term memory, self-similarity, and regime changes in time series data.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

# Optional dependencies
pip install pmdarima     # ARIMA support
pip install arch         # GARCH support
pip install prophet      # Prophet support
pip install pymc arviz   # Bayesian forecasting
pip install torch        # LSTM support
```

## Quick Start

```python
import fractime as ft
import numpy as np

# Create time series data
prices = np.random.randn(500).cumsum() + 100

# Fit forecaster
forecaster = ft.FractalForecaster()
forecaster.fit(prices)

# Generate forecast
result = forecaster.predict(n_steps=30)

# Plot
fig = ft.plot_forecast(
    prices=prices,
    forecast=result['forecast'],
    paths=result['paths'],
    confidence_intervals=result,
    title="30-Step Forecast"
)
fig.show()

# View results
print(f"Forecast: {result['forecast'][-1]:.2f}")
print(f"95% CI: [{result['lower'][-1]:.2f}, {result['upper'][-1]:.2f}]")
```

## API Reference

### Core Classes

#### FractalForecaster

Main forecasting class using fractal-based time series analysis.

```python
forecaster = ft.FractalForecaster(lookback=252, method='rs')
forecaster.fit(prices, dates=None)
result = forecaster.predict(n_steps=None, end_date=None, period=None, n_paths=1000, confidence=0.95)
```

**Attributes:** `hurst`, `fractal_dim`

#### FractalAnalyzer

Analyze fractal properties of time series.

```python
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)          # Hurst exponent
dim = analyzer.compute_fractal_dimension(prices) # Fractal dimension
patterns = analyzer.analyze_patterns(prices)    # Pattern analysis
analysis = analyzer.analyze(prices)             # Complete analysis
```

#### CrossDimensionalAnalyzer

Analyze fractal properties across multiple time series.

```python
from fractime import CrossDimensionalAnalyzer

analyzer = CrossDimensionalAnalyzer()
analyzer.add_dimension('Stock A', prices_a)
analyzer.add_dimension('Stock B', prices_b)

correlation = analyzer.compute_cross_correlation()
hurst_exp = analyzer.compute_hurst_exponents()
```

### Simulation

#### FractalSimulator

Direct path simulation using fractal methods.

```python
analyzer = ft.FractalAnalyzer()
simulator = ft.FractalSimulator(prices, analyzer)
paths, metadata = simulator.simulate_paths(n_steps=30, n_paths=1000)
```

#### TradingTimeWarper

Time warping for trading applications.

```python
warper = ft.TradingTimeWarper()
```

#### PathAnalyzer

Analyze simulated paths.

```python
path_analyzer = ft.PathAnalyzer()
```

### Baseline Models

All baseline models follow a consistent API: `fit(prices, dates=None)` and `predict(n_steps, end_date=None)`.

```python
from fractime.baselines import (
    ARIMAForecaster,    # Auto-ARIMA with pmdarima
    GARCHForecaster,    # GARCH volatility modeling
    ProphetForecaster,  # Facebook Prophet
    ETSForecaster,      # Exponential smoothing
    VARForecaster,      # Vector autoregression
    LSTMForecaster,     # PyTorch LSTM with MC dropout
)
```

#### ARIMAForecaster

```python
arima = ARIMAForecaster(seasonal=False, max_p=5, max_q=5)
arima.fit(prices)
result = arima.predict(n_steps=30)
```

#### GARCHForecaster

```python
garch = GARCHForecaster(p=1, q=1)
garch.fit(prices)
result = garch.predict(n_steps=30)
```

#### ProphetForecaster

```python
prophet = ProphetForecaster()
prophet.fit(prices, dates=dates)
result = prophet.predict(n_steps=30)
```

#### ETSForecaster

```python
ets = ETSForecaster(trend='add', seasonal=None, damped=False)
ets.fit(prices)
result = ets.predict(n_steps=30)
```

#### VARForecaster

```python
var = VARForecaster(maxlags=15)
var.fit(multivariate_data)
result = var.predict(n_steps=30)
```

#### LSTMForecaster

```python
lstm = LSTMForecaster(hidden_size=50, num_layers=2, dropout=0.2)
lstm.fit(prices, epochs=100)
result = lstm.predict(n_steps=30, n_simulations=100)
```

### Ensemble Methods

#### StackingForecaster

Meta-learning ensemble using cross-validation.

```python
from fractime import StackingForecaster

stacker = StackingForecaster(
    base_models=[model1, model2, model3],
    meta_learner='ridge',  # 'ridge', 'linear', or 'rf'
    n_splits=5
)
stacker.fit(prices)
result = stacker.predict(n_steps=30)
weights = stacker.get_model_weights()
```

#### BoostingForecaster

Sequential error correction ensemble.

```python
from fractime import BoostingForecaster

booster = BoostingForecaster(
    base_model_configs=[(ModelClass, params), ...],
    n_estimators=5,
    learning_rate=0.1
)
booster.fit(prices)
result = booster.predict(n_steps=30)
```

### Backtesting

#### WalkForwardValidator

Walk-forward validation framework.

```python
from fractime.backtesting import WalkForwardValidator

validator = WalkForwardValidator(
    model=FractalForecaster(),
    initial_window=252,
    step_size=20,
    forecast_horizon=10
)
results = validator.run(prices, dates)
```

**Returns:** `metrics`, `forecasts`, `actuals`, `parameter_history`

#### compare_models

Compare multiple models with walk-forward validation.

```python
from fractime.backtesting import compare_models

comparison = compare_models(
    models={'Fractal': FractalForecaster(), 'ARIMA': ARIMAForecaster()},
    prices=prices,
    dates=dates,
    initial_window=100,
    step_size=20,
    forecast_horizon=10
)
```

#### ForecastMetrics

Compute comprehensive forecast metrics.

```python
from fractime.backtesting import ForecastMetrics

metrics = ForecastMetrics.compute_all(
    forecasts=predictions,
    actuals=actual_values,
    current_prices=current,
    lower=lower_bound,
    upper=upper_bound
)
```

**Metrics:** RMSE, MAE, MAPE, MSE, directional accuracy, coverage, calibration error, CRPS

#### DualPenaltyScorer

Balance accuracy vs overfitting.

```python
from fractime.backtesting import DualPenaltyScorer

scorer = DualPenaltyScorer()
```

### Model Selection

#### AutoSelector

Automatic model selection.

```python
from fractime.selection import AutoSelector

selector = AutoSelector()
best_model = selector.select_best(prices, dates)
```

#### ModelRegistry

Catalog available models.

```python
from fractime.selection import ModelRegistry, register_model, get_global_registry

registry = get_global_registry()
```

#### Statistical Tests

```python
from fractime.selection import diebold_mariano_test, model_confidence_set

dm_stat, p_value = diebold_mariano_test(errors1, errors2)
mcs_result = model_confidence_set(model_errors_dict)
```

#### EnsembleForecaster

```python
from fractime.selection import EnsembleForecaster, WeightedEnsemble, create_ensemble

ensemble = create_ensemble(models, weights)
```

### Bayesian Forecasting

Requires PyMC installation.

```python
from fractime import BayesianFractalForecaster
from fractime.bayesian import BayesianFractalModel

forecaster = BayesianFractalForecaster()
forecaster.fit(prices)
result = forecaster.predict(n_steps=30)
```

### Data Sources

Unified interface for accessing time series data.

```python
from fractime.data_sources import (
    get_data_source,
    list_sources,
    get_data_with_fallback,
    DataSource,
    DataSourceConfig,
    TimeSeriesData,
)

# List available data sources
sources = list_sources()

# Get data with automatic fallback
data = get_data_with_fallback(symbol='AAPL', asset_type='equity')
```

### Visualization

#### plot_forecast_interactive

Interactive Plotly visualization with probability weighting.

```python
chart = ft.plot_forecast_interactive(
    prices,
    result,
    dates=None,
    title="Forecast",
    top_n_paths=20,
    use_weighted_ci=True
)
chart.show()
```

#### plot_forecast

Static matplotlib visualization.

```python
fig = ft.plot_forecast(
    prices,
    forecast=None,
    paths=None,
    confidence_intervals=None,
    title="Forecast"
)
fig.show()
```

#### print_forecast_summary

Print formatted forecast summary.

```python
ft.print_forecast_summary(result, current_price=prices[-1], show_paths=10)
```

### Utilities

```python
from fractime import get_yahoo_data

# Fetch data from Yahoo Finance
prices, dates = get_yahoo_data(ticker='AAPL', period='2y')
```

## Date-Based Forecasting

```python
import fractime as ft
import polars as pl

# Your daily price data with dates
dates = pl.date_range(end=pl.datetime(2025, 11, 19), interval='1d', eager=True).tail(500)
prices = np.random.randn(500).cumsum() + 100

# Fit with dates
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates.to_numpy())

# Forecast to specific date
result = forecaster.predict(end_date='2025-11-27')

# Forecast by period
result = forecaster.predict(period='7d')   # 7 days
result = forecaster.predict(period='2w')   # 2 weeks
result = forecaster.predict(period='1M')   # 1 month

# Traditional step-based
result = forecaster.predict(n_steps=30)
```

**Supported periods:** `'7d'`, `'2w'`, `'1M'`, `'12h'`, `'30m'`, etc.

## Examples

### Backtesting

```python
from fractime import FractalForecaster
from fractime.baselines import ARIMAForecaster, ETSForecaster
from fractime.backtesting import WalkForwardValidator, compare_models

# Single model validation
validator = WalkForwardValidator(
    model=FractalForecaster(),
    initial_window=252,
    step_size=30,
    forecast_horizon=30
)
results = validator.run(prices, dates)

print(f"RMSE: {results['metrics']['rmse']:.2f}")
print(f"Directional Accuracy: {results['metrics']['direction_accuracy']:.2%}")

# Multi-model comparison
comparison = compare_models(
    models={
        'Fractal': FractalForecaster(),
        'ARIMA': ARIMAForecaster(),
        'ETS': ETSForecaster()
    },
    prices=prices,
    dates=dates,
    initial_window=100,
    step_size=20,
    forecast_horizon=10
)

for name, metrics in comparison.items():
    print(f"{name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
```

### Ensemble Forecasting

```python
from fractime import FractalForecaster, StackingForecaster
from fractime.baselines import ARIMAForecaster, ETSForecaster

# Create and fit base models
models = [
    FractalForecaster().fit(prices),
    ARIMAForecaster().fit(prices),
    ETSForecaster().fit(prices)
]

# Create stacking ensemble
stacker = StackingForecaster(base_models=models, meta_learner='ridge')
stacker.fit(prices)

result = stacker.predict(n_steps=30)
print(f"Ensemble forecast: {result['forecast'][-1]:.2f}")
print(f"Model weights: {stacker.get_model_weights()}")
```

## Why Fractal-Based Forecasting?

Traditional methods (ARIMA, exponential smoothing) assume:
- Normal distributions
- Statistical independence
- Short-term memory only

FracTime recognizes that real time series have:
- **Long-term memory** - Past affects distant future (Hurst exponent)
- **Self-similarity** - Patterns repeat across time scales
- **Regime changes** - Trending to mean-reverting shifts
- **Fat tails** - Extreme events are more common

## How It Works

1. **Fractal Analysis**: Computes Hurst exponent and fractal dimension
   - H > 0.5: Trending (persistent)
   - H < 0.5: Mean-reverting (anti-persistent)
   - H = 0.5: Random walk

2. **Pattern Recognition**: Finds self-similar patterns across time scales

3. **Path Simulation**: Generates scenarios using Fractional Brownian Motion

4. **Probability Weighting**: Weights paths by Hurst consistency, volatility similarity, and multi-scale pattern matching

5. **Uncertainty Quantification**: Provides confidence intervals and probability distributions

## Development

```bash
# Run tests
pytest

# Format
black fractime/ tests/

# Lint
ruff check fractime/

# Type check
mypy fractime/
```

## License

MIT - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{fractime2024,
  title = {FracTime: Fractal-Based Time Series Forecasting},
  year = {2024},
  url = {https://github.com/Wayy-Research/fractime}
}
```
