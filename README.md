# FracTime

**FracTime** is a Python library for advanced time series forecasting using fractal geometry and chaos theory principles.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Why FracTime?](#why-fractime)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [Fractal Analysis](#fractal-analysis)
  - [Path Simulation](#path-simulation)
  - [Forecasting](#forecasting)
  - [Backtesting](#backtesting)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## Why FracTime?

Traditional forecasting methods assume normal distributions and statistical independence. **FracTime takes a different approach:**

- **Fractal-Based Analysis**: Captures long-term memory and self-similarity that traditional methods miss
- **Multiple Forecasting Methods**: Statistical (ARIMA), Fractal (ST-FRSR, Pattern Projection), and ML (Random Forest, XGBoost)
- **Probability-Weighted Scenarios**: Generate multiple future paths with likelihood scores
- **Production-Ready**: Numba-optimized for performance

---

## Installation

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using pip
pip install -e .
```

### Requirements
- Python >= 3.10
- Core: NumPy, SciPy, Pandas, Polars
- ML: Scikit-learn, Statsmodels
- Visualization: Plotly, Matplotlib
- Performance: Numba

---

## Quick Start

```python
import fractime as ft
import numpy as np

# Load data (using built-in Yahoo Finance loader)
data = ft.get_yahoo_data("AAPL", start_date="2020-01-01")
prices = data['Close'].values

# Analyze fractal properties
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
fractal_dim = analyzer.compute_fractal_dimension(prices)

print(f"Hurst Exponent: {hurst:.3f}")
print(f"Fractal Dimension: {fractal_dim:.3f}")

if hurst > 0.5:
    print("→ Trending behavior (persistent)")
elif hurst < 0.5:
    print("→ Mean-reverting behavior (anti-persistent)")
else:
    print("→ Random walk behavior")

# Generate probability-weighted forecast paths
simulator = ft.FractalSimulator(prices, analyzer)
paths, metadata = simulator.simulate_paths(
    n_steps=30,      # Forecast 30 days ahead
    n_paths=1000,    # Generate 1000 scenarios
)

# Most likely outcome
print(f"\nMost likely forecast: ${paths[0, -1]:.2f}")
print(f"Median forecast: ${np.median(paths[:, -1]):.2f}")
print(f"95% CI: ${np.percentile(paths[:, -1], 2.5):.2f} - ${np.percentile(paths[:, -1], 97.5):.2f}")
```

---

## Core Components

### Fractal Analysis

Analyze time series fractal properties to understand market behavior:

```python
import fractime as ft

# Load your data
data = ft.get_yahoo_data("SPY", start_date="2020-01-01")
prices = data['Close'].values

# Initialize analyzer
analyzer = ft.FractalAnalyzer()

# Compute fractal metrics
hurst = analyzer.compute_hurst(prices)
fractal_dim = analyzer.compute_fractal_dimension(prices)

# Analyze patterns
patterns = analyzer.analyze_patterns(prices, full_analysis=True)
print(f"Hurst: {patterns['hurst']:.3f}")
print(f"Fractal Dimension: {patterns['fractal_dim']:.3f}")
print(f"Found {len(patterns['self_similar_patterns'])} patterns")
```

**Key Metrics:**
- **Hurst Exponent**: Measures long-term memory (H > 0.5 = trending, H < 0.5 = mean-reverting)
- **Fractal Dimension**: Measures complexity (higher = more jagged movements)
- **Self-Similar Patterns**: Recurring structures across time scales

### Path Simulation

Generate probability-weighted future scenarios:

```python
import fractime as ft
import numpy as np

# Initialize with historical data
data = ft.get_yahoo_data("AAPL", start_date="2020-01-01")
prices = data['Close'].values

analyzer = ft.FractalAnalyzer()
simulator = ft.FractalSimulator(prices, analyzer)

# Simulate future paths
paths, metadata = simulator.simulate_paths(
    n_steps=30,          # Days to forecast
    n_paths=1000,        # Number of scenarios
    pattern_weight=0.3,  # Weight for pattern-based generation
    use_trading_time=True  # Use volume-based time warping
)

# Analyze outcomes
final_prices = paths[:, -1]
print(f"Expected value: ${np.mean(final_prices):.2f}")
print(f"Standard deviation: ${np.std(final_prices):.2f}")
print(f"Probability of gain: {np.mean(final_prices > prices[-1]):.1%}")

# Risk metrics
var_95 = np.percentile(final_prices, 5)
print(f"Value at Risk (95%): ${var_95:.2f} ({(var_95/prices[-1]-1):.1%})")
```

**Simulation Features:**
- Pattern-based path generation
- Trading time warping (volume/volatility adjusted)
- Cross-dimensional analysis (price + volume)
- Fast and GPU-accelerated variants

### Forecasting

FracTime offers 3 categories of forecasting methods:

#### 1. Statistical Methods

```python
from fractime.forecasting import ARIMAForecaster, ExponentialSmoothingForecaster

# ARIMA
arima = ARIMAForecaster(p=1, d=1, q=1)
arima.fit(X=prices[:-30].reshape(-1, 1), y=prices[:-30])
forecast = arima.predict(X=np.zeros((30, 1)))

# Exponential Smoothing
ets = ExponentialSmoothingForecaster()
ets.fit(X=prices[:-30].reshape(-1, 1), y=prices[:-30])
forecast = ets.predict(X=np.zeros((30, 1)))
```

#### 2. Fractal Methods (Unique to FracTime)

```python
from fractime.forecasting import (
    StateTransitionFRSRForecaster,  # Regime-switching
    FractalProjectionForecaster,     # Hurst-based
    FractalClassificationForecaster  # Pattern-based
)

# State-Transition FRSR (Regime-Switching)
st_frsr = StateTransitionFRSRForecaster(n_states=3, window_size=20)
st_frsr.fit(X=prices[:-30].reshape(-1, 1), y=prices[:-30])
forecast = st_frsr.predict(X=np.zeros((30, 1)))

# Fractal Projection
proj = FractalProjectionForecaster(pattern_length=10)
proj.fit(X=prices[:-30].reshape(-1, 1), y=prices[:-30])
forecast = proj.predict(X=np.zeros((30, 1)))

# Fractal Classification
fclass = FractalClassificationForecaster(n_classes=4, window_size=5)
fclass.fit(X=prices[:-30].reshape(-1, 1), y=prices[:-30])
forecast = fclass.predict(X=np.zeros((30, 1)))
```

#### 3. Machine Learning Methods

```python
from fractime.forecasting import RandomForestForecaster, XGBoostForecaster

# Prepare features
def prepare_features(prices, n_lags=5):
    import polars as pl
    df = pl.DataFrame({'price': prices})
    for i in range(1, n_lags + 1):
        df = df.with_columns(pl.col('price').shift(i).alias(f'lag_{i}'))
    return df.drop_nulls()

data_df = prepare_features(prices, n_lags=5)
X = data_df.select([f'lag_{i}' for i in range(1, 6)]).to_numpy()
y = data_df['price'].to_numpy()

# Random Forest
rf = RandomForestForecaster(n_estimators=100, max_depth=10)
rf.fit(X[:-30], y[:-30])
forecast = rf.predict(X[-30:])

# XGBoost (requires xgboost package)
try:
    xgb = XGBoostForecaster(n_estimators=100)
    xgb.fit(X[:-30], y[:-30])
    forecast = xgb.predict(X[-30:])
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
```

### Backtesting

Rigorously test forecasting methods:

```python
import polars as pl
from fractime.forecasting import (
    ARIMAForecaster,
    StateTransitionFRSRForecaster,
    RandomForestForecaster
)

# Prepare data
def prepare_data(prices, n_lags=5):
    df = pl.DataFrame({'price': prices})
    for i in range(1, n_lags + 1):
        df = df.with_columns(pl.col('price').shift(i).alias(f'lag_{i}'))
    return df.drop_nulls()

data = ft.get_yahoo_data("AAPL", start_date="2020-01-01")
prices = data['Close'].values
prepared_data = prepare_data(prices, n_lags=5)

# Create forecasters
forecasters = {
    'ARIMA': ARIMAForecaster(p=1, d=1, q=1),
    'ST-FRSR': StateTransitionFRSRForecaster(),
    'Random Forest': RandomForestForecaster(n_estimators=100)
}

# Run backtest using built-in function
results = ft.run_backtest(
    prices=prices,
    forecasters=forecasters,
    train_size=252,        # 1 year training
    test_size=21,          # 1 month testing
    step_size=21,          # Re-train monthly
    verbose=True
)

# View results
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2%}")
```

---

## Examples

### Example 1: Compare Multiple Forecasters

```python
import fractime as ft
import numpy as np
from fractime.forecasting import *

# Load data
data = ft.get_yahoo_data("SPY", start_date="2020-01-01")
prices = data['Close'].values

# Create forecasters
forecasters = {
    'ARIMA': ARIMAForecaster(p=1, d=1, q=1),
    'ETS': ExponentialSmoothingForecaster(),
    'ST-FRSR': StateTransitionFRSRForecaster(),
    'Fractal Projection': FractalProjectionForecaster(),
    'Random Forest': RandomForestForecaster(n_estimators=100),
    'KNN': KNNForecaster(n_neighbors=5)
}

# Train and forecast
horizon = 30
results = {}

for name, forecaster in forecasters.items():
    try:
        # Split data
        train_prices = prices[:-horizon]
        X_train = train_prices[:-1].reshape(-1, 1)
        y_train = train_prices[1:]
        X_test = np.zeros((horizon, 1))

        # Fit and predict
        forecaster.fit(X_train, y_train)
        forecast = forecaster.predict(X_test)

        # Store results
        results[name] = forecast[-1] if isinstance(forecast, np.ndarray) else forecast
        change = (results[name] / prices[-1] - 1) * 100

        print(f"{name:20s}: ${results[name]:7.2f} ({change:+5.1f}%)")
    except Exception as e:
        print(f"{name:20s}: Error - {str(e)[:50]}")

# Ensemble forecast
ensemble = np.mean(list(results.values()))
print(f"\n{'Ensemble':20s}: ${ensemble:7.2f} ({(ensemble/prices[-1]-1)*100:+5.1f}%)")
```

### Example 2: Risk Analysis with Path Simulation

```python
import fractime as ft
import numpy as np

# Load data
data = ft.get_yahoo_data("TSLA", start_date="2020-01-01")
prices = data['Close'].values

# Simulate paths
analyzer = ft.FractalAnalyzer()
simulator = ft.FractalSimulator(prices, analyzer)

paths, metadata = simulator.simulate_paths(
    n_steps=30,
    n_paths=10000,
    use_trading_time=True
)

# Calculate risk metrics
final_prices = paths[:, -1]
current_price = prices[-1]

# Expected values
expected_price = np.mean(final_prices)
median_price = np.median(final_prices)

# Probabilities
prob_gain = np.mean(final_prices > current_price)
prob_10pct_gain = np.mean(final_prices > current_price * 1.10)
prob_10pct_loss = np.mean(final_prices < current_price * 0.90)

# Risk metrics
var_95 = np.percentile(final_prices, 5)
cvar_95 = np.mean(final_prices[final_prices <= var_95])

print(f"Current Price: ${current_price:.2f}")
print(f"\n30-Day Forecast:")
print(f"  Expected: ${expected_price:.2f} ({(expected_price/current_price-1):.1%})")
print(f"  Median: ${median_price:.2f}")
print(f"  95% CI: ${np.percentile(final_prices, 2.5):.2f} - ${np.percentile(final_prices, 97.5):.2f}")
print(f"\nProbabilities:")
print(f"  Any gain: {prob_gain:.1%}")
print(f"  >10% gain: {prob_10pct_gain:.1%}")
print(f"  >10% loss: {prob_10pct_loss:.1%}")
print(f"\nRisk Metrics:")
print(f"  VaR (95%): ${var_95:.2f} ({(var_95/current_price-1):.1%})")
print(f"  CVaR (95%): ${cvar_95:.2f} ({(cvar_95/current_price-1):.1%})")
```

### Example 3: Custom Data (Non-Financial)

```python
import fractime as ft
import numpy as np

# Example: Energy consumption data
# Replace with your own time series
energy_consumption = np.random.randn(500).cumsum() + 100

# Analyze fractal properties
analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(energy_consumption)
fractal_dim = analyzer.compute_fractal_dimension(energy_consumption)

print(f"Hurst Exponent: {hurst:.3f}")
print(f"Fractal Dimension: {fractal_dim:.3f}")

if hurst > 0.5:
    print("→ Persistent patterns (trends continue)")
else:
    print("→ Mean-reverting patterns (spikes revert)")

# Forecast
from fractime.forecasting import StateTransitionFRSRForecaster

forecaster = StateTransitionFRSRForecaster()
forecaster.fit(
    X=energy_consumption[:-30].reshape(-1, 1),
    y=energy_consumption[:-30]
)
forecast = forecaster.predict(X=np.zeros((30, 1)))

print(f"\n7-day forecast average: {np.mean(forecast[:7]):.1f}")
print(f"Current 7-day average: {np.mean(energy_consumption[-7:]):.1f}")
```

---

## API Reference

### FractalAnalyzer

Analyze fractal properties of time series.

```python
analyzer = ft.FractalAnalyzer()
```

**Methods:**
- `compute_hurst(prices)` → float: Calculate Hurst exponent (R/S analysis)
- `compute_fractal_dimension(prices, quick_mode=False)` → float: Calculate fractal dimension (box-counting)
- `analyze_patterns(prices, full_analysis=True)` → dict: Comprehensive analysis
- `get_patterns(prices, max_patterns=20)` → list: Extract self-similar patterns

### FractalSimulator

Generate probability-weighted future paths.

```python
simulator = ft.FractalSimulator(prices, analyzer, volumes=None)
```

**Methods:**
- `simulate_paths(n_steps, n_paths=1000, **kwargs)` → (paths, metadata): Main simulation
  - `pattern_weight`: Weight for pattern-based generation (0-1)
  - `use_trading_time`: Enable volume-based time warping
  - `cloud_paths`: Number of high-probability paths to return
- `simulate_paths_fast(n_steps, n_paths=100)` → paths: Faster variant
- `analyze_path_distributions(paths)` → dict: Statistical analysis of paths

### Forecasting Classes

All forecasters inherit from `BaseForecaster` and implement:
- `fit(X, y)` → self: Train the model
- `predict(X)` → predictions: Make forecasts
- `get_params()` → dict: Get model parameters

**Statistical:**
- `ARIMAForecaster(p=1, d=1, q=1)`
- `SARIMAForecaster(p=1, d=1, q=1, P=0, D=0, Q=0, s=0)`
- `ExponentialSmoothingForecaster(trend=None, seasonal=None)`

**Fractal (Unique to FracTime):**
- `StateTransitionFRSRForecaster(n_states=3, window_size=10)`
- `FractalProjectionForecaster(pattern_length=10, similarity_threshold=0.8)`
- `FractalClassificationForecaster(n_classes=4, window_size=5)`
- `RescaledRangeForecaster(window_size=10, n_lags=5)`

**Machine Learning:**
- `RandomForestForecaster(n_estimators=100, max_depth=None)`
- `XGBoostForecaster(n_estimators=100, max_depth=6, learning_rate=0.1)` *requires xgboost*
- `SVRForecaster(kernel='rbf', C=1.0, epsilon=0.1)`
- `KNNForecaster(n_neighbors=5, weights='uniform')`

### Utility Functions

```python
# Load data from Yahoo Finance
data = ft.get_yahoo_data(symbol, start_date, end_date=None)

# Run backtesting
results = ft.run_backtest(prices, forecasters, train_size, test_size, step_size, verbose=True)
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fractime --cov-report=html

# Run specific test
pytest tests/test_core.py::test_hurst_calculation
```

### Code Quality

```bash
# Format code
black fractime/ tests/

# Lint code
ruff check fractime/ tests/

# Type checking
mypy fractime/
```

### Project Structure

```
fractime/
├── fractime/
│   ├── __init__.py
│   ├── core.py            # Fractal analysis & simulation
│   ├── optimization.py    # Numba-accelerated functions
│   ├── forecasting/       # Forecasting methods
│   │   ├── base.py
│   │   ├── statistical.py
│   │   ├── fractal.py
│   │   └── ml.py
│   └── backtester.py      # Backtesting framework
├── tests/
├── examples/
└── pyproject.toml
```

---

## Mathematical Background

### Hurst Exponent (H)

Measures long-term memory in time series:

- **H > 0.5**: Persistent (trending) - price increases likely followed by increases
- **H = 0.5**: Random walk (Brownian motion) - no memory
- **H < 0.5**: Anti-persistent (mean-reverting) - price increases likely followed by decreases

### Fractal Dimension (D)

Quantifies complexity of price movements:

- **D = 2 - H** for time series
- Higher D → more jagged, complex movements
- Lower D → smoother, more persistent trends

### Fractional Brownian Motion (FBM)

Generalization of Brownian motion with memory parameter H. Used to generate realistic price paths that capture the fractal nature of financial markets.

---

## Citation

If you use FracTime in research, please cite:

```bibtex
@software{fractime2024,
  title = {FracTime: Time Series Forecasting with Fractal Geometry},
  year = {2024},
  url = {https://github.com/Wayy-Research/fractime},
  version = {0.1.0}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest`)
5. Format code (`black`, `ruff`)
6. Submit a pull request

---

## Disclaimer

**FracTime is for research and educational purposes only.**

- Past performance does not guarantee future results
- All trading decisions are your own responsibility
- No warranty of any kind is provided
- Consult financial professionals before investing

Financial markets are inherently unpredictable. Use FracTime as one tool in your analysis toolkit.

---

**Built with Python | Powered by Fractal Geometry | Inspired by Mandelbrot**
