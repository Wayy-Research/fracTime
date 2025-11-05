# FracTime

**FracTime** is a Python library for advanced time series forecasting that leverages fractal geometry and chaos theory principles for market analysis and prediction.

This is a core library designed to be integrated into your own applications—whether you're building web dashboards, trading systems, research tools, or data pipelines.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [Data Loading](#data-loading)
  - [Fractal Analysis](#fractal-analysis)
  - [Path Simulation](#path-simulation)
  - [Path Analysis & Clustering](#path-analysis--clustering)
  - [Visualization](#visualization)
  - [Forecasting Methods](#forecasting-methods)
  - [Backtesting Framework](#backtesting-framework)
- [Advanced Examples](#advanced-examples)
- [Using FracTime in Your Projects](#using-fractime-in-your-projects)
- [Development](#development)
- [Theoretical Framework](#theoretical-framework)
- [License](#license)

---

## Introduction

FracTime brings together cutting-edge concepts from fractal geometry, chaos theory, and quantitative finance to provide a comprehensive toolkit for time series analysis and forecasting. Unlike traditional methods that assume normal distributions and market efficiency, FracTime embraces the fractal nature of financial markets as described by Benoit Mandelbrot.

**Why FracTime?**

- **Fractal-Based**: Captures market memory and self-similarity across time scales
- **Multiple Forecasting Methods**: Compare statistical, fractal, and ML approaches
- **Comprehensive Backtesting**: Robust framework for evaluating forecast accuracy
- **Flexible Data Sources**: Built-in support for Yahoo Finance, Alpha Vantage, crypto exchanges, and economic data
- **Production-Ready**: Numba-optimized performance for real-world applications

---

## Key Features

### Pattern Recognition
- **Hurst Exponent Calculation**: Measure long-term memory and trend persistence using R/S analysis
- **Fractal Dimension Estimation**: Quantify complexity and self-similarity of price movements
- **Self-Similarity Detection**: Identify recurring patterns across different time scales
- **Multi-Scale Analysis**: Analyze price movements at various temporal resolutions

### Advanced Simulation
- **Fractional Brownian Motion**: Generate paths using spectral synthesis methods
- **Pattern-Weighted Generation**: Create scenarios based on historical pattern similarities
- **Non-Parametric Bootstrapping**: Resample historical data with recency bias
- **Regime Detection**: Identify market states using Hidden Markov Models

### Path Analysis
- **K-Means Clustering**: Group similar price trajectories
- **Probability Density Estimation**: Calculate likelihood of different outcomes
- **Representative Paths**: Select typical trajectories for each cluster
- **Empirical Confidence Intervals**: Data-driven uncertainty quantification

### Forecasting Suite
- **Statistical Methods**: ARIMA, SARIMA, Exponential Smoothing
- **Fractal Methods**: State-transition FRSR, Fractal Projection, Rescaled Range
- **Machine Learning**: Random Forest, XGBoost, SVR, KNN
- **Custom Forecasters**: Extensible base class for your own methods

---

## Installation

### Using uv (Recommended)

FracTime uses `uv` for fast, reliable dependency management:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
uv pip install -e .

# Install development dependencies (for contributing)
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### Requirements

- Python >= 3.10
- NumPy, SciPy, Pandas, Polars
- Scikit-learn, Statsmodels
- Plotly for visualizations
- Numba for performance optimization

---

## Quick Start

Here's a minimal example to get you started:

```python
import fractime as ft
import numpy as np

# Load market data
data = ft.get_yahoo_data("^GSPC", "2020-01-01", "2024-01-01")
prices = data['Close'].values

# Analyze fractal properties
analyzer = ft.FractalAnalyzer()
hurst = analyzer.calculate_hurst_exponent(prices)
fractal_dim = analyzer.calculate_fractal_dimension(prices)

print(f"Hurst Exponent: {hurst:.3f}")
print(f"Fractal Dimension: {fractal_dim:.3f}")

if hurst > 0.5:
    print("Market shows trending behavior (persistent)")
elif hurst < 0.5:
    print("Market shows mean-reverting behavior (anti-persistent)")
else:
    print("Market shows random walk behavior")

# Simulate future paths
simulator = ft.FractalSimulator()
paths = simulator.simulate_paths(
    initial_price=prices[-1],
    n_steps=30,      # Forecast 30 days ahead
    n_paths=1000,    # Generate 1000 scenarios
    dt=1/252,        # Daily time step
    hurst=hurst      # Use calculated Hurst exponent
)

# Analyze simulated paths
path_analyzer = ft.PathAnalyzer()
stats = path_analyzer.calculate_statistics(paths)

print(f"\n30-Day Forecast Statistics:")
print(f"Expected Price: ${stats['mean']:.2f}")
print(f"95% Confidence Interval: ${stats['ci_lower']:.2f} - ${stats['ci_upper']:.2f}")
print(f"Downside Risk (5th percentile): ${stats['percentile_5']:.2f}")

# Visualize results
visualizer = ft.FractalVisualizer()
fig = visualizer.plot_simulation_results(paths, data, stats)
fig.show()
```

---

## Core Components

### Data Loading

FracTime provides multiple data sources through a unified interface:

#### Yahoo Finance (Free, No API Key Required)

```python
import fractime as ft

# Get stock data
data = ft.get_yahoo_data("AAPL", "2020-01-01", "2024-01-01")

# Get index data
sp500 = ft.get_yahoo_data("^GSPC", "2020-01-01", "2024-01-01")

# Get cryptocurrency data
btc = ft.get_yahoo_data("BTC-USD", "2020-01-01", "2024-01-01")
```

#### Multiple Data Sources

```python
from fractime.data_sources import get_data_with_fallback, list_sources

# List all available data sources
sources = list_sources()
print(f"Available sources: {sources}")

# Get data with automatic fallback
data = get_data_with_fallback(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2024-01-01",
    preferred_sources=["alpha_vantage", "yahoo", "twelve_data"]
)
```

**Supported Data Sources:**
- **Equities**: Yahoo Finance, Alpha Vantage, Twelve Data, Finnhub, Tiingo
- **Crypto**: Binance, CoinGecko, Kraken
- **Economic Data**: FRED (Federal Reserve), World Bank, ECB
- **Forex & Commodities**: Various providers

### Fractal Analysis

Analyze the fractal properties of time series data:

```python
import fractime as ft
import numpy as np

# Load your data
prices = ft.get_yahoo_data("AAPL", "2020-01-01", "2024-01-01")['Close'].values

# Initialize analyzer
analyzer = ft.FractalAnalyzer()

# Calculate Hurst exponent (measures trend persistence)
hurst = analyzer.calculate_hurst_exponent(prices)
print(f"Hurst Exponent: {hurst:.3f}")
# H > 0.5: Trending (persistent)
# H = 0.5: Random walk
# H < 0.5: Mean-reverting (anti-persistent)

# Calculate fractal dimension (measures complexity)
fractal_dim = analyzer.calculate_fractal_dimension(prices)
print(f"Fractal Dimension: {fractal_dim:.3f}")
# Higher values indicate more complex, jagged price movements

# Find self-similar patterns
returns = np.diff(np.log(prices))
patterns = analyzer.find_similar_patterns(
    returns,
    pattern_length=20,
    n_patterns=5,
    threshold=0.8
)
print(f"Found {len(patterns)} similar patterns")

# Analyze volatility scaling
scaling_exponent = analyzer.analyze_volatility_scaling(returns)
print(f"Volatility Scaling Exponent: {scaling_exponent:.3f}")
```

**Key Methods:**

- `calculate_hurst_exponent()`: R/S analysis for long-term memory
- `calculate_fractal_dimension()`: Box-counting dimension
- `find_similar_patterns()`: Pattern matching across time scales
- `analyze_volatility_scaling()`: How volatility changes with time horizon
- `detect_regime_changes()`: Identify shifts in market behavior

### Path Simulation

Generate realistic future price scenarios:

```python
import fractime as ft

# Initialize simulator
simulator = ft.FractalSimulator()
current_price = 100.0

# Method 1: Fractional Brownian Motion
paths_fbm = simulator.simulate_paths(
    initial_price=current_price,
    n_steps=30,        # 30 days ahead
    n_paths=1000,      # 1000 scenarios
    dt=1/252,          # Daily time step
    hurst=0.6,         # Trending market
    volatility=0.02    # 2% daily volatility
)

# Method 2: Pattern-Based Simulation
historical_prices = ft.get_yahoo_data("SPY", "2020-01-01", "2024-01-01")['Close'].values
paths_pattern = simulator.simulate_patterns(
    historical_prices=historical_prices,
    n_paths=1000,
    forecast_horizon=30,
    pattern_length=20,
    recency_weight=0.7  # Favor recent patterns
)

# Method 3: Bootstrap Simulation
paths_bootstrap = simulator.bootstrap_paths(
    historical_returns=np.diff(np.log(historical_prices)),
    initial_price=current_price,
    n_steps=30,
    n_paths=1000,
    block_size=5       # Block bootstrap for autocorrelation
)

# Method 4: Regime-Based Simulation
paths_regime = simulator.simulate_with_regimes(
    historical_prices=historical_prices,
    n_paths=1000,
    forecast_horizon=30,
    n_regimes=3        # Identify 3 market regimes
)

print(f"Generated {paths_fbm.shape[0]} paths with {paths_fbm.shape[1]} time steps")
```

**Simulation Methods:**

- `simulate_paths()`: Fractional Brownian motion (FBM)
- `simulate_patterns()`: Pattern-weighted path generation
- `bootstrap_paths()`: Non-parametric resampling
- `simulate_with_regimes()`: Regime-switching models

### Path Analysis & Clustering

Analyze and cluster simulated paths:

```python
import fractime as ft

# Generate paths (from previous example)
simulator = ft.FractalSimulator()
paths = simulator.simulate_paths(
    initial_price=100,
    n_steps=30,
    n_paths=1000,
    dt=1/252
)

# Initialize path analyzer
path_analyzer = ft.PathAnalyzer()

# Calculate comprehensive statistics
stats = path_analyzer.calculate_statistics(paths)
print(f"Mean final price: ${stats['mean']:.2f}")
print(f"Median final price: ${stats['median']:.2f}")
print(f"Standard deviation: ${stats['std']:.2f}")
print(f"95% CI: [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")
print(f"Probability of profit: {stats['prob_above_initial']:.1%}")

# Cluster paths into scenarios
n_clusters = 5
clusters = path_analyzer.cluster_paths(paths, n_clusters=n_clusters)

# Get representative path for each cluster
representatives = path_analyzer.get_representative_paths(paths, clusters)

# Analyze each cluster
for i in range(n_clusters):
    cluster_paths = paths[clusters == i]
    cluster_stats = path_analyzer.calculate_statistics(cluster_paths)
    cluster_prob = np.mean(clusters == i)

    print(f"\nCluster {i+1} ({cluster_prob:.1%} of scenarios):")
    print(f"  Mean return: {cluster_stats['total_return']:.1%}")
    print(f"  Max drawdown: {cluster_stats['max_drawdown']:.1%}")
    print(f"  Volatility: {cluster_stats['volatility']:.1%}")

# Calculate probability density
prices = np.linspace(paths.min(), paths.max(), 100)
density = path_analyzer.estimate_density(paths[:, -1], prices)
```

**Analysis Methods:**

- `calculate_statistics()`: Mean, median, CI, percentiles, Sharpe ratio
- `cluster_paths()`: K-means clustering of trajectories
- `get_representative_paths()`: Select typical path for each cluster
- `estimate_density()`: Kernel density estimation
- `calculate_drawdown()`: Maximum drawdown analysis

### Visualization

Create publication-quality visualizations:

```python
import fractime as ft

# Initialize visualizer
visualizer = ft.FractalVisualizer()

# Plot 1: Simulation results with confidence intervals
fig1 = visualizer.plot_simulation_results(
    paths=paths,
    historical_data=historical_prices,
    stats=stats,
    show_ci=True,
    show_percentiles=[5, 25, 75, 95]
)
fig1.show()

# Plot 2: Clustered paths
fig2 = visualizer.plot_clustered_paths(
    paths=paths,
    clusters=clusters,
    representatives=representatives,
    show_legend=True
)
fig2.show()

# Plot 3: Probability density
fig3 = visualizer.plot_density(
    paths=paths,
    historical_data=historical_prices,
    bins=50
)
fig3.show()

# Plot 4: Fractal analysis results
fig4 = visualizer.plot_fractal_analysis(
    prices=historical_prices,
    hurst=hurst,
    fractal_dim=fractal_dim,
    regimes=regimes
)
fig4.show()

# Plot 5: Heatmap of path distribution
fig5 = visualizer.plot_path_heatmap(
    paths=paths,
    resolution=100
)
fig5.show()

# Save figures
fig1.write_html("simulation_results.html")
fig1.write_image("simulation_results.png", width=1200, height=600)
```

**Visualization Methods:**

- `plot_simulation_results()`: Paths with confidence bands
- `plot_clustered_paths()`: Color-coded scenario clusters
- `plot_density()`: Probability distribution of outcomes
- `plot_fractal_analysis()`: Hurst exponent and regime charts
- `plot_path_heatmap()`: 2D density visualization

### Forecasting Methods

Compare multiple forecasting approaches:

```python
from fractime.forecasting import (
    ARIMAForecaster,
    StateTransitionFRSRForecaster,
    FractalProjectionForecaster,
    RandomForestForecaster,
    XGBoostForecaster
)
import polars as pl

# Prepare your data (create lag features)
def prepare_data(prices, n_lags=5):
    df = pl.DataFrame({'price': prices})
    for i in range(1, n_lags + 1):
        df = df.with_columns(
            pl.col('price').shift(i).alias(f'lag_{i}')
        )
    return df.drop_nulls()

data = prepare_data(historical_prices, n_lags=5)
feature_cols = [f'lag_{i}' for i in range(1, 6)]

# Initialize forecasters
forecasters = {
    'ARIMA': ARIMAForecaster(p=1, d=1, q=1),
    'ST-FRSR': StateTransitionFRSRForecaster(n_patterns=10),
    'Fractal Projection': FractalProjectionForecaster(hurst=0.6),
    'Random Forest': RandomForestForecaster(n_estimators=100),
    'XGBoost': XGBoostForecaster(n_estimators=100),
}

# Train and forecast with each method
results = {}
for name, forecaster in forecasters.items():
    # Fit the model
    forecaster.fit(
        data=data,
        target_col='price',
        feature_cols=feature_cols
    )

    # Make predictions
    forecast = forecaster.predict(
        data=data[-1:],
        horizon=30
    )

    results[name] = forecast
    print(f"{name}: 30-day forecast = {forecast[-1]:.2f}")

# Ensemble forecast (average of all methods)
ensemble_forecast = np.mean([results[name] for name in results], axis=0)
print(f"\nEnsemble: 30-day forecast = {ensemble_forecast[-1]:.2f}")
```

**Available Forecasters:**

**Statistical Methods:**
- `ARIMAForecaster`: Auto-regressive integrated moving average
- `SARIMAForecaster`: Seasonal ARIMA
- `ExponentialSmoothingForecaster`: ETS models

**Fractal Methods:**
- `StateTransitionFRSRForecaster`: Fractal regime-switching
- `FractalProjectionForecaster`: Hurst-based projection
- `FractalClassificationForecaster`: Pattern classification
- `RescaledRangeForecaster`: R/S analysis forecasting
- `FractalInterpolationForecaster`: Multi-scale interpolation

**Machine Learning:**
- `RandomForestForecaster`: Ensemble of decision trees
- `XGBoostForecaster`: Gradient boosting
- `SVRForecaster`: Support vector regression
- `KNNForecaster`: K-nearest neighbors

### Backtesting Framework

Rigorously evaluate forecasting methods:

```python
from fractime.backtester import TimeSeriesBacktester
from fractime.forecasting import *

# Create forecasters dictionary
forecasters = {
    'ARIMA(1,1,1)': ARIMAForecaster(p=1, d=1, q=1),
    'ST-FRSR': StateTransitionFRSRForecaster(),
    'Random Forest': RandomForestForecaster(n_estimators=100),
    'XGBoost': XGBoostForecaster(n_estimators=100),
}

# Initialize backtester
backtester = TimeSeriesBacktester(forecasters)

# Run backtest
results = backtester.run_backtest(
    data=prepared_data,
    target_col='price',
    feature_cols=feature_cols,
    window_size=252,          # 1 year training window
    step_size=21,             # Re-train monthly
    forecast_horizon=5,       # 5-day ahead forecast
    expanding_window=False    # Use rolling window
)

# Generate report
report = backtester.generate_report()
print(report)

# Plot results
fig = backtester.plot_results()
fig.show()

# Access detailed metrics
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  RMSE: {metrics['avg_rmse']:.4f}")
    print(f"  MAE: {metrics['avg_mae']:.4f}")
    print(f"  MAPE: {metrics['avg_mape']:.2%}")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
```

**Backtesting Features:**

- **Rolling/Expanding Windows**: Train on historical data, test on future
- **Multiple Metrics**: RMSE, MAE, MAPE, directional accuracy, Sharpe ratio
- **Walk-Forward Analysis**: Realistic out-of-sample testing
- **Visualization**: Compare forecast accuracy across methods
- **Custom Evaluation**: Add your own metrics

---

## Advanced Examples

### Example 1: Market Regime Detection

```python
import fractime as ft
import numpy as np

# Load data
prices = ft.get_yahoo_data("SPY", "2010-01-01", "2024-01-01")['Close'].values

# Initialize analyzer
analyzer = ft.FractalAnalyzer()

# Detect regimes using HMM
regimes = analyzer.detect_regime_changes(
    prices,
    n_regimes=3,      # Bull, bear, sideways
    lookback=252      # 1-year rolling window
)

# Analyze each regime
returns = np.diff(np.log(prices))
for regime_id in range(3):
    mask = regimes[1:] == regime_id
    regime_returns = returns[mask]

    print(f"\nRegime {regime_id + 1}:")
    print(f"  Frequency: {np.mean(mask):.1%}")
    print(f"  Avg Return: {np.mean(regime_returns) * 252:.1%} (annualized)")
    print(f"  Volatility: {np.std(regime_returns) * np.sqrt(252):.1%}")
    print(f"  Sharpe: {np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(252):.2f}")

# Forecast regime probabilities
regime_probs = analyzer.forecast_regime_probabilities(prices, horizon=30)
print(f"\n30-Day Regime Probabilities:")
for i, prob in enumerate(regime_probs):
    print(f"  Regime {i+1}: {prob:.1%}")
```

### Example 2: Multi-Asset Portfolio Simulation

```python
import fractime as ft
import numpy as np

# Load multiple assets
assets = ['SPY', 'TLT', 'GLD', 'QQQ']
data = {}
for symbol in assets:
    data[symbol] = ft.get_yahoo_data(symbol, "2020-01-01", "2024-01-01")['Close'].values

# Calculate correlation matrix
returns_matrix = np.array([np.diff(np.log(data[symbol])) for symbol in assets])
corr_matrix = np.corrcoef(returns_matrix)

# Simulate correlated paths
simulator = ft.FractalSimulator()
portfolio_paths = {}

for i, symbol in enumerate(assets):
    # Get asset-specific Hurst exponent
    analyzer = ft.FractalAnalyzer()
    hurst = analyzer.calculate_hurst_exponent(data[symbol])

    # Simulate paths
    portfolio_paths[symbol] = simulator.simulate_paths(
        initial_price=data[symbol][-1],
        n_steps=30,
        n_paths=1000,
        dt=1/252,
        hurst=hurst,
        correlation_matrix=corr_matrix[i]
    )

# Calculate portfolio value (equal weight)
weights = np.array([0.25, 0.25, 0.25, 0.25])
portfolio_value = sum(
    weights[i] * portfolio_paths[symbol]
    for i, symbol in enumerate(assets)
)

# Analyze portfolio
path_analyzer = ft.PathAnalyzer()
portfolio_stats = path_analyzer.calculate_statistics(portfolio_value)

print("Portfolio Forecast (30 days):")
print(f"Expected Return: {portfolio_stats['total_return']:.1%}")
print(f"Portfolio Volatility: {portfolio_stats['volatility']:.1%}")
print(f"Sharpe Ratio: {portfolio_stats['sharpe_ratio']:.2f}")
print(f"VaR (95%): {portfolio_stats['var_95']:.1%}")
```

### Example 3: Trading Strategy Backtest

```python
import fractime as ft
import numpy as np

# Load data
data = ft.get_yahoo_data("AAPL", "2020-01-01", "2024-01-01")
prices = data['Close'].values

# Strategy: Buy when Hurst > 0.6 (trending), Sell when Hurst < 0.4 (mean-reverting)
analyzer = ft.FractalAnalyzer()
positions = []
lookback = 60

for i in range(lookback, len(prices)):
    window = prices[i-lookback:i]
    hurst = analyzer.calculate_hurst_exponent(window)

    if hurst > 0.6:
        positions.append(1)   # Long
    elif hurst < 0.4:
        positions.append(-1)  # Short
    else:
        positions.append(0)   # Neutral

positions = np.array(positions)

# Calculate strategy returns
returns = np.diff(np.log(prices[lookback:]))
strategy_returns = positions[:-1] * returns

# Performance metrics
cumulative_return = np.exp(np.sum(strategy_returns)) - 1
sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
max_drawdown = np.min(np.minimum.accumulate(np.cumsum(strategy_returns)) - np.cumsum(strategy_returns))

print(f"Strategy Performance:")
print(f"Total Return: {cumulative_return:.1%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.1%}")
print(f"Win Rate: {np.mean(strategy_returns > 0):.1%}")

# Compare to buy-and-hold
bh_return = prices[-1] / prices[lookback] - 1
print(f"\nBuy & Hold Return: {bh_return:.1%}")
print(f"Alpha: {cumulative_return - bh_return:.1%}")
```

---

## Using FracTime in Your Projects

### As a Library Dependency

Add FracTime to your project:

```bash
# Using uv
uv pip install fractime

# Or add to your pyproject.toml
[project]
dependencies = [
    "fractime>=0.1.0",
]
```

### Building Applications with FracTime

FracTime is designed as a core library that can be integrated into various applications:

#### Web Dashboard Example (Streamlit)

```python
import streamlit as st
import fractime as ft
import numpy as np

st.title("FracTime Market Analysis Dashboard")

# User inputs
symbol = st.text_input("Enter stock symbol:", "AAPL")
start_date = st.date_input("Start date:", value=pd.to_datetime("2020-01-01"))

if st.button("Analyze"):
    # Load data
    data = ft.get_yahoo_data(symbol, str(start_date), None)
    prices = data['Close'].values

    # Analyze
    analyzer = ft.FractalAnalyzer()
    hurst = analyzer.calculate_hurst_exponent(prices)
    fractal_dim = analyzer.calculate_fractal_dimension(prices)

    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Hurst Exponent", f"{hurst:.3f}")
    col2.metric("Fractal Dimension", f"{fractal_dim:.3f}")

    # Simulate paths
    simulator = ft.FractalSimulator()
    paths = simulator.simulate_paths(
        initial_price=prices[-1],
        n_steps=30,
        n_paths=1000,
        dt=1/252,
        hurst=hurst
    )

    # Visualize
    visualizer = ft.FractalVisualizer()
    fig = visualizer.plot_simulation_results(paths, data, {})
    st.plotly_chart(fig, use_container_width=True)
```

#### Trading Bot Integration

```python
import fractime as ft
import alpaca_trade_api as tradeapi

class FractalTradingBot:
    def __init__(self, api_key, secret_key):
        self.api = tradeapi.REST(api_key, secret_key)
        self.analyzer = ft.FractalAnalyzer()
        self.simulator = ft.FractalSimulator()

    def analyze_and_trade(self, symbol, lookback=60):
        # Get historical data
        data = ft.get_yahoo_data(symbol, lookback_days=lookback)
        prices = data['Close'].values

        # Calculate Hurst exponent
        hurst = self.analyzer.calculate_hurst_exponent(prices)

        # Trading logic
        if hurst > 0.6:  # Strong trending
            # Simulate future paths
            paths = self.simulator.simulate_paths(
                initial_price=prices[-1],
                n_steps=5,
                n_paths=1000,
                hurst=hurst
            )

            # Calculate probability of uptrend
            prob_up = np.mean(paths[:, -1] > prices[-1])

            if prob_up > 0.65:
                self.api.submit_order(
                    symbol=symbol,
                    qty=100,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                return f"BUY signal: {prob_up:.1%} probability of uptrend"

        return "No trade signal"

# Usage
bot = FractalTradingBot(api_key="your_key", secret_key="your_secret")
signal = bot.analyze_and_trade("AAPL")
print(signal)
```

#### Research Pipeline

```python
import fractime as ft
import pandas as pd
from typing import Dict, List

class FractalResearchPipeline:
    """Batch analysis pipeline for research."""

    def __init__(self):
        self.analyzer = ft.FractalAnalyzer()

    def analyze_universe(self, symbols: List[str], start_date: str) -> pd.DataFrame:
        """Analyze fractal properties across multiple assets."""
        results = []

        for symbol in symbols:
            try:
                # Load data
                data = ft.get_yahoo_data(symbol, start_date)
                prices = data['Close'].values

                # Calculate metrics
                hurst = self.analyzer.calculate_hurst_exponent(prices)
                fractal_dim = self.analyzer.calculate_fractal_dimension(prices)
                vol = np.std(np.diff(np.log(prices))) * np.sqrt(252)

                results.append({
                    'symbol': symbol,
                    'hurst': hurst,
                    'fractal_dim': fractal_dim,
                    'volatility': vol,
                    'market_type': 'trending' if hurst > 0.5 else 'mean_reverting'
                })

            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

        return pd.DataFrame(results)

# Usage
pipeline = FractalResearchPipeline()
sp500_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
results = pipeline.analyze_universe(sp500_symbols, "2020-01-01")

# Find most trending stocks
trending = results.nlargest(10, 'hurst')
print("Most trending stocks:")
print(trending[['symbol', 'hurst', 'market_type']])
```

### Application Ideas

- **Portfolio Optimization**: Use fractal analysis to identify complementary assets
- **Risk Management**: Calculate regime-specific VaR and stress scenarios
- **Market Microstructure**: Analyze high-frequency patterns and self-similarity
- **Derivatives Pricing**: Incorporate fractal volatility into option models
- **Economic Forecasting**: Apply to macro indicators and commodity prices

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fractime --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run a specific test
pytest tests/test_core.py::test_sample_data

# Run in verbose mode
pytest -v
```

### Code Quality

```bash
# Format code (automatically fix style issues)
black fractime/ tests/

# Lint code (check for errors)
ruff check fractime/ tests/

# Type checking (verify type annotations)
mypy fractime/

# Run all quality checks
black fractime/ tests/ && ruff check fractime/ tests/ && mypy fractime/
```

### Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Format code: `black fractime/ tests/`
6. Submit a pull request

**Contribution Guidelines:**
- Add tests for new features
- Update documentation
- Follow existing code style
- Ensure type hints are included
- Write descriptive commit messages

### Project Structure

```
fractime/
├── fractime/              # Main package
│   ├── __init__.py        # Package exports
│   ├── core.py            # Core fractal analysis & simulation
│   ├── data_loader.py     # Data loading utilities
│   ├── backtester.py      # Backtesting framework
│   ├── optimization.py    # Performance-optimized functions
│   ├── forecasting/       # Forecasting methods
│   │   ├── base.py        # Base forecaster class
│   │   ├── statistical.py # ARIMA, ETS, etc.
│   │   ├── fractal.py     # Fractal forecasters
│   │   └── ml.py          # ML forecasters
│   └── data_sources/      # Data source connectors
│       ├── equities.py    # Stock data sources
│       ├── crypto.py      # Crypto data sources
│       ├── economic.py    # Economic indicators
│       └── registry.py    # Source management
├── tests/                 # Test suite
├── examples/              # Example scripts
├── CLAUDE.md              # Development guide
├── README.md              # This file
└── pyproject.toml         # Project configuration
```

## API Reference Summary

### FractalAnalyzer

**Core Methods:**
- `calculate_hurst_exponent(prices, min_lag=2, max_lag=100)` - R/S analysis
- `calculate_fractal_dimension(prices)` - Box-counting dimension
- `find_similar_patterns(returns, pattern_length, n_patterns, threshold)` - Pattern matching
- `analyze_volatility_scaling(returns, scales)` - Multi-scale volatility
- `detect_regime_changes(prices, n_regimes, lookback)` - HMM regime detection
- `forecast_regime_probabilities(prices, horizon)` - Regime transition forecast

### FractalSimulator

**Simulation Methods:**
- `simulate_paths(initial_price, n_steps, n_paths, dt, hurst, volatility)` - FBM
- `simulate_patterns(historical_prices, n_paths, forecast_horizon, pattern_length)` - Pattern-based
- `bootstrap_paths(historical_returns, initial_price, n_steps, n_paths, block_size)` - Bootstrap
- `simulate_with_regimes(historical_prices, n_paths, forecast_horizon, n_regimes)` - Regime-based

### PathAnalyzer

**Analysis Methods:**
- `calculate_statistics(paths)` - Comprehensive path statistics
- `cluster_paths(paths, n_clusters)` - K-means clustering
- `get_representative_paths(paths, clusters)` - Representative selection
- `estimate_density(values, grid_points)` - KDE probability density
- `calculate_drawdown(paths)` - Drawdown analysis

### FractalVisualizer

**Plotting Methods:**
- `plot_simulation_results(paths, historical_data, stats, show_ci, show_percentiles)` - Main results
- `plot_clustered_paths(paths, clusters, representatives)` - Clustered scenarios
- `plot_density(paths, historical_data, bins)` - Probability distribution
- `plot_fractal_analysis(prices, hurst, fractal_dim, regimes)` - Analysis results
- `plot_path_heatmap(paths, resolution)` - 2D density heatmap

### TimeSeriesBacktester

**Backtesting Methods:**
- `run_backtest(data, target_col, feature_cols, window_size, step_size, forecast_horizon)` - Main backtest
- `generate_report()` - Performance report
- `plot_results()` - Visualization

---

## Examples Directory

The `examples/` directory contains complete working examples:

### forecasting_comparison.py

Complete script demonstrating:
- Loading data from multiple sources
- Comparing statistical, fractal, and ML forecasters
- Running walk-forward backtests
- Generating performance reports
- Creating visualizations

**Usage:**
```bash
# Basic usage (S&P 500 index)
python examples/forecasting_comparison.py

# Custom symbol and parameters
python examples/forecasting_comparison.py \
    --ticker AAPL \
    --window-size 60 \
    --forecast-horizon 5 \
    --output-dir results/

# With expanding window
python examples/forecasting_comparison.py \
    --ticker BTC-USD \
    --expanding-window \
    --n-lags 10
```

---

## Theoretical Framework

### The Fractal Market Hypothesis

FracTime is built on the Fractal Market Hypothesis which challenges traditional market theories by recognizing that:

1. **Markets Have Memory**: Long-range dependence means past price movements influence future behavior, not just recent events
2. **Self-Similarity**: Price movements exhibit similar patterns across different time scales (minutes, hours, days, months)
3. **Power-Law Distributions**: Returns follow heavy-tailed distributions rather than normal distributions, making extreme events more common than classical models predict
4. **Scale-Invariant Volatility**: Volatility clustering occurs across all time scales in a self-similar manner

### Trading Time vs. Clock Time

Following Mandelbrot's insights, FracTime implements the concept that markets operate on their own internal time scale:

- **Non-Uniform Time**: Market time "speeds up" during high volatility/volume periods and "slows down" during quiet periods
- **Time Warping**: The relationship between clock time and trading time is non-linear and varies with market conditions
- **Statistical Tractability**: When prices are resampled to uniform trading time (based on volume or volatility), their statistical properties become more stable and predictable

### Multi-Scale Analysis

Our methodology analyzes financial time series through multiple components:

#### Fractal Pattern Recognition
- Uses Hurst exponent calculation to quantify long-term memory
- Applies normalized cross-correlation to find self-similar patterns
- Identifies recurring market structures across different time horizons

#### Scaling Analysis
- Quantifies how volatility scales with time: σ(t) ~ t^H where H is the Hurst exponent
- Measures self-similarity across scales using fractal dimension
- Tests for multi-fractal behavior in different market conditions

#### Regime Identification
- Uses Hidden Markov Models to identify distinct fractal regimes (bull, bear, sideways)
- Predicts transitions between regimes based on pattern evolution
- Calculates regime-specific statistics and forecast distributions

#### Distribution Modeling
- Forecasts entire probability distributions, not just point estimates
- Conditions forecasts on current fractal regime
- Captures fat-tails and asymmetries in return distributions

### Applications

This approach enables novel analytical capabilities:

- **Scale-Dependent Analysis**: Adapt strategies to different time horizons (intraday, daily, weekly)
- **Regime-Specific Forecasting**: Generate different predictions for trending vs mean-reverting markets
- **Trading Time Optimization**: Identify optimal times to enter/exit based on market activity
- **Improved Risk Models**: Better capture extreme events with power-law distributions
- **Early Warning Systems**: Detect regime changes before they fully materialize

By integrating these concepts, FracTime offers a more nuanced understanding of market dynamics than traditional forecasting methods, particularly during periods of high volatility and regime transitions.

### Mathematical Foundation

**Hurst Exponent (H):**
- H > 0.5: Persistent (trending) behavior - price increases likely to be followed by increases
- H = 0.5: Random walk (Brownian motion) - no memory
- H < 0.5: Anti-persistent (mean-reverting) behavior - price increases likely to be followed by decreases

**Fractal Dimension (D):**
- D = 2 - H for time series
- Higher D indicates more complex, jagged price movements
- Lower D indicates smoother, more persistent trends

**Fractional Brownian Motion (FBM):**
- Generalization of Brownian motion with memory parameter H
- Used to generate realistic price paths with long-range dependence
- Captures the fractal nature of financial markets

## Advanced Theoretical Concepts

### Multi-Dimensional Fractal Analysis

FracTime extends beyond one-dimensional time series analysis by incorporating:

- **Multi-dimensional fractal equations** that simulate market behavior across multiple factors simultaneously
- Cross-correlation analysis between different fractal dimensions
- Factor-based decomposition of market movements using fractal geometry
- Higher-dimensional attractors that capture complex market dynamics

### Quantum Finance Applications

Drawing from quantum mechanics principles applied to finance:

#### Quantum Price Levels (QPLs)

The framework implements Quantum Price Levels (QPLs) from Chapter 10 of Quantum Finance theory:

- **Quantum Finance Schrödinger Equation (QFSE)** is used to derive price levels that act as support and resistance
- QPLs represent probabilistic price barriers with quantum properties
- Market prices tend to stabilize around these quantum levels during specific regimes

#### Computational Methods

The implementation utilizes advanced numerical methods:

- **Finite difference methods (FDM)** to solve the QFSE for practical application
- **Path integral techniques** that connect quantum mechanics to path simulation
- Discretization approaches that balance computational efficiency with model accuracy

#### Practical Applications

These quantum finance concepts enhance practical market analysis by:

- Providing more robust support/resistance levels based on quantum probabilities
- Identifying price regions with higher probability of reversal or continuation
- Quantifying the energy required for price to break through quantum barriers
- Predicting regime changes using quantum tunneling analogies

By integrating these advanced theoretical concepts, FracTime provides a unified framework that connects classical fractal analysis with quantum finance principles for a more comprehensive understanding of market behavior.

---

## Frequently Asked Questions

### General Questions

**Q: Is FracTime suitable for live trading?**

A: FracTime is a research and analysis library. While it provides robust forecasting and simulation capabilities, always:
- Thoroughly backtest any strategy
- Start with paper trading
- Understand the risks involved
- Never risk more than you can afford to lose

**Q: What markets does FracTime support?**

A: FracTime works with any time series data. Built-in data sources include:
- Equities (stocks, ETFs, indices)
- Cryptocurrencies
- Forex
- Commodities
- Economic indicators

**Q: How accurate are the forecasts?**

A: Forecast accuracy varies by market conditions and time horizon. FracTime provides:
- Multiple forecasting methods to compare
- Comprehensive backtesting framework
- Probabilistic forecasts (distributions, not just point estimates)
- Tools to evaluate accuracy on historical data

Use the backtesting framework to evaluate accuracy for your specific use case.

### Technical Questions

**Q: Why is the Hurst exponent different from other implementations?**

A: There are multiple methods to calculate the Hurst exponent (R/S analysis, DFA, Wavelet, etc.). FracTime uses R/S analysis with optimized lag selection. Results may vary slightly from other implementations but should be in the same range (typically 0.3-0.7 for financial data).

**Q: How many data points do I need for reliable analysis?**

A: Recommended minimums:
- Hurst exponent: 100+ data points
- Fractal dimension: 50+ data points
- Regime detection: 200+ data points
- Backtesting: 500+ data points

More data generally leads to more stable estimates.

**Q: Can I use my own data instead of downloading from APIs?**

A: Yes! All FracTime functions accept NumPy arrays or Pandas/Polars DataFrames:

```python
import fractime as ft
import numpy as np

# Your custom data
my_prices = np.array([100, 102, 101, 105, 103, ...])

# Use with FracTime
analyzer = ft.FractalAnalyzer()
hurst = analyzer.calculate_hurst_exponent(my_prices)
```

**Q: How do I speed up backtesting?**

A: Several options:
1. Reduce `window_size` or increase `step_size`
2. Use fewer forecasting methods
3. Reduce `n_paths` for simulation-based methods
4. Use faster models (ARIMA instead of XGBoost)
5. Process multiple assets in parallel with multiprocessing

**Q: What's the difference between expanding and sliding windows?**

A:
- **Sliding window**: Fixed-size training window that moves forward (e.g., always use last 252 days)
- **Expanding window**: Growing training window that includes all historical data up to that point

Expanding windows use more data but may include outdated patterns. Sliding windows adapt faster to regime changes.

### Troubleshooting

**Q: I'm getting "insufficient data" errors**

A: Ensure your time series has enough data points (see "How many data points..." above). Also check for:
- NaN or missing values
- Constant values (zero variance)
- Data in correct format (NumPy array, Pandas Series, etc.)

**Q: Simulations are taking too long**

A: Reduce computational complexity:
```python
# Faster settings
paths = simulator.simulate_paths(
    initial_price=100,
    n_steps=20,      # Reduce from 30
    n_paths=500,     # Reduce from 1000
    dt=1/252
)
```

**Q: Forecast accuracy is poor**

A: Consider:
1. Try different forecasting methods (ensemble often works best)
2. Adjust feature engineering (more/fewer lags)
3. Check for regime changes (use regime-specific models)
4. Verify data quality and sufficient history
5. Some markets are harder to forecast than others

**Q: Installation issues with dependencies**

A: Try:
```bash
# Use uv for better dependency resolution
uv pip install -e .

# Or install dependencies individually
pip install numpy scipy pandas polars scikit-learn statsmodels plotly numba
```

---

## Performance Considerations

### Optimization Tips

1. **Use Numba-accelerated functions**: Core computations are JIT-compiled with Numba
2. **Vectorize operations**: Operate on entire arrays instead of loops
3. **Reduce simulation paths**: Start with 500-1000 paths for exploration
4. **Cache results**: Save expensive computations (Hurst exponents, regimes)
5. **Parallel processing**: Use multiprocessing for multiple assets

### Memory Usage

For large-scale simulations:
```python
# Memory-efficient approach
import numpy as np

# Instead of keeping all paths in memory
paths = simulator.simulate_paths(n_paths=10000, n_steps=100)  # Large memory usage

# Process in batches
batch_size = 1000
results = []
for _ in range(10):
    batch_paths = simulator.simulate_paths(n_paths=batch_size, n_steps=100)
    results.append(path_analyzer.calculate_statistics(batch_paths))
```

---

## Citation

If you use FracTime in academic research, please cite:

```bibtex
@software{fractime2024,
  title = {FracTime: Advanced Time Series Forecasting with Fractal Geometry},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/fractime},
  version = {0.1.0}
}
```

---

## License

FracTime is released under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Support & Contact

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/yourusername/fractime/issues)
- **Discussions**: Join the community on [GitHub Discussions](https://github.com/yourusername/fractime/discussions)
- **Documentation**: Full API docs at [docs site](https://fractime.readthedocs.io)

---

## Acknowledgments

FracTime is inspired by the pioneering work of:
- **Benoit Mandelbrot**: Fractal geometry and the fractal market hypothesis
- **Edgar Peters**: Fractal market analysis and chaos theory in finance
- **Hurst, Black, Scholes**: Foundational work in quantitative finance

Special thanks to the open-source community and contributors.

---

## Disclaimer

**IMPORTANT**: FracTime is provided for research and educational purposes only.

- Past performance does not guarantee future results
- All trading and investment decisions are your own responsibility
- The authors are not liable for any financial losses
- Always consult with financial professionals before making investment decisions
- This software comes with no warranty of any kind

Financial markets are inherently unpredictable. Use FracTime as one tool among many in your analysis toolkit.

---

**Built with Python | Powered by Fractal Geometry | Inspired by Mandelbrot**