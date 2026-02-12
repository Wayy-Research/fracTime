# FracTime

**Fractal-based time series analysis and forecasting.**

FracTime uses fractal geometry to analyze and forecast time series data. It computes the Hurst exponent (via R/S, DFA, or wavelet methods), detects regime transitions with HMMs, estimates how long regimes will persist, and generates probabilistic forecasts through Monte Carlo simulation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Installation

```bash
pip install fractime
```

From source:

```bash
git clone https://github.com/wayy-research/fracTime.git
cd fracTime
pip install -e .
```

---

## Quick Start

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, 600))

# 1. Analyze fractal properties
analyzer = ft.Analyzer(prices, method='dfa')
print(f"Hurst: {analyzer.hurst}")          # hurst=0.62
print(f"Regime: {analyzer.regime}")        # 'trending'

# 2. Understand regime persistence
hra = ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
print(f"Hurst regime: {hra.current_regime}")
print(f"Expected remaining: {hra.expected_remaining:.0f} days")
print(f"Recommended horizon: {hra.recommended_horizon} days")

# 3. Forecast with regime-informed horizon
result = ft.Forecaster(prices).predict(steps=hra.recommended_horizon)
print(f"Forecast: {result.forecast[-1]:.2f}")

# 4. Visualize
ft.plot(result)
```

---

## Best Practices

### Choosing a Hurst Method

| Method | When to Use | Trade-off |
|--------|-------------|-----------|
| `'dfa'` | **Default choice.** Robust to trends and non-stationarity. Best for financial data. | Slower than R/S |
| `'rs'` | Quick exploratory analysis. Classic Mandelbrot approach. | Biased by trends |
| `'wavelet'` | Multi-scale analysis. Good when you need scale-specific Hurst. | Requires `PyWavelets` |

```python
# Recommended: DFA for production analysis
analyzer = ft.Analyzer(prices, method='dfa', window=63)

# Quick exploration
analyzer = ft.Analyzer(prices, method='rs')

# Multi-scale research
analyzer = ft.Analyzer(prices, method='wavelet')
```

### Recommended Workflow

```
1. Analyze         ft.Analyzer(prices, method='dfa')
                   -> Hurst, fractal dim, volatility, regime

2. Regime check    ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
                   -> How long will this regime last?
                   -> What forecast horizon should I use?

3. Forecast        ft.Forecaster(prices).predict(steps=horizon)
                   -> Probabilistic paths weighted by fractal similarity

4. Risk manage     ft.RegimeStrategy(bull_allocation=1.0, bear_allocation=0.3)
                   -> Position sizing based on HMM regime detection
```

### Window Size Guidelines

| Window | Period | Use Case |
|--------|--------|----------|
| 21 | ~1 month | Short-term trading signals |
| 63 | ~1 quarter | Default for `Analyzer` rolling analysis |
| 126 | ~6 months | Medium-term regime analysis |
| 252 | ~1 year | Default for `HurstRegimeAnalyzer`. Long-term regime persistence |

### Key Principle: Fractal Features for Risk, Not Prediction

Empirical testing shows fractal properties are most valuable for **risk management** (position sizing, horizon selection, regime awareness) rather than direct price prediction. Build strategies around regime detection and adaptive allocation, not Hurst-based directional bets.

---

## Core Concepts

### The Hurst Exponent

The Hurst exponent (H) measures long-term memory in a time series:

| Hurst Value | Behavior | Implication |
|-------------|----------|-------------|
| H < 0.45 | Mean-reverting | Moves tend to reverse. Contrarian strategies. |
| 0.45 <= H <= 0.55 | Random walk | No exploitable memory. |
| H > 0.55 | Trending | Moves tend to continue. Momentum strategies. |

```python
analyzer = ft.Analyzer(prices, method='dfa')
h = analyzer.hurst.value
ci = analyzer.hurst.ci(0.95)
print(f"H = {h:.3f}, 95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
```

### Fractal Dimension

Measures complexity/roughness of the series (1.0 = smooth, 2.0 = space-filling):

```python
analyzer = ft.Analyzer(prices)
print(f"Fractal dim: {analyzer.fractal_dim}")
```

### Market Regimes

FracTime offers two complementary regime detection approaches:

**Hurst-based regimes** (from `Analyzer` or `HurstRegimeAnalyzer`):
- `trending` (H > 0.55) / `random` (0.45-0.55) / `mean_reverting` (H < 0.45)
- Based on the memory structure of the process

**HMM-based regimes** (from `RegimeDetector`):
- `bull` / `bear` / `sideways`
- Based on return distributions

Use both together: HMM tells you *what* the market is doing, Hurst tells you *how persistent* that behavior is.

---

## API Reference

### Analyzer

Computes fractal properties with lazy evaluation and caching.

```python
analyzer = ft.Analyzer(
    prices,
    dates=None,           # Optional date array
    method='dfa',         # 'rs', 'dfa', or 'wavelet'
    window=63,            # Rolling window size
    n_samples=1000,       # Bootstrap samples for CIs
    min_scale=10,         # Min scale for fractal analysis
    max_scale=100,        # Max scale for fractal analysis
)
```

| Property | Type | Description |
|----------|------|-------------|
| `hurst` | `Metric` | Hurst exponent (0-1) |
| `fractal_dim` | `Metric` | Fractal dimension (1-2) |
| `volatility` | `Metric` | Annualized volatility |
| `regime` | `str` | `'trending'`, `'mean_reverting'`, or `'random'` |
| `regime_probabilities` | `dict` | Probability distribution over regimes |
| `result` | `AnalysisResult` | All metrics as a structured object |

Each `Metric` has three views:

```python
analyzer.hurst.value          # Point estimate: 0.67
analyzer.hurst.rolling        # Polars DataFrame with rolling values
analyzer.hurst.ci(0.95)       # Bootstrap CI: (0.61, 0.73)
analyzer.hurst.std            # Bootstrap standard error
float(analyzer.hurst)         # Use as float directly
```

```python
# Multi-dimensional analysis
analyzer = ft.Analyzer({'price': prices, 'volume': volumes})
analyzer['price'].hurst
analyzer.coherence.value      # Cross-dimensional fractal coherence
```

---

### HurstRegimeAnalyzer

Fits an HMM to the rolling Hurst series to detect regime transitions, estimate how long regimes persist, and recommend forecast horizons.

```python
hra = ft.HurstRegimeAnalyzer(
    prices,
    dates=None,                        # Optional date array
    method='dfa',                      # 'rs', 'dfa', or 'wavelet'
    window=252,                        # Rolling Hurst window
    n_regimes=3,                       # 2 or 3 HMM states
    random_state=42,
    trending_threshold=0.55,           # H above -> trending
    mean_reverting_threshold=0.45,     # H below -> mean_reverting
)
```

| Property | Type | Description |
|----------|------|-------------|
| `current_regime` | `str` | `'trending'`, `'random'`, or `'mean_reverting'` |
| `current_hurst` | `float` | Latest rolling H estimate |
| `regime_age` | `int` | Days in current regime |
| `regime_stability` | `float` | Confidence in current regime (0-1) |
| `expected_duration` | `float` | Expected total duration (HMM + empirical blend) |
| `expected_remaining` | `float` | Expected remaining days in regime |
| `recommended_horizon` | `int` | Suggested forecast horizon |
| `regime_history` | `DataFrame` | Full history: date, hurst, regime, age, probability |
| `change_points` | `list[int]` | Indices where regime changed |
| `transition_matrix` | `ndarray` | P[i,j] between Hurst regimes |

```python
hra = ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
print(hra.summary())

# Use recommended horizon for forecasting
horizon = hra.recommended_horizon
result = ft.Forecaster(prices).predict(steps=horizon)
```

---

### RegimeDetector

HMM-based market regime detection from returns (bull/bear/sideways).

```python
detector = ft.RegimeDetector(n_regimes=2, random_state=42)
detector.fit(returns)

# Current regime
regime, prob = detector.get_current_regime(returns)

# Transition probabilities
print(detector.transition_matrix)
print(detector.expected_duration())
print(detector.summary())
```

---

### RegimeStrategy

Regime-based position sizing for risk management.

```python
strategy = ft.RegimeStrategy(
    bull_allocation=1.0,
    bear_allocation=0.3,
    sideways_allocation=0.5,
    vol_scale=True,           # Scale by inverse volatility
    vol_target=0.15,
)

results = strategy.backtest(prices)
print(results.summary())
print(f"Sharpe: {results.sharpe:.2f}, Max DD: {results.max_drawdown:.1%}")

# Quick one-liner
results = ft.quick_backtest(prices)
```

---

### Forecaster

Probabilistic forecasting via fractal-weighted Monte Carlo.

```python
model = ft.Forecaster(
    prices,
    dates=None,
    exogenous=None,              # {'VIX': vix_prices, ...}
    analyzer=None,               # Pre-computed Analyzer (for reuse)
    lookback=252,
    method='rs',
    time_warp=False,             # Mandelbrot trading time
    path_weights={               # How to weight paths
        'hurst': 0.3,
        'volatility': 0.3,
        'pattern': 0.4,
    },
)

result = model.predict(steps=30, n_paths=1000, confidence=0.95)

result.forecast       # Median forecast
result.mean           # Mean forecast
result.ci(0.90)       # 90% CI: (lower_array, upper_array)
result.paths          # All paths: (n_paths, n_steps)
result.to_frame()     # Export to Polars DataFrame
```

---

### Simulator

Direct Monte Carlo path generation.

```python
sim = ft.Simulator(prices, time_warp=False)

paths = sim.generate(n_paths=1000, steps=30, method='auto')
# Methods: 'auto', 'fbm', 'pattern', 'bootstrap'
```

| Method | Best For |
|--------|----------|
| `auto` | General use (selects based on data) |
| `fbm` | Short histories, strong fractal properties |
| `pattern` | Long histories with repeating patterns |
| `bootstrap` | Preserving exact historical distribution |

---

### Ensemble

Combine multiple forecasters.

```python
ensemble = ft.Ensemble(
    prices,
    models=[
        ft.Forecaster(prices, method='rs'),
        ft.Forecaster(prices, method='dfa'),
        ft.Forecaster(prices, time_warp=True),
    ],
    strategy='weighted',   # 'average', 'weighted', 'stacking', 'boosting'
)

result = ensemble.predict(steps=30, n_paths=500)
```

---

### Advanced Analytics

Specialized fractal functions for research:

```python
import fractime as ft
import numpy as np

# Wavelet Hurst estimation
h = ft.compute_wavelet_hurst(prices, wavelet='db4')
h_rolling = ft.compute_rolling_wavelet_hurst(prices, window=252)

# Multifractal Asymmetric Detrended Cross-Correlation (MF-ADCCA)
rho_q, h_xy = ft.compute_mf_adcca(prices_x, prices_y)

# Quantum Price Levels (support/resistance from Schrodinger equation)
levels = ft.compute_qpl(prices, n_levels=5)

# Fractal coherence across dimensions
coherence = ft.compute_fractal_coherence(prices, volume, method='dfa')

# Asymmetric Fractal Trend Dimension (up vs down complexity)
d_plus, d_minus = ft.compute_ftd(prices, window=63)

# DTW-corrected beta (handles lead/lag between assets)
beta = ft.compute_dtw_beta(stock_returns, market_returns)
```

---

## Examples

### Full Analysis Pipeline

```python
import fractime as ft
import numpy as np

prices = np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, 600)) * 100

# Step 1: Fractal analysis
analyzer = ft.Analyzer(prices, method='dfa')
print(analyzer.summary())

# Step 2: Regime persistence
hra = ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
print(f"Regime: {hra.current_regime} (age: {hra.regime_age}d)")
print(f"Stability: {hra.regime_stability:.2f}")
print(f"Remaining: ~{hra.expected_remaining:.0f}d")

# Step 3: Forecast with informed horizon
horizon = hra.recommended_horizon
model = ft.Forecaster(prices, method='dfa')
result = model.predict(steps=horizon)

# Step 4: Examine results
ci = result.ci(0.95)
print(f"{horizon}-day forecast: {result.forecast[-1]:.2f}")
print(f"95% CI: ({ci[0][-1]:.2f}, {ci[1][-1]:.2f})")
```

### Regime-Based Risk Management

```python
import fractime as ft
import numpy as np

prices = np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, 600)) * 100

# HMM regime detection for position sizing
strategy = ft.RegimeStrategy(
    bull_allocation=1.0,
    bear_allocation=0.3,
    vol_scale=True,
    vol_target=0.15,
)
results = strategy.backtest(prices)
print(results.summary())

# Combine with Hurst regime for horizon awareness
hra = ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
if hra.regime_stability > 0.6:
    print(f"Stable {hra.current_regime} regime, hold position")
else:
    print(f"Unstable regime, reduce exposure")
```

### Ensemble Forecast with Method Diversity

```python
import fractime as ft

ensemble = ft.Ensemble(
    prices,
    models=[
        ft.Forecaster(prices, method='rs'),
        ft.Forecaster(prices, method='dfa'),
        ft.Forecaster(prices, time_warp=True),
    ],
    strategy='weighted',
)

result = ensemble.predict(steps=30, n_paths=500)
ci = result.ci(0.95)
print(f"Forecast: {result.forecast[-1]:.2f}")
print(f"95% CI: ({ci[0][-1]:.2f}, {ci[1][-1]:.2f})")
```

### Multi-Asset Fractal Screening

```python
import fractime as ft

assets = {'SPY': spy_prices, 'TLT': bond_prices, 'GLD': gold_prices}

for name, prices in assets.items():
    analyzer = ft.Analyzer(prices, method='dfa')
    hra = ft.HurstRegimeAnalyzer(prices, method='dfa', window=252)
    print(
        f"{name}: H={analyzer.hurst.value:.2f} "
        f"regime={hra.current_regime} "
        f"remaining={hra.expected_remaining:.0f}d "
        f"horizon={hra.recommended_horizon}d"
    )
```

### Rolling Regime Detection with Polars

```python
import fractime as ft
import polars as pl

analyzer = ft.Analyzer(prices, dates=dates, method='dfa', window=63)
rolling = analyzer.hurst.rolling

# Classify each period
classified = rolling.with_columns(
    pl.when(pl.col('value') > 0.55)
      .then(pl.lit('trending'))
      .when(pl.col('value') < 0.45)
      .then(pl.lit('mean_reverting'))
      .otherwise(pl.lit('random'))
      .alias('regime')
)

print(classified.group_by('regime').count())
```

### Forecast Comparison

```python
import fractime as ft
import numpy as np

train, test = prices[:-30], prices[-30:]

# Compare methods
results = {}
for label, kwargs in [
    ('RS', {'method': 'rs'}),
    ('DFA', {'method': 'dfa'}),
    ('Time Warp', {'method': 'rs', 'time_warp': True}),
]:
    model = ft.Forecaster(train, **kwargs)
    pred = model.predict(steps=30)
    rmse = np.sqrt(np.mean((test - pred.forecast) ** 2))
    results[label] = rmse
    print(f"{label}: RMSE = {rmse:.4f}")
```

---

## API Summary

```python
import fractime as ft

# Classes
ft.Analyzer                 # Fractal analysis (Hurst, fractal dim, volatility)
ft.Forecaster               # Probabilistic Monte Carlo forecasting
ft.Simulator                # Direct path generation
ft.Ensemble                 # Multi-model combination
ft.RegimeDetector           # HMM regime detection (bull/bear/sideways)
ft.RegimeStrategy           # Regime-based position sizing
ft.HurstRegimeAnalyzer      # Hurst regime persistence & horizon estimation

# Convenience functions
ft.analyze(prices)          # Quick analysis
ft.forecast(prices)         # Quick forecast
ft.plot(obj)                # Plot any FracTime object
ft.quick_backtest(prices)   # Quick regime strategy backtest

# Advanced analytics
ft.compute_wavelet_hurst(prices)
ft.compute_rolling_wavelet_hurst(prices)
ft.compute_mf_adcca(x, y)
ft.compute_qpl(prices, n_levels=5)
ft.compute_fractal_coherence(price, volume)
ft.compute_ftd(prices)
ft.compute_dtw_alignment(x, y)
ft.compute_dtw_beta(stock_returns, market_returns)

# Result types
ft.Metric                   # Single metric with point/rolling/distribution views
ft.AnalysisResult           # Complete analysis result
ft.ForecastResult           # Forecast with paths and CIs
ft.BacktestResult           # Strategy backtest metrics
```

---

## Performance

FracTime is optimized for speed:

- **Numba JIT** for Hurst, DFA, fractal dimension, fBm generation
- **Lazy computation** with caching (only compute what you access)
- **Polars** for fast DataFrame operations

| Operation | Time (600 data points) |
|-----------|------------------------|
| Hurst (point, DFA) | ~5ms |
| Hurst (rolling, 252 window) | ~100ms |
| Hurst (bootstrap, 1000 samples) | ~500ms |
| HurstRegimeAnalyzer | ~200ms |
| Forecast (1000 paths, 30 steps) | ~200ms |

Tips:
1. **Reuse analyzers** across forecasters: `ft.Forecaster(prices, analyzer=analyzer)`
2. **Use `method='fbm'`** for path generation with short histories
3. **Reduce `n_samples`** if bootstrap precision isn't critical

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{fractime,
  title = {FracTime: Fractal-based Time Series Analysis and Forecasting},
  author = {Wayy Research},
  year = {2024},
  url = {https://github.com/wayy-research/fracTime}
}
```
