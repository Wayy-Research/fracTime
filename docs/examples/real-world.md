# Advanced Usage Examples

Real-world applications and advanced features.

---

## Rolling Regime Detection

Track how market regimes change over time.

```python
import fractime as ft
import polars as pl
import numpy as np
import datetime

# Sample data
np.random.seed(42)
n = 500
prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i)
         for i in range(n)]

# Analyze with rolling window
analyzer = ft.Analyzer(prices, dates=dates, window=63)

# Get rolling Hurst
rolling = analyzer.hurst.rolling

# Classify each period
result = rolling.with_columns(
    pl.when(pl.col('value') > 0.55)
      .then(pl.lit('trending'))
      .when(pl.col('value') < 0.45)
      .then(pl.lit('mean_reverting'))
      .otherwise(pl.lit('random'))
      .alias('regime')
)

# Count regimes
print("Regime Distribution:")
print(result.group_by('regime').count())

# Find regime transitions
result = result.with_columns(
    pl.col('regime').shift(1).alias('prev_regime')
)
transitions = result.filter(pl.col('regime') != pl.col('prev_regime'))
print(f"\nRegime transitions: {len(transitions)}")
```

---

## Multi-Asset Analysis

Analyze relationships between multiple assets.

```python
import fractime as ft
import numpy as np

# Generate correlated assets
np.random.seed(42)
n = 500

# Common factor
factor = np.random.randn(n).cumsum()

# Assets with different exposures
stock_a = 100 + factor * 2 + np.random.randn(n).cumsum() * 0.5
stock_b = 100 + factor * 1.5 + np.random.randn(n).cumsum() * 0.8
volume = np.random.randint(1000, 10000, n).astype(float)

# Multi-dimensional analysis
analyzer = ft.Analyzer({
    'stock_a': stock_a,
    'stock_b': stock_b,
    'volume': volume,
})

# Compare Hurst exponents
print("Hurst Exponents:")
for dim in analyzer.dimensions:
    h = analyzer[dim].hurst.value
    regime = analyzer[dim].regime
    print(f"  {dim}: H={h:.4f} ({regime})")

# Cross-dimensional coherence
print(f"\nCoherence: {analyzer.coherence.value:.4f}")
```

---

## Conditional Forecasting

Forecast based on current regime.

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

# Analyze current state
analyzer = ft.Analyzer(prices)
regime = analyzer.regime

print(f"Current regime: {regime}")

# Adjust strategy based on regime
if regime == 'trending':
    # Use default (momentum-aware)
    model = ft.Forecaster(prices)
elif regime == 'mean_reverting':
    # Emphasize volatility matching
    model = ft.Forecaster(prices, path_weights={
        'hurst': 0.3,
        'volatility': 0.5,
        'pattern': 0.2,
    })
else:
    # Use ensemble for uncertain regime
    model = ft.Ensemble(prices)

result = model.predict(steps=30)
print(f"30-day forecast: {result.forecast[-1]:.2f}")
```

---

## Risk Metrics

Calculate comprehensive risk metrics from forecasts.

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)
current = prices[-1]

# Generate many paths for robust statistics
model = ft.Forecaster(prices)
result = model.predict(steps=30, n_paths=5000)

final_values = result.paths[:, -1]
returns = (final_values / current - 1) * 100

print("=" * 50)
print("Risk Metrics (30-day horizon)")
print("=" * 50)

# Basic statistics
print(f"Expected return: {np.mean(returns):.2f}%")
print(f"Volatility (std): {np.std(returns):.2f}%")
print(f"Skewness: {((returns - returns.mean())**3).mean() / returns.std()**3:.2f}")

# Value at Risk
print(f"\nValue at Risk:")
print(f"  VaR 95%: {np.percentile(returns, 5):.2f}%")
print(f"  VaR 99%: {np.percentile(returns, 1):.2f}%")

# Expected Shortfall (CVaR)
var_95 = np.percentile(returns, 5)
es_95 = np.mean(returns[returns <= var_95])
print(f"  ES 95%: {es_95:.2f}%")

# Probability of extreme moves
print(f"\nExtreme move probabilities:")
print(f"  P(loss > 10%): {np.mean(returns < -10):.1%}")
print(f"  P(loss > 20%): {np.mean(returns < -20):.1%}")
print(f"  P(gain > 10%): {np.mean(returns > 10):.1%}")
print(f"  P(gain > 20%): {np.mean(returns > 20):.1%}")
```

---

## Forecast Horizon Sensitivity

Analyze how forecast uncertainty grows with horizon.

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)
model = ft.Forecaster(prices)

horizons = [5, 10, 20, 30, 60]

print("Forecast Horizon Sensitivity")
print("=" * 60)
print(f"{'Horizon':>8} {'Forecast':>12} {'Std':>10} {'95% CI Width':>15}")
print("-" * 60)

for h in horizons:
    result = model.predict(steps=h, n_paths=1000)
    lower, upper = result.ci(0.95)
    ci_width = upper[-1] - lower[-1]

    print(f"{h:>8} {result.forecast[-1]:>12.2f} {result.std[-1]:>10.2f} {ci_width:>15.2f}")
```

---

## Exogenous Variables

Incorporate external predictors.

```python
import fractime as ft
import numpy as np

# Generate correlated data
np.random.seed(42)
n = 500

# VIX-like volatility index (leading indicator)
vix = 20 + np.random.randn(n).cumsum() * 0.5
vix = np.maximum(vix, 10)  # Floor at 10

# Prices inversely related to VIX
prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02 - (vix - 20) * 0.001)

# Forecast with exogenous
model = ft.Forecaster(
    prices,
    exogenous={'VIX': vix}
)

result = model.predict(steps=30)
print(f"Forecast with VIX: {result.forecast[-1]:.2f}")

# Compare to without
model_baseline = ft.Forecaster(prices)
result_baseline = model_baseline.predict(steps=30)
print(f"Forecast without VIX: {result_baseline.forecast[-1]:.2f}")
```

---

## Time Warping Analysis

Compare forecasts with and without time warping.

```python
import fractime as ft
import numpy as np

# Generate volatile data
np.random.seed(42)
n = 500

# Variable volatility
vol = np.where(np.arange(n) % 100 < 50, 0.01, 0.03)
returns = np.random.randn(n) * vol
prices = 100 * np.cumprod(1 + returns)

# Without time warp
model_normal = ft.Forecaster(prices, time_warp=False)
result_normal = model_normal.predict(steps=30, n_paths=1000)

# With time warp
model_warped = ft.Forecaster(prices, time_warp=True)
result_warped = model_warped.predict(steps=30, n_paths=1000)

print("Time Warp Comparison")
print("=" * 50)
print(f"{'':15} {'Normal':>15} {'Time Warped':>15}")
print("-" * 50)
print(f"{'Forecast':15} {result_normal.forecast[-1]:>15.2f} {result_warped.forecast[-1]:>15.2f}")
print(f"{'Std Dev':15} {result_normal.std[-1]:>15.2f} {result_warped.std[-1]:>15.2f}")

# CI width
_, upper_n = result_normal.ci(0.95)
lower_n, _ = result_normal.ci(0.95)
_, upper_w = result_warped.ci(0.95)
lower_w, _ = result_warped.ci(0.95)

print(f"{'95% CI Width':15} {upper_n[-1]-lower_n[-1]:>15.2f} {upper_w[-1]-lower_w[-1]:>15.2f}")
```

---

## Bootstrap Confidence Interval Stability

Check how CI estimates stabilize with more samples.

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

sample_sizes = [100, 500, 1000, 2000, 5000]

print("CI Stability vs Bootstrap Samples")
print("=" * 55)
print(f"{'Samples':>8} {'Hurst':>10} {'CI Low':>10} {'CI High':>10} {'Width':>10}")
print("-" * 55)

for n_samples in sample_sizes:
    analyzer = ft.Analyzer(prices, n_samples=n_samples)
    ci = analyzer.hurst.ci(0.95)

    print(f"{n_samples:>8} {analyzer.hurst.value:>10.4f} {ci[0]:>10.4f} {ci[1]:>10.4f} {ci[1]-ci[0]:>10.4f}")
```

---

## Production Workflow

Complete production-ready workflow.

```python
import fractime as ft
import numpy as np
import datetime
import json

def analyze_and_forecast(prices, dates=None, config=None):
    """
    Production workflow for fractal analysis and forecasting.

    Args:
        prices: Historical price array
        dates: Optional date array
        config: Optional configuration dict

    Returns:
        dict with analysis and forecast results
    """
    config = config or {}

    # 1. Analyze
    analyzer = ft.Analyzer(
        prices,
        dates=dates,
        method=config.get('method', 'rs'),
        n_samples=config.get('n_samples', 1000),
    )

    # 2. Forecast
    if config.get('use_ensemble', False):
        model = ft.Ensemble(prices, dates=dates)
    else:
        model = ft.Forecaster(
            prices,
            dates=dates,
            method=config.get('method', 'rs'),
            time_warp=config.get('time_warp', False),
        )

    result = model.predict(
        steps=config.get('horizon', 30),
        n_paths=config.get('n_paths', 1000),
    )

    # 3. Compute metrics
    lower, upper = result.ci(0.95)
    current = prices[-1]

    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'analysis': {
            'hurst': float(analyzer.hurst.value),
            'hurst_ci': [float(x) for x in analyzer.hurst.ci(0.95)],
            'fractal_dim': float(analyzer.fractal_dim.value),
            'volatility': float(analyzer.volatility.value),
            'regime': analyzer.regime,
            'regime_probs': {k: float(v) for k, v in analyzer.regime_probabilities.items()},
        },
        'forecast': {
            'horizon': config.get('horizon', 30),
            'current_price': float(current),
            'forecast_price': float(result.forecast[-1]),
            'expected_return': float((result.forecast[-1] / current - 1) * 100),
            'ci_95_lower': float(lower[-1]),
            'ci_95_upper': float(upper[-1]),
            'prob_increase': float(np.mean(result.paths[:, -1] > current)),
        },
        'config': config,
    }

    return output

# Example usage
np.random.seed(42)
prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

result = analyze_and_forecast(
    prices,
    config={
        'method': 'rs',
        'horizon': 30,
        'n_paths': 2000,
        'use_ensemble': False,
        'time_warp': True,
    }
)

print(json.dumps(result, indent=2))
```

---

## Next Steps

- [API Reference](../api/analyzer.md) - Complete documentation
- [Core Concepts](../guide/concepts.md) - Theory background
- [Forecasting Guide](../guide/forecasting.md) - Detailed forecasting
