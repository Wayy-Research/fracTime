# Simulator API Reference

The `Simulator` class generates Monte Carlo paths using fractal-aware methods.

---

## Overview

```python
import fractime as ft

sim = ft.Simulator(prices)
paths = sim.generate(n_paths=1000, steps=30)
```

---

## Constructor

```python
ft.Simulator(
    data,
    dates=None,
    method='auto',
    time_warp=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array | required | Historical price series |
| `dates` | array | None | Corresponding dates |
| `method` | str | 'auto' | Simulation method |
| `time_warp` | bool | False | Use trading time |

---

## Methods

### generate()

Generate simulated paths.

```python
paths = sim.generate(
    n_paths=1000,   # Number of paths to generate
    steps=30,       # Length of each path
    method=None,    # Override default method
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_paths` | int | 1000 | Number of paths |
| `steps` | int | 30 | Steps per path |
| `method` | str | None | Override simulation method |

**Returns:** numpy array of shape `(n_paths, steps)`

---

## Simulation Methods

### Auto (Default)

Automatically selects the best method based on data characteristics:

```python
sim = ft.Simulator(prices, method='auto')
```

Selection logic:

- Long history (>500 points): Uses pattern matching
- Shorter history: Uses fBm

### Fractional Brownian Motion (fBm)

Generate paths using fBm with estimated Hurst exponent:

```python
sim = ft.Simulator(prices, method='fbm')
paths = sim.generate(n_paths=1000, steps=30)
```

Properties:

- Preserves long-term memory (Hurst exponent)
- Good for shorter histories
- Faster than pattern matching

### Pattern Matching

Match historical patterns and extend them:

```python
sim = ft.Simulator(prices, method='pattern')
paths = sim.generate(n_paths=1000, steps=30)
```

Properties:

- Uses actual historical patterns
- Better captures real market dynamics
- Requires longer history

### Bootstrap

Resample historical returns:

```python
sim = ft.Simulator(prices, method='bootstrap')
paths = sim.generate(n_paths=1000, steps=30)
```

Properties:

- Preserves return distribution exactly
- No memory structure (shuffled returns)
- Good for comparison/baseline

---

## Time Warping

Enable Mandelbrot's trading time concept:

```python
sim = ft.Simulator(prices, time_warp=True)
paths = sim.generate(n_paths=1000, steps=30)
```

Effects:

- Volatility-dependent time scaling
- More realistic volatility clustering
- Paths respect market microstructure

---

## Properties

### hurst

Estimated Hurst exponent used for simulation.

```python
print(sim.hurst)  # float
```

### volatility

Estimated volatility used for simulation.

```python
print(sim.volatility)  # float
```

---

## Examples

### Basic Path Generation

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

sim = ft.Simulator(prices)
paths = sim.generate(n_paths=1000, steps=30)

print(f"Paths shape: {paths.shape}")  # (1000, 30)
```

### Compare Methods

```python
sim = ft.Simulator(prices)

fbm_paths = sim.generate(n_paths=1000, steps=30, method='fbm')
pattern_paths = sim.generate(n_paths=1000, steps=30, method='pattern')
bootstrap_paths = sim.generate(n_paths=1000, steps=30, method='bootstrap')

# Compare final distributions
print(f"fBm final std: {fbm_paths[:, -1].std():.2f}")
print(f"Pattern final std: {pattern_paths[:, -1].std():.2f}")
print(f"Bootstrap final std: {bootstrap_paths[:, -1].std():.2f}")
```

### With Time Warping

```python
# Without time warping
sim_normal = ft.Simulator(prices, time_warp=False)
paths_normal = sim_normal.generate(n_paths=1000, steps=30)

# With time warping
sim_warped = ft.Simulator(prices, time_warp=True)
paths_warped = sim_warped.generate(n_paths=1000, steps=30)

# Compare - time-warped paths have more realistic volatility structure
```

### Custom Workflow

```python
import fractime as ft
import numpy as np

# Generate paths
sim = ft.Simulator(prices, method='fbm')
paths = sim.generate(n_paths=5000, steps=30)

# Compute your own statistics
final_values = paths[:, -1]
current_price = prices[-1]

# Probability of >10% gain
prob_big_gain = np.mean(final_values > current_price * 1.10)
print(f"P(>10% gain): {prob_big_gain:.1%}")

# Value at Risk
var_95 = np.percentile(final_values, 5)
print(f"95% VaR: {var_95:.2f}")

# Expected shortfall
es_95 = np.mean(final_values[final_values <= var_95])
print(f"95% ES: {es_95:.2f}")
```

---

## Method Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| General use | `'auto'` |
| Short history (<200 points) | `'fbm'` |
| Long history, want realism | `'pattern'` |
| Need exact return distribution | `'bootstrap'` |
| High volatility regime | Add `time_warp=True` |

---

## See Also

- [Forecaster](forecaster.md) - Higher-level forecasting
- [Ensemble](ensemble.md) - Combine multiple models
- [Core Concepts](../guide/concepts.md) - Fractional Brownian motion
