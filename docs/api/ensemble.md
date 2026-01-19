# Ensemble API Reference

The `Ensemble` class combines multiple forecasters for robust predictions.

---

## Overview

```python
import fractime as ft

ensemble = ft.Ensemble(prices)
result = ensemble.predict(steps=30)
```

---

## Constructor

```python
ft.Ensemble(
    data,
    dates=None,
    models=None,
    strategy='weighted',
    meta_learner='ridge',
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array | required | Historical price series |
| `dates` | array | None | Corresponding dates |
| `models` | list | None | List of Forecaster instances |
| `strategy` | str | 'weighted' | Combination strategy |
| `meta_learner` | str | 'ridge' | For stacking strategy |

---

## Strategies

### average

Simple average of all model predictions.

```python
ensemble = ft.Ensemble(prices, strategy='average')
```

- Each model contributes equally
- Simple and robust
- May not be optimal

### weighted (Default)

Diversity-based weighting.

```python
ensemble = ft.Ensemble(prices, strategy='weighted')
```

- Models with diverse (uncorrelated) forecasts get higher weights
- Encourages ensemble diversity
- Good general-purpose choice

### stacking

Meta-learner combines predictions.

```python
ensemble = ft.Ensemble(
    prices,
    strategy='stacking',
    meta_learner='ridge',
)
```

**Meta-learner options:**

| Value | Description |
|-------|-------------|
| `'ridge'` | Ridge regression (handles multicollinearity) |
| `'linear'` | Linear regression |
| `'rf'` | Random forest (non-linear combination) |

### boosting

Sequential error correction.

```python
ensemble = ft.Ensemble(prices, strategy='boosting')
```

- Each model corrects previous errors
- Weights decrease for later models
- Good for complex patterns

---

## Methods

### predict()

Generate ensemble forecast.

```python
result = ensemble.predict(
    steps=30,       # Forecast horizon
    n_paths=1000,   # Paths per model
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | int | 30 | Number of steps |
| `n_paths` | int | 1000 | Paths per model |

**Returns:** `ForecastResult` object

**Note:** Total paths = `n_paths × n_models`

---

## Properties

### models

List of forecasters in the ensemble.

```python
for model in ensemble.models:
    print(f"Hurst: {model.hurst}")
```

### n_models

Number of models in the ensemble.

```python
print(f"Ensemble has {ensemble.n_models} models")
```

---

## Default Models

When `models=None`, the ensemble creates:

1. R/S method forecaster
2. DFA method forecaster
3. Weighted path forecaster

```python
ensemble = ft.Ensemble(prices)  # Creates 3 default models
```

---

## Custom Models

```python
models = [
    ft.Forecaster(prices, method='rs'),
    ft.Forecaster(prices, method='dfa'),
    ft.Forecaster(prices, time_warp=True),
    ft.Forecaster(prices, path_weights={
        'hurst': 0.7,
        'volatility': 0.2,
        'pattern': 0.1,
    }),
]

ensemble = ft.Ensemble(prices, models=models)
```

---

## Result Metadata

The `ForecastResult` includes ensemble-specific metadata:

```python
result = ensemble.predict(steps=30)

# Strategy used
print(result.metadata['strategy'])

# Number of models
print(result.metadata['n_models'])

# Model weights (for weighted strategy)
if 'model_weights' in result.metadata:
    print(result.metadata['model_weights'])
```

---

## Examples

### Basic Ensemble

```python
import fractime as ft
import numpy as np

prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

ensemble = ft.Ensemble(prices)
result = ensemble.predict(steps=30, n_paths=500)

print(f"Forecast: {result.forecast[-1]:.2f}")
print(f"95% CI: {result.ci(0.95)}")
```

### Custom Models

```python
models = [
    ft.Forecaster(prices, method='rs'),
    ft.Forecaster(prices, method='dfa'),
]

ensemble = ft.Ensemble(prices, models=models)
result = ensemble.predict(steps=30)
```

### Compare Strategies

```python
strategies = ['average', 'weighted', 'stacking', 'boosting']

for strategy in strategies:
    ensemble = ft.Ensemble(prices, strategy=strategy)
    result = ensemble.predict(steps=30, n_paths=300)
    print(f"{strategy}: {result.forecast[-1]:.2f}")
```

### Accessing Model Weights

```python
ensemble = ft.Ensemble(prices, strategy='weighted')
result = ensemble.predict(steps=30)

weights = result.metadata.get('model_weights', [])
for i, (model, weight) in enumerate(zip(ensemble.models, weights)):
    print(f"Model {i+1}: weight={weight:.3f}")
```

### Diverse Ensemble

```python
# Create intentionally diverse models
models = [
    # Different methods
    ft.Forecaster(prices, method='rs'),
    ft.Forecaster(prices, method='dfa'),

    # Different configurations
    ft.Forecaster(prices, time_warp=True),
    ft.Forecaster(prices, path_weights={'hurst': 0.8, 'volatility': 0.1, 'pattern': 0.1}),
]

ensemble = ft.Ensemble(
    prices,
    models=models,
    strategy='weighted',
)

result = ensemble.predict(steps=30, n_paths=500)
print(f"Total paths: {result.n_paths}")  # 2000 (500 × 4)
```

---

## Performance Tips

1. **Reduce paths per model** for ensembles:
   ```python
   result = ensemble.predict(n_paths=300)  # vs 1000 for single model
   ```

2. **Limit number of models** to 3-5 for efficiency

3. **Use parallel computation** when available (automatic)

---

## See Also

- [Forecaster](forecaster.md) - Single model forecasting
- [ForecastResult](results.md#forecastresult) - Result object
- [Ensemble Guide](../guide/ensemble.md) - Usage guide
