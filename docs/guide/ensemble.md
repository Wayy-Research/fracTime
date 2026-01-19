# Ensemble Methods

Combining multiple forecasters for robust predictions.

---

## Why Ensembles?

Single models can be brittle. Ensembles provide:

- **Reduced variance** by averaging predictions
- **Better generalization** by capturing different patterns
- **Robustness** to model misspecification

---

## Basic Usage

```python
import fractime as ft

# Create ensemble with default models
ensemble = ft.Ensemble(prices)

# Generate forecast
result = ensemble.predict(steps=30, n_paths=500)

# Access results (same as Forecaster)
print(f"Forecast: {result.forecast[-1]:.2f}")
print(f"95% CI: {result.ci(0.95)}")
```

---

## Default Ensemble

When you create an Ensemble without specifying models, it creates:

1. **R/S Forecaster** - Rescaled Range method
2. **DFA Forecaster** - Detrended Fluctuation Analysis
3. **Weighted Forecaster** - Custom path weights

```python
ensemble = ft.Ensemble(prices)
print(f"Number of models: {ensemble.n_models}")  # 3
```

---

## Custom Models

Specify your own combination of forecasters:

```python
models = [
    ft.Forecaster(prices, method='rs'),
    ft.Forecaster(prices, method='dfa'),
    ft.Forecaster(prices, time_warp=True),
]

ensemble = ft.Ensemble(prices, models=models)
result = ensemble.predict(steps=30)
```

---

## Combination Strategies

### Average

Simple average of all models:

```python
ensemble = ft.Ensemble(prices, strategy='average')
```

Each model contributes equally to the final forecast.

### Weighted (Default)

Models weighted by forecast diversity:

```python
ensemble = ft.Ensemble(prices, strategy='weighted')
result = ensemble.predict(steps=30)

# See model weights
print(result.metadata['model_weights'])
```

Models with more diverse (less correlated) forecasts get higher weights.

### Stacking

Meta-learner combines model predictions:

```python
ensemble = ft.Ensemble(
    prices,
    strategy='stacking',
    meta_learner='ridge'  # or 'linear', 'rf'
)
```

| Meta-Learner | Description |
|--------------|-------------|
| `'ridge'` | Ridge regression (default) |
| `'linear'` | Linear regression |
| `'rf'` | Random forest |

### Boosting

Sequential error correction:

```python
ensemble = ft.Ensemble(prices, strategy='boosting')
```

Each model corrects the residuals of previous models.

---

## Ensemble Parameters

```python
ensemble = ft.Ensemble(
    data,                  # Price series (required)
    dates=None,            # Optional date array
    models=None,           # List of Forecaster instances
    strategy='weighted',   # Combination strategy
    meta_learner='ridge',  # For stacking strategy
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | array | required | Historical price series |
| `dates` | array | None | Corresponding dates |
| `models` | list | None | Custom forecasters |
| `strategy` | str | 'weighted' | How to combine models |
| `meta_learner` | str | 'ridge' | For stacking only |

---

## Accessing Ensemble State

```python
# List of models
models = ensemble.models

# Number of models
n = ensemble.n_models

# Individual model forecasts
for model in ensemble.models:
    result = model.predict(steps=30)
    print(f"Hurst: {model.hurst}")
```

---

## Choosing a Strategy

| Strategy | Best When |
|----------|-----------|
| **average** | Models are equally good, want simple combination |
| **weighted** | Want diversity-based weighting, general use |
| **stacking** | Have validation data, want learned combination |
| **boosting** | Series has complex structure, want iterative refinement |

---

## Example: Comparing Strategies

```python
import fractime as ft
import numpy as np

# Your data
prices = ...

# Try different strategies
strategies = ['average', 'weighted', 'stacking', 'boosting']

results = {}
for strategy in strategies:
    ensemble = ft.Ensemble(prices, strategy=strategy)
    result = ensemble.predict(steps=30, n_paths=500)
    results[strategy] = result.forecast

# Compare final forecasts
for name, forecast in results.items():
    print(f"{name}: {forecast[-1]:.2f}")
```

---

## Example: Diverse Model Ensemble

Create an ensemble with deliberately diverse models:

```python
models = [
    # Different Hurst methods
    ft.Forecaster(prices, method='rs'),
    ft.Forecaster(prices, method='dfa'),

    # With time warping
    ft.Forecaster(prices, time_warp=True),

    # Different path weights
    ft.Forecaster(prices, path_weights={
        'hurst': 0.7,
        'volatility': 0.2,
        'pattern': 0.1,
    }),
]

ensemble = ft.Ensemble(prices, models=models, strategy='weighted')
result = ensemble.predict(steps=30, n_paths=500)

print(f"Model weights: {result.metadata['model_weights']}")
```

---

## Performance Considerations

### Computation Time

Ensemble forecasting takes longer because each model generates paths:

```
Total time ≈ n_models × single_model_time
```

### Reducing Paths

Use fewer paths per model when using ensembles:

```python
# Instead of 1000 paths for single model
# Use 300-500 paths per model in ensemble
result = ensemble.predict(steps=30, n_paths=300)
```

---

## Next Steps

- [API Reference: Ensemble](../api/ensemble.md) - Complete documentation
- [Forecaster Guide](forecasting.md) - Single model forecasting
- [Examples](../examples/comparison.md) - Real-world comparisons
