# Model Selection Guide

Automatic model selection and statistical testing.

## AutoSelector

Automatically select the best model for your data:

```python
from fractime.selection import AutoSelector

selector = AutoSelector()
result = selector.select_best(prices, dates)

print(f"Best model: {result.model_name}")
print(f"Score: {result.score:.4f}")

# Use the selected model
forecaster = result.model
result = forecaster.predict(n_steps=30)
```

## Model Registry

Catalog and manage available models:

```python
from fractime.selection import ModelRegistry, get_global_registry

# Get global registry
registry = get_global_registry()

# List available models
models = registry.list_models()

# Get a model by name
model_class = registry.get('FractalForecaster')
```

### Register Custom Models

```python
from fractime.selection import register_model

@register_model('MyCustomModel')
class MyCustomModel:
    def fit(self, prices, dates=None):
        ...
    def predict(self, n_steps):
        ...
```

## Statistical Tests

### Diebold-Mariano Test

Test if one model significantly outperforms another:

```python
from fractime.selection import diebold_mariano_test

dm_stat, p_value = diebold_mariano_test(
    errors_model1,
    errors_model2,
    h=1  # forecast horizon
)

if p_value < 0.05:
    print("Models are significantly different")
```

### Model Confidence Set

Find the set of models that cannot be statistically distinguished:

```python
from fractime.selection import model_confidence_set

model_errors = {
    'Fractal': fractal_errors,
    'ARIMA': arima_errors,
    'ETS': ets_errors,
}

mcs = model_confidence_set(model_errors, alpha=0.1)
print(f"Models in confidence set: {mcs}")
```

## Ensemble Creation

Combine multiple models:

```python
from fractime.selection import create_ensemble, WeightedEnsemble

# Equal-weighted ensemble
ensemble = create_ensemble([model1, model2, model3])

# Custom weights
ensemble = WeightedEnsemble(
    models=[model1, model2, model3],
    weights=[0.5, 0.3, 0.2]
)

ensemble.fit(prices)
result = ensemble.predict(n_steps=30)
```
