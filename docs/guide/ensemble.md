# Ensemble Methods

Combining multiple models for robust predictions.

## Why Ensembles?

- Reduce variance by averaging predictions
- Capture different aspects of the data
- More robust to model misspecification

## Stacking Ensemble

Meta-learning approach that learns optimal model weights:

```python
from fractime import FractalForecaster, StackingForecaster
from fractime.baselines import ARIMAForecaster, ETSForecaster

# Create base models
models = [
    FractalForecaster(),
    ARIMAForecaster(),
    ETSForecaster()
]

# Create stacking ensemble
stacker = StackingForecaster(
    base_models=models,
    meta_learner='ridge',  # 'ridge', 'linear', or 'rf'
    n_splits=5
)

# Fit (fits base models + meta-learner)
stacker.fit(prices)

# Predict
result = stacker.predict(n_steps=30)

# Check learned weights
weights = stacker.get_model_weights()
print(f"Model contributions: {weights}")
```

### Meta-Learner Options

| Meta-Learner | Description |
|--------------|-------------|
| `'ridge'` | Ridge regression (default, handles multicollinearity) |
| `'linear'` | Linear regression |
| `'rf'` | Random forest (captures non-linear combinations) |

## Boosting Ensemble

Sequential error correction where each model focuses on previous mistakes:

```python
from fractime import BoostingForecaster
from fractime import FractalForecaster
from fractime.baselines import ARIMAForecaster, ETSForecaster

# Define model configurations
configs = [
    (FractalForecaster, {}),
    (ARIMAForecaster, {}),
    (ETSForecaster, {'trend': 'add'})
]

# Create boosting ensemble
booster = BoostingForecaster(
    base_model_configs=configs,
    n_estimators=5,
    learning_rate=0.1
)

# Fit
booster.fit(prices)

# Predict
result = booster.predict(n_steps=30)

# Model weights
weights = booster.get_model_weights()
```

### How Boosting Works

1. First model fits the original series
2. Calculate residuals (errors)
3. Second model fits the residuals
4. Repeat, each model correcting previous errors
5. Final prediction is weighted sum

## Choosing Between Methods

| Method | Best When |
|--------|-----------|
| **Stacking** | Models capture different patterns; want optimal linear combination |
| **Boosting** | Series has complex structure; want iterative refinement |

## Advanced: Custom Ensembles

```python
from fractime.selection import EnsembleForecaster

class MyEnsemble(EnsembleForecaster):
    def combine_predictions(self, predictions):
        # Custom combination logic
        ...
```
