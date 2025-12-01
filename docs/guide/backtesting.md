# Backtesting Guide

Rigorous model validation with walk-forward testing.

## Walk-Forward Validation

Walk-forward validation simulates how a model would have performed in real-time:

1. Train on historical data up to time T
2. Forecast N steps ahead
3. Move forward, expand training window
4. Repeat

```python
from fractime.backtesting import WalkForwardValidator
from fractime import FractalForecaster

validator = WalkForwardValidator(
    model=FractalForecaster(),
    initial_window=252,    # Initial training size
    step_size=20,          # Steps between refits
    forecast_horizon=10    # Forecast horizon
)

results = validator.run(prices, dates)
```

### Results Structure

```python
results = {
    'metrics': {
        'rmse': float,
        'mae': float,
        'mape': float,
        'mse': float,
        'direction_accuracy': float,
        'coverage': float,
    },
    'forecasts': np.array,
    'actuals': np.array,
    'parameter_history': list,
}
```

## Metrics

### ForecastMetrics

Compute comprehensive metrics:

```python
from fractime.backtesting import ForecastMetrics

metrics = ForecastMetrics.compute_all(
    forecasts=predictions,
    actuals=actual_values,
    current_prices=current,
    lower=lower_bounds,
    upper=upper_bounds
)
```

### Individual Metrics

```python
from fractime.backtesting import (
    compute_rmse,
    compute_mae,
    compute_mape,
    compute_direction_accuracy,
    compute_coverage,
    compute_crps
)

rmse = compute_rmse(forecasts, actuals)
direction = compute_direction_accuracy(forecasts, actuals, current)
```

| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| MAPE | Mean absolute percentage error |
| Direction Accuracy | % correct up/down predictions |
| Coverage | % actuals within confidence interval |
| CRPS | Continuous ranked probability score |

## Model Comparison

Compare multiple models on the same data:

```python
from fractime.backtesting import compare_models
from fractime import FractalForecaster
from fractime.baselines import ARIMAForecaster, ETSForecaster

comparison = compare_models(
    models={
        'Fractal': FractalForecaster(),
        'ARIMA': ARIMAForecaster(),
        'ETS': ETSForecaster()
    },
    prices=prices,
    dates=dates,
    initial_window=252,
    step_size=20,
    forecast_horizon=10
)

# Print results
for name, metrics in comparison.items():
    print(f"{name}:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Direction: {metrics['direction_accuracy']:.2%}")
```

## Dual Penalty Scoring

Balance accuracy vs overfitting:

```python
from fractime.backtesting import DualPenaltyScorer

scorer = DualPenaltyScorer()
score = scorer.score(in_sample_error, out_sample_error)
```
