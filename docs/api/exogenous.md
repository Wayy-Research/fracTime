# Exogenous Predictors API Reference

FracTime supports exogenous (external) predictors to condition forecasts on external variables like market indicators, economic data, or correlated time series.

## Quick Example

```python
import fractime as ft
import pandas as pd

# Prepare exogenous data
exogenous = pd.DataFrame({
    'VIX': vix_prices,
    'TLT': bond_prices,
    'GLD': gold_prices
})

# Fit with exogenous
forecaster = ft.FractalForecaster(use_exogenous=True)
forecaster.fit(prices, dates=dates, exogenous=exogenous)

# View analysis
print(forecaster.get_exogenous_summary())

# Predict (automatically uses exogenous)
result = forecaster.predict(n_steps=30)
```

## FractalForecaster Exogenous Parameters

When creating a `FractalForecaster`, you can enable exogenous support:

```python
forecaster = ft.FractalForecaster(
    use_exogenous=True,              # Enable exogenous predictors
    exog_max_lags=10,                # Maximum lags to search
    exog_min_correlation=0.1,        # Minimum |correlation| to include
    exog_adjustment_strength=0.3     # How strongly to adjust probabilities
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_exogenous` | bool | False | Enable exogenous predictor support |
| `exog_max_lags` | int | 10 | Maximum number of lags to search for each variable |
| `exog_min_correlation` | float | 0.1 | Minimum absolute correlation to include a variable |
| `exog_adjustment_strength` | float | 0.3 | How strongly exogenous state affects path probabilities (0-1) |

## fit() with Exogenous

```python
forecaster.fit(
    prices,              # Target time series
    dates=dates,         # Optional dates
    exogenous=exog_df    # Exogenous data (DataFrame, dict, or array)
)
```

The `exogenous` parameter accepts:

- **pandas DataFrame**: Column names become variable names
- **dict**: Keys become variable names, values are arrays
- **numpy array**: Variables named `exog_0`, `exog_1`, etc.

## get_exogenous_summary()

After fitting, get the analysis results:

```python
summary = forecaster.get_exogenous_summary()
print(summary)
```

Returns:
```python
{
    'fitted': True,
    'n_variables': 3,
    'n_included': 2,
    'variables': {
        'VIX': {
            'best_lag': 0,
            'correlation': -0.26,
            'included': True
        },
        'TLT': {
            'best_lag': 1,
            'correlation': 0.24,
            'included': True
        },
        'GLD': {
            'best_lag': 2,
            'correlation': 0.08,
            'included': False  # Below min_correlation
        }
    }
}
```

## ExogenousHandler

Low-level class for exogenous variable preprocessing.

```python
from fractime import ExogenousHandler

handler = ExogenousHandler(
    max_lags=10,              # Maximum lags to consider
    min_correlation=0.1,      # Minimum correlation threshold
    use_differences=True,     # Use returns/differences
    scale_features=True       # Standardize features
)

# Fit to data
handler.fit(target_prices, exogenous_data)

# Get feature matrix
X, y_aligned = handler.get_feature_matrix(target_prices)

# Get summary
print(handler.get_summary())
```

::: fractime.ExogenousHandler
    options:
      show_source: false

## ExogenousRegimeModifier

Adjusts path probabilities based on exogenous regime conditions.

```python
from fractime import ExogenousRegimeModifier

modifier = ExogenousRegimeModifier(
    n_regimes=3,           # Number of regimes to identify
    model_type='ridge'     # 'ridge', 'elasticnet', 'rf', 'gbm'
)

# Fit to returns and features
modifier.fit(target_returns, exog_features)

# Get return adjustment for current state
expected_return, regime = modifier.predict_return_adjustment(current_features)

# Adjust path probabilities
adjusted_probs = modifier.adjust_path_probabilities(
    paths,
    original_probs,
    current_features,
    adjustment_strength=0.5
)
```

::: fractime.ExogenousRegimeModifier
    options:
      show_source: false

## ExogenousForecastAdjuster

Incorporate external forecasts into path generation.

```python
from fractime import ExogenousForecastAdjuster

adjuster = ExogenousForecastAdjuster()

# Add external forecasts
adjuster.add_exog_forecast('VIX', vix_forecast, weight=1.0)
adjuster.add_exog_forecast('TLT', bond_forecast, weight=0.5)

# Adjust paths
adjusted_paths = adjuster.adjust_paths(
    paths,
    exog_handler,
    regime_modifier,
    adjustment_strength=0.3
)
```

::: fractime.ExogenousForecastAdjuster
    options:
      show_source: false

## compute_exogenous_fractal_coherence

Analyze fractal alignment between target and exogenous series.

```python
from fractime import compute_exogenous_fractal_coherence

coherence = compute_exogenous_fractal_coherence(
    target=spy_prices,
    exogenous=vix_prices,
    window_sizes=[21, 63, 126]  # Analysis windows
)

print(f"Overall coherence: {coherence['overall_coherence']:.3f}")
print(f"Hurst correlation: {coherence['hurst_correlation']:.3f}")
print(f"Volatility correlation: {coherence['volatility_correlation']:.3f}")
print(f"Scale coherence: {coherence['scale_coherence']}")
```

::: fractime.compute_exogenous_fractal_coherence
    options:
      show_source: false

## Complete Example

```python
import fractime as ft
import pandas as pd
import numpy as np

# Fetch data
spy = ft.get_yahoo_data('SPY')
vix = ft.get_yahoo_data('^VIX')
tlt = ft.get_yahoo_data('TLT')

# Align dates
common_dates = spy.index.intersection(vix.index).intersection(tlt.index)
spy = spy.loc[common_dates]
exogenous = pd.DataFrame({
    'VIX': vix.loc[common_dates].values,
    'TLT': tlt.loc[common_dates].values
}, index=common_dates)

# Split
train_end = len(spy) - 30
train_prices = spy.values[:train_end]
train_exog = exogenous.iloc[:train_end]
train_dates = spy.index[:train_end].values

# Fit with exogenous
forecaster = ft.FractalForecaster(
    use_exogenous=True,
    exog_adjustment_strength=0.4
)
forecaster.fit(train_prices, dates=train_dates, exogenous=train_exog)

# Analyze exogenous relationships
summary = forecaster.get_exogenous_summary()
print("\nExogenous Analysis:")
for name, info in summary['variables'].items():
    print(f"  {name}: lag={info['best_lag']}, corr={info['correlation']:.3f}")

# Analyze fractal coherence
coherence = ft.compute_exogenous_fractal_coherence(
    spy.values[:train_end],
    exogenous['VIX'].values[:train_end]
)
print(f"\nVIX Fractal Coherence: {coherence['overall_coherence']:.3f}")

# Forecast
result = forecaster.predict(n_steps=30)

# Visualize
fig = ft.plot_forecast(
    train_prices[-100:],
    result,
    dates=train_dates[-100:],
    title="Forecast with Exogenous Predictors (VIX, TLT)"
)
fig.show()
```

## Best Practices

1. **Choose relevant exogenous variables**: Select variables with economic rationale for predicting your target
2. **Check correlations**: Review `get_exogenous_summary()` to see which variables are included
3. **Tune adjustment strength**: Start with 0.3 and adjust based on backtesting results
4. **Align time series**: Ensure all exogenous data aligns with target dates
5. **Use returns for stationarity**: The handler uses log returns by default
6. **Consider lag structure**: Some relationships are lagged (e.g., economic indicators lead markets)
