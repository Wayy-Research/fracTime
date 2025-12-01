# Real-World Data Example

Working with actual market data.

## Setup

```python
import fractime as ft
from fractime.baselines import ARIMAForecaster, ETSForecaster
from fractime.backtesting import compare_models
import numpy as np
```

## Fetch Market Data

```python
# S&P 500
spy_prices, spy_dates = ft.get_yahoo_data('SPY', period='5y')

# Tech stocks
aapl_prices, aapl_dates = ft.get_yahoo_data('AAPL', period='5y')
msft_prices, msft_dates = ft.get_yahoo_data('MSFT', period='5y')

print(f"SPY: {len(spy_prices)} days, {spy_dates[0]} to {spy_dates[-1]}")
```

## Analyze Market Regimes

```python
analyzer = ft.FractalAnalyzer()

# Compute Hurst for different periods
def rolling_hurst(prices, window=252):
    hursts = []
    for i in range(window, len(prices)):
        h = analyzer.compute_hurst(prices[i-window:i])
        hursts.append(h)
    return np.array(hursts)

spy_hurst = rolling_hurst(spy_prices)

print(f"Average Hurst: {np.mean(spy_hurst):.3f}")
print(f"Current Hurst: {spy_hurst[-1]:.3f}")
print(f"Min Hurst: {np.min(spy_hurst):.3f}")
print(f"Max Hurst: {np.max(spy_hurst):.3f}")
```

## Multi-Asset Analysis

```python
from fractime import CrossDimensionalAnalyzer

cda = CrossDimensionalAnalyzer()
cda.add_dimension('SPY', spy_prices[-252:])
cda.add_dimension('AAPL', aapl_prices[-252:])
cda.add_dimension('MSFT', msft_prices[-252:])

# Cross-correlation
corr = cda.compute_cross_correlation()
print("\nCross-Correlation Matrix:")
print(corr)

# Hurst exponents
hursts = cda.compute_hurst_exponents()
print("\nHurst Exponents:")
for name, h in hursts.items():
    regime = "trending" if h > 0.5 else "mean-reverting"
    print(f"  {name}: {h:.3f} ({regime})")
```

## Forecast Individual Assets

```python
# Forecast SPY
forecaster = ft.FractalForecaster()
forecaster.fit(spy_prices, dates=spy_dates)

# 30-day forecast
result = forecaster.predict(n_steps=30, n_paths=1000)

current = spy_prices[-1]
forecast_end = result['forecast'][-1]
pct_change = (forecast_end / current - 1) * 100

print(f"\nSPY 30-Day Forecast:")
print(f"  Current: ${current:.2f}")
print(f"  Forecast: ${forecast_end:.2f} ({pct_change:+.1f}%)")
print(f"  95% CI: [${result['lower'][-1]:.2f}, ${result['upper'][-1]:.2f}]")

# Probability of gain
probs = result['probabilities']
final_values = result['paths'][:, -1]
prob_gain = np.sum(probs[final_values > current])
print(f"  P(gain): {prob_gain:.1%}")
```

## Backtest and Compare

```python
# Compare models on recent data
comparison = compare_models(
    models={
        'Fractal': ft.FractalForecaster(),
        'ARIMA': ARIMAForecaster(),
        'ETS': ETSForecaster()
    },
    prices=spy_prices,
    dates=spy_dates,
    initial_window=504,    # 2 years
    step_size=20,
    forecast_horizon=5
)

print("\nBacktest Results (5-day forecast):")
print("-" * 50)
for name, metrics in comparison.items():
    print(f"{name}:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Direction: {metrics['direction_accuracy']:.1%}")
```

## Ensemble Approach

```python
from fractime import StackingForecaster

# Create ensemble
models = [
    ft.FractalForecaster().fit(spy_prices),
    ARIMAForecaster().fit(spy_prices),
    ETSForecaster().fit(spy_prices)
]

stacker = StackingForecaster(base_models=models, meta_learner='ridge')
stacker.fit(spy_prices)

# Ensemble forecast
ensemble_result = stacker.predict(n_steps=30)

print(f"\nEnsemble Forecast: ${ensemble_result['forecast'][-1]:.2f}")
print(f"Model weights: {stacker.get_model_weights()}")
```

## Visualize

```python
# Interactive chart
chart = ft.plot_forecast_interactive(
    prices=spy_prices,
    result=result,
    dates=spy_dates,
    title="SPY 30-Day Forecast",
    top_n_paths=30
)
chart.show()
```
