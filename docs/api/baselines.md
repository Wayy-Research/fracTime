# Baseline Models API Reference

All baseline models follow the same interface:

- `fit(prices, dates=None)` - Fit the model
- `predict(n_steps, end_date=None)` - Generate forecast

## ARIMAForecaster

::: fractime.baselines.ARIMAForecaster
    options:
      show_source: false

## GARCHForecaster

::: fractime.baselines.GARCHForecaster
    options:
      show_source: false

## ProphetForecaster

::: fractime.baselines.ProphetForecaster
    options:
      show_source: false

## ETSForecaster

::: fractime.baselines.ETSForecaster
    options:
      show_source: false

## VARForecaster

::: fractime.baselines.VARForecaster
    options:
      show_source: false

## LSTMForecaster

::: fractime.baselines.LSTMForecaster
    options:
      show_source: false
