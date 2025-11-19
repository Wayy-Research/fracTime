# Visualization Fix - Missing `dates` Parameter

## The Problem

Your code was creating a bad chart because the `dates` parameter was missing from `plot_forecast_interactive()`.

### Your Original Code (BROKEN):
```python
chart = ft.plot_forecast_interactive(
    prices = prices,
    result = result,
    title = "Prob-weighted Forecast Paths - BNB",
    top_n_paths = 50
)  # ← Missing dates parameter!
```

### What Was Happening:
- **Historical data**: Used integer indices (0, 1, 2, ..., 2148)
- **Forecast data**: Used actual dates from result (2025-11-19, 2025-11-20, ...)
- **Result**: Mismatch between historical (integers) and forecast (dates) caused rendering issues

## The Solution

### Corrected Code:
```python
chart = ft.plot_forecast_interactive(
    prices = prices.to_numpy(),
    result = result,
    dates = dates.to_numpy(),  # ← ADD THIS LINE!
    title = "Prob-weighted Forecast Paths - BNB",
    top_n_paths = 50
)
```

## Complete Working Example

See `bnb_forecast_example.py` in this directory for a full working example.

```python
from wrdata import DataStream
import fractime as ft

# Get data
stream = DataStream()
df = stream.get('BNB-USD', start='2020-01-01', end='2025-11-19')

dates = df['timestamp']
prices = df['close']

# Fit
forecaster = ft.FractalForecaster()
forecaster.fit(prices.to_numpy(), dates=dates.to_numpy())

# Forecast
result = forecaster.predict(end_date='2025-11-27')

# Visualize - WITH dates parameter!
chart = ft.plot_forecast_interactive(
    prices=prices.to_numpy(),
    result=result,
    dates=dates.to_numpy(),  # ← REQUIRED!
    title="Prob-weighted Forecast Paths - BNB",
    top_n_paths=50
)

chart.show()
```

## Expected Visualization

The chart should now display:

✓ **X-Axis**: Proper dates (2020-01-01, 2020-06-01, ..., 2025-11-27)
✓ **Historical Data**: Black line showing actual prices
✓ **Probability Cloud**: Blue paths (opacity varies by probability)
✓ **High-Probability Paths**: Orange-red gradient (top 50 paths)
✓ **Weighted Forecast**: Red dashed line
✓ **95% Confidence**: Green shaded region
✓ **Clean Legend**: 4 entries (not 50+)

## Key Takeaways

1. **Always pass `dates` parameter** when you have datetime data
2. **Convert to numpy** using `.to_numpy()` for polars/pandas Series
3. **Pass same dates** to both `fit()` and `plot_forecast_interactive()`

## Output

The working example created: `/home/rcgalbo/wayy-research/wayy-fin/wf/examples/bnb_forecast.html`

Open this file in a browser to see the correctly rendered visualization.
