# Visualization API Reference

FracTime provides interactive visualizations using wrchart.

---

## Overview

```python
import fractime as ft

# Plot any FracTime object
ft.plot(result)
ft.plot(analyzer)
ft.plot(analyzer.hurst)

# Plot forecast with Monte Carlo paths
ft.plot_forecast(prices, result)
```

---

## plot()

The universal plotting function that handles any FracTime object.

```python
ft.plot(
    obj,            # Object to plot
    view=None,      # For Metric: 'point', 'rolling', 'distribution'
    title=None,     # Custom title
    show=True,      # Display immediately
    **kwargs        # Additional wrchart arguments
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obj` | various | required | FracTime object to plot |
| `view` | str | None | View type for Metric objects |
| `title` | str | None | Custom plot title |
| `show` | bool | True | Display immediately |
| `width` | int | varies | Chart width in pixels |
| `height` | int | varies | Chart height in pixels |
| `theme` | str | 'dark' | Theme ('dark' or 'light') |

### Supported Objects

| Object Type | Default Visualization |
|-------------|----------------------|
| `ForecastResult` | Forecast with confidence bands |
| `AnalysisResult` | Analysis dashboard |
| `Analyzer` | Analysis dashboard |
| `Metric` | Auto-detect best view |

### Returns

wrchart chart object (ForecastChart, MultiPanelChart, or Chart).

---

## plot_forecast()

Plot forecast with Monte Carlo paths and path density visualization.

```python
ft.plot_forecast(
    prices,                 # Historical price data
    result,                 # Forecast result dict
    dates=None,             # Historical dates (optional)
    title=None,             # Chart title
    colorscale='viridis',   # Color scale for path density
    show_percentiles=True,  # Show percentile lines
    **kwargs                # Additional arguments
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prices` | np.ndarray | required | Historical price data |
| `result` | dict | required | Forecast result with paths, probabilities |
| `dates` | np.ndarray | None | Historical dates |
| `title` | str | None | Chart title |
| `colorscale` | str | 'viridis' | Color scale ('viridis', 'plasma', 'inferno', 'hot') |
| `show_percentiles` | bool | True | Show percentile lines |

### Returns

`ForecastChart` object.

---

## Plotting ForecastResult

```python
model = ft.Forecaster(prices)
result = model.predict(steps=30)

# Basic plot
ft.plot(result)

# Custom title
ft.plot(result, title="30-Day Price Forecast")

# Don't show immediately
chart = ft.plot(result, show=False)
chart.to_html("forecast.html")
```

### What's Shown

- **95% confidence band** (outer band)
- **50% confidence band** (inner band)
- **Median forecast** (solid line)
- **Mean forecast** (dashed line)

---

## Plotting AnalysisResult

```python
analyzer = ft.Analyzer(prices)

# Plot analysis dashboard
ft.plot(analyzer)

# Or use the result directly
ft.plot(analyzer.result)
```

### What's Shown

- **Hurst gauge** - Value from 0 to 1
- **Fractal dimension gauge** - Value from 1 to 2
- **Volatility gauge** - Annualized percentage
- **Regime bar chart** - Probability distribution

---

## Plotting Metric

Metrics support three views:

### Auto-Detect

```python
ft.plot(analyzer.hurst)  # Auto-selects best view
```

Selection logic:
1. If rolling data exists → rolling view
2. Else if distribution exists → distribution view
3. Else → point view

### Point View

Gauge chart showing the point estimate.

```python
ft.plot(analyzer.hurst, view='point')
```

### Rolling View

Time series of rolling values.

```python
ft.plot(analyzer.hurst, view='rolling')
```

Shows:
- Rolling values over time
- Line chart with the metric values

### Distribution View

Histogram of bootstrap samples.

```python
ft.plot(analyzer.hurst, view='distribution')
```

Shows:
- Bootstrap distribution as bar chart
- Point estimate in title

---

## Customization

### wrchart Arguments

Pass additional arguments to wrchart:

```python
ft.plot(
    result,
    title="My Forecast",
    height=600,
    width=1000,
    theme='dark',  # or 'light'
)
```

### Common Layout Options

| Option | Description |
|--------|-------------|
| `height` | Figure height in pixels |
| `width` | Figure width in pixels |
| `theme` | Theme ('dark' or 'light') |
| `colorscale` | For forecasts: 'viridis', 'plasma', 'inferno', 'hot' |

---

## Saving Charts

### HTML (Interactive)

```python
chart = ft.plot(result, show=False)
chart.to_html("forecast.html")
```

---

## Examples

### Forecast Plot

```python
import fractime as ft

model = ft.Forecaster(prices)
result = model.predict(steps=30)

ft.plot(result, title="30-Day Forecast")
```

### Forecast with Monte Carlo Paths

```python
import fractime as ft

forecaster = ft.FractalForecaster(lookback=252)
forecaster.fit(prices)
result = forecaster.predict(n_steps=30, n_paths=500)

chart = ft.plot_forecast(
    prices[-100:],
    result,
    title="Monte Carlo Forecast",
    colorscale='plasma'
)
chart.to_html("forecast.html")
```

### Analysis Dashboard

```python
analyzer = ft.Analyzer(prices)
ft.plot(analyzer, title="Fractal Analysis")
```

### Rolling Hurst

```python
analyzer = ft.Analyzer(prices, dates=dates, window=63)
ft.plot(analyzer.hurst, view='rolling', title="Rolling Hurst Exponent")
```

### Bootstrap Distribution

```python
analyzer = ft.Analyzer(prices, n_samples=2000)
ft.plot(analyzer.hurst, view='distribution', title="Hurst Distribution")
```

### Save to HTML

```python
chart = ft.plot(result, show=False)
chart.to_html("forecast.html")
```

### Custom Styling

```python
ft.plot(
    result,
    title="Price Forecast",
    height=500,
    width=900,
    theme='light',
)
```

---

## Working with Charts

The returned chart is a wrchart chart object:

```python
chart = ft.plot(result, show=False)

# Display the chart
chart.show()

# Save to HTML
chart.to_html("forecast.html")

# In Streamlit
chart.streamlit()
```

For ForecastChart, you can chain configuration:

```python
chart = ft.plot_forecast(prices, result, show=False)
chart.colorscale('plasma')
chart.show_percentiles(True)
chart.show_weighted_forecast(True)
chart.show()
```

---

## See Also

- [ForecastResult](results.md#forecastresult) - Forecast data
- [AnalysisResult](results.md#analysisresult) - Analysis data
- [Metric](results.md#metric) - Individual metrics
