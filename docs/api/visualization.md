# Visualization API Reference

FracTime provides interactive visualizations using Plotly.

---

## Overview

```python
import fractime as ft

# Plot any FracTime object
ft.plot(result)
ft.plot(analyzer)
ft.plot(analyzer.hurst)
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
    **kwargs        # Additional Plotly layout args
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obj` | various | required | FracTime object to plot |
| `view` | str | None | View type for Metric objects |
| `title` | str | None | Custom plot title |
| `show` | bool | True | Display immediately |
| `**kwargs` | - | - | Passed to Plotly layout |

### Supported Objects

| Object Type | Default Visualization |
|-------------|----------------------|
| `ForecastResult` | Forecast with confidence bands |
| `AnalysisResult` | Analysis dashboard |
| `Analyzer` | Analysis dashboard |
| `Metric` | Auto-detect best view |

### Returns

Plotly `Figure` object.

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
fig = ft.plot(result, show=False)
fig.write_html("forecast.html")
```

### What's Shown

- **95% confidence band** (light blue)
- **50% confidence band** (darker blue)
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
- **Regime pie chart** - Probability distribution

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
- Horizontal line at current point estimate

### Distribution View

Histogram of bootstrap samples.

```python
ft.plot(analyzer.hurst, view='distribution')
```

Shows:
- Bootstrap distribution histogram
- Vertical line at point estimate
- 95% CI boundaries

---

## Customization

### Plotly Layout Arguments

Pass additional arguments to Plotly:

```python
ft.plot(
    result,
    title="My Forecast",
    height=600,
    width=1000,
    template='plotly_white',  # Light theme
)
```

### Common Layout Options

| Option | Description |
|--------|-------------|
| `height` | Figure height in pixels |
| `width` | Figure width in pixels |
| `template` | Plotly theme |

### Available Templates

- `'plotly_dark'` (default)
- `'plotly_white'`
- `'plotly'`
- `'seaborn'`
- `'ggplot2'`

---

## Saving Figures

### HTML (Interactive)

```python
fig = ft.plot(result, show=False)
fig.write_html("forecast.html")
```

### Static Image

Requires kaleido: `pip install kaleido`

```python
fig = ft.plot(result, show=False)
fig.write_image("forecast.png", width=1200, height=700)
fig.write_image("forecast.pdf")
fig.write_image("forecast.svg")
```

### JSON (Web Embedding)

```python
fig = ft.plot(result, show=False)
json_str = fig.to_json()
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

### Save Multiple Formats

```python
fig = ft.plot(result, show=False)

# Interactive HTML
fig.write_html("forecast.html")

# Static PNG
fig.write_image("forecast.png", width=1200, height=700)

# For reports
fig.write_image("forecast.pdf")
```

### Custom Styling

```python
ft.plot(
    result,
    title="Price Forecast",
    height=500,
    width=900,
    template='plotly_white',
)
```

---

## Working with Figures

The returned figure is a standard Plotly `Figure` object:

```python
fig = ft.plot(result, show=False)

# Modify traces
fig.data[0].name = "Custom Name"

# Add annotations
fig.add_annotation(
    x=15,
    y=result.forecast[14],
    text="Midpoint",
)

# Update layout
fig.update_layout(
    font=dict(size=14),
)

# Show modified figure
fig.show()
```

---

## See Also

- [ForecastResult](results.md#forecastresult) - Forecast data
- [AnalysisResult](results.md#analysisresult) - Analysis data
- [Metric](results.md#metric) - Individual metrics
