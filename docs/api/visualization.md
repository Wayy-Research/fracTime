# Visualization API Reference

FracTime provides fully interactive visualizations using Plotly, with special support for path density visualization that shows probability clusters.

## Quick Example

```python
import fractime as ft

# Generate forecast
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)
result = forecaster.predict(n_steps=30)

# Interactive density plot
fig = ft.plot_forecast(
    prices[-100:],
    result,
    dates=dates[-100:],
    colorscale='Viridis',
    show_percentiles=True
)
fig.show()
```

## Main Functions

### plot_forecast

The primary plotting function with path density visualization.

```python
fig = ft.plot_forecast(
    prices,                      # Historical prices
    result,                      # Result dict from predict()
    dates=None,                  # Optional dates array
    title="Fractal Forecast",    # Chart title
    show_all_paths=True,         # Show individual paths
    max_paths=500,               # Max paths to display
    colorscale='Viridis',        # Color scale for density
    show_percentiles=True,       # Show percentile lines
    show_density_heatmap=False   # Add 2D density heatmap
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prices` | array | required | Historical price data |
| `result` | dict | required | Result from `forecaster.predict()` |
| `dates` | array | None | Historical dates (optional) |
| `title` | str | "Fractal Forecast..." | Chart title |
| `show_all_paths` | bool | True | Show individual paths colored by density |
| `max_paths` | int | 500 | Maximum paths to display (for performance) |
| `colorscale` | str | 'Viridis' | Color scale: 'Viridis', 'Plasma', 'Inferno', 'Hot' |
| `show_percentiles` | bool | True | Show 5th, 25th, 50th, 75th, 95th percentile lines |
| `show_density_heatmap` | bool | False | Add 2D density heatmap overlay |

**Returns:** Plotly Figure object

::: fractime.plot_forecast
    options:
      show_source: false

### plot_forecast_interactive

Enhanced version that defaults to density-based visualization.

::: fractime.plot_forecast_interactive
    options:
      show_source: false

### print_forecast_summary

Print a formatted text summary of forecast results.

```python
ft.print_forecast_summary(result, current_price=prices[-1])
```

::: fractime.print_forecast_summary
    options:
      show_source: false

## Colorscales

FracTime supports multiple colorscales for density visualization:

| Colorscale | Description | Best For |
|------------|-------------|----------|
| `Viridis` | Purple → Teal → Yellow | Default, perceptually uniform |
| `Plasma` | Blue → Pink → Yellow | High contrast |
| `Inferno` | Black → Red → Yellow | Dark themes |
| `Hot` | Blue → Red → Yellow | Traditional heat map |

```python
# Compare colorscales
for scale in ['Viridis', 'Plasma', 'Inferno', 'Hot']:
    fig = ft.plot_forecast(prices, result, colorscale=scale)
    fig.write_html(f"forecast_{scale.lower()}.html")
```

## Density Functions

### compute_path_density

Compute 2D histogram of path distribution.

::: fractime.visualization.compute_path_density
    options:
      show_source: false

### compute_path_colors_by_density

Assign density scores to paths using kernel density estimation.

::: fractime.visualization.compute_path_colors_by_density
    options:
      show_source: false

## FractalVisualizer Class

Advanced visualizations for fractal analysis.

::: fractime.FractalVisualizer
    options:
      show_source: false

### Methods

#### plot_cross_dimensional_analysis

Create 6-panel dashboard for cross-dimensional fractal analysis.

```python
visualizer = ft.FractalVisualizer()
fig = visualizer.plot_cross_dimensional_analysis(
    prices=prices,
    volumes=volumes,
    cross_dim_results=cross_dim_results,
    dates=dates
)
```

#### plot_trading_time_analysis

Visualize trading time vs clock time with time dilation markers.

```python
fig = visualizer.plot_trading_time_analysis(
    prices=prices,
    time_map=time_map,
    dates=dates
)
```

#### plot_analysis_and_forecast

Comprehensive analysis with path density visualization.

```python
fig = visualizer.plot_analysis_and_forecast(
    historical_prices=prices,
    simulation_results=(paths, path_analysis),
    analysis_results={'hurst': 0.6, 'fractal_dim': 1.4},
    dates=dates
)
```

## Saving Visualizations

All visualizations return Plotly Figure objects that can be:

```python
# Display in Jupyter
fig.show()

# Save as HTML (interactive)
fig.write_html("forecast.html")

# Save as static image (requires kaleido)
fig.write_image("forecast.png", width=1200, height=700)

# Get JSON for web embedding
json_str = fig.to_json()
```
