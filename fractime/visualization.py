"""
Visualization for fractal analysis and forecasting.

Simple plot() function that handles any fractime result type.

Examples:
    >>> import fractime as ft
    >>> result = ft.Forecaster(prices).predict(steps=30)
    >>> ft.plot(result)
"""

from __future__ import annotations

from typing import Union, Optional, Any, TYPE_CHECKING
import numpy as np

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    from .result import ForecastResult, AnalysisResult, Metric
    from .analyzer import Analyzer


def plot(
    obj: Union['ForecastResult', 'AnalysisResult', 'Analyzer', 'Metric'],
    view: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Any:
    """
    Plot any fractime result.

    Automatically detects the type of object and creates an appropriate
    visualization.

    Args:
        obj: Object to plot (ForecastResult, AnalysisResult, Analyzer, or Metric)
        view: For Metric: 'point', 'rolling', or 'distribution'
        title: Custom plot title
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to wrchart

    Returns:
        wrchart chart object (ForecastChart, MultiPanelChart, or Chart)

    Examples:
        Plot forecast:
            >>> result = model.predict(steps=30)
            >>> ft.plot(result)

        Plot analysis:
            >>> analyzer = ft.Analyzer(prices)
            >>> ft.plot(analyzer)

        Plot single metric:
            >>> ft.plot(analyzer.hurst)
            >>> ft.plot(analyzer.hurst, view='rolling')
            >>> ft.plot(analyzer.hurst, view='distribution')
    """
    # Import result types
    from .result import ForecastResult, AnalysisResult, Metric
    from .analyzer import Analyzer

    # Detect type and dispatch
    if isinstance(obj, ForecastResult):
        chart = _plot_forecast(obj, title, **kwargs)
    elif isinstance(obj, AnalysisResult):
        chart = _plot_analysis(obj, title, **kwargs)
    elif isinstance(obj, Analyzer):
        chart = _plot_analysis(obj.result, title, **kwargs)
    elif isinstance(obj, Metric):
        chart = _plot_metric(obj, view, title, **kwargs)
    else:
        raise TypeError(f"Cannot plot object of type {type(obj)}")

    if show:
        chart.show()

    return chart


def _plot_forecast(
    result: 'ForecastResult',
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """Plot forecast with confidence intervals and path density."""
    from wrchart import ForecastChart

    n_steps = result.n_steps

    # Build result dict for ForecastChart
    # Get 25th and 75th percentiles for inner CI
    q25 = result.quantile(0.25)
    q75 = result.quantile(0.75)

    # Create paths array from percentiles if available
    # ForecastChart expects paths array, but we can construct one from percentiles
    # Use forecast, mean, and percentile bounds as representative paths
    paths = np.vstack([
        result.forecast,
        result.mean,
        result.upper,
        result.lower,
        q75,
        q25,
    ])

    # Create probabilities (weight forecast and mean higher)
    probabilities = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])

    result_dict = {
        'paths': paths,
        'probabilities': probabilities,
        'weighted_forecast': result.forecast,
    }

    # Add dates if available
    if result.dates is not None:
        result_dict['dates'] = result.dates

    # Create a dummy historical price (last known value repeated)
    # Since ForecastResult doesn't include historical, we create a minimal one
    last_value = result.forecast[0] if len(result.forecast) > 0 else 100.0
    historical = np.array([last_value])

    # Create ForecastChart
    width = kwargs.pop('width', 1000)
    height = kwargs.pop('height', 700)
    theme = kwargs.pop('theme', 'dark')
    colorscale = kwargs.pop('colorscale', 'viridis')

    chart = ForecastChart(
        width=width,
        height=height,
        theme=theme,
        title=title or 'Fractal Forecast',
    )

    chart.set_data(historical, result_dict)
    chart.colorscale(colorscale)
    chart.show_percentiles(True)
    chart.show_weighted_forecast(True)

    return chart


def _plot_analysis(
    result: 'AnalysisResult',
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """Plot analysis dashboard using MultiPanelChart with gauges and bars."""
    from wrchart import MultiPanelChart
    from wrchart.multipanel import GaugePanel, BarPanel

    width = kwargs.pop('width', 1200)
    height = kwargs.pop('height', 800)
    theme = kwargs.pop('theme', 'dark')

    chart = MultiPanelChart(
        rows=2,
        cols=2,
        width=width,
        height=height,
        title=title or 'Fractal Analysis',
        theme=theme,
    )

    # Hurst gauge (row 0, col 0)
    # Thresholds: < 0.45 mean-reverting (red), 0.45-0.55 random (yellow), > 0.55 trending (green)
    hurst_thresholds = [
        (0.45, "#F44336"),   # Red - mean reverting
        (0.55, "#FFC107"),   # Yellow - random walk
        (1.0, "#4CAF50"),    # Green - trending
    ]
    chart.add_panel(GaugePanel(
        title=f'Hurst Exponent: {result.hurst.value:.3f}',
        value=result.hurst.value,
        min_value=0,
        max_value=1,
        thresholds=hurst_thresholds,
        label='Hurst',
        row=0,
        col=0,
    ))

    # Fractal dimension gauge (row 0, col 1)
    # Range 1-2, higher = more complex
    fd_thresholds = [
        (1.33, "#4CAF50"),   # Green - smooth
        (1.66, "#FFC107"),   # Yellow - moderate
        (2.0, "#F44336"),    # Red - complex/rough
    ]
    chart.add_panel(GaugePanel(
        title=f'Fractal Dimension: {result.fractal_dim.value:.3f}',
        value=result.fractal_dim.value,
        min_value=1,
        max_value=2,
        thresholds=fd_thresholds,
        label='Fractal Dim',
        row=0,
        col=1,
    ))

    # Volatility gauge (row 1, col 0)
    vol_pct = result.volatility.value * 100  # Convert to percentage
    vol_thresholds = [
        (20, "#4CAF50"),    # Green - low volatility
        (40, "#FFC107"),    # Yellow - moderate
        (100, "#F44336"),   # Red - high volatility
    ]
    chart.add_panel(GaugePanel(
        title=f'Volatility: {result.volatility.value:.1%}',
        value=vol_pct,
        min_value=0,
        max_value=100,
        thresholds=vol_thresholds,
        label='Volatility',
        unit='%',
        row=1,
        col=0,
    ))

    # Regime probabilities bar chart (row 1, col 1)
    # Replace pie chart with horizontal bar chart
    probs = result.regime_probabilities
    categories = list(probs.keys())
    values = [probs[k] * 100 for k in categories]  # Convert to percentages

    # Colors for regime types
    regime_colors = ['#4CAF50', '#2196F3', '#F44336']  # Green (trending), Blue (random), Red (mean-reverting)

    chart.add_panel(BarPanel(
        title=f'Regime: {result.regime}',
        categories=categories,
        values=values,
        colors=regime_colors[:len(categories)],
        show_values=True,
        row=1,
        col=1,
    ))

    return chart


def _plot_metric(
    metric: 'Metric',
    view: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """Plot a single metric."""
    if view is None:
        # Auto-detect best view
        try:
            _ = metric.rolling
            view = 'rolling'
        except ValueError:
            try:
                _ = metric.distribution
                view = 'distribution'
            except ValueError:
                view = 'point'

    if view == 'point':
        return _plot_metric_point(metric, title, **kwargs)
    elif view == 'rolling':
        return _plot_metric_rolling(metric, title, **kwargs)
    elif view == 'distribution':
        return _plot_metric_distribution(metric, title, **kwargs)
    else:
        raise ValueError(f"Unknown view: {view}")


def _plot_metric_point(
    metric: 'Metric',
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """Plot metric as gauge indicator."""
    from wrchart import MultiPanelChart
    from wrchart.multipanel import GaugePanel

    width = kwargs.pop('width', 400)
    height = kwargs.pop('height', 400)
    theme = kwargs.pop('theme', 'dark')

    chart = MultiPanelChart(
        rows=1,
        cols=1,
        width=width,
        height=height,
        title=title or metric._name.replace('_', ' ').title(),
        theme=theme,
    )

    # Determine range based on metric value
    max_val = max(1, metric.value * 1.5)

    chart.add_panel(GaugePanel(
        title=metric._name.replace('_', ' ').title(),
        value=metric.value,
        min_value=0,
        max_value=max_val,
        label=metric._name,
        row=0,
        col=0,
    ))

    return chart


def _plot_metric_rolling(
    metric: 'Metric',
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """Plot metric rolling values."""
    from wrchart import MultiPanelChart
    from wrchart.multipanel import LinePanel

    width = kwargs.pop('width', 800)
    height = kwargs.pop('height', 500)
    theme = kwargs.pop('theme', 'dark')

    rolling = metric.rolling

    # Get x and y data
    if 'date' in rolling.columns:
        # Convert dates to numeric indices for plotting
        x = list(range(len(rolling)))
        x_label = 'Date'
    else:
        x = rolling['index'].to_list()
        x_label = 'Index'

    y = rolling['value'].to_list()

    chart = MultiPanelChart(
        rows=1,
        cols=1,
        width=width,
        height=height,
        title=title or f'{metric._name.replace("_", " ").title()} Over Time',
        theme=theme,
    )

    chart.add_panel(LinePanel(
        title='',
        x_data=x,
        y_data=y,
        colors=['#636EFA'],
        line_widths=[2],
        show_zero_line=False,
        y_label=metric._name.replace('_', ' ').title(),
        x_label=x_label,
        row=0,
        col=0,
    ))

    return chart


def _plot_metric_distribution(
    metric: 'Metric',
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """Plot metric bootstrap distribution."""
    from wrchart import MultiPanelChart
    from wrchart.multipanel import BarPanel

    width = kwargs.pop('width', 800)
    height = kwargs.pop('height', 500)
    theme = kwargs.pop('theme', 'dark')

    dist = metric.distribution

    # Create histogram bins
    n_bins = 30
    counts, bin_edges = np.histogram(dist, bins=n_bins)

    # Use bin centers as categories
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    categories = [f'{x:.3f}' for x in bin_centers]

    chart = MultiPanelChart(
        rows=1,
        cols=1,
        width=width,
        height=height,
        title=title or f'{metric._name.replace("_", " ").title()} Distribution',
        theme=theme,
    )

    chart.add_panel(BarPanel(
        title=f'Point estimate: {metric.value:.3f}',
        categories=categories,
        values=counts.tolist(),
        colors=['#636EFA'],
        show_values=False,
        row=0,
        col=0,
    ))

    return chart


def plot_forecast(
    prices: np.ndarray,
    result: dict,
    dates: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    colorscale: str = 'viridis',
    show_percentiles: bool = True,
    **kwargs
) -> Any:
    """
    Plot forecast with Monte Carlo paths and path density visualization.

    This is the primary forecast visualization function that displays:
    - Historical price data
    - Monte Carlo simulation paths with density-based coloring
    - Percentile lines (5th, 25th, 50th, 75th, 95th)
    - Weighted forecast line

    Args:
        prices: Historical price data (numpy array)
        result: Forecast result dict containing:
            - paths: (n_paths, n_steps) array of Monte Carlo paths
            - probabilities: (n_paths,) path probabilities (optional)
            - weighted_forecast: (n_steps,) weighted forecast (optional)
        dates: Historical dates (optional)
        title: Chart title
        colorscale: Color scale for path density ('viridis', 'plasma', 'inferno', 'hot')
        show_percentiles: Whether to show percentile lines
        **kwargs: Additional arguments (width, height, theme)

    Returns:
        ForecastChart object

    Example:
        >>> import fractime as ft
        >>> import numpy as np
        >>>
        >>> prices = np.array([100, 101, 102, ...])
        >>> result = forecaster.predict(n_steps=30, n_paths=500)
        >>> chart = ft.plot_forecast(prices, result, title="My Forecast")
        >>> chart.show()
    """
    from wrchart import ForecastChart

    width = kwargs.pop('width', 1000)
    height = kwargs.pop('height', 700)
    theme = kwargs.pop('theme', 'dark')

    chart = ForecastChart(
        width=width,
        height=height,
        theme=theme,
        title=title or 'Fractal Forecast',
    )

    chart.set_data(prices, result, dates=dates)
    chart.colorscale(colorscale)
    chart.show_percentiles(show_percentiles)
    chart.show_weighted_forecast(True)

    return chart
