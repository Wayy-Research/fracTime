"""
Visualization for fractal analysis and forecasting.

Simple plot() function that handles any fractime result type.

Examples:
    >>> import fractime as ft
    >>> result = ft.Forecaster(prices).predict(steps=30)
    >>> ft.plot(result)
"""

from __future__ import annotations

from typing import Union, Optional, TYPE_CHECKING
import numpy as np

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import plotly.graph_objects as go
    from .result import ForecastResult, AnalysisResult, Metric
    from .analyzer import Analyzer


def plot(
    obj: Union['ForecastResult', 'AnalysisResult', 'Analyzer', 'Metric'],
    view: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> 'go.Figure':
    """
    Plot any fractime result.

    Automatically detects the type of object and creates an appropriate
    visualization.

    Args:
        obj: Object to plot (ForecastResult, AnalysisResult, Analyzer, or Metric)
        view: For Metric: 'point', 'rolling', or 'distribution'
        title: Custom plot title
        show: Whether to display the plot immediately
        **kwargs: Additional arguments passed to Plotly

    Returns:
        Plotly Figure object

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
    import plotly.graph_objects as go

    # Import result types
    from .result import ForecastResult, AnalysisResult, Metric
    from .analyzer import Analyzer

    # Detect type and dispatch
    if isinstance(obj, ForecastResult):
        fig = _plot_forecast(obj, title, **kwargs)
    elif isinstance(obj, AnalysisResult):
        fig = _plot_analysis(obj, title, **kwargs)
    elif isinstance(obj, Analyzer):
        fig = _plot_analysis(obj.result, title, **kwargs)
    elif isinstance(obj, Metric):
        fig = _plot_metric(obj, view, title, **kwargs)
    else:
        raise TypeError(f"Cannot plot object of type {type(obj)}")

    if show:
        fig.show()

    return fig


def _plot_forecast(
    result: 'ForecastResult',
    title: Optional[str] = None,
    **kwargs
) -> 'go.Figure':
    """Plot forecast with confidence intervals and path density."""
    import plotly.graph_objects as go

    n_steps = result.n_steps
    x = list(range(1, n_steps + 1))

    # Use dates if available
    if result.dates is not None:
        try:
            x = result.dates.tolist()
        except:
            pass

    fig = go.Figure()

    # Plot confidence bands
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(result.upper) + list(result.lower[::-1]),
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='95% CI',
        showlegend=True,
    ))

    # Plot 50% CI
    q25 = result.quantile(0.25)
    q75 = result.quantile(0.75)
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(q75) + list(q25[::-1]),
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.4)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='50% CI',
        showlegend=True,
    ))

    # Plot main forecast
    fig.add_trace(go.Scatter(
        x=x,
        y=result.forecast,
        mode='lines',
        name='Forecast',
        line=dict(color='rgb(99, 110, 250)', width=2),
    ))

    # Plot mean
    fig.add_trace(go.Scatter(
        x=x,
        y=result.mean,
        mode='lines',
        name='Mean',
        line=dict(color='rgb(239, 85, 59)', width=1, dash='dash'),
    ))

    # Layout
    fig.update_layout(
        title=title or 'Fractal Forecast',
        xaxis_title='Step' if result.dates is None else 'Date',
        yaxis_title='Value',
        template='plotly_dark',
        hovermode='x unified',
        **kwargs
    )

    return fig


def _plot_analysis(
    result: 'AnalysisResult',
    title: Optional[str] = None,
    **kwargs
) -> 'go.Figure':
    """Plot analysis dashboard."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Hurst Exponent: {result.hurst.value:.3f}',
            f'Fractal Dimension: {result.fractal_dim.value:.3f}',
            f'Volatility: {result.volatility.value:.1%}',
            f'Regime: {result.regime}'
        ),
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'pie'}],
        ]
    )

    # Hurst gauge
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=result.hurst.value,
        domain={'row': 0, 'column': 0},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': 'rgb(99, 110, 250)'},
            'steps': [
                {'range': [0, 0.45], 'color': 'rgba(239, 85, 59, 0.3)'},
                {'range': [0.45, 0.55], 'color': 'rgba(255, 255, 255, 0.1)'},
                {'range': [0.55, 1], 'color': 'rgba(0, 204, 150, 0.3)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': 0.5
            }
        },
    ), row=1, col=1)

    # Fractal dimension gauge
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=result.fractal_dim.value,
        domain={'row': 0, 'column': 1},
        gauge={
            'axis': {'range': [1, 2]},
            'bar': {'color': 'rgb(0, 204, 150)'},
        },
    ), row=1, col=2)

    # Volatility gauge
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=result.volatility.value,
        number={'suffix': '%', 'valueformat': '.1f'},
        domain={'row': 1, 'column': 0},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': 'rgb(239, 85, 59)'},
        },
    ), row=2, col=1)

    # Regime pie chart
    probs = result.regime_probabilities
    fig.add_trace(go.Pie(
        labels=list(probs.keys()),
        values=list(probs.values()),
        marker={'colors': ['rgb(0, 204, 150)', 'rgb(99, 110, 250)', 'rgb(239, 85, 59)']},
    ), row=2, col=2)

    fig.update_layout(
        title=title or 'Fractal Analysis',
        template='plotly_dark',
        height=600,
        **kwargs
    )

    return fig


def _plot_metric(
    metric: 'Metric',
    view: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> 'go.Figure':
    """Plot a single metric."""
    import plotly.graph_objects as go

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
) -> 'go.Figure':
    """Plot metric as indicator."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=metric.value,
        title={'text': metric._name.replace('_', ' ').title()},
        gauge={
            'axis': {'range': [0, max(1, metric.value * 1.5)]},
            'bar': {'color': 'rgb(99, 110, 250)'},
        },
    ))

    fig.update_layout(
        title=title or f'{metric._name}',
        template='plotly_dark',
        **kwargs
    )

    return fig


def _plot_metric_rolling(
    metric: 'Metric',
    title: Optional[str] = None,
    **kwargs
) -> 'go.Figure':
    """Plot metric rolling values."""
    import plotly.graph_objects as go

    rolling = metric.rolling

    # Get x and y
    if 'date' in rolling.columns:
        x = rolling['date'].to_list()
    else:
        x = rolling['index'].to_list()
    y = rolling['value'].to_list()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=metric._name,
        line=dict(color='rgb(99, 110, 250)', width=2),
    ))

    # Add reference line at current value
    fig.add_hline(
        y=metric.value,
        line_dash='dash',
        line_color='rgba(239, 85, 59, 0.7)',
        annotation_text=f'Current: {metric.value:.3f}',
    )

    fig.update_layout(
        title=title or f'{metric._name.replace("_", " ").title()} Over Time',
        xaxis_title='Date' if 'date' in rolling.columns else 'Index',
        yaxis_title=metric._name.replace('_', ' ').title(),
        template='plotly_dark',
        **kwargs
    )

    return fig


def _plot_metric_distribution(
    metric: 'Metric',
    title: Optional[str] = None,
    **kwargs
) -> 'go.Figure':
    """Plot metric bootstrap distribution."""
    import plotly.graph_objects as go

    dist = metric.distribution

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=dist,
        nbinsx=50,
        name='Distribution',
        marker_color='rgb(99, 110, 250)',
        opacity=0.7,
    ))

    # Add point estimate line
    fig.add_vline(
        x=metric.value,
        line_color='rgb(239, 85, 59)',
        line_width=2,
        annotation_text=f'Point: {metric.value:.3f}',
    )

    # Add CI lines
    try:
        ci = metric.ci(0.95)
        fig.add_vline(x=ci[0], line_dash='dash', line_color='rgba(255, 255, 255, 0.5)')
        fig.add_vline(x=ci[1], line_dash='dash', line_color='rgba(255, 255, 255, 0.5)')
    except:
        pass

    fig.update_layout(
        title=title or f'{metric._name.replace("_", " ").title()} Distribution',
        xaxis_title=metric._name.replace('_', ' ').title(),
        yaxis_title='Count',
        template='plotly_dark',
        **kwargs
    )

    return fig
