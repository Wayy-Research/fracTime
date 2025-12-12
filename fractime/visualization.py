"""
Visualization module for fractal time series analysis.

This module provides interactive visualizations using Plotly
for fractal patterns, forecasts, and analysis results.
All plots are interactive by default.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.ndimage import gaussian_filter


def compute_path_density(
    paths: np.ndarray,
    n_time_bins: int = 50,
    n_price_bins: int = 100,
    probabilities: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D density of paths over time and price.

    Args:
        paths: Array of shape (n_paths, n_steps)
        n_time_bins: Number of bins along time axis
        n_price_bins: Number of bins along price axis
        probabilities: Optional path probabilities for weighting

    Returns:
        Tuple of (density, time_edges, price_edges)
    """
    n_paths, n_steps = paths.shape

    # Flatten paths to get all (time, price) points
    time_indices = np.tile(np.arange(n_steps), n_paths)
    price_values = paths.flatten()

    # Create weights based on path probabilities
    if probabilities is not None:
        weights = np.repeat(probabilities, n_steps)
    else:
        weights = None

    # Compute 2D histogram
    price_range = (np.min(price_values), np.max(price_values))

    density, time_edges, price_edges = np.histogram2d(
        time_indices,
        price_values,
        bins=[n_time_bins, n_price_bins],
        range=[[0, n_steps], price_range],
        weights=weights,
        density=True
    )

    # Apply Gaussian smoothing for nicer visualization
    density = gaussian_filter(density.T, sigma=1.5)

    return density, time_edges, price_edges


def compute_path_colors_by_density(
    paths: np.ndarray,
    probabilities: np.ndarray = None,
    method: str = 'endpoint'
) -> np.ndarray:
    """
    Compute colors for each path based on density/probability.

    Args:
        paths: Array of shape (n_paths, n_steps)
        probabilities: Optional path probabilities
        method: 'endpoint' uses final values, 'full' uses full path clustering

    Returns:
        Array of density scores for each path (0-1 range)
    """
    n_paths = paths.shape[0]

    if probabilities is not None:
        # Normalize probabilities to 0-1
        scores = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min() + 1e-10)
        return scores

    if method == 'endpoint':
        # Use kernel density estimation on endpoint values
        endpoints = paths[:, -1]
        kde = stats.gaussian_kde(endpoints)
        density_scores = kde(endpoints)
        # Normalize to 0-1
        scores = (density_scores - density_scores.min()) / (density_scores.max() - density_scores.min() + 1e-10)
    else:
        # Full path similarity - compute pairwise distances
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(10, n_paths))
        nn.fit(paths)
        distances, _ = nn.kneighbors(paths)
        # Average distance to neighbors - lower = denser region
        avg_dist = distances.mean(axis=1)
        # Convert to density score (inverse distance)
        scores = 1 / (1 + avg_dist)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    return scores


def density_to_color(score: float, colorscale: str = 'Viridis') -> str:
    """
    Convert a density score (0-1) to a color string.

    Args:
        score: Density score between 0 and 1
        colorscale: Color scale to use

    Returns:
        RGBA color string
    """
    # Define color scales
    colorscales = {
        'Viridis': [
            (0.0, (68, 1, 84)),      # Dark purple
            (0.25, (59, 82, 139)),   # Blue-purple
            (0.5, (33, 145, 140)),   # Teal
            (0.75, (94, 201, 98)),   # Green
            (1.0, (253, 231, 37))    # Yellow
        ],
        'Plasma': [
            (0.0, (13, 8, 135)),     # Dark blue
            (0.25, (126, 3, 168)),   # Purple
            (0.5, (204, 71, 120)),   # Pink
            (0.75, (248, 149, 64)),  # Orange
            (1.0, (240, 249, 33))    # Yellow
        ],
        'Inferno': [
            (0.0, (0, 0, 4)),        # Black
            (0.25, (87, 16, 110)),   # Purple
            (0.5, (188, 55, 84)),    # Red
            (0.75, (249, 142, 9)),   # Orange
            (1.0, (252, 255, 164))   # Yellow-white
        ],
        'Hot': [
            (0.0, (10, 10, 40)),     # Dark blue
            (0.25, (80, 30, 100)),   # Purple
            (0.5, (200, 50, 50)),    # Red
            (0.75, (255, 150, 50)),  # Orange
            (1.0, (255, 255, 200))   # Light yellow
        ]
    }

    scale = colorscales.get(colorscale, colorscales['Viridis'])

    # Interpolate color
    for i in range(len(scale) - 1):
        t0, c0 = scale[i]
        t1, c1 = scale[i + 1]
        if t0 <= score <= t1:
            # Linear interpolation
            t = (score - t0) / (t1 - t0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return f'rgb({r},{g},{b})'

    # Default to last color
    return f'rgb({scale[-1][1][0]},{scale[-1][1][1]},{scale[-1][1][2]})'


def plot_forecast(
    prices: np.ndarray,
    result: dict,
    dates: np.ndarray = None,
    title: str = "Fractal Forecast with Path Density",
    show_all_paths: bool = True,
    max_paths: int = 500,
    colorscale: str = 'Viridis',
    show_percentiles: bool = True,
    show_density_heatmap: bool = False
) -> go.Figure:
    """
    Create interactive forecast visualization with path density coloring.

    This replaces the static matplotlib version with a fully interactive
    Plotly visualization that shows path density through color.

    Args:
        prices: Historical price data
        result: Result dict from forecaster.predict()
        dates: Historical dates (optional)
        title: Chart title
        show_all_paths: Show individual paths colored by density
        max_paths: Maximum paths to display (for performance)
        colorscale: Color scale for density ('Viridis', 'Plasma', 'Inferno', 'Hot')
        show_percentiles: Show percentile lines
        show_density_heatmap: Show 2D density heatmap overlay

    Returns:
        Plotly Figure object
    """
    from .utils import _ensure_numpy_array
    import polars as pl
    from datetime import datetime

    prices = _ensure_numpy_array(prices)
    paths = result['paths']
    probabilities = result.get('probabilities', None)

    n_hist = len(prices)
    n_paths, n_steps = paths.shape

    # Prepare x-axis dates
    forecast_dates = result.get('dates', None)

    if dates is not None:
        dates = _ensure_numpy_array(dates)
        x_hist = dates[-n_hist:]
        if np.issubdtype(x_hist.dtype, np.datetime64):
            x_hist = pd.to_datetime(x_hist)
    else:
        x_hist = np.arange(n_hist)

    if forecast_dates is not None:
        x_forecast = _ensure_numpy_array(forecast_dates)
        if np.issubdtype(x_forecast.dtype, np.datetime64):
            x_forecast = pd.to_datetime(x_forecast)
    elif dates is not None:
        last_date = x_hist[-1]
        if isinstance(last_date, (datetime, np.datetime64, pd.Timestamp)):
            x_forecast = pd.date_range(start=last_date, periods=n_steps + 1, freq='B')[1:]
        else:
            x_forecast = np.arange(n_steps) + n_hist
    else:
        x_forecast = np.arange(n_steps) + n_hist

    # Create figure
    fig = go.Figure()

    # Compute path density scores
    density_scores = compute_path_colors_by_density(paths, probabilities)

    # Sort paths by density so high-density paths are drawn on top
    sort_idx = np.argsort(density_scores)
    sorted_paths = paths[sort_idx]
    sorted_scores = density_scores[sort_idx]

    # Prepare forecast x-axis with connection to historical
    last_hist_price = prices[-1]
    last_hist_x = x_hist[-1]

    try:
        x_forecast_plot = np.concatenate([[last_hist_x], x_forecast])
    except (TypeError, Exception):
        x_forecast_plot = x_forecast

    # Add density heatmap if requested
    if show_density_heatmap:
        density, time_edges, price_edges = compute_path_density(
            paths,
            n_time_bins=min(50, n_steps),
            n_price_bins=100,
            probabilities=probabilities
        )

        fig.add_trace(go.Heatmap(
            z=density,
            x=x_forecast,
            y=np.linspace(price_edges[0], price_edges[-1], density.shape[0]),
            colorscale=colorscale,
            opacity=0.5,
            showscale=True,
            colorbar=dict(title='Density', x=1.02),
            name='Path Density',
            hoverinfo='skip'
        ))

    # Add individual paths with density-based coloring
    if show_all_paths:
        # Subsample if too many paths
        if n_paths > max_paths:
            # Sample proportionally to density to keep more high-density paths
            sample_probs = sorted_scores / sorted_scores.sum()
            sample_idx = np.random.choice(
                n_paths, size=max_paths, replace=False, p=sample_probs
            )
            display_paths = sorted_paths[sample_idx]
            display_scores = sorted_scores[sample_idx]
            # Re-sort by score
            resort_idx = np.argsort(display_scores)
            display_paths = display_paths[resort_idx]
            display_scores = display_scores[resort_idx]
        else:
            display_paths = sorted_paths
            display_scores = sorted_scores

        # Draw paths from low to high density
        for i, (path, score) in enumerate(zip(display_paths, display_scores)):
            color = density_to_color(score, colorscale)
            # Opacity increases with density
            opacity = 0.2 + 0.6 * score
            # Width increases slightly with density
            width = 0.5 + 1.5 * score

            path_with_connection = np.concatenate([[last_hist_price], path])

            fig.add_trace(go.Scatter(
                x=x_forecast_plot,
                y=path_with_connection,
                mode='lines',
                line=dict(color=color, width=width),
                opacity=opacity,
                showlegend=False,
                hovertemplate=f'Density: {score:.3f}<br>Value: %{{y:.2f}}<extra></extra>'
            ))

    # Add percentile lines
    if show_percentiles:
        percentiles = [5, 25, 50, 75, 95]
        percentile_colors = {
            5: 'rgba(100, 100, 255, 0.8)',
            25: 'rgba(100, 150, 255, 0.8)',
            50: 'rgba(255, 100, 100, 1.0)',
            75: 'rgba(100, 150, 255, 0.8)',
            95: 'rgba(100, 100, 255, 0.8)'
        }
        percentile_widths = {5: 1, 25: 1.5, 50: 3, 75: 1.5, 95: 1}
        percentile_dashes = {5: 'dot', 25: 'dash', 50: 'solid', 75: 'dash', 95: 'dot'}

        percentile_values = np.percentile(paths, percentiles, axis=0)

        for i, p in enumerate(percentiles):
            pval = percentile_values[i]
            pval_with_connection = np.concatenate([[last_hist_price], pval])

            fig.add_trace(go.Scatter(
                x=x_forecast_plot,
                y=pval_with_connection,
                mode='lines',
                name=f'{p}th Percentile',
                line=dict(
                    color=percentile_colors[p],
                    width=percentile_widths[p],
                    dash=percentile_dashes[p]
                ),
                hovertemplate=f'{p}th Percentile<br>Value: %{{y:.2f}}<extra></extra>'
            ))

    # Add probability-weighted forecast
    if 'weighted_forecast' in result:
        weighted = result['weighted_forecast']
        weighted_with_connection = np.concatenate([[last_hist_price], weighted])
        fig.add_trace(go.Scatter(
            x=x_forecast_plot,
            y=weighted_with_connection,
            mode='lines',
            name='Weighted Forecast',
            line=dict(color='white', width=4),
            hovertemplate='Weighted Forecast<br>Value: %{y:.2f}<extra></extra>'
        ))
        # Add inner line for visibility
        fig.add_trace(go.Scatter(
            x=x_forecast_plot,
            y=weighted_with_connection,
            mode='lines',
            name='Weighted Forecast',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add historical prices
    fig.add_trace(go.Scatter(
        x=x_hist,
        y=prices,
        mode='lines',
        name='Historical',
        line=dict(color='white', width=3),
        hovertemplate='Historical<br>Value: %{y:.2f}<extra></extra>'
    ))

    # Layout with dark theme for better density visualization
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title='Price',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        plot_bgcolor='rgb(20, 20, 30)',
        paper_bgcolor='rgb(20, 20, 30)',
        font=dict(color='white'),
        hovermode='closest',
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white')
        )
    )

    # Add colorbar annotation
    if show_all_paths:
        fig.add_annotation(
            text=f"Path Density ({colorscale})<br>Yellow = High Probability<br>Purple = Low Probability",
            xref="paper", yref="paper",
            x=1.0, y=0.5,
            showarrow=False,
            font=dict(size=10, color='white'),
            bgcolor='rgba(0,0,0,0.5)',
            borderpad=5
        )

    return fig


def plot_forecast_interactive(
    prices: np.ndarray,
    result: dict,
    dates: np.ndarray = None,
    title: str = "Probability-Weighted Forecast",
    top_n_paths: int = 20,
    show_probability_cloud: bool = True,
    use_weighted_ci: bool = True,
    colorscale: str = 'Viridis',
    show_density: bool = True
) -> go.Figure:
    """
    Create interactive Plotly visualization with path density coloring.

    This is an enhanced version that shows path clusters and density.
    For the simplest visualization, use plot_forecast() instead.

    Args:
        prices: Historical price data
        result: Result dict from forecaster.predict()
        dates: Historical dates (optional)
        title: Chart title
        top_n_paths: Number of highest-probability paths to highlight
        show_probability_cloud: Show all paths as density cloud
        use_weighted_ci: Use probability-weighted confidence intervals
        colorscale: Color scale for density visualization
        show_density: Show density-based coloring (vs simple opacity)

    Returns:
        Plotly Figure object
    """
    # If show_density is True, use the new density-based plot
    if show_density:
        return plot_forecast(
            prices=prices,
            result=result,
            dates=dates,
            title=title,
            show_all_paths=show_probability_cloud,
            max_paths=500,
            colorscale=colorscale,
            show_percentiles=True,
            show_density_heatmap=False
        )

    # Otherwise, fall back to the original implementation
    from .utils import _ensure_numpy_array
    import polars as pl
    from datetime import datetime

    prices = _ensure_numpy_array(prices)
    paths = result['paths']
    probabilities = result['probabilities']
    weighted_forecast = result.get('weighted_forecast', result['forecast'])

    n_hist = len(prices)
    n_forecast = paths.shape[1]

    # Handle dates
    forecast_dates = result.get('dates', None)

    if dates is not None or forecast_dates is not None:
        if dates is not None:
            dates = _ensure_numpy_array(dates)
            x_hist = dates[-n_hist:]
            if len(x_hist) > 0 and np.issubdtype(x_hist.dtype, np.datetime64):
                x_hist = pd.to_datetime(x_hist)
        else:
            x_hist = np.arange(n_hist)

        if forecast_dates is not None:
            x_forecast = _ensure_numpy_array(forecast_dates)
            if len(x_forecast) > 0 and np.issubdtype(x_forecast.dtype, np.datetime64):
                x_forecast = pd.to_datetime(x_forecast)
        else:
            if dates is not None:
                last_date = x_hist[-1]
                if isinstance(last_date, (datetime, np.datetime64)):
                    last_date_pl = pl.Series([last_date]).cast(pl.Datetime).item()
                    x_forecast = pl.datetime_range(
                        start=last_date_pl,
                        end=None,
                        interval='1d',
                        eager=True
                    ).slice(1, n_forecast).to_numpy()
                    x_forecast = pd.to_datetime(x_forecast)
                else:
                    x_forecast = np.arange(n_forecast) + n_hist
            else:
                x_forecast = np.arange(n_forecast) + n_hist
    else:
        x_hist = np.arange(n_hist)
        x_forecast = np.arange(n_forecast) + n_hist

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=x_hist,
        y=prices,
        mode='lines',
        name='Historical',
        line=dict(color='black', width=2),
        hovertemplate='<b>Historical</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    # Connection point
    last_hist_price = prices[-1]
    if hasattr(x_hist, '__len__') and len(x_hist) > 0:
        last_hist_date = x_hist[-1]
        try:
            x_forecast_plot = np.concatenate([[last_hist_date], x_forecast])
        except (TypeError, Exception):
            x_forecast_plot = x_forecast
    else:
        x_forecast_plot = x_forecast

    # Probability cloud
    if show_probability_cloud:
        density_scores = compute_path_colors_by_density(paths, probabilities)
        sort_idx = np.argsort(density_scores)

        for i in sort_idx:
            score = density_scores[i]
            color = density_to_color(score, colorscale)
            opacity = 0.1 + 0.4 * score

            path_with_connection = np.concatenate([[last_hist_price], paths[i]])
            fig.add_trace(go.Scatter(
                x=x_forecast_plot,
                y=path_with_connection,
                mode='lines',
                line=dict(color=color, width=0.8),
                opacity=opacity,
                showlegend=False,
                hoverinfo='skip'
            ))

    # High-probability paths
    top_indices = np.argsort(probabilities)[-top_n_paths:][::-1]

    for i, idx in enumerate(top_indices):
        prob = probabilities[idx]
        path = paths[idx]
        width = 2.5 if i == 0 else max(2.0 - i * 0.05, 1.0)

        path_with_connection = np.concatenate([[last_hist_price], path])

        fig.add_trace(go.Scatter(
            x=x_forecast_plot,
            y=path_with_connection,
            mode='lines',
            name='High-Prob Paths' if i == 0 else None,
            line=dict(color=f'rgba(255, {100 + i * 5}, 0, 0.8)', width=width),
            hovertemplate=f'Path #{i+1}<br>Prob: {prob:.5f}<br>Value: %{{y:.2f}}<extra></extra>',
            legendgroup='high_prob',
            showlegend=(i == 0)
        ))

    # Weighted forecast
    weighted_with_connection = np.concatenate([[last_hist_price], weighted_forecast])
    fig.add_trace(go.Scatter(
        x=x_forecast_plot,
        y=weighted_with_connection,
        mode='lines',
        name='Weighted Forecast',
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate='<b>Weighted Forecast</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    # Confidence intervals
    if use_weighted_ci and 'weighted_upper' in result:
        upper_ci = result['weighted_upper']
        lower_ci = result['weighted_lower']
    else:
        upper_ci = result['upper']
        lower_ci = result['lower']

    upper_with_connection = np.concatenate([[last_hist_price], upper_ci])
    lower_with_connection = np.concatenate([[last_hist_price], lower_ci])

    fig.add_trace(go.Scatter(
        x=x_forecast_plot,
        y=upper_with_connection,
        mode='lines',
        name='95% CI',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_forecast_plot,
        y=lower_with_connection,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        name='95% Confidence',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title='Date', type='date' if dates is not None else None),
        yaxis_title='Value',
        hovermode='closest',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )

    return fig


class FractalVisualizer:
    """Creates interactive visualizations of fractal analysis and simulations."""

    @staticmethod
    def plot_cross_dimensional_analysis(
        prices: np.ndarray,
        volumes: np.ndarray,
        cross_dim_results: Dict,
        dates: np.ndarray = None
    ) -> go.Figure:
        """
        Create visualization of cross-dimensional fractal analysis.

        Args:
            prices: Price time series
            volumes: Volume time series
            cross_dim_results: Results from cross-dimensional analysis
            dates: Optional dates array

        Returns:
            Plotly figure with visualization
        """
        if dates is None:
            dates = np.arange(len(prices))

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Price and Volume",
                "Price-Volume Correlation",
                "Fractal Metrics by Dimension",
                "Regime Classification",
                "Cross-Dimensional Coherence",
                "Correlation Heatmap"
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "heatmap"}],
            ],
            column_widths=[0.6, 0.4],
            row_heights=[0.4, 0.3, 0.3]
        )

        # 1. Price and Volume plot
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                name="Price",
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                name="Volume",
                marker=dict(color='rgba(100,100,100,0.3)'),
                opacity=0.3
            ),
            row=1, col=1
        )

        # 2. Price-Volume correlation
        price_returns = np.diff(np.log(prices))
        volume_returns = np.diff(np.log(volumes + 1))

        window = min(30, len(price_returns) // 5)
        rolling_corr = np.zeros(len(price_returns) - window + 1)

        for i in range(len(rolling_corr)):
            if i + window <= len(price_returns):
                try:
                    corr = np.corrcoef(
                        price_returns[i:i+window],
                        volume_returns[i:i+window]
                    )[0, 1]
                    rolling_corr[i] = corr
                except:
                    rolling_corr[i] = 0

        corr_dates = dates[window:]
        fig.add_trace(
            go.Scatter(
                x=corr_dates,
                y=rolling_corr,
                name="P-V Correlation",
                line=dict(color='purple', width=1.5)
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[0, 0],
                name="Zero",
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Fractal Metrics
        fractal_dims = cross_dim_results.get('fractal_dimensions', {})
        hurst_exps = cross_dim_results.get('hurst_exponents', {})
        dimensions = list(fractal_dims.keys())

        fig.add_trace(
            go.Bar(
                x=dimensions,
                y=[fractal_dims.get(dim, 0) for dim in dimensions],
                name="Fractal Dimension",
                marker_color='blue'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(
                x=dimensions,
                y=[hurst_exps.get(dim, 0) for dim in dimensions],
                name="Hurst Exponent",
                marker_color='red'
            ),
            row=2, col=1
        )

        # 4. Regime Classification
        regime_info = cross_dim_results.get('regime', {})
        current_regime = regime_info.get('regime', 0)
        n_regimes = regime_info.get('n_regimes', 3)
        regime_names = ["Trending", "Mean-Reverting", "Random Walk"]

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_regime,
                title={"text": f"Current: {regime_names[current_regime]}"},
                gauge={
                    'axis': {'range': [0, n_regimes-1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 2], 'color': "orange"},
                        {'range': [2, 3], 'color': "lightgray"}
                    ]
                },
                delta={'reference': 1}
            ),
            row=2, col=2
        )

        # 5. Coherence
        coherence = cross_dim_results.get('fractal_coherence', {}).get('overall', 0)
        fig.add_trace(
            go.Bar(
                x=['Overall Coherence'],
                y=[coherence],
                name="Coherence",
                marker_color='green'
            ),
            row=3, col=1
        )

        # 6. Correlation Heatmap
        corr_matrix = np.array(cross_dim_results.get('cross_correlation', [[1, 0], [0, 1]]))
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=dimensions,
                y=dimensions,
                colorscale='Viridis',
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            ),
            row=3, col=2
        )

        fig.update_layout(
            title="Cross-Dimensional Fractal Analysis",
            height=1000,
            width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def plot_trading_time_analysis(
        prices: np.ndarray,
        time_map: Dict,
        dates: np.ndarray = None
    ) -> go.Figure:
        """Create visualization showing trading time vs clock time analysis."""
        if dates is None:
            dates = np.arange(len(prices))

        dilation_factors = time_map['dilation_factors']
        trading_time = time_map['trading_time_values']

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Price Series with Time Dilation Markers",
                "Trading Time vs Clock Time Mapping",
                "Time Dilation Factors"
            ),
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Color by dilation
        scaled_dilation = (dilation_factors - np.min(dilation_factors)) / (
            np.max(dilation_factors) - np.min(dilation_factors) + 1e-10
        )
        colors = [
            'rgba(0,0,255,0.7)' if s < 0.33 else
            'rgba(0,128,0,0.7)' if s < 0.66 else
            'rgba(255,0,0,0.7)'
            for s in scaled_dilation
        ]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name='Price',
                line=dict(width=1, color='rgba(100,100,100,0.8)')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode='markers',
                name='Time Dilation',
                marker=dict(
                    size=dilation_factors * 5,
                    color=colors,
                    symbol='circle'
                ),
                hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Dilation: %{marker.size:.2f}"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(trading_time)),
                y=trading_time,
                mode='lines',
                name='Trading Time',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )

        linear_time = np.linspace(0, trading_time[-1], len(trading_time))
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(trading_time)),
                y=linear_time,
                mode='lines',
                name='Linear Time',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dilation_factors,
                mode='lines',
                name='Time Dilation',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[1, 1],
                mode='lines',
                name='Neutral',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=3, col=1
        )

        fig.update_layout(
            title="Trading Time Analysis: Market Time Dilation",
            height=900,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def plot_analysis_and_forecast(
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray
    ) -> go.Figure:
        """
        Create comprehensive visualization with path density.

        This now uses the new density-based visualization.
        """
        paths, path_analysis = simulation_results

        # Create a result dict compatible with plot_forecast
        result = {
            'paths': paths,
            'probabilities': path_analysis.get('path_probabilities', np.ones(len(paths)) / len(paths)),
            'forecast': np.median(paths, axis=0),
            'weighted_forecast': path_analysis.get('most_likely_path', np.median(paths, axis=0)),
            'upper': np.percentile(paths, 95, axis=0),
            'lower': np.percentile(paths, 5, axis=0),
            'dates': pd.date_range(start=dates[-1], periods=paths.shape[1] + 1, freq='B')[1:]
        }

        # Use the new density-based plot
        fig = plot_forecast(
            prices=historical_prices,
            result=result,
            dates=dates,
            title="Fractal Pattern Analysis with Path Density",
            show_all_paths=True,
            colorscale='Viridis',
            show_percentiles=True
        )

        # Add statistics annotation
        stats_text = (
            f"<b>Fractal Statistics</b><br>"
            f"Hurst: {analysis_results.get('hurst', 0):.3f}<br>"
            f"Fractal Dim: {analysis_results.get('fractal_dim', 0):.3f}<br>"
            f"Paths: {paths.shape[0]:,}<br>"
            f"Horizon: {paths.shape[1]} steps"
        )

        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color='white'),
            bgcolor='rgba(0,0,0,0.7)',
            borderpad=8,
            align='left'
        )

        return fig

    def plot_quantum_analysis(self, prices, quantum_results, dates=None):
        """Plot quantum analysis results."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                "Price History",
                "Quantum Price Levels",
                "Multidimensional Fractal Analysis"
            ],
            vertical_spacing=0.1,
            row_heights=[0.3, 0.3, 0.4]
        )

        if dates is not None:
            fig.add_trace(go.Scatter(x=dates, y=prices, name="Price"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(y=prices, name="Price"), row=1, col=1)

        qpl = quantum_results['price_levels']['levels']
        for level in qpl:
            fig.add_trace(
                go.Scatter(
                    x=[0, len(prices)],
                    y=[level['price'], level['price']],
                    name=f"QPL: {level['price']:.2f}",
                    line=dict(dash="dash", width=1, color=f"rgba(255, 0, 0, {level['probability']:.2f})")
                ),
                row=2, col=1
            )

        multi_results = quantum_results['multi_dimensional']
        fig.add_trace(
            go.Heatmap(
                z=multi_results['cross_correlations'],
                x=['Price', 'Volume'],
                y=['Price', 'Volume'],
                colorscale='Viridis',
                name="Cross-Correlations"
            ),
            row=3, col=1
        )

        fig.update_layout(height=800, title_text="Quantum Fractal Analysis")
        return fig

    def plot_high_density_forecast(
        self,
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray
    ) -> go.Figure:
        """
        Create high-performance density visualization.

        This now uses the new unified density-based visualization.
        """
        paths, path_analysis = simulation_results

        result = {
            'paths': paths,
            'probabilities': path_analysis.get('path_probabilities', np.ones(len(paths)) / len(paths)),
            'forecast': np.median(paths, axis=0),
            'weighted_forecast': path_analysis.get('most_likely_path', np.median(paths, axis=0)),
            'upper': np.percentile(paths, 95, axis=0),
            'lower': np.percentile(paths, 5, axis=0),
            'dates': pd.date_range(start=dates[-1], periods=paths.shape[1] + 1, freq='B')[1:]
        }

        return plot_forecast(
            prices=historical_prices,
            result=result,
            dates=dates,
            title="High Density Path Visualization",
            show_all_paths=True,
            max_paths=500,
            colorscale='Plasma',
            show_percentiles=True,
            show_density_heatmap=False
        )


def print_forecast_summary(result: dict, current_price: float = None, show_paths: int = 5):
    """
    Print a nicely formatted summary of forecast results.

    Args:
        result: Result dictionary from forecaster.predict()
        current_price: Current/last price for comparison (optional)
        show_paths: Number of top probability paths to display (default 5)
    """
    import datetime

    if not isinstance(result, dict):
        raise TypeError(
            f"Expected 'result' to be a dict from forecaster.predict(), "
            f"but got {type(result).__name__}."
        )

    required_keys = ['forecast', 'weighted_forecast', 'paths', 'probabilities']
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        raise ValueError(f"Result missing required keys: {missing_keys}")

    print("\n" + "=" * 70)
    print("FORECAST SUMMARY")
    print("=" * 70)

    n_steps = len(result['forecast'])
    if 'dates' in result:
        dates = result['dates']
        print(f"\nPeriod: {dates[0]} to {dates[-1]} ({n_steps} steps)")
    else:
        print(f"\nSteps: {n_steps}")

    if current_price is not None:
        if isinstance(current_price, np.ndarray):
            current_price = float(current_price.item() if current_price.size == 1 else current_price[-1])
        print(f"Current Price: ${float(current_price):.2f}")

    print("\n" + "-" * 70)
    print("POINT FORECASTS (at final step)")
    print("-" * 70)

    final_median = result['forecast'][-1]
    final_weighted = result['weighted_forecast'][-1]
    final_mean = result['mean'][-1]

    print(f"  Median Forecast:           ${final_median:.2f}")
    print(f"  Probability-Weighted:      ${final_weighted:.2f}  <- Recommended")
    print(f"  Mean:                      ${final_mean:.2f}")

    if current_price is not None:
        change_pct = ((final_weighted - current_price) / current_price) * 100
        direction = "+" if change_pct > 0 else ""
        print(f"\n  Expected Change:           {direction}{change_pct:.2f}%")

    print("\n" + "-" * 70)
    print("95% CONFIDENCE INTERVALS")
    print("-" * 70)

    std_lower = result['lower'][-1]
    std_upper = result['upper'][-1]
    print(f"  Standard CI:      [${std_lower:.2f}, ${std_upper:.2f}]")

    if 'weighted_lower' in result:
        weighted_lower = result['weighted_lower'][-1]
        weighted_upper = result['weighted_upper'][-1]
        print(f"  Weighted CI:      [${weighted_lower:.2f}, ${weighted_upper:.2f}]  <- Recommended")

    print("\n" + "-" * 70)
    print(f"TOP {show_paths} MOST LIKELY PATHS")
    print("-" * 70)

    paths = result['paths']
    probs = result['probabilities']
    top_indices = np.argsort(probs)[-show_paths:][::-1]

    print(f"  {'Rank':<6} {'Probability':<15} {'Final Value':<15}")
    print("  " + "-" * 50)

    for rank, idx in enumerate(top_indices, 1):
        prob = probs[idx]
        final_val = paths[idx, -1]
        bar_length = int(prob * 1000)
        bar = "*" * min(bar_length, 30)
        print(f"  #{rank:<5} {prob:.6f}        ${final_val:>8.2f}       {bar}")

    print("\n" + "=" * 70)
