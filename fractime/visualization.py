"""
Visualization module for fractal time series analysis.

This module provides tools for creating interactive visualizations
of fractal patterns, forecasts, and analysis results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

        # 1. Price and Volume plot with shared axis

        # Create a secondary y-axis for volume
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                name="Price",
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )

        # Add volume as bars
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
        # Get log returns for price and volume
        price_returns = np.diff(np.log(prices))
        volume_returns = np.diff(np.log(volumes+1))  # Add 1 to avoid log(0)

        # Calculate rolling correlation
        window = min(30, len(price_returns)//5)
        rolling_corr = np.zeros(len(price_returns) - window + 1)

        for i in range(len(rolling_corr)):
            if i+window <= len(price_returns):
                try:
                    corr = np.corrcoef(
                        price_returns[i:i+window],
                        volume_returns[i:i+window]
                    )[0, 1]
                    rolling_corr[i] = corr
                except:
                    rolling_corr[i] = 0

        # Plot rolling correlation
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

        # Add zero reference line
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[0, 0],
                name="Zero Correlation",
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Fractal Metrics by Dimension
        fractal_dims = cross_dim_results.get('fractal_dimensions', {})
        hurst_exps = cross_dim_results.get('hurst_exponents', {})

        dimensions = list(fractal_dims.keys())

        # Fractal dimensions
        fig.add_trace(
            go.Bar(
                x=dimensions,
                y=[fractal_dims.get(dim, 0) for dim in dimensions],
                name="Fractal Dimension",
                marker_color='blue'
            ),
            row=2, col=1
        )

        # Add Hurst exponents
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
        confidence = regime_info.get('confidence', 0.5)
        n_regimes = regime_info.get('n_regimes', 3)

        # Define regime names
        regime_names = ["Trending", "Mean-Reverting", "Random Walk"]

        # Plot regime gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_regime,
                title={"text": f"Current Regime: {regime_names[current_regime]}"},
                gauge={
                    'axis': {'range': [0, n_regimes-1], 'tickvals': list(range(n_regimes))},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 2], 'color': "orange"},
                        {'range': [2, 3], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': current_regime
                    }
                },
                delta={'reference': 1, 'increasing': {'color': "green"}}
            ),
            row=2, col=2
        )

        # 5. Cross-Dimensional Coherence
        coherence = cross_dim_results.get('fractal_coherence', {}).get('overall', 0)

        # Create labels and values for coherence
        coherence_labels = ['Overall Coherence']
        coherence_values = [coherence]

        fig.add_trace(
            go.Bar(
                x=coherence_labels,
                y=coherence_values,
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

        # Update layout
        fig.update_layout(
            title="Cross-Dimensional Fractal Analysis",
            height=1000,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)

        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Correlation", row=1, col=2)

        fig.update_xaxes(title_text="Dimension", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)

        fig.update_xaxes(title_text="Metric", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=1)

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

        # Get time dilation factors
        dilation_factors = time_map['dilation_factors']
        trading_time = time_map['trading_time_values']

        # Create figure with subplots
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

        # Add price series with markers sized by time dilation
        # Colors for volatility: blue (slow) to red (fast time)
        colors = []
        scaled_dilation = (dilation_factors - np.min(dilation_factors)) / (np.max(dilation_factors) - np.min(dilation_factors))

        for i in range(len(scaled_dilation)):
            if scaled_dilation[i] < 0.33:
                colors.append('rgba(0,0,255,0.7)')  # Blue for slow regions
            elif scaled_dilation[i] < 0.66:
                colors.append('rgba(0,128,0,0.7)')  # Green for medium regions
            else:
                colors.append('rgba(255,0,0,0.7)')  # Red for fast regions

        # Plot price with markers
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

        # Add time dilation as markers on price
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode='markers',
                name='Time Dilation',
                marker=dict(
                    size=dilation_factors * 5,  # Scale marker size
                    color=colors,
                    symbol='circle',
                    line=dict(width=0)
                ),
                hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Time Dilation: %{marker.size:.2f}"
            ),
            row=1, col=1
        )

        # Plot trading time vs clock time
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

        # Add reference line (y=x)
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

        # Plot time dilation factors
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

        # Add a horizontal line at dilation = 1 (neutral)
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]],
                y=[1, 1],
                mode='lines',
                name='Neutral Time',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title="Trading Time Analysis: Market Time Dilation",
            height=900,
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Trading Time", row=2, col=1)
        fig.update_yaxes(title_text="Dilation Factor", row=3, col=1)

        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    @staticmethod
    def plot_analysis_and_forecast(
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray
    ) -> go.Figure:
        """Create comprehensive visualization with return distribution comparison."""
        paths, path_analysis = simulation_results

        # For very large path counts, use density visualization instead of individual paths
        use_density_plot = paths.shape[0] > 5000

        # Create future dates for forecast
        last_date = dates[-1]
        forecast_dates = pd.date_range(
            start=last_date,
            periods=paths.shape[1]+1,
            freq='B'
        )

        # Create subplots with proper specifications
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Forecast with Probability Cloud',
                'Return Distribution Comparison',
                'Historical Pattern Matches',
                'Fractal Statistics'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],  # First row: price plot and distribution
                [{"type": "xy"}, {"type": "table"}]  # Second row: patterns and stats
            ],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Plot historical prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=historical_prices,
                name='Historical',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        if use_density_plot:
            # DENSITY VISUALIZATION APPROACH - with professional styling and preserved volatility

            # First, select a few representative paths to show the jaggedness
            num_sample_paths = 3  # Show a few sample paths
            if paths.shape[0] > num_sample_paths:
                # Get some diverse paths by sampling from different percentiles
                sample_indices = []
                for p in np.linspace(25, 75, num_sample_paths):
                    # Find path closest to this percentile at the end point
                    target_value = np.percentile(paths[:, -1], p)
                    idx = np.argmin(np.abs(paths[:, -1] - target_value))
                    sample_indices.append(idx)

                sample_paths = paths[sample_indices]

            # Calculate percentiles at each time step
            percentiles = [5, 25, 50, 75, 95]
            percentile_paths = np.zeros((len(percentiles), paths.shape[1]))

            for t in range(paths.shape[1]):
                percentile_paths[:, t] = np.percentile(paths[:, t], percentiles)

            # Plot the 90% confidence band with shading
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[4]]),  # 95th
                    name='95th percentile',
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[0]]),  # 5th
                    name='5th percentile',
                    fill='tonexty',
                    fillcolor='rgba(255,82,82,0.15)',  # Very light red
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[3]]),  # 75th
                    name='75th percentile',
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[1]]),  # 25th
                    name='25th percentile',
                    fill='tonexty',
                    fillcolor='rgba(255,82,82,0.3)',  # Medium light red
                    line=dict(width=0.5, color='rgba(255,82,82,0.0)'),
                    showlegend=True
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], percentile_paths[2]]),  # 50th
                    name='Median forecast',
                    line=dict(color='rgb(255,82,82)', width=2)  # Solid red line
                ),
                row=1, col=1
            )

            # Add sample jagged paths to show volatility
            for i, path in enumerate(sample_paths):
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=np.concatenate([[historical_prices[-1]], path]),
                        name=f'Sample path {i+1}',
                        opacity=0.7,
                        line=dict(color='rgba(255,82,82,0.7)', width=0.7, dash='dot'),
                        showlegend=(i==0)  # Only show one in legend
                    ),
                    row=1, col=1
                )

            # Plot most likely path if available and different from median
            if 'most_likely_path' in path_analysis:
                most_likely = path_analysis['most_likely_path']
                # Only add if it's significantly different from the median
                median_path = percentile_paths[2]
                if np.mean(np.abs(most_likely - median_path)) > 0.01 * np.mean(median_path):
                    if 'cluster_probs' in path_analysis:
                        prob = path_analysis["cluster_probs"][np.argmax(path_analysis["cluster_probs"])]
                    else:
                        prob = np.max(path_analysis['path_probabilities'])

                    fig.add_trace(
                        go.Scatter(
                            x=forecast_dates,
                            y=np.concatenate([[historical_prices[-1]], most_likely]),
                            name=f'Most likely path ({prob:.0%})',
                            line=dict(color='rgb(128,0,0)', width=2, dash='dot')  # Darker red, dotted
                        ),
                        row=1, col=1
                    )
        else:
            # INDIVIDUAL PATHS APPROACH - for when there are fewer paths
            # Plot all paths with probability-based coloring
            cloud_paths = paths
            path_probs = path_analysis['path_probabilities']

            # Sort paths by probability
            sort_idx = np.argsort(path_probs)
            cloud_paths = cloud_paths[sort_idx]
            path_probs = path_probs[sort_idx]

            # Find 95th percentile probability to identify top paths
            prob_95th = np.percentile(path_probs, 95)

            # Plot paths from lowest to highest probability
            for i, path in enumerate(cloud_paths):
                prob = path_probs[i]

                # Aggressive exponential scaling to emphasize only the highest probability paths
                # Scale relative to 95th percentile for better highlighting
                rel_prob = (prob / prob_95th) ** 3  # Cube for more aggressive scaling

                if prob > prob_95th:
                    # Top 5% paths: Purple to Red transition
                    r = 1.0
                    b = max(0.0, 1.0 - (prob - prob_95th) / (np.max(path_probs) - prob_95th))
                    opacity = 0.8
                    width = 2.0
                else:
                    # Bottom 95% paths: Light blue with very low opacity
                    r = 0.0
                    b = 1.0
                    opacity = 0.03 + 0.07 * rel_prob  # Very subtle
                    width = 0.5

                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=np.concatenate([[historical_prices[-1]], path]),
                        name=f'Path (p={prob:.1%})',
                        line=dict(
                            color=f'rgba({int(r*255)},0,{int(b*255)},{opacity})',
                            width=width
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # Plot most likely path if available
            if 'most_likely_path' in path_analysis:
                most_likely = path_analysis['most_likely_path']
                if 'cluster_probs' in path_analysis:
                    prob = path_analysis["cluster_probs"][np.argmax(path_analysis["cluster_probs"])]
                else:
                    prob = np.max(path_probs)

                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=np.concatenate([[historical_prices[-1]], most_likely]),
                        name=f'Most Likely Path ({prob:.0%})',
                        line=dict(color='red', width=3)
                    ),
                    row=1, col=1
                )

        # 2. Return distribution comparison
        historical_returns = np.diff(np.log(historical_prices))
        all_forecast_returns = np.diff(np.log(paths), axis=1).flatten()

        # Plot return distributions
        fig.add_trace(
            go.Histogram(
                x=historical_returns,
                name='Historical Returns',
                nbinsx=30,
                opacity=0.7,
                histnorm='probability',
                marker_color='blue'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Histogram(
                x=all_forecast_returns,
                name='Forecast Returns',
                nbinsx=30,
                opacity=0.7,
                histnorm='probability',
                marker_color='red'
            ),
            row=1, col=2
        )

        # Calculate distribution statistics
        hist_mean = np.mean(historical_returns)
        hist_std = np.std(historical_returns)
        fore_mean = np.mean(all_forecast_returns)
        fore_std = np.std(all_forecast_returns)

        # Add distribution statistics annotation
        stats_text = (
            f"Historical: μ={hist_mean:.3%}, σ={hist_std:.3%}<br>"
            f"Forecast: μ={fore_mean:.3%}, σ={fore_std:.3%}"
        )

        fig.add_annotation(
            text=stats_text,
            xref="x2", yref="y2",
            x=min(min(historical_returns), min(all_forecast_returns)),
            y=0.9,
            showarrow=False,
            font=dict(size=10)
        )

        # Update return distribution layout
        fig.update_xaxes(
            title_text="Returns",
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Probability",
            row=1, col=2
        )

        # 3. Add pattern matches if available
        if 'pattern_matches' in path_analysis and len(path_analysis['pattern_matches']) > 0:
            pattern_matches = path_analysis['pattern_matches']
            for i, pattern in enumerate(pattern_matches[:5]):  # Show up to 5 patterns
                match_start = pattern['historical_idx']
                match_end = match_start + pattern['segment_length']
                match_pattern = historical_prices[match_start:match_end]

                fig.add_trace(
                    go.Scatter(
                        x=dates[match_start:match_end],
                        y=match_pattern,
                        name=f'Pattern {i+1}',
                        line=dict(dash='dot')
                    ),
                    row=2, col=1
                )

        # 4. Statistics table
        prob = 0
        if 'cluster_probs' in path_analysis and len(path_analysis['cluster_probs']) > 0:
            prob = max(path_analysis['cluster_probs'])
        elif 'path_probabilities' in path_analysis:
            prob = max(path_analysis['path_probabilities'])

        stats_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Hurst Exponent', f"{analysis_results['hurst']:.3f}", '>0.5 suggests trend persistence'],
            ['Fractal Dimension', f"{analysis_results['fractal_dim']:.3f}", 'higher = more volatile'],
            ['Pattern Matches', str(len(path_analysis.get('pattern_matches', []))), 'matching forecast paths'],
            ['Forecast Paths', f"{paths.shape[0]:,}", 'simulated trajectories'],
            ['Forecast Horizon', f"{paths.shape[1]} steps", 'prediction length']
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>', '<b>Interpretation</b>'],
                    font=dict(size=12),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*stats_data)),
                    font=dict(size=11),
                    align='left'
                )
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            title_text="Fractal Pattern Analysis and Forecast",
            title_x=0.5
        )

        # Add explanatory annotations
        fig.add_annotation(
            text="Gray cloud shows possible paths<br>Red line shows most likely trajectory",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10)
        )

        fig.add_annotation(
            text="Historical patterns matching<br>forecast trajectories",
            xref="paper", yref="paper",
            x=0.02, y=0.45,
            showarrow=False,
            font=dict(size=10)
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

        # Add price history
        if dates is not None:
            fig.add_trace(go.Scatter(x=dates, y=prices, name="Price"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(y=prices, name="Price"), row=1, col=1)

        # Add quantum price levels
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

        # Add hurst exponent heatmap
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

        fig.update_layout(
            height=800,
            title_text="Quantum Fractal Analysis"
        )

        return fig

    def plot_high_density_forecast(
        self,
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray
    ) -> go.Figure:
        """Create high-performance density visualization showing sample paths with percentile coloring."""
        paths, path_analysis = simulation_results

        # Create future dates for forecast
        last_date = dates[-1]
        forecast_dates = pd.date_range(
            start=last_date,
            periods=paths.shape[1]+1,
            freq='B'
        )

        # Create figure
        fig = go.Figure()

        # Add historical prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=historical_prices,
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )

        # Calculate percentiles at each time step
        percentiles = [5, 25, 50, 75, 95]
        percentile_paths = np.zeros((len(percentiles), paths.shape[1]))

        for t in range(paths.shape[1]):
            percentile_paths[:, t] = np.percentile(paths[:, t], percentiles)

        # Define color scale
        def get_color(percentile, opacity=0.15):
            """Get color based on percentile (blue->green->red)."""
            if percentile < 50:
                # Blue (0,0,255) to Green (0,255,0)
                ratio = percentile / 50
                r = 0
                g = int(255 * ratio)
                b = int(255 * (1 - ratio))
            else:
                # Green (0,255,0) to Red (255,0,0)
                ratio = (percentile - 50) / 50
                r = int(255 * ratio)
                g = int(255 * (1 - ratio))
                b = 0
            return f'rgba({r},{g},{b},{opacity})'

        # Add sample paths - limit to 100 for performance
        max_samples = min(100, paths.shape[0])
        sample_indices = np.linspace(0, paths.shape[0]-1, max_samples, dtype=int)

        # Calculate final values for each path to determine percentiles
        final_values = paths[sample_indices, -1]
        ranks = np.argsort(np.argsort(final_values)) / len(final_values) * 100

        # Plot sampled paths
        for i, idx in enumerate(sample_indices):
            percentile = ranks[i]
            color = get_color(percentile)
            path = paths[idx]

            # No legend entries for individual paths
            fig.add_trace(
                go.Scatter(  # Use regular Scatter instead of Scattergl for compatibility
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], path]),
                    name=f'Path {i+1}',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='none'  # Disable hover for performance
                )
            )

        # Add shaded areas for percentile bands
        # 5-95 percentile band
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[4]]),  # 95th
                name='95th percentile',
                line=dict(color='rgba(255,0,0,0.5)', width=1.5, dash='dot'),
                showlegend=True
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[0]]),  # 5th
                name='5th percentile',
                fill='tonexty',
                fillcolor='rgba(150,150,150,0.2)',  # Light gray
                line=dict(color='rgba(0,0,255,0.5)', width=1.5, dash='dot'),
                showlegend=True
            )
        )

        # 25-75 percentile band (interquartile range)
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[3]]),  # 75th
                name='75th percentile',
                line=dict(color='rgba(255,0,0,0.8)', width=1.5),
                showlegend=True
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[1]]),  # 25th
                name='25th percentile',
                fill='tonexty',
                fillcolor='rgba(150,150,150,0.4)',  # Darker gray
                line=dict(color='rgba(0,0,255,0.8)', width=1.5),
                showlegend=True
            )
        )

        # Median path
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.concatenate([[historical_prices[-1]], percentile_paths[2]]),  # 50th
                name='Median forecast',
                line=dict(color='purple', width=2),
                showlegend=True
            )
        )

        # Add most likely path if available
        if 'most_likely_path' in path_analysis:
            most_likely = path_analysis['most_likely_path']

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=np.concatenate([[historical_prices[-1]], most_likely]),
                    name='Most likely path',
                    line=dict(color='black', width=2.5),
                    showlegend=True
                )
            )

        # Configure layout
        fig.update_layout(
            title="High Density Path Visualization",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig


def plot_forecast(prices: np.ndarray,
                 forecast: np.ndarray = None,
                 paths: np.ndarray = None,
                 confidence_intervals: dict = None,
                 title: str = "Fractal Forecast",
                 dates: np.ndarray = None,
                 show_patterns: bool = False):
    """
    Simple, clean plotting function for forecasts.

    Args:
        prices: Historical price data
        forecast: Point forecast (optional)
        paths: Simulated paths array (optional, shape: n_paths x n_steps)
        confidence_intervals: Dict with 'lower' and 'upper' bounds (optional)
        title: Plot title
        dates: Date array for x-axis (optional)
        show_patterns: Show path distribution (default False)

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import polars as pl
    from datetime import datetime, timedelta
    from .utils import _ensure_numpy_array

    prices = _ensure_numpy_array(prices)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare x-axis
    n_hist = len(prices)
    if dates is not None:
        dates = _ensure_numpy_array(dates)
        x_hist = dates[-n_hist:]

        # Generate future dates
        if forecast is not None or paths is not None:
            n_forecast = len(forecast) if forecast is not None else paths.shape[1]
            last_date = x_hist[-1]

            # Check if date is datetime-like (datetime, np.datetime64)
            if isinstance(last_date, (datetime, np.datetime64)):
                # Use polars for date range generation
                last_date_pl = pl.Series([last_date]).cast(pl.Datetime)[0]
                x_forecast = pl.datetime_range(
                    start=last_date_pl,
                    end=None,
                    interval='1d',
                    eager=True
                ).slice(1, n_forecast).to_numpy()
            else:
                x_forecast = np.arange(n_forecast) + n_hist
    else:
        x_hist = np.arange(n_hist)
        if forecast is not None or paths is not None:
            n_forecast = len(forecast) if forecast is not None else paths.shape[1]
            x_forecast = np.arange(n_forecast) + n_hist

    # Plot historical prices
    ax.plot(x_hist, prices, 'k-', linewidth=2, label='Historical', alpha=0.8)

    # Plot paths if provided
    if paths is not None:
        paths = _ensure_numpy_array(paths)
        if show_patterns:
            # Show all paths with transparency
            for i in range(min(100, len(paths))):
                ax.plot(x_forecast, paths[i], 'b-', alpha=0.05, linewidth=0.5)
        else:
            # Show percentile bands
            p10 = np.percentile(paths, 10, axis=0)
            p90 = np.percentile(paths, 90, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p75 = np.percentile(paths, 75, axis=0)

            ax.fill_between(x_forecast, p10, p90, alpha=0.2, color='blue', label='80% Range')
            ax.fill_between(x_forecast, p25, p75, alpha=0.3, color='blue', label='50% Range')

    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        lower = _ensure_numpy_array(confidence_intervals['lower'])
        upper = _ensure_numpy_array(confidence_intervals['upper'])
        ax.fill_between(x_forecast, lower, upper, alpha=0.3, color='green', label='95% CI')

    # Plot forecast
    if forecast is not None:
        forecast = _ensure_numpy_array(forecast)
        ax.plot(x_forecast, forecast, 'r-', linewidth=2, label='Forecast', alpha=0.8)

        # Connect historical to forecast
        ax.plot([x_hist[-1], x_forecast[0]], [prices[-1], forecast[0]],
                'r--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Format dates if using datetime
    if dates is not None and isinstance(x_hist[0], (datetime, np.datetime64)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def plot_forecast_interactive(
    prices: np.ndarray,
    result: dict,
    dates: np.ndarray = None,
    title: str = "Probability-Weighted Forecast",
    top_n_paths: int = 20,
    show_probability_cloud: bool = True,
    use_weighted_ci: bool = True
):
    """
    Create interactive Plotly visualization showing probability-weighted forecast paths.

    Shows high-probability paths as distinct "branches" with probability clouds,
    making it easy to see which futures are most likely based on fractal similarity.

    Visualization features:
    - Light blue probability cloud (all paths with opacity by probability)
    - High-probability paths in orange-red gradient (darker = higher probability)
    - Clean lines only (no markers) for easy viewing
    - Line thickness varies by probability rank
    - Top 3 paths labeled with exact probabilities

    Args:
        prices: Historical price data
        result: Result dict from forecaster.predict() (must include 'paths' and 'probabilities')
        dates: Historical date array for x-axis (optional, auto-extracted from result if available)
        title: Chart title
        top_n_paths: Number of highest-probability paths to show clearly (default 20)
        show_probability_cloud: Show probability density cloud (default True)
        use_weighted_ci: Use probability-weighted CI instead of simple quantiles (default True)

    Returns:
        Plotly figure object (call .show() to display or .write_html() to save)

    Example:
        >>> # Dates handled automatically
        >>> forecaster.fit(prices, dates=historical_dates)
        >>> result = forecaster.predict(end_date='2025-11-27')
        >>> fig = ft.plot_forecast_interactive(prices, result)
        >>> fig.show()
    """
    import polars as pl
    from datetime import datetime
    from .utils import _ensure_numpy_array

    prices = _ensure_numpy_array(prices)
    paths = result['paths']
    probabilities = result['probabilities']
    weighted_forecast = result.get('weighted_forecast', result['forecast'])

    n_hist = len(prices)
    n_forecast = paths.shape[1]

    # Auto-extract forecast dates from result if available
    forecast_dates = result.get('dates', None)

    # Prepare x-axis
    if dates is not None or forecast_dates is not None:
        # Use provided dates or generate if we have historical dates
        if dates is not None:
            dates = _ensure_numpy_array(dates)
            x_hist = dates[-n_hist:]
            # Convert numpy datetime64 to pandas Timestamp for Plotly compatibility
            if len(x_hist) > 0 and np.issubdtype(x_hist.dtype, np.datetime64):
                x_hist = pd.to_datetime(x_hist)
        else:
            # No historical dates provided, use indices
            x_hist = np.arange(n_hist)

        # Use forecast dates from result
        if forecast_dates is not None:
            x_forecast = _ensure_numpy_array(forecast_dates)
            # Convert numpy datetime64 to pandas Timestamp for Plotly compatibility
            if len(x_forecast) > 0 and np.issubdtype(x_forecast.dtype, np.datetime64):
                x_forecast = pd.to_datetime(x_forecast)
        else:
            # Try to generate from historical dates
            if dates is not None:
                last_date = x_hist[-1]
                if isinstance(last_date, (datetime, np.datetime64)):
                    # Use polars for date range generation
                    last_date_pl = pl.Series([last_date]).cast(pl.Datetime).item()
                    x_forecast = pl.datetime_range(
                        start=last_date_pl,
                        end=None,
                        interval='1d',
                        eager=True
                    ).slice(1, n_forecast).to_numpy()
                    # Convert to pandas Timestamp for Plotly compatibility
                    x_forecast = pd.to_datetime(x_forecast)
                else:
                    x_forecast = np.arange(n_forecast) + n_hist
            else:
                x_forecast = np.arange(n_forecast) + n_hist
    else:
        x_hist = np.arange(n_hist)
        x_forecast = np.arange(n_forecast) + n_hist

    # Create figure
    fig = go.Figure()

    # 1. Historical data
    fig.add_trace(go.Scatter(
        x=x_hist,
        y=prices,
        mode='lines',
        name='Historical',
        line=dict(color='black', width=2),
        hovertemplate='<b>Historical</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    # Prepare forecast x-axis with connection point
    # Prepend last historical date to forecast dates for visual continuity
    last_hist_price = prices[-1]
    if hasattr(x_hist, '__len__') and len(x_hist) > 0:
        last_hist_date = x_hist[-1]
        # Check if dtypes are compatible before concatenating
        # (e.g., both datetime or both numeric)
        try:
            # Attempt to concatenate - will fail if dtypes incompatible
            x_forecast_plot = np.concatenate([[last_hist_date], x_forecast])
        except (TypeError, Exception) as e:
            # Dtypes incompatible (e.g., int vs datetime) - don't prepend
            # Catches DTypePromotionError and other concatenation errors
            if 'DType' in str(type(e).__name__) or 'promoted' in str(e):
                x_forecast_plot = x_forecast
            else:
                raise
    else:
        x_forecast_plot = x_forecast

    # 2. Probability cloud (all paths with low opacity)
    if show_probability_cloud:
        # Show all paths as light background cloud
        for i in range(len(paths)):
            # Opacity based on probability (more visible for higher probability)
            opacity = min(probabilities[i] * 2000, 0.9)  # Increased multiplier and cap for better visibility
            # Prepend last historical price for visual continuity
            path_with_connection = np.concatenate([[last_hist_price], paths[i]])
            fig.add_trace(go.Scatter(
                x=x_forecast_plot,
                y=path_with_connection,
                mode='lines',
                line=dict(color='lightblue', width=1.2),  # Increased width
                opacity=opacity,
                showlegend=False,
                hoverinfo='skip'
            ))

    # 3. High-probability paths (top N)
    top_indices = np.argsort(probabilities)[-top_n_paths:][::-1]  # Highest first

    # Color gradient for high-probability paths
    # Use orange-red gradient to distinguish from blue cloud
    colors = []
    for i, idx in enumerate(top_indices):
        # Gradient from dark orange/red (highest) to lighter orange (lowest of top N)
        intensity = 1.0 - (i / top_n_paths) * 0.5  # 1.0 to 0.5
        r = int(255 * intensity)
        g = int(140 * (1 - intensity * 0.5))  # Varies from ~70 to 140
        b = 0
        colors.append(f'rgba({r}, {g}, {b}, 0.8)')

    for i, idx in enumerate(top_indices):
        prob = probabilities[idx]
        path = paths[idx]
        final_value = path[-1]

        # Width based on rank (thicker for higher probability)
        width = 2.5 if i == 0 else max(2.0 - i * 0.05, 1.0)

        # Prepend last historical price for visual continuity
        path_with_connection = np.concatenate([[last_hist_price], path])

        fig.add_trace(go.Scatter(
            x=x_forecast_plot,
            y=path_with_connection,
            mode='lines',
            name='High-Probability Paths',  # Same name for all - groups in legend
            line=dict(color=colors[i], width=width),
            hovertemplate=f'<b>Path #{i+1}</b><br>' +
                         f'Probability: {prob:.5f}<br>' +
                         f'Value: %{{y:.2f}}<br>' +
                         f'Final: {final_value:.2f}<extra></extra>',
            legendgroup='high_prob',
            showlegend=(i == 0)  # Only show legend for first path
        ))

    # 4. Weighted forecast
    # Prepend last historical price for visual continuity
    weighted_with_connection = np.concatenate([[last_hist_price], weighted_forecast])
    fig.add_trace(go.Scatter(
        x=x_forecast_plot,
        y=weighted_with_connection,
        mode='lines',
        name='Weighted Forecast',
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate='<b>Weighted Forecast</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    # 5. Confidence intervals
    # Use probability-weighted CI if requested and available
    if use_weighted_ci and 'weighted_upper' in result and 'weighted_lower' in result:
        upper_ci = result['weighted_upper']
        lower_ci = result['weighted_lower']
        ci_label = '95% Weighted CI'
    else:
        upper_ci = result['upper']
        lower_ci = result['lower']
        ci_label = '95% CI'

    # Prepend last historical price for visual continuity
    upper_with_connection = np.concatenate([[last_hist_price], upper_ci])
    lower_with_connection = np.concatenate([[last_hist_price], lower_ci])

    fig.add_trace(go.Scatter(
        x=x_forecast_plot,
        y=upper_with_connection,
        mode='lines',
        name=ci_label,
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
        name=ci_label.replace('CI', 'Confidence'),
        hoverinfo='skip'
    ))

    # 6. Annotations for top 3 paths at final point
    for i in range(min(3, len(top_indices))):
        idx = top_indices[i]
        prob = probabilities[idx]
        final_value = paths[idx, -1]

        fig.add_annotation(
            x=x_forecast[-1],
            y=final_value,
            text=f"#{i+1}: {prob:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=colors[i],
            ax=40 if i == 0 else 35,
            ay=-30 * (i+1),
            font=dict(size=10, color=colors[i], family='monospace'),
            bgcolor='white',
            bordercolor=colors[i],
            borderwidth=1,
            borderpad=2
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family='Arial Black')
        ),
        xaxis=dict(
            title='Date',
            type='date',
            tickformat='%Y-%m-%d',
            tickangle=-45
        ),
        yaxis_title='Value',
        hovermode='closest',
        template='plotly_white',
        height=600,
        autosize=True,  # Responsive width for Jupyter
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        font=dict(family='Arial', size=12)
    )

    return fig


def print_forecast_summary(result: dict, current_price: float = None, show_paths: int = 5):
    """
    Print a nicely formatted summary of forecast results.

    Args:
        result: Result dictionary from forecaster.predict()
        current_price: Current/last price for comparison (optional)
        show_paths: Number of top probability paths to display (default 5)

    Example:
        >>> result = forecaster.predict(end_date='2025-11-27')
        >>> ft.print_forecast_summary(result, current_price=prices[-1])
    """
    import datetime

    # Validate input type
    if not isinstance(result, dict):
        raise TypeError(
            f"Expected 'result' to be a dict from forecaster.predict(), "
            f"but got {type(result).__name__}. "
            f"Usage: ft.print_forecast_summary(result, current_price=prices[-1])"
        )

    # Check for required keys
    required_keys = ['forecast', 'weighted_forecast', 'paths', 'probabilities']
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        raise ValueError(
            f"Result dictionary is missing required keys: {missing_keys}. "
            f"Make sure you're passing the result from forecaster.predict()"
        )

    print("\n" + "=" * 70)
    print("FORECAST SUMMARY")
    print("=" * 70)

    # Forecast period
    n_steps = len(result['forecast'])
    if 'dates' in result:
        dates = result['dates']
        print(f"\nPeriod: {dates[0]} to {dates[-1]} ({n_steps} steps)")
    else:
        print(f"\nSteps: {n_steps}")

    # Current price comparison
    if current_price is not None:
        # Ensure it's a scalar (not an array)
        if isinstance(current_price, np.ndarray):
            current_price = float(current_price.item() if current_price.size == 1 else current_price[-1])
        print(f"Current Price: ${float(current_price):.2f}")

    # Point forecasts
    print("\n" + "-" * 70)
    print("POINT FORECASTS (at final step)")
    print("-" * 70)

    final_median = result['forecast'][-1]
    final_weighted = result['weighted_forecast'][-1]
    final_mean = result['mean'][-1]

    print(f"  Median Forecast:           ${final_median:.2f}")
    print(f"  Probability-Weighted:      ${final_weighted:.2f}  ← Recommended")
    print(f"  Mean:                      ${final_mean:.2f}")

    if current_price is not None:
        change_pct = ((final_weighted - current_price) / current_price) * 100
        direction = "↑" if change_pct > 0 else "↓"
        print(f"\n  Expected Change:           {direction} {abs(change_pct):.2f}%")

    # Confidence intervals
    print("\n" + "-" * 70)
    print("95% CONFIDENCE INTERVALS (at final step)")
    print("-" * 70)

    std_lower = result['lower'][-1]
    std_upper = result['upper'][-1]
    std_width = std_upper - std_lower

    print(f"  Standard CI:      [${std_lower:.2f}, ${std_upper:.2f}]  (width: ${std_width:.2f})")

    if 'weighted_lower' in result and 'weighted_upper' in result:
        weighted_lower = result['weighted_lower'][-1]
        weighted_upper = result['weighted_upper'][-1]
        weighted_width = weighted_upper - weighted_lower

        print(f"  Weighted CI:      [${weighted_lower:.2f}, ${weighted_upper:.2f}]  (width: ${weighted_width:.2f})  ← Recommended")

        width_diff = ((weighted_width - std_width) / std_width) * 100
        if abs(width_diff) > 1:
            print(f"\n  Weighted CI is {abs(width_diff):.1f}% {'narrower' if width_diff < 0 else 'wider'}")

    # Statistics
    print("\n" + "-" * 70)
    print("STATISTICS")
    print("-" * 70)

    final_std = result['std'][-1]
    print(f"  Standard Deviation:        ${final_std:.2f}")
    print(f"  Number of Paths:           {len(result['probabilities'])}")

    # Path probabilities
    print("\n" + "-" * 70)
    print(f"TOP {show_paths} MOST LIKELY PATHS")
    print("-" * 70)

    paths = result['paths']
    probs = result['probabilities']

    # Sort by probability
    top_indices = np.argsort(probs)[-show_paths:][::-1]

    print(f"  {'Rank':<6} {'Probability':<15} {'Final Value':<15} {'Change':<10}")
    print("  " + "-" * 60)

    for rank, idx in enumerate(top_indices, 1):
        prob = probs[idx]
        final_val = paths[idx, -1]

        if current_price is not None:
            change = ((final_val - current_price) / current_price) * 100
            change_str = f"{change:+.2f}%"
        else:
            change_str = "-"

        # Visual probability bar
        bar_length = int(prob * 1000)  # Scale up for visibility
        bar = "█" * min(bar_length, 50)

        print(f"  #{rank:<5} {prob:.6f} ({prob*100:.3f}%)  ${final_val:>8.2f}  {change_str:>9}  {bar}")

    # Visual forecast range
    print("\n" + "-" * 70)
    print("FORECAST RANGE VISUALIZATION")
    print("-" * 70)

    if 'weighted_lower' in result and 'weighted_upper' in result:
        lower = result['weighted_lower'][-1]
        upper = result['weighted_upper'][-1]
        ci_label = "Weighted 95% CI"
    else:
        lower = result['lower'][-1]
        upper = result['upper'][-1]
        ci_label = "Standard 95% CI"

    forecast = final_weighted

    # Create ASCII visualization
    range_width = upper - lower
    if range_width > 0:
        # Scale to 60 characters
        scale = 60 / range_width

        lower_pos = 0
        forecast_pos = int((forecast - lower) * scale)
        upper_pos = 60

        # Build visualization
        viz = [" "] * 61
        viz[lower_pos] = "["
        viz[upper_pos] = "]"
        viz[forecast_pos] = "●"

        print(f"\n  ${lower:.2f}  {''.join(viz)}  ${upper:.2f}")
        print(f"  {ci_label}: {''.join([' '] * forecast_pos)}↑")
        print(f"  {' ' * (len(ci_label) + 2)}{''.join([' '] * forecast_pos)}Forecast: ${forecast:.2f}")

    print("\n" + "=" * 70)
