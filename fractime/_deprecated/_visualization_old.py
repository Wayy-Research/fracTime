"""
Visualization module for fractal time series analysis.

This module provides interactive visualizations using wrchart
for fractal patterns, forecasts, and analysis results.
All plots are interactive by default with TradingView-style aesthetics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union

import wrchart as wrc
from wrchart import ForecastChart, MultiPanelChart
from wrchart.multipanel import LinePanel, BarPanel, HeatmapPanel, GaugePanel, AreaPanel
from wrchart.forecast.utils import (
    compute_path_density,
    compute_path_colors_by_density,
)


def plot_forecast(
    prices: np.ndarray,
    result: dict,
    dates: np.ndarray = None,
    title: str = "Fractal Forecast with Path Density",
    show_all_paths: bool = True,
    max_paths: int = 500,
    colorscale: str = 'viridis',
    show_percentiles: bool = True,
    show_density_heatmap: bool = False,
) -> ForecastChart:
    """
    Create interactive forecast visualization with path density coloring.

    Args:
        prices: Historical price data
        result: Result dict from forecaster.predict()
        dates: Historical dates (optional)
        title: Chart title
        show_all_paths: Show individual paths colored by density
        max_paths: Maximum paths to display (for performance)
        colorscale: Color scale for density ('viridis', 'plasma', 'inferno', 'hot')
        show_percentiles: Show percentile lines
        show_density_heatmap: Show 2D density heatmap overlay (not yet supported)

    Returns:
        ForecastChart object (call .show() to display, .to_html() to save)
    """
    from .utils import _ensure_numpy_array

    prices = _ensure_numpy_array(prices)

    chart = ForecastChart(
        width=1000,
        height=700,
        theme="dark",
        title=title,
    )

    chart.set_data(prices, result, dates)
    chart.colorscale(colorscale)
    chart.max_paths(max_paths)
    chart.show_percentiles(show_percentiles)

    return chart


def plot_forecast_interactive(
    prices: np.ndarray,
    result: dict,
    dates: np.ndarray = None,
    title: str = "Probability-Weighted Forecast",
    top_n_paths: int = 20,
    show_probability_cloud: bool = True,
    use_weighted_ci: bool = True,
    colorscale: str = 'viridis',
    show_density: bool = True,
) -> ForecastChart:
    """
    Create interactive visualization with path density coloring.

    This is an enhanced version that shows path clusters and density.

    Args:
        prices: Historical price data
        result: Result dict from forecaster.predict()
        dates: Historical dates (optional)
        title: Chart title
        top_n_paths: Number of highest-probability paths to highlight
        show_probability_cloud: Show all paths as density cloud
        use_weighted_ci: Use probability-weighted confidence intervals
        colorscale: Color scale for density visualization
        show_density: Show density-based coloring

    Returns:
        ForecastChart object (call .show() to display)
    """
    return plot_forecast(
        prices=prices,
        result=result,
        dates=dates,
        title=title,
        show_all_paths=show_probability_cloud,
        max_paths=500,
        colorscale=colorscale,
        show_percentiles=True,
    )


class FractalVisualizer:
    """Creates interactive visualizations of fractal analysis and simulations."""

    @staticmethod
    def plot_cross_dimensional_analysis(
        prices: np.ndarray,
        volumes: np.ndarray,
        cross_dim_results: Dict,
        dates: np.ndarray = None,
    ) -> MultiPanelChart:
        """
        Create visualization of cross-dimensional fractal analysis.

        Args:
            prices: Price time series
            volumes: Volume time series
            cross_dim_results: Results from cross-dimensional analysis
            dates: Optional dates array

        Returns:
            MultiPanelChart object (call .show() to display)
        """
        if dates is None:
            dates = np.arange(len(prices))

        # Create chart
        chart = MultiPanelChart(
            rows=3,
            cols=2,
            width=1200,
            height=1000,
            title="Cross-Dimensional Fractal Analysis",
            theme="dark",
            row_heights=[0.4, 0.3, 0.3],
            col_widths=[0.6, 0.4],
        )

        # 1. Price and Volume (top left)
        x_data = list(range(len(prices)))
        chart.add_panel(LinePanel(
            title="Price and Volume",
            x_data=x_data,
            y_data=[prices.tolist(), volumes.tolist()],
            colors=["#1976d2", "rgba(100, 100, 100, 0.3)"],
            line_widths=[1.5, 0.5],
            labels=["Price", "Volume"],
            row=0, col=0,
        ))

        # 2. Price-Volume Correlation (top right)
        price_returns = np.diff(np.log(prices))
        volume_returns = np.diff(np.log(volumes + 1))
        window = min(30, len(price_returns) // 5)
        rolling_corr = []

        for i in range(len(price_returns) - window + 1):
            try:
                corr = np.corrcoef(
                    price_returns[i:i+window],
                    volume_returns[i:i+window]
                )[0, 1]
                rolling_corr.append(corr if not np.isnan(corr) else 0)
            except:
                rolling_corr.append(0)

        chart.add_panel(LinePanel(
            title="Price-Volume Correlation",
            x_data=list(range(len(rolling_corr))),
            y_data=rolling_corr,
            colors=["purple"],
            show_zero_line=True,
            row=0, col=1,
        ))

        # 3. Fractal Metrics (middle left)
        fractal_dims = cross_dim_results.get('fractal_dimensions', {})
        hurst_exps = cross_dim_results.get('hurst_exponents', {})
        dimensions = list(fractal_dims.keys())

        if dimensions:
            chart.add_panel(BarPanel(
                title="Fractal Metrics by Dimension",
                categories=dimensions,
                values=[
                    [fractal_dims.get(dim, 0) for dim in dimensions],
                    [hurst_exps.get(dim, 0) for dim in dimensions],
                ],
                colors=["#1976d2", "#E53935"],
                labels=["Fractal Dimension", "Hurst Exponent"],
                row=1, col=0,
            ))

        # 4. Regime Classification (middle right)
        regime_info = cross_dim_results.get('regime', {})
        current_regime = regime_info.get('regime', 0)
        n_regimes = regime_info.get('n_regimes', 3)
        regime_names = ["Trending", "Mean-Reverting", "Random Walk"]

        chart.add_panel(GaugePanel(
            title="Regime Classification",
            value=current_regime,
            min_value=0,
            max_value=n_regimes - 1,
            thresholds=[
                (1, "#4CAF50"),
                (2, "#FFC107"),
                (3, "#9E9E9E"),
            ],
            label=regime_names[min(current_regime, 2)],
            row=1, col=1,
        ))

        # 5. Coherence (bottom left)
        coherence = cross_dim_results.get('fractal_coherence', {}).get('overall', 0)
        chart.add_panel(BarPanel(
            title="Cross-Dimensional Coherence",
            categories=["Overall"],
            values=[[coherence]],
            colors=["#4CAF50"],
            show_values=True,
            row=2, col=0,
        ))

        # 6. Correlation Heatmap (bottom right)
        corr_matrix = np.array(cross_dim_results.get('cross_correlation', [[1, 0], [0, 1]]))
        chart.add_panel(HeatmapPanel(
            title="Correlation Heatmap",
            data=corr_matrix.tolist(),
            x_labels=dimensions[:corr_matrix.shape[1]] if dimensions else None,
            y_labels=dimensions[:corr_matrix.shape[0]] if dimensions else None,
            colorscale="viridis",
            v_min=-1,
            v_max=1,
            row=2, col=1,
        ))

        return chart

    @staticmethod
    def plot_trading_time_analysis(
        prices: np.ndarray,
        time_map: Dict,
        dates: np.ndarray = None,
    ) -> MultiPanelChart:
        """
        Create visualization showing trading time vs clock time analysis.

        Args:
            prices: Price time series
            time_map: Dict with 'dilation_factors' and 'trading_time_values'
            dates: Optional dates array

        Returns:
            MultiPanelChart object (call .show() to display)
        """
        if dates is None:
            dates = np.arange(len(prices))

        dilation_factors = time_map['dilation_factors']
        trading_time = time_map['trading_time_values']

        chart = MultiPanelChart(
            rows=3,
            cols=1,
            width=1000,
            height=900,
            title="Trading Time Analysis: Market Time Dilation",
            theme="dark",
            row_heights=[0.5, 0.25, 0.25],
        )

        x_data = list(range(len(prices)))

        # 1. Price with dilation (top)
        chart.add_panel(LinePanel(
            title="Price Series with Time Dilation Markers",
            x_data=x_data,
            y_data=prices.tolist(),
            colors=["rgba(100, 100, 100, 0.8)"],
            line_widths=[1],
            row=0, col=0,
        ))

        # 2. Trading time mapping (middle)
        linear_time = np.linspace(0, trading_time[-1], len(trading_time))
        chart.add_panel(LinePanel(
            title="Trading Time vs Clock Time Mapping",
            x_data=list(range(len(trading_time))),
            y_data=[trading_time.tolist(), linear_time.tolist()],
            colors=["purple", "gray"],
            line_widths=[2, 1],
            labels=["Trading Time", "Linear Time"],
            row=1, col=0,
        ))

        # 3. Time dilation factors (bottom)
        chart.add_panel(AreaPanel(
            title="Time Dilation Factors",
            x_data=x_data[:len(dilation_factors)],
            y_data=dilation_factors.tolist(),
            color="rgba(255, 0, 0, 0.3)",
            line_color="red",
            baseline=1,
            show_baseline=True,
            row=2, col=0,
        ))

        return chart

    @staticmethod
    def plot_analysis_and_forecast(
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray,
    ) -> ForecastChart:
        """
        Create comprehensive visualization with path density.

        Args:
            historical_prices: Historical price data
            simulation_results: Tuple of (paths, path_analysis)
            analysis_results: Dict with fractal metrics
            dates: Dates array

        Returns:
            ForecastChart object (call .show() to display)
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

        chart = ForecastChart(
            width=1000,
            height=700,
            theme="dark",
            title="Fractal Pattern Analysis with Path Density",
        )

        chart.set_data(historical_prices, result, dates)
        chart.colorscale("viridis")
        chart.show_percentiles(True)

        # Add statistics annotation
        stats_text = (
            f"Hurst: {analysis_results.get('hurst', 0):.3f}<br>"
            f"Fractal Dim: {analysis_results.get('fractal_dim', 0):.3f}<br>"
            f"Paths: {paths.shape[0]:,}<br>"
            f"Horizon: {paths.shape[1]} steps"
        )
        chart.add_annotation(stats_text, x=0.02, y=0.98, font_size=10)

        return chart

    def plot_quantum_analysis(
        self,
        prices: np.ndarray,
        quantum_results: Dict,
        dates: np.ndarray = None,
    ) -> MultiPanelChart:
        """
        Plot quantum analysis results.

        Args:
            prices: Price time series
            quantum_results: Results from quantum analysis
            dates: Optional dates array

        Returns:
            MultiPanelChart object (call .show() to display)
        """
        if dates is None:
            dates = np.arange(len(prices))

        chart = MultiPanelChart(
            rows=3,
            cols=1,
            width=1000,
            height=800,
            title="Quantum Fractal Analysis",
            theme="dark",
            row_heights=[0.3, 0.3, 0.4],
        )

        x_data = list(range(len(prices)))

        # 1. Price History
        chart.add_panel(LinePanel(
            title="Price History",
            x_data=x_data,
            y_data=prices.tolist(),
            colors=["#1976d2"],
            row=0, col=0,
        ))

        # 2. Quantum Price Levels
        qpl = quantum_results.get('price_levels', {}).get('levels', [])
        if qpl:
            price_levels = [level['price'] for level in qpl]
            probabilities = [level['probability'] for level in qpl]
            chart.add_panel(BarPanel(
                title="Quantum Price Levels",
                categories=[f"${p:.2f}" for p in price_levels[:10]],
                values=[probabilities[:10]],
                colors=["rgba(255, 0, 0, 0.7)"],
                row=1, col=0,
            ))

        # 3. Cross-correlations heatmap
        multi_results = quantum_results.get('multi_dimensional', {})
        cross_corr = multi_results.get('cross_correlations', [[1, 0], [0, 1]])
        chart.add_panel(HeatmapPanel(
            title="Cross-Correlations",
            data=cross_corr if isinstance(cross_corr, list) else cross_corr.tolist(),
            x_labels=['Price', 'Volume'],
            y_labels=['Price', 'Volume'],
            colorscale="viridis",
            row=2, col=0,
        ))

        return chart

    def plot_high_density_forecast(
        self,
        historical_prices: np.ndarray,
        simulation_results: Tuple[np.ndarray, Dict],
        analysis_results: Dict,
        dates: np.ndarray,
    ) -> ForecastChart:
        """
        Create high-performance density visualization.

        Args:
            historical_prices: Historical price data
            simulation_results: Tuple of (paths, path_analysis)
            analysis_results: Dict with fractal metrics
            dates: Dates array

        Returns:
            ForecastChart object (call .show() to display)
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

        chart = ForecastChart(
            width=1000,
            height=700,
            theme="dark",
            title="High Density Path Visualization",
        )

        chart.set_data(historical_prices, result, dates)
        chart.colorscale("plasma")
        chart.max_paths(500)
        chart.show_percentiles(True)

        return chart


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
