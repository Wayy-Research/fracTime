"""
Static publication-quality plots for FracTime research.

This module generates matplotlib figures suitable for academic publication,
including model comparison charts, calibration diagrams, and forecast
visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Figure = None

from ..experiment import ExperimentResult, ForecastRecord
from ..metrics import ForecastMetrics


PUBLICATION_STYLE = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

MODEL_COLORS = {
    "fractime_rs": "#1f77b4",
    "fractime_dfa": "#2ca02c",
    "fractime_ensemble": "#17becf",
    "arima": "#ff7f0e",
    "garch": "#d62728",
    "prophet": "#9467bd",
    "ets": "#8c564b",
    "lstm": "#e377c2",
}


def _check_matplotlib() -> None:
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_model_comparison(
    metrics: Dict[str, Dict[str, ForecastMetrics]],
    metric_name: str = "rmse",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot model comparison bar chart.

    Args:
        metrics: Nested dict of model -> horizon -> metrics
        metric_name: Metric to compare ('rmse', 'mae', 'direction_accuracy', etc.)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    _check_matplotlib()

    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        models = list(metrics.keys())
        horizons = []
        for model_metrics in metrics.values():
            horizons.extend(model_metrics.keys())
        horizons = sorted(set(h for h in horizons if h != "overall"))

        x = np.arange(len(horizons))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = []
            for h in horizons:
                if h in metrics[model]:
                    val = getattr(metrics[model][h], metric_name, np.nan)
                    values.append(val if not np.isnan(val) else 0)
                else:
                    values.append(0)

            color = MODEL_COLORS.get(model, f"C{i}")
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=color)

        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel(metric_name.upper().replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels([h.upper() for h in horizons])
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Model Comparison: {metric_name.upper()}")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig


def plot_calibration_diagram(
    metrics: Dict[str, Dict[str, ForecastMetrics]],
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot calibration diagram (reliability diagram).

    Shows observed coverage vs expected coverage for each model.

    Args:
        metrics: Nested dict of model -> horizon -> metrics
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    _check_matplotlib()

    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        expected = [0.50, 0.80, 0.95]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")

        for i, (model, model_metrics) in enumerate(metrics.items()):
            if "overall" not in model_metrics:
                continue

            m = model_metrics["overall"]
            observed = [
                m.coverage_50 if not np.isnan(m.coverage_50) else 0.5,
                m.coverage_80 if not np.isnan(m.coverage_80) else 0.8,
                m.coverage_95 if not np.isnan(m.coverage_95) else 0.95,
            ]

            color = MODEL_COLORS.get(model, f"C{i}")
            ax.plot(expected, observed, "o-", label=model, color=color, markersize=8)

        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title("Calibration Diagram")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc="lower right")
        ax.set_aspect("equal")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig


def plot_cumulative_metrics(
    forecasts: List[ForecastRecord],
    model_name: str,
    metric: str = "error",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot cumulative metrics over time for a model.

    Args:
        forecasts: List of ForecastRecord objects
        model_name: Model to plot
        metric: 'error' (MAE), 'squared_error' (MSE), or 'direction'
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    _check_matplotlib()

    model_forecasts = [f for f in forecasts if f.model_name == model_name and f.actual is not None]
    if not model_forecasts:
        return None

    model_forecasts.sort(key=lambda f: f.forecast_date)

    dates = [f.forecast_date for f in model_forecasts]
    errors = np.array([f.forecast_mean - f.actual for f in model_forecasts])
    current_prices = np.array([f.current_price for f in model_forecasts])
    actuals = np.array([f.actual for f in model_forecasts])

    if metric == "error":
        values = np.abs(errors)
        ylabel = "Cumulative MAE"
    elif metric == "squared_error":
        values = errors**2
        ylabel = "Cumulative MSE"
    elif metric == "direction":
        pred_dir = np.sign(np.array([f.forecast_mean for f in model_forecasts]) - current_prices)
        actual_dir = np.sign(actuals - current_prices)
        values = (pred_dir == actual_dir).astype(float)
        ylabel = "Cumulative Direction Accuracy"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    cumulative = np.cumsum(values) / np.arange(1, len(values) + 1)

    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        color = MODEL_COLORS.get(model_name, "C0")
        ax.plot(dates, cumulative, color=color, linewidth=2)

        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name}: {ylabel} Over Time")

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig


def plot_regime_analysis(
    forecasts: List[ForecastRecord],
    ticker: str,
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot regime analysis for a specific asset.

    Shows price with Hurst-based regime classification over time.

    Args:
        forecasts: List of ForecastRecord objects
        ticker: Asset ticker to analyze
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    _check_matplotlib()

    ticker_forecasts = [
        f for f in forecasts
        if f.ticker == ticker and f.metadata and "hurst" in f.metadata
    ]
    if not ticker_forecasts:
        return None

    ticker_forecasts.sort(key=lambda f: f.forecast_date)

    dates = [f.forecast_date for f in ticker_forecasts]
    prices = [f.current_price for f in ticker_forecasts]
    hursts = [f.metadata.get("hurst", 0.5) for f in ticker_forecasts]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})

        ax1.plot(dates, prices, "b-", linewidth=1)
        ax1.set_ylabel("Price")
        ax1.set_title(f"{ticker}: Price and Market Regime")

        ax2.plot(dates, hursts, "g-", linewidth=1)
        ax2.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="H=0.5 (Random Walk)")
        ax2.axhline(0.55, color="r", linestyle=":", alpha=0.5, label="H>0.55 (Trending)")
        ax2.axhline(0.45, color="b", linestyle=":", alpha=0.5, label="H<0.45 (Mean Reverting)")
        ax2.set_ylabel("Hurst Exponent")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0.3, 0.7)
        ax2.legend(loc="upper right")

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig


def plot_forecast_paths(
    forecast: ForecastRecord,
    n_paths: int = 100,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot forecast paths with confidence intervals.

    Args:
        forecast: ForecastRecord with paths
        n_paths: Number of paths to display
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    _check_matplotlib()

    if forecast.paths is None or len(forecast.paths) == 0:
        return None

    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        paths = forecast.paths[:n_paths]
        for path in paths:
            ax.plot([0, 1], [forecast.current_price, path], "b-", alpha=0.1)

        ax.axhline(forecast.actual, color="r", linewidth=2, label="Actual")
        ax.axhline(forecast.forecast_mean, color="g", linewidth=2, label="Forecast Mean")

        ax.axhspan(forecast.lower_95, forecast.upper_95, alpha=0.2, color="gray", label="95% CI")
        ax.axhspan(forecast.lower_50, forecast.upper_50, alpha=0.3, color="gray", label="50% CI")

        ax.scatter([0], [forecast.current_price], color="black", s=100, zorder=5, label="Current")
        ax.scatter([1], [forecast.actual], color="red", s=100, zorder=5)

        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title(f"{forecast.ticker}: {forecast.model_name} Forecast (h={forecast.horizon})")
        ax.legend(loc="best")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Current", f"+{forecast.horizon}d"])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig


def plot_hurst_analysis(
    prices: np.ndarray,
    hurst: float,
    scales: Optional[np.ndarray] = None,
    rs_values: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Plot Hurst exponent analysis visualization.

    Shows the R/S analysis log-log plot with regression line.

    Args:
        prices: Price series
        hurst: Computed Hurst exponent
        scales: Scale values (log)
        rs_values: R/S values (log)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib Figure object
    """
    _check_matplotlib()

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax1 = axes[0, 0]
        ax1.plot(prices, "b-", linewidth=0.5)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.set_title("Price Series")

        ax2 = axes[0, 1]
        returns = np.diff(np.log(prices))
        ax2.plot(returns, "g-", linewidth=0.5, alpha=0.7)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Log Returns")
        ax2.set_title("Log Returns")

        ax3 = axes[1, 0]
        ax3.hist(returns, bins=50, density=True, alpha=0.7, color="blue")
        x = np.linspace(returns.min(), returns.max(), 100)
        from scipy import stats
        ax3.plot(x, stats.norm.pdf(x, np.mean(returns), np.std(returns)), "r-", label="Normal")
        ax3.set_xlabel("Log Return")
        ax3.set_ylabel("Density")
        ax3.set_title("Return Distribution")
        ax3.legend()

        ax4 = axes[1, 1]
        if scales is not None and rs_values is not None:
            ax4.scatter(scales, rs_values, alpha=0.7)
            slope, intercept = np.polyfit(scales, rs_values, 1)
            ax4.plot(scales, slope * scales + intercept, "r-",
                     label=f"H = {hurst:.3f}")
        else:
            n_scales = 20
            min_scale, max_scale = 10, min(len(prices) // 4, 200)
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
            ax4.text(0.5, 0.5, f"H = {hurst:.3f}", transform=ax4.transAxes,
                     fontsize=16, ha="center", va="center")

        ax4.set_xlabel("log(Scale)")
        ax4.set_ylabel("log(R/S)")
        ax4.set_title("R/S Analysis")
        ax4.legend()

        regime = "Trending" if hurst > 0.55 else ("Mean Reverting" if hurst < 0.45 else "Random Walk")
        fig.suptitle(f"Hurst Analysis: H = {hurst:.3f} ({regime})", fontsize=14)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig


class ResearchPlotter:
    """Generate all publication figures from experiment results.

    Example:
        >>> plotter = ResearchPlotter(results)
        >>> plotter.generate_all()
    """

    def __init__(
        self,
        results: ExperimentResult,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the plotter.

        Args:
            results: Experiment results
            output_dir: Output directory for figures
        """
        _check_matplotlib()

        self.results = results
        self.output_dir = output_dir or results.config.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> Dict[str, Path]:
        """Generate all publication figures.

        Returns:
            Dictionary mapping figure name to file path
        """
        paths: Dict[str, Path] = {}

        for metric in ["rmse", "mae", "direction_accuracy"]:
            path = self.output_dir / f"comparison_{metric}.png"
            fig = plot_model_comparison(
                self.results.metrics,
                metric_name=metric,
                save_path=path,
            )
            if fig:
                paths[f"comparison_{metric}"] = path
                plt.close(fig)

        path = self.output_dir / "calibration.png"
        fig = plot_calibration_diagram(self.results.metrics, save_path=path)
        if fig:
            paths["calibration"] = path
            plt.close(fig)

        for model in self.results.config.models[:3]:
            path = self.output_dir / f"cumulative_{model}.png"
            fig = plot_cumulative_metrics(
                self.results.forecasts,
                model_name=model,
                metric="error",
                save_path=path,
            )
            if fig:
                paths[f"cumulative_{model}"] = path
                plt.close(fig)

        return paths
