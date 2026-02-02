"""
Visualization components for FracTime research.

Includes:
    - static_plots: Publication-quality matplotlib figures
    - threejs: Interactive Three.js site generator
"""

from __future__ import annotations

from .static_plots import (
    plot_model_comparison,
    plot_calibration_diagram,
    plot_cumulative_metrics,
    plot_regime_analysis,
    plot_forecast_paths,
    plot_hurst_analysis,
    ResearchPlotter,
)

__all__ = [
    "plot_model_comparison",
    "plot_calibration_diagram",
    "plot_cumulative_metrics",
    "plot_regime_analysis",
    "plot_forecast_paths",
    "plot_hurst_analysis",
    "ResearchPlotter",
]
