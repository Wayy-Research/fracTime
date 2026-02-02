"""
FracTime Scientific Research Framework.

This module provides a comprehensive framework for comparing FracTime forecasting
against baseline models with results suitable for academic publication.

Key Components:
    - config: Experiment configuration (ExperimentConfig, TEST_UNIVERSE)
    - data_collector: Unified data fetching with caching (DataCollector)
    - experiment: Walk-forward validation framework (ExperimentRunner)
    - metrics: Extended evaluation metrics (CRPS, calibration, etc.)
    - statistical_tests: Significance testing (Diebold-Mariano, MCS)
    - report_generator: LaTeX/Markdown report generation
    - visualization: Static plots and interactive Three.js site

Example:
    >>> from fractime.research import ExperimentConfig, DataCollector, ExperimentRunner
    >>> config = ExperimentConfig.default()
    >>> collector = DataCollector(config)
    >>> data = collector.fetch_all()
    >>> runner = ExperimentRunner(config)
    >>> results = runner.run_all(data)
"""

from __future__ import annotations

from .config import (
    ExperimentConfig,
    AssetConfig,
    TEST_UNIVERSE,
    HORIZONS,
    DEFAULT_CONFIG,
)
from .data_collector import DataCollector
from .experiment import ExperimentRunner, ExperimentResult
from .metrics import MetricsCalculator, ForecastMetrics
from .statistical_tests import (
    diebold_mariano_test,
    model_confidence_set,
    DMTestResult,
    MCSResult,
)
from .report_generator import ReportGenerator
from .tca import TCACalculator, TransactionCosts, compute_net_metrics

__all__ = [
    # Config
    "ExperimentConfig",
    "AssetConfig",
    "TEST_UNIVERSE",
    "HORIZONS",
    "DEFAULT_CONFIG",
    # Data
    "DataCollector",
    # Experiments
    "ExperimentRunner",
    "ExperimentResult",
    # Metrics
    "MetricsCalculator",
    "ForecastMetrics",
    # Statistical tests
    "diebold_mariano_test",
    "model_confidence_set",
    "DMTestResult",
    "MCSResult",
    # Report generation
    "ReportGenerator",
    # Transaction Cost Analysis (NEW in v0.5)
    "TCACalculator",
    "TransactionCosts",
    "compute_net_metrics",
]
