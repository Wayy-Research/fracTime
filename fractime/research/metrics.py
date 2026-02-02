"""
Extended evaluation metrics for FracTime research.

This module provides comprehensive forecast evaluation metrics including:
- Point forecast: RMSE, MAE, MAPE, MdAPE
- Probabilistic: CRPS (Continuous Ranked Probability Score)
- Calibration: Coverage at various confidence levels
- Directional: Percentage of correct direction predictions
- Trading: Sharpe ratio of directional signals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class ForecastMetrics:
    """Container for all forecast evaluation metrics.

    Attributes:
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        mdape: Median Absolute Percentage Error
        crps: Continuous Ranked Probability Score
        direction_accuracy: Percentage of correct direction predictions
        coverage_50: Coverage at 50% confidence level
        coverage_80: Coverage at 80% confidence level
        coverage_95: Coverage at 95% confidence level
        calibration_error: Deviation from expected coverage
        mean_interval_width: Average prediction interval width
        sharpe_ratio: Sharpe ratio of directional trading signals
        n_forecasts: Number of forecasts evaluated
    """

    rmse: float = np.nan
    mae: float = np.nan
    mape: float = np.nan
    mdape: float = np.nan
    crps: float = np.nan
    direction_accuracy: float = np.nan
    coverage_50: float = np.nan
    coverage_80: float = np.nan
    coverage_95: float = np.nan
    calibration_error: float = np.nan
    mean_interval_width: float = np.nan
    sharpe_ratio: float = np.nan
    n_forecasts: int = 0
    additional: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rmse": float(self.rmse) if not np.isnan(self.rmse) else None,
            "mae": float(self.mae) if not np.isnan(self.mae) else None,
            "mape": float(self.mape) if not np.isnan(self.mape) else None,
            "mdape": float(self.mdape) if not np.isnan(self.mdape) else None,
            "crps": float(self.crps) if not np.isnan(self.crps) else None,
            "direction_accuracy": float(self.direction_accuracy) if not np.isnan(self.direction_accuracy) else None,
            "coverage_50": float(self.coverage_50) if not np.isnan(self.coverage_50) else None,
            "coverage_80": float(self.coverage_80) if not np.isnan(self.coverage_80) else None,
            "coverage_95": float(self.coverage_95) if not np.isnan(self.coverage_95) else None,
            "calibration_error": float(self.calibration_error) if not np.isnan(self.calibration_error) else None,
            "mean_interval_width": float(self.mean_interval_width) if not np.isnan(self.mean_interval_width) else None,
            "sharpe_ratio": float(self.sharpe_ratio) if not np.isnan(self.sharpe_ratio) else None,
            "n_forecasts": self.n_forecasts,
            "additional": self.additional,
        }

    def __repr__(self) -> str:
        return (
            f"ForecastMetrics(rmse={self.rmse:.4f}, mae={self.mae:.4f}, "
            f"direction={self.direction_accuracy:.2%}, coverage_95={self.coverage_95:.2%})"
        )


def compute_rmse(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    if len(forecasts) == 0 or len(actuals) == 0:
        return np.nan
    errors = forecasts - actuals
    return float(np.sqrt(np.mean(errors**2)))


def compute_mae(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    if len(forecasts) == 0 or len(actuals) == 0:
        return np.nan
    return float(np.mean(np.abs(forecasts - actuals)))


def compute_mape(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    if len(forecasts) == 0 or len(actuals) == 0:
        return np.nan
    mask = actuals != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((forecasts[mask] - actuals[mask]) / actuals[mask]))) * 100


def compute_mdape(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Median Absolute Percentage Error."""
    if len(forecasts) == 0 or len(actuals) == 0:
        return np.nan
    mask = actuals != 0
    if not mask.any():
        return np.nan
    return float(np.median(np.abs((forecasts[mask] - actuals[mask]) / actuals[mask]))) * 100


def compute_crps(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    paths: Optional[np.ndarray] = None,
) -> float:
    """Compute Continuous Ranked Probability Score.

    CRPS is a proper scoring rule that generalizes MAE to probabilistic forecasts.
    Lower values indicate better calibration.

    Args:
        forecasts: Point forecasts (n_forecasts,)
        actuals: Actual values (n_forecasts,)
        paths: Monte Carlo paths (n_forecasts, n_paths) - if provided,
               uses empirical distribution; otherwise assumes Gaussian

    Returns:
        Mean CRPS value
    """
    if len(forecasts) == 0 or len(actuals) == 0:
        return np.nan

    n = len(forecasts)
    crps_values = np.zeros(n)

    if paths is not None and paths.ndim == 2:
        for i in range(n):
            if i < len(paths):
                samples = np.sort(paths[i])
                m = len(samples)
                actual = actuals[i]

                indices = np.arange(1, m + 1)
                crps_values[i] = (
                    np.mean(np.abs(samples - actual))
                    - 0.5 * np.mean(np.abs(np.diff(samples)))
                )
            else:
                crps_values[i] = np.abs(forecasts[i] - actuals[i])
    else:
        std = np.std(forecasts - actuals) if len(forecasts) > 1 else 1.0
        if std == 0:
            std = 1.0

        for i in range(n):
            z = (actuals[i] - forecasts[i]) / std
            crps_values[i] = std * (
                z * (2 * stats.norm.cdf(z) - 1)
                + 2 * stats.norm.pdf(z)
                - 1 / np.sqrt(np.pi)
            )

    return float(np.mean(crps_values))


def compute_direction_accuracy(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
) -> float:
    """Compute directional accuracy (percentage of correct up/down predictions).

    Args:
        forecasts: Point forecasts
        actuals: Actual values
        current_prices: Prices at forecast time

    Returns:
        Percentage of correct direction predictions (0-1)
    """
    if len(forecasts) == 0 or len(actuals) == 0 or len(current_prices) == 0:
        return np.nan

    forecast_direction = np.sign(forecasts - current_prices)
    actual_direction = np.sign(actuals - current_prices)

    non_zero_mask = actual_direction != 0
    if not non_zero_mask.any():
        return np.nan

    correct = forecast_direction[non_zero_mask] == actual_direction[non_zero_mask]
    return float(np.mean(correct))


def compute_coverage(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute empirical coverage of prediction intervals.

    Args:
        actuals: Actual values
        lower: Lower bounds of prediction intervals
        upper: Upper bounds of prediction intervals

    Returns:
        Proportion of actuals within intervals (0-1)
    """
    if len(actuals) == 0 or len(lower) == 0 or len(upper) == 0:
        return np.nan

    within = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(within))


def compute_calibration_error(
    observed_coverages: List[float],
    expected_coverages: List[float],
) -> float:
    """Compute calibration error as mean absolute deviation from expected coverage.

    Args:
        observed_coverages: List of observed coverage rates
        expected_coverages: List of expected coverage rates

    Returns:
        Mean absolute calibration error
    """
    if not observed_coverages or not expected_coverages:
        return np.nan

    observed = np.array(observed_coverages)
    expected = np.array(expected_coverages)
    return float(np.mean(np.abs(observed - expected)))


def compute_sharpe_ratio(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute Sharpe ratio of directional trading signals.

    Simulates a simple strategy: go long if forecast > current, short otherwise.

    Args:
        forecasts: Point forecasts
        actuals: Actual values
        current_prices: Prices at forecast time
        risk_free_rate: Annualized risk-free rate (default 0)

    Returns:
        Annualized Sharpe ratio (assuming 252 trading days)
    """
    if len(forecasts) < 2 or len(actuals) < 2:
        return np.nan

    signals = np.sign(forecasts - current_prices)
    actual_returns = (actuals - current_prices) / current_prices
    strategy_returns = signals * actual_returns

    if np.std(strategy_returns) == 0:
        return np.nan

    excess_returns = strategy_returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return float(sharpe * np.sqrt(252))


def compute_pit_values(
    actuals: np.ndarray,
    paths: np.ndarray,
) -> np.ndarray:
    """Compute Probability Integral Transform (PIT) values.

    PIT values should be uniformly distributed if the forecast distribution
    is well-calibrated.

    Args:
        actuals: Actual values (n_forecasts,)
        paths: Monte Carlo paths (n_forecasts, n_paths)

    Returns:
        PIT values (n_forecasts,)
    """
    if len(actuals) == 0 or paths is None or paths.size == 0:
        return np.array([])

    n = min(len(actuals), len(paths))
    pit_values = np.zeros(n)

    for i in range(n):
        pit_values[i] = np.mean(paths[i] <= actuals[i])

    return pit_values


def compute_pit_calibration(pit_values: np.ndarray) -> Tuple[float, float]:
    """Test uniformity of PIT values using Kolmogorov-Smirnov test.

    Args:
        pit_values: PIT values from compute_pit_values

    Returns:
        Tuple of (KS statistic, p-value)
    """
    if len(pit_values) == 0:
        return np.nan, np.nan

    ks_stat, p_value = stats.kstest(pit_values, "uniform")
    return float(ks_stat), float(p_value)


class MetricsCalculator:
    """Calculator for comprehensive forecast evaluation metrics.

    This class provides a unified interface for computing all evaluation
    metrics used in the FracTime research study.

    Example:
        >>> calc = MetricsCalculator()
        >>> metrics = calc.compute_all(forecasts, actuals, current_prices)
        >>> print(metrics.rmse, metrics.direction_accuracy)
    """

    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        pass

    def compute_all(
        self,
        forecasts: np.ndarray,
        actuals: np.ndarray,
        current_prices: Optional[np.ndarray] = None,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        lower_50: Optional[np.ndarray] = None,
        upper_50: Optional[np.ndarray] = None,
        lower_80: Optional[np.ndarray] = None,
        upper_80: Optional[np.ndarray] = None,
        paths: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
    ) -> ForecastMetrics:
        """Compute all evaluation metrics.

        Args:
            forecasts: Point forecasts
            actuals: Actual values
            current_prices: Prices at forecast time (for direction/Sharpe)
            lower: Lower bounds of 95% CI
            upper: Upper bounds of 95% CI
            lower_50: Lower bounds of 50% CI
            upper_50: Upper bounds of 50% CI
            lower_80: Lower bounds of 80% CI
            upper_80: Upper bounds of 80% CI
            paths: Monte Carlo paths for probabilistic metrics
            confidence_level: Primary confidence level for coverage

        Returns:
            ForecastMetrics with all computed values
        """
        forecasts = np.asarray(forecasts)
        actuals = np.asarray(actuals)

        n = min(len(forecasts), len(actuals))
        if n == 0:
            return ForecastMetrics(n_forecasts=0)

        forecasts = forecasts[:n]
        actuals = actuals[:n]

        metrics = ForecastMetrics(n_forecasts=n)

        metrics.rmse = compute_rmse(forecasts, actuals)
        metrics.mae = compute_mae(forecasts, actuals)
        metrics.mape = compute_mape(forecasts, actuals)
        metrics.mdape = compute_mdape(forecasts, actuals)

        if current_prices is not None:
            current_prices = np.asarray(current_prices)[:n]
            metrics.direction_accuracy = compute_direction_accuracy(
                forecasts, actuals, current_prices
            )
            metrics.sharpe_ratio = compute_sharpe_ratio(
                forecasts, actuals, current_prices
            )

        if lower is not None and upper is not None:
            lower = np.asarray(lower)[:n]
            upper = np.asarray(upper)[:n]
            metrics.coverage_95 = compute_coverage(actuals, lower, upper)
            metrics.mean_interval_width = float(np.mean(upper - lower))

        if lower_50 is not None and upper_50 is not None:
            lower_50 = np.asarray(lower_50)[:n]
            upper_50 = np.asarray(upper_50)[:n]
            metrics.coverage_50 = compute_coverage(actuals, lower_50, upper_50)

        if lower_80 is not None and upper_80 is not None:
            lower_80 = np.asarray(lower_80)[:n]
            upper_80 = np.asarray(upper_80)[:n]
            metrics.coverage_80 = compute_coverage(actuals, lower_80, upper_80)

        observed = []
        expected = []
        if not np.isnan(metrics.coverage_50):
            observed.append(metrics.coverage_50)
            expected.append(0.50)
        if not np.isnan(metrics.coverage_80):
            observed.append(metrics.coverage_80)
            expected.append(0.80)
        if not np.isnan(metrics.coverage_95):
            observed.append(metrics.coverage_95)
            expected.append(0.95)

        if observed:
            metrics.calibration_error = compute_calibration_error(observed, expected)

        if paths is not None:
            paths = np.asarray(paths)
            if paths.ndim == 2 and len(paths) >= n:
                paths = paths[:n]
                metrics.crps = compute_crps(forecasts, actuals, paths)

                pit_values = compute_pit_values(actuals, paths)
                if len(pit_values) > 0:
                    ks_stat, p_value = compute_pit_calibration(pit_values)
                    metrics.additional["pit_ks_statistic"] = ks_stat
                    metrics.additional["pit_p_value"] = p_value
            else:
                metrics.crps = compute_crps(forecasts, actuals, None)
        else:
            metrics.crps = compute_crps(forecasts, actuals, None)

        return metrics

    def compare_models(
        self,
        model_metrics: Dict[str, ForecastMetrics],
        primary_metric: str = "rmse",
    ) -> Dict[str, Any]:
        """Compare metrics across models.

        Args:
            model_metrics: Dictionary mapping model name to ForecastMetrics
            primary_metric: Metric to use for ranking

        Returns:
            Dictionary with comparison results and rankings
        """
        if not model_metrics:
            return {}

        comparison = {
            "models": list(model_metrics.keys()),
            "rankings": {},
            "best_model": {},
        }

        metric_names = [
            "rmse", "mae", "mape", "crps",
            "direction_accuracy", "coverage_95", "sharpe_ratio"
        ]

        for metric_name in metric_names:
            values = {}
            for model, metrics in model_metrics.items():
                val = getattr(metrics, metric_name, np.nan)
                if not np.isnan(val):
                    values[model] = val

            if values:
                lower_is_better = metric_name in ["rmse", "mae", "mape", "crps", "calibration_error"]

                sorted_models = sorted(
                    values.keys(),
                    key=lambda m: values[m],
                    reverse=not lower_is_better
                )
                comparison["rankings"][metric_name] = sorted_models
                comparison["best_model"][metric_name] = sorted_models[0]

        if primary_metric in comparison["rankings"]:
            comparison["overall_ranking"] = comparison["rankings"][primary_metric]
            comparison["overall_best"] = comparison["best_model"][primary_metric]

        return comparison
