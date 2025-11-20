"""
Forecast accuracy and calibration metrics.

Provides comprehensive metrics for evaluating time series forecasts:
- Point forecast accuracy: RMSE, MAE, MAPE
- Directional accuracy: Did we predict the right direction?
- Probabilistic: CRPS (Continuous Ranked Probability Score)
- Calibration: Are confidence intervals properly calibrated?
"""

import numpy as np
from typing import Dict, Optional


def compute_rmse(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        forecasts: Predicted values
        actuals: Actual observed values

    Returns:
        RMSE value (lower is better)
    """
    return np.sqrt(np.mean((forecasts - actuals) ** 2))


def compute_mae(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        forecasts: Predicted values
        actuals: Actual observed values

    Returns:
        MAE value (lower is better)
    """
    return np.mean(np.abs(forecasts - actuals))


def compute_mape(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        forecasts: Predicted values
        actuals: Actual observed values

    Returns:
        MAPE value in percentage (lower is better)
    """
    # Avoid division by zero
    mask = actuals != 0
    if not np.any(mask):
        return np.inf

    return np.mean(np.abs((actuals[mask] - forecasts[mask]) / actuals[mask])) * 100


def compute_direction_accuracy(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray
) -> float:
    """
    Directional accuracy: Did we predict the right direction of change?

    Args:
        forecasts: Predicted values
        actuals: Actual observed values
        current_prices: Current prices (for computing direction)

    Returns:
        Proportion of correct directional predictions (0 to 1)
    """
    predicted_direction = np.sign(forecasts - current_prices)
    actual_direction = np.sign(actuals - current_prices)

    # Count correct predictions
    correct = np.sum(predicted_direction == actual_direction)

    return correct / len(forecasts)


def compute_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    actuals: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Coverage: What proportion of actuals fall within the confidence interval?

    For a well-calibrated 95% CI, this should be close to 0.95.

    Args:
        lower: Lower confidence bound
        upper: Upper confidence bound
        actuals: Actual observed values
        confidence_level: Expected coverage (default 0.95)

    Returns:
        Actual coverage proportion
    """
    in_interval = (actuals >= lower) & (actuals <= upper)
    return np.mean(in_interval)


def compute_crps(forecast_samples: np.ndarray, actuals: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score.

    A proper scoring rule that rewards calibrated probabilistic forecasts.
    Lower is better.

    Args:
        forecast_samples: Array of forecast samples (n_samples x n_steps)
        actuals: Actual observed values (n_steps,)

    Returns:
        Mean CRPS across all time steps
    """
    n_steps = len(actuals)
    crps_values = np.zeros(n_steps)

    for t in range(n_steps):
        # Forecast distribution at time t
        samples = forecast_samples[:, t]

        # Sort samples
        sorted_samples = np.sort(samples)
        n_samples = len(sorted_samples)

        # Empirical CDF
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        # where X, X' are independent samples from forecast distribution
        # and y is the actual value

        # Compute E|X - y|
        term1 = np.mean(np.abs(sorted_samples - actuals[t]))

        # Compute E|X - X'|
        # For empirical distribution: mean of all pairwise differences
        term2 = 0.0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                term2 += np.abs(sorted_samples[i] - sorted_samples[j])

        term2 = term2 / (n_samples * (n_samples - 1) / 2)

        crps_values[t] = term1 - 0.5 * term2

    return np.mean(crps_values)


class ForecastMetrics:
    """
    Comprehensive forecast evaluation metrics.

    Computes all relevant metrics for forecast accuracy and calibration.
    """

    @staticmethod
    def compute_all(
        forecasts: np.ndarray,
        actuals: np.ndarray,
        current_prices: Optional[np.ndarray] = None,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        forecast_samples: Optional[np.ndarray] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute all available metrics.

        Args:
            forecasts: Point forecasts
            actuals: Actual observed values
            current_prices: Current prices (for directional accuracy)
            lower: Lower confidence bound (for coverage)
            upper: Upper confidence bound (for coverage)
            forecast_samples: Forecast sample paths (for CRPS)
            confidence_level: Expected confidence level

        Returns:
            Dictionary of all computed metrics
        """
        # Compute MSE and RMSE
        errors = forecasts - actuals
        mse = np.mean(errors ** 2)

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': compute_mae(forecasts, actuals),
            'mape': compute_mape(forecasts, actuals),
        }

        # Directional accuracy (if current prices provided)
        if current_prices is not None:
            metrics['direction_accuracy'] = compute_direction_accuracy(
                forecasts, actuals, current_prices
            )

        # Coverage (if CI provided)
        if lower is not None and upper is not None:
            metrics['coverage'] = compute_coverage(lower, upper, actuals, confidence_level)

            # Calibration error
            metrics['calibration_error'] = abs(metrics['coverage'] - confidence_level)

            # CI width (narrower is better if well-calibrated)
            metrics['mean_ci_width'] = np.mean(upper - lower)

        # CRPS (if samples provided)
        if forecast_samples is not None:
            metrics['crps'] = compute_crps(forecast_samples, actuals)

        return metrics

    @staticmethod
    def print_summary(metrics: Dict[str, float]):
        """
        Print a nicely formatted summary of metrics.

        Args:
            metrics: Dictionary of computed metrics
        """
        print("\n" + "=" * 60)
        print("FORECAST ACCURACY METRICS")
        print("=" * 60)

        print("\nPoint Forecast Accuracy:")
        print(f"  RMSE:                    {metrics['rmse']:.4f}")
        print(f"  MAE:                     {metrics['mae']:.4f}")

        if 'mape' in metrics and not np.isinf(metrics['mape']):
            print(f"  MAPE:                    {metrics['mape']:.2f}%")

        if 'direction_accuracy' in metrics:
            print(f"\nDirectional Accuracy:      {metrics['direction_accuracy']:.2%}")
            random_baseline = 0.50
            improvement = metrics['direction_accuracy'] - random_baseline
            print(f"  (vs random 50% baseline: {improvement:+.2%})")

        if 'coverage' in metrics:
            print(f"\nProbabilistic Calibration:")
            print(f"  95% CI Coverage:         {metrics['coverage']:.2%}")
            print(f"  Calibration Error:       {metrics['calibration_error']:.4f}")
            print(f"  Mean CI Width:           {metrics['mean_ci_width']:.4f}")

            # Assess calibration quality
            if abs(metrics['calibration_error']) < 0.05:
                assessment = "✓ Well calibrated"
            elif abs(metrics['calibration_error']) < 0.10:
                assessment = "⚠ Moderately calibrated"
            else:
                assessment = "✗ Poorly calibrated"

            print(f"  Assessment:              {assessment}")

        if 'crps' in metrics:
            print(f"\nContinuous Ranked Probability Score:")
            print(f"  CRPS:                    {metrics['crps']:.4f}")

        print("=" * 60)
