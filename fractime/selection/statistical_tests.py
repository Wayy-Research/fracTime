"""
Statistical significance tests for model comparison.

Provides tests to determine if performance differences between models
are statistically significant, not just due to chance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def diebold_mariano_test(
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    actuals: np.ndarray,
    loss_function: str = 'mse',
    h: int = 1
) -> Dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests the null hypothesis that two forecasts have equal accuracy.
    A significant result means one forecast is significantly better.

    Args:
        forecast1: Forecasts from model 1
        forecast2: Forecasts from model 2
        actuals: Actual observed values
        loss_function: 'mse' (default), 'mae', or 'mape'
        h: Forecast horizon (for autocorrelation adjustment)

    Returns:
        Dictionary with:
            - statistic: DM test statistic
            - p_value: Two-sided p-value
            - better_model: Which model is better (1 or 2)
            - significant: Whether difference is significant (p < 0.05)

    Reference:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
        Journal of Business & Economic Statistics, 13(3), 253-263.

    Example:
        >>> result = diebold_mariano_test(forecast1, forecast2, actuals)
        >>> if result['significant']:
        >>>     print(f"Model {result['better_model']} is significantly better")
    """
    # Compute loss differential
    loss1 = _compute_loss(forecast1, actuals, loss_function)
    loss2 = _compute_loss(forecast2, actuals, loss_function)
    loss_diff = loss1 - loss2

    # Mean loss differential
    d_bar = np.mean(loss_diff)

    # Variance of loss differential (accounting for autocorrelation)
    n = len(loss_diff)
    gamma_0 = np.var(loss_diff, ddof=1)

    # Adjust for autocorrelation at lags 1, ..., h-1
    if h > 1:
        gamma_sum = 0
        for k in range(1, h):
            gamma_k = np.mean(
                (loss_diff[k:] - d_bar) * (loss_diff[:-k] - d_bar)
            )
            gamma_sum += gamma_k
        var_d = (gamma_0 + 2 * gamma_sum) / n
    else:
        var_d = gamma_0 / n

    # DM test statistic
    if var_d <= 0:
        # Degenerate case: no variance
        dm_stat = 0.0
        p_value = 1.0
    else:
        dm_stat = d_bar / np.sqrt(var_d)

        # Two-sided p-value (asymptotically N(0,1))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    # Determine which model is better
    if dm_stat > 0:
        better_model = 2  # Model 2 has lower loss
    else:
        better_model = 1  # Model 1 has lower loss

    return {
        'statistic': dm_stat,
        'p_value': p_value,
        'better_model': better_model,
        'significant': p_value < 0.05,
        'mean_loss_diff': d_bar,
        'interpretation': _interpret_dm_test(dm_stat, p_value, better_model)
    }


def _compute_loss(forecast: np.ndarray, actual: np.ndarray, loss_function: str) -> np.ndarray:
    """Compute loss for each forecast."""
    error = forecast - actual

    if loss_function == 'mse':
        return error ** 2
    elif loss_function == 'mae':
        return np.abs(error)
    elif loss_function == 'mape':
        return np.abs(error / actual) * 100
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")


def _interpret_dm_test(dm_stat: float, p_value: float, better_model: int) -> str:
    """Generate interpretation of DM test results."""
    if p_value < 0.01:
        significance = "very strong"
    elif p_value < 0.05:
        significance = "strong"
    elif p_value < 0.10:
        significance = "moderate"
    else:
        significance = "no"

    if p_value < 0.10:
        return f"Model {better_model} is significantly better ({significance} evidence, p={p_value:.4f})"
    else:
        return f"No significant difference between models (p={p_value:.4f})"


def model_confidence_set(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    alpha: float = 0.05,
    loss_function: str = 'mse'
) -> Dict:
    """
    Model Confidence Set (MCS) for selecting superior models.

    Identifies a set of models that are not significantly worse than
    the best model. This is more robust than selecting a single "best" model.

    Args:
        forecasts: Dictionary mapping model names to forecasts
        actuals: Actual observed values
        alpha: Significance level (default 0.05)
        loss_function: Loss function to use

    Returns:
        Dictionary with:
            - superior_set: List of models in MCS
            - eliminated: Order in which models were eliminated
            - p_values: P-values for each elimination

    Reference:
        Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
        Econometrica, 79(2), 453-497.

    Example:
        >>> mcs = model_confidence_set(forecasts, actuals)
        >>> print(f"Superior models: {mcs['superior_set']}")
    """
    model_names = list(forecasts.keys())
    n_models = len(model_names)

    # Compute losses for each model
    losses = {}
    for name, forecast in forecasts.items():
        losses[name] = _compute_loss(forecast, actuals, loss_function)

    # Track which models are still in the set
    remaining = set(model_names)
    eliminated = []
    p_values = []

    # Iteratively eliminate worst models
    while len(remaining) > 1:
        # Compute relative losses
        mean_losses = {name: np.mean(losses[name]) for name in remaining}

        # Find model with worst (highest) mean loss
        worst_model = max(remaining, key=lambda x: mean_losses[x])

        # Test if worst model is significantly worse
        # Compare against all other remaining models
        is_significantly_worse = False
        max_p_value = 0

        for other_model in remaining:
            if other_model == worst_model:
                continue

            # Pairwise DM test
            dm_result = diebold_mariano_test(
                forecasts[worst_model],
                forecasts[other_model],
                actuals,
                loss_function
            )

            max_p_value = max(max_p_value, dm_result['p_value'])

            # If significantly worse than any model, eliminate
            if dm_result['significant'] and dm_result['better_model'] == 2:
                is_significantly_worse = True

        # Eliminate if significantly worse or if no models left
        if is_significantly_worse or len(remaining) == 2:
            remaining.remove(worst_model)
            eliminated.append(worst_model)
            p_values.append(max_p_value)
        else:
            # Can't eliminate any more models
            break

    return {
        'superior_set': list(remaining),
        'eliminated': eliminated,
        'p_values': p_values,
        'n_superior': len(remaining),
        'interpretation': f"Model Confidence Set contains {len(remaining)} superior model(s)"
    }


def compare_all_pairs(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    loss_function: str = 'mse'
) -> Dict[Tuple[str, str], Dict]:
    """
    Pairwise comparison of all models using Diebold-Mariano test.

    Args:
        forecasts: Dictionary mapping model names to forecasts
        actuals: Actual observed values
        loss_function: Loss function to use

    Returns:
        Dictionary mapping (model1, model2) → DM test results

    Example:
        >>> results = compare_all_pairs(forecasts, actuals)
        >>> for (m1, m2), result in results.items():
        >>>     if result['significant']:
        >>>         print(f"{m1} vs {m2}: {result['interpretation']}")
    """
    model_names = list(forecasts.keys())
    results = {}

    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            dm_result = diebold_mariano_test(
                forecasts[model1],
                forecasts[model2],
                actuals,
                loss_function
            )

            results[(model1, model2)] = dm_result

    return results


def print_pairwise_comparison(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    loss_function: str = 'mse'
) -> None:
    """
    Print formatted pairwise comparison results.

    Args:
        forecasts: Dictionary mapping model names to forecasts
        actuals: Actual observed values
        loss_function: Loss function to use
    """
    results = compare_all_pairs(forecasts, actuals, loss_function)

    print("=" * 80)
    print("PAIRWISE MODEL COMPARISON (Diebold-Mariano Test)")
    print("=" * 80)
    print(f"Loss function: {loss_function.upper()}")
    print()

    for (model1, model2), result in results.items():
        print(f"{model1} vs {model2}:")
        print(f"  {result['interpretation']}")
        print()

    print("=" * 80)


def test_ensemble_vs_individuals(
    ensemble_forecast: np.ndarray,
    individual_forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray
) -> Dict:
    """
    Test if an ensemble forecast is significantly better than individual models.

    Args:
        ensemble_forecast: Ensemble forecast
        individual_forecasts: Dictionary of individual model forecasts
        actuals: Actual observed values

    Returns:
        Dictionary with test results for each model
    """
    results = {}

    for model_name, forecast in individual_forecasts.items():
        dm_result = diebold_mariano_test(
            forecast,
            ensemble_forecast,
            actuals
        )

        results[model_name] = dm_result

    # Summary
    n_better = sum(1 for r in results.values() if r['better_model'] == 2 and r['significant'])
    n_total = len(results)

    return {
        'individual_results': results,
        'n_ensemble_better': n_better,
        'n_total': n_total,
        'ensemble_improves': n_better > n_total / 2
    }
