"""
Statistical tests for forecast comparison in FracTime research.

This module provides significance tests for comparing forecast accuracy:
- Diebold-Mariano test with HAC variance estimator
- Model Confidence Set (simplified implementation)
- Multiple testing correction (Bonferroni, Holm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class DMTestResult:
    """Result from Diebold-Mariano test.

    Attributes:
        model1: Name of first model
        model2: Name of second model
        statistic: DM test statistic
        p_value: Two-sided p-value
        p_value_one_sided: One-sided p-value (model1 better)
        conclusion: Human-readable conclusion
        mean_loss_diff: Mean difference in loss (model1 - model2)
        significant_at_05: Whether significant at 5% level
        significant_at_01: Whether significant at 1% level
    """

    model1: str
    model2: str
    statistic: float
    p_value: float
    p_value_one_sided: float
    conclusion: str
    mean_loss_diff: float
    significant_at_05: bool
    significant_at_01: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model1": self.model1,
            "model2": self.model2,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "p_value_one_sided": float(self.p_value_one_sided),
            "conclusion": self.conclusion,
            "mean_loss_diff": float(self.mean_loss_diff),
            "significant_at_05": self.significant_at_05,
            "significant_at_01": self.significant_at_01,
        }


@dataclass
class MCSResult:
    """Result from Model Confidence Set procedure.

    Attributes:
        superior_models: List of models in the superior set
        eliminated_models: Dictionary of eliminated models with p-values
        p_values: Dictionary of p-values for all models
        confidence_level: Confidence level used
    """

    superior_models: List[str]
    eliminated_models: Dict[str, float]
    p_values: Dict[str, float]
    confidence_level: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "superior_models": self.superior_models,
            "eliminated_models": self.eliminated_models,
            "p_values": self.p_values,
            "confidence_level": self.confidence_level,
        }


def _compute_hac_variance(d: np.ndarray, bandwidth: Optional[int] = None) -> float:
    """Compute HAC (Heteroskedasticity and Autocorrelation Consistent) variance.

    Uses Newey-West estimator with Bartlett kernel.

    Args:
        d: Loss differential series
        bandwidth: Bandwidth for HAC (default: floor(n^(1/3)))

    Returns:
        HAC variance estimate
    """
    n = len(d)
    if n < 2:
        return np.nan

    if bandwidth is None:
        bandwidth = int(np.floor(n ** (1 / 3)))

    d_demean = d - np.mean(d)

    gamma_0 = np.mean(d_demean**2)

    gamma_j_sum = 0.0
    for j in range(1, bandwidth + 1):
        weight = 1 - j / (bandwidth + 1)
        gamma_j = np.mean(d_demean[j:] * d_demean[:-j])
        gamma_j_sum += 2 * weight * gamma_j

    variance = gamma_0 + gamma_j_sum
    return max(variance, 1e-10)


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    loss_function: str = "squared",
    h: int = 1,
) -> DMTestResult:
    """Perform Diebold-Mariano test for comparing forecast accuracy.

    Tests null hypothesis that two forecasts have equal expected loss.

    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        loss_function: 'squared' (MSE) or 'absolute' (MAE)
        h: Forecast horizon (for HAC bandwidth adjustment)

    Returns:
        DMTestResult with test statistics and conclusion
    """
    errors1 = np.asarray(errors1)
    errors2 = np.asarray(errors2)

    n = min(len(errors1), len(errors2))
    if n < 10:
        return DMTestResult(
            model1=model1_name,
            model2=model2_name,
            statistic=np.nan,
            p_value=np.nan,
            p_value_one_sided=np.nan,
            conclusion="Insufficient data for test",
            mean_loss_diff=np.nan,
            significant_at_05=False,
            significant_at_01=False,
        )

    errors1 = errors1[:n]
    errors2 = errors2[:n]

    if loss_function == "squared":
        loss1 = errors1**2
        loss2 = errors2**2
    elif loss_function == "absolute":
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    d = loss1 - loss2
    d_mean = np.mean(d)

    bandwidth = max(h - 1, int(np.floor(n ** (1 / 3))))
    variance = _compute_hac_variance(d, bandwidth)

    if variance <= 0 or np.isnan(variance):
        return DMTestResult(
            model1=model1_name,
            model2=model2_name,
            statistic=np.nan,
            p_value=np.nan,
            p_value_one_sided=np.nan,
            conclusion="Could not compute variance",
            mean_loss_diff=d_mean,
            significant_at_05=False,
            significant_at_01=False,
        )

    dm_stat = d_mean / np.sqrt(variance / n)

    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    p_value_one_sided = 1 - stats.norm.cdf(dm_stat)

    significant_05 = p_value < 0.05
    significant_01 = p_value < 0.01

    if p_value >= 0.05:
        conclusion = f"No significant difference between {model1_name} and {model2_name}"
    elif d_mean < 0:
        conclusion = f"{model1_name} significantly outperforms {model2_name}"
    else:
        conclusion = f"{model2_name} significantly outperforms {model1_name}"

    return DMTestResult(
        model1=model1_name,
        model2=model2_name,
        statistic=float(dm_stat),
        p_value=float(p_value),
        p_value_one_sided=float(p_value_one_sided),
        conclusion=conclusion,
        mean_loss_diff=float(d_mean),
        significant_at_05=significant_05,
        significant_at_01=significant_01,
    )


def model_confidence_set(
    loss_matrix: Dict[str, np.ndarray],
    alpha: float = 0.10,
    bootstrap_samples: int = 1000,
) -> MCSResult:
    """Compute Model Confidence Set (simplified implementation).

    The MCS procedure identifies the set of models that contains the best
    model with a given confidence level.

    This is a simplified implementation that uses pairwise DM tests.
    For the full MCS procedure, see Hansen et al. (2011).

    Args:
        loss_matrix: Dictionary mapping model names to loss arrays
        alpha: Significance level (default 0.10)
        bootstrap_samples: Number of bootstrap samples (not used in simplified version)

    Returns:
        MCSResult with superior model set
    """
    models = list(loss_matrix.keys())

    if len(models) < 2:
        return MCSResult(
            superior_models=models,
            eliminated_models={},
            p_values={m: 1.0 for m in models},
            confidence_level=1 - alpha,
        )

    mean_losses = {m: np.mean(loss_matrix[m]) for m in models}
    sorted_models = sorted(models, key=lambda m: mean_losses[m])

    eliminated: Dict[str, float] = {}
    p_values: Dict[str, float] = {}

    remaining = sorted_models.copy()

    for model in reversed(sorted_models):
        if model not in remaining or len(remaining) <= 1:
            continue

        max_p_value = 0.0
        best_model = remaining[0]

        for other in remaining:
            if other == model:
                continue

            errors_model = loss_matrix[model]
            errors_other = loss_matrix[other]

            n = min(len(errors_model), len(errors_other))
            d = errors_model[:n] - errors_other[:n]
            d_mean = np.mean(d)
            d_std = np.std(d, ddof=1)

            if d_std > 0:
                t_stat = d_mean / (d_std / np.sqrt(n))
                p_val = 1 - stats.norm.cdf(t_stat)
            else:
                p_val = 0.5

            max_p_value = max(max_p_value, p_val)

        p_values[model] = max_p_value

        if max_p_value < alpha:
            eliminated[model] = max_p_value
            remaining.remove(model)

    for model in remaining:
        if model not in p_values:
            p_values[model] = 1.0

    return MCSResult(
        superior_models=remaining,
        eliminated_models=eliminated,
        p_values=p_values,
        confidence_level=1 - alpha,
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], float]:
    """Apply Bonferroni correction for multiple testing.

    Args:
        p_values: List of p-values to correct
        alpha: Family-wise error rate

    Returns:
        Tuple of (list of rejection decisions, adjusted alpha)
    """
    m = len(p_values)
    if m == 0:
        return [], alpha

    adjusted_alpha = alpha / m
    rejections = [p < adjusted_alpha for p in p_values]
    return rejections, adjusted_alpha


def holm_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """Apply Holm-Bonferroni step-down correction for multiple testing.

    More powerful than Bonferroni while still controlling FWER.

    Args:
        p_values: List of p-values to correct
        alpha: Family-wise error rate

    Returns:
        Tuple of (list of rejection decisions, adjusted p-values)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    rejections = [False] * m
    adjusted_p = [1.0] * m

    for i, idx in enumerate(sorted_indices):
        adjusted = sorted_p[i] * (m - i)
        adjusted_p[idx] = min(adjusted, 1.0)

        if sorted_p[i] <= alpha / (m - i):
            rejections[idx] = True
        else:
            break

    return rejections, adjusted_p


def pairwise_dm_tests(
    model_errors: Dict[str, np.ndarray],
    baseline_model: str,
    loss_function: str = "squared",
    h: int = 1,
) -> Dict[str, DMTestResult]:
    """Perform pairwise Diebold-Mariano tests against a baseline model.

    Args:
        model_errors: Dictionary mapping model names to error arrays
        baseline_model: Name of baseline model for comparison
        loss_function: 'squared' or 'absolute'
        h: Forecast horizon

    Returns:
        Dictionary mapping model names to DMTestResult against baseline
    """
    if baseline_model not in model_errors:
        raise ValueError(f"Baseline model '{baseline_model}' not in model_errors")

    results: Dict[str, DMTestResult] = {}
    baseline_errors = model_errors[baseline_model]

    for model_name, errors in model_errors.items():
        if model_name == baseline_model:
            continue

        result = diebold_mariano_test(
            baseline_errors,
            errors,
            model1_name=baseline_model,
            model2_name=model_name,
            loss_function=loss_function,
            h=h,
        )
        results[model_name] = result

    return results


def compute_loss_matrix_from_forecasts(
    forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    loss_function: str = "squared",
) -> Dict[str, np.ndarray]:
    """Compute loss matrix from forecasts and actuals.

    Args:
        forecasts: Dictionary mapping model names to forecast arrays
        actuals: Array of actual values
        loss_function: 'squared' or 'absolute'

    Returns:
        Dictionary mapping model names to loss arrays
    """
    actuals = np.asarray(actuals)
    loss_matrix: Dict[str, np.ndarray] = {}

    for model_name, forecast in forecasts.items():
        forecast = np.asarray(forecast)
        n = min(len(forecast), len(actuals))
        errors = forecast[:n] - actuals[:n]

        if loss_function == "squared":
            loss_matrix[model_name] = errors**2
        elif loss_function == "absolute":
            loss_matrix[model_name] = np.abs(errors)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

    return loss_matrix
