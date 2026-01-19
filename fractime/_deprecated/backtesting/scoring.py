"""
Dual penalty scoring for model selection.

Implements the core principle: Balance accuracy against overfitting.

Score = α·Accuracy - β·Overfitting

This provides a single metric for comparing models that captures both:
1. How well the model forecasts (accuracy)
2. Whether the model is overfitting (stability, complexity)

The goal is to select models that generalize well, not just fit the training data.
"""

import numpy as np
from typing import Dict, Optional, List


class DualPenaltyScorer:
    """
    Score models by balancing forecast accuracy against overfitting.

    The dual penalty approach ensures we select models that:
    - Make accurate predictions (high accuracy)
    - Generalize well (low overfitting)

    Parameters:
        alpha: Weight on accuracy component (default 1.0)
        beta: Weight on overfitting penalty (default 0.5)
        normalization: How to normalize scores ('minmax' or 'zscore')

    Example:
        >>> from fractime.backtesting import WalkForwardValidator, DualPenaltyScorer
        >>> from fractime import FractalForecaster
        >>>
        >>> # Run validation
        >>> validator = WalkForwardValidator(FractalForecaster())
        >>> results = validator.run(prices, dates)
        >>>
        >>> # Score the model
        >>> scorer = DualPenaltyScorer(alpha=1.0, beta=0.5)
        >>> score = scorer.score(results)
        >>>
        >>> print(f"Combined Score: {score['total_score']:.4f}")
        >>> print(f"  Accuracy:     {score['accuracy_score']:.4f}")
        >>> print(f"  Overfitting:  {score['overfitting_penalty']:.4f}")
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        normalization: str = 'minmax'
    ):
        self.alpha = alpha
        self.beta = beta
        self.normalization = normalization

    def score(self, validation_results: Dict) -> Dict[str, float]:
        """
        Compute dual penalty score from validation results.

        Args:
            validation_results: Output from WalkForwardValidator.run()
                Must contain 'metrics' and optionally 'parameter_history'

        Returns:
            Dictionary with:
                - total_score: Combined score (higher is better)
                - accuracy_score: Accuracy component
                - overfitting_penalty: Overfitting component
                - components: Breakdown of individual metrics
        """
        metrics = validation_results['metrics']
        parameter_history = validation_results.get('parameter_history', [])
        forecasts = validation_results.get('forecasts', [])

        # Compute accuracy score (higher is better)
        accuracy_score = self._compute_accuracy_score(metrics)

        # Compute overfitting penalty (lower is better)
        overfitting_penalty = self._compute_overfitting_penalty(
            metrics,
            parameter_history,
            forecasts
        )

        # Combined score
        total_score = self.alpha * accuracy_score - self.beta * overfitting_penalty

        return {
            'total_score': total_score,
            'accuracy_score': accuracy_score,
            'overfitting_penalty': overfitting_penalty,
            'components': {
                'accuracy': self._accuracy_components(metrics),
                'overfitting': self._overfitting_components(
                    metrics, parameter_history, forecasts
                )
            }
        }

    def _compute_accuracy_score(self, metrics: Dict[str, float]) -> float:
        """
        Combine multiple accuracy metrics into single score.

        Higher is better. Normalized to [0, 1] range.
        """
        components = []

        # Direction accuracy (already 0-1, higher is better)
        if 'direction_accuracy' in metrics:
            # Normalize: 50% = 0, 100% = 1
            direction_score = max(0, (metrics['direction_accuracy'] - 0.5) * 2)
            components.append(('direction', direction_score, 0.4))

        # RMSE (lower is better, so invert)
        # We normalize by converting to R² equivalent
        if 'rmse' in metrics and 'actuals' in metrics:
            # Simple normalization: compare to naive baseline
            # This assumes naive forecast = last value, with typical RMSE
            # For financial data, good models achieve RMSE < 2% of price level
            # We'll use relative RMSE if we have the data
            rmse = metrics['rmse']
            # Lower RMSE = higher score
            # Typical RMSE for good models: 0.01-0.05 (1-5%)
            # Exponential decay: high penalty for RMSE > 0.05
            rmse_score = np.exp(-20 * rmse)
            components.append(('rmse', rmse_score, 0.3))

        # Calibration (how well-calibrated are confidence intervals?)
        if 'calibration_error' in metrics:
            # calibration_error is abs(coverage - 0.95)
            # Perfect calibration = 0, normalized to [0, 1]
            calibration_score = max(0, 1 - metrics['calibration_error'] / 0.1)
            components.append(('calibration', calibration_score, 0.3))

        # Weighted average of components
        if not components:
            return 0.5  # Neutral score if no metrics

        total_weight = sum(weight for _, _, weight in components)
        accuracy_score = sum(score * weight for _, score, weight in components) / total_weight

        return accuracy_score

    def _compute_overfitting_penalty(
        self,
        metrics: Dict[str, float],
        parameter_history: List[Dict],
        forecasts: List
    ) -> float:
        """
        Compute overfitting penalty from stability metrics.

        Lower is better. Normalized to [0, 1] range.
        """
        penalties = []

        # 1. Parameter instability
        if parameter_history and len(parameter_history) > 10:
            param_variance = self._compute_parameter_variance(parameter_history)
            # High variance = overfitting
            # Hurst should be stable, typical variance < 0.01 for good models
            variance_penalty = min(1.0, param_variance * 100)
            penalties.append(('param_variance', variance_penalty, 0.4))

        # 2. Forecast variance (excessive uncertainty)
        if forecasts and len(forecasts) > 10:
            forecast_variance = self._compute_forecast_variance(forecasts)
            # Normalize by mean forecast level
            forecasts_array = np.array([f[0] if len(f) > 0 else 0 for f in forecasts])
            mean_level = np.mean(np.abs(forecasts_array))
            if mean_level > 0:
                relative_variance = forecast_variance / (mean_level ** 2)
                # Typical relative variance: 0.001-0.01 for stable models
                variance_penalty = min(1.0, relative_variance * 100)
                penalties.append(('forecast_variance', variance_penalty, 0.3))

        # 3. Coverage over-confidence (CIs too narrow = overfitting)
        if 'coverage' in metrics:
            # Under-coverage suggests overfitting (too confident)
            coverage = metrics['coverage']
            if coverage < 0.95:
                # Severe penalty for under-coverage
                undercoverage_penalty = (0.95 - coverage) * 2
                penalties.append(('undercoverage', undercoverage_penalty, 0.3))

        # Weighted average of penalties
        if not penalties:
            return 0.0  # No penalty if we can't measure overfitting

        total_weight = sum(weight for _, _, weight in penalties)
        overfitting_penalty = sum(penalty * weight for _, penalty, weight in penalties) / total_weight

        return overfitting_penalty

    def _compute_parameter_variance(self, parameter_history: List[Dict]) -> float:
        """
        Measure parameter stability over time.

        High variance = parameters jumping around = overfitting
        """
        if not parameter_history:
            return 0.0

        # Extract Hurst values
        if 'hurst_mean' in parameter_history[0]:
            # Bayesian model
            hurst_values = [p['hurst_mean'] for p in parameter_history]
        elif 'hurst' in parameter_history[0]:
            # Classical model
            hurst_values = [p['hurst'] for p in parameter_history]
        else:
            return 0.0

        # Compute rolling variance to detect instability
        window = min(20, len(hurst_values) // 4)
        if window < 5:
            return np.var(hurst_values)

        rolling_vars = []
        for i in range(len(hurst_values) - window):
            rolling_vars.append(np.var(hurst_values[i:i+window]))

        # Average rolling variance
        return np.mean(rolling_vars)

    def _compute_forecast_variance(self, forecasts: List) -> float:
        """
        Measure forecast variance over time.

        Excessive variance suggests the model is too reactive (overfitting noise).
        """
        if not forecasts:
            return 0.0

        # Extract first-step forecasts
        forecasts_array = np.array([f[0] if len(f) > 0 else np.nan for f in forecasts])
        forecasts_array = forecasts_array[~np.isnan(forecasts_array)]

        if len(forecasts_array) < 2:
            return 0.0

        return np.var(forecasts_array)

    def _accuracy_components(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract accuracy component breakdown for transparency."""
        components = {}

        if 'direction_accuracy' in metrics:
            components['direction'] = max(0, (metrics['direction_accuracy'] - 0.5) * 2)

        if 'rmse' in metrics:
            components['rmse'] = np.exp(-20 * metrics['rmse'])

        if 'calibration_error' in metrics:
            components['calibration'] = max(0, 1 - metrics['calibration_error'] / 0.1)

        return components

    def _overfitting_components(
        self,
        metrics: Dict[str, float],
        parameter_history: List[Dict],
        forecasts: List
    ) -> Dict[str, float]:
        """Extract overfitting penalty breakdown for transparency."""
        components = {}

        if parameter_history and len(parameter_history) > 10:
            param_var = self._compute_parameter_variance(parameter_history)
            components['parameter_variance'] = min(1.0, param_var * 100)

        if forecasts and len(forecasts) > 10:
            forecast_var = self._compute_forecast_variance(forecasts)
            forecasts_array = np.array([f[0] if len(f) > 0 else 0 for f in forecasts])
            mean_level = np.mean(np.abs(forecasts_array))
            if mean_level > 0:
                components['forecast_variance'] = min(1.0, (forecast_var / (mean_level ** 2)) * 100)

        if 'coverage' in metrics and metrics['coverage'] < 0.95:
            components['undercoverage'] = (0.95 - metrics['coverage']) * 2

        return components


def compare_models(
    models_or_results: Dict[str, any],
    prices: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    initial_window: Optional[int] = None,
    step_size: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    scorer: Optional[DualPenaltyScorer] = None
) -> Dict:
    """
    Compare multiple models using dual penalty scoring.

    This function has two modes:

    1. Pass pre-computed validation results:
       compare_models(results_dict)

    2. Run walk-forward validation on models:
       compare_models(models_dict, prices, dates, initial_window, step_size, forecast_horizon)

    Args:
        models_or_results: Either:
            - Dictionary mapping model names to validation results (mode 1)
            - Dictionary mapping model names to model instances (mode 2)
        prices: Price series (required for mode 2)
        dates: Date array (optional for mode 2)
        initial_window: Initial training window size (mode 2)
        step_size: Steps between refits (mode 2)
        forecast_horizon: Forecast horizon (mode 2)
        scorer: DualPenaltyScorer instance (creates default if None)

    Returns:
        For mode 1: Dictionary with rankings, scores, best_model, comparison_table
        For mode 2: Dictionary with model names as keys and metrics as values

    Example (mode 1):
        >>> results = {
        ...     'fractal': validator1.run(prices),
        ...     'bayesian': validator2.run(prices),
        ...     'arima': validator3.run(prices)
        ... }
        >>> comparison = compare_models(results)
        >>> print(comparison['comparison_table'])

    Example (mode 2):
        >>> models = {
        ...     'Fractal': FractalForecaster(),
        ...     'ARIMA': ARIMAForecaster()
        ... }
        >>> comparison = compare_models(models, prices, dates,
        ...     initial_window=100, step_size=20, forecast_horizon=10)
        >>> print(comparison['Fractal']['mae'])
    """
    # Check if we're in mode 2 (models + walk-forward validation)
    if prices is not None:
        # Mode 2: Run walk-forward validation on models
        from .validator import WalkForwardValidator

        if initial_window is None or step_size is None or forecast_horizon is None:
            raise ValueError(
                "When comparing models with walk-forward validation, "
                "must provide initial_window, step_size, and forecast_horizon"
            )

        results = {}
        for model_name, model in models_or_results.items():
            try:
                validator = WalkForwardValidator(
                    model,
                    initial_window=initial_window,
                    step_size=step_size,
                    forecast_horizon=forecast_horizon,
                    verbose=False
                )
                validation_result = validator.run(prices, dates)
                results[model_name] = validation_result['metrics']
            except Exception as e:
                import warnings
                warnings.warn(f"Model {model_name} failed: {e}")
                results[model_name] = {'mae': np.inf, 'mse': np.inf, 'rmse': np.inf}

        return results

    # Mode 1: Score pre-computed results
    results_dict = models_or_results

    if scorer is None:
        scorer = DualPenaltyScorer(alpha=1.0, beta=0.5)

    # Score all models
    scores = {}
    for model_name, results in results_dict.items():
        scores[model_name] = scorer.score(results)

    # Rank by total score
    rankings = sorted(
        scores.items(),
        key=lambda x: x[1]['total_score'],
        reverse=True  # Higher is better
    )

    # Format comparison table
    comparison_table = _format_comparison_table(rankings)

    return {
        'rankings': rankings,
        'scores': scores,
        'best_model': rankings[0][0] if rankings else None,
        'comparison_table': comparison_table
    }


def _format_comparison_table(rankings: List) -> str:
    """Format model comparison as readable table."""
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL COMPARISON - DUAL PENALTY SCORING")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Model':<20} {'Total Score':<15} {'Accuracy':<12} {'Overfitting':<12}")
    lines.append("-" * 80)

    for rank, (model_name, score_dict) in enumerate(rankings, 1):
        total = score_dict['total_score']
        accuracy = score_dict['accuracy_score']
        overfitting = score_dict['overfitting_penalty']

        lines.append(
            f"{rank:<6} {model_name:<20} {total:>10.4f}     "
            f"{accuracy:>8.4f}     {overfitting:>8.4f}"
        )

    lines.append("=" * 80)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Higher Total Score is better (accuracy - overfitting)")
    lines.append("  - Accuracy: [0, 1], higher is better")
    lines.append("  - Overfitting: [0, 1], lower is better")
    lines.append("")

    return "\n".join(lines)
