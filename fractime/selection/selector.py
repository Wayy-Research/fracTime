"""
Automated model selection.

Automatically selects the best forecasting model for a given dataset
using walk-forward validation and dual penalty scoring.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .registry import ModelRegistry, get_global_registry
from ..backtesting import WalkForwardValidator, DualPenaltyScorer, compare_models


@dataclass
class SelectionResult:
    """
    Result of automated model selection.

    Attributes:
        best_model_name: Name of best model
        best_model: Instantiated best model (fitted)
        best_score: Score of best model
        all_scores: Scores for all models tested
        validation_results: Full validation results for all models
        comparison: Model comparison results
    """
    best_model_name: str
    best_model: Any
    best_score: float
    all_scores: Dict[str, float]
    validation_results: Dict[str, Dict]
    comparison: Dict


class AutoSelector:
    """
    Automatically select the best forecasting model for a dataset.

    Uses walk-forward validation and dual penalty scoring to:
    1. Test multiple models on the data
    2. Score each model (accuracy - overfitting)
    3. Select the best performer
    4. Optionally fit the best model on full data

    Parameters:
        models: List of model names to test (if None, tests all available)
        registry: ModelRegistry to use (if None, uses global registry)
        validation_params: Parameters for WalkForwardValidator
        scoring_params: Parameters for DualPenaltyScorer
        verbose: Print progress and results

    Example:
        >>> from fractime.selection import AutoSelector
        >>>
        >>> selector = AutoSelector(verbose=True)
        >>> result = selector.select_best(prices, dates)
        >>>
        >>> print(f"Best model: {result.best_model_name}")
        >>> print(f"Score: {result.best_score:.4f}")
        >>>
        >>> # Use the best model for forecasting
        >>> forecast = result.best_model.predict(n_steps=10)
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        registry: Optional[ModelRegistry] = None,
        validation_params: Optional[Dict[str, Any]] = None,
        scoring_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        self.models = models
        self.registry = registry or get_global_registry()
        self.validation_params = validation_params or {
            'initial_window': 252,
            'step_size': 10,
            'forecast_horizon': 1,
            'window_type': 'expanding',
            'verbose': False
        }
        self.scoring_params = scoring_params or {
            'alpha': 1.0,
            'beta': 0.5
        }
        self.verbose = verbose

    def select_best(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        fit_on_full_data: bool = True
    ) -> SelectionResult:
        """
        Select the best model for the given data.

        Args:
            prices: Historical prices
            dates: Optional dates
            fit_on_full_data: Whether to refit best model on full dataset

        Returns:
            SelectionResult with best model and detailed results
        """
        if self.verbose:
            print("=" * 80)
            print("AUTOMATED MODEL SELECTION")
            print("=" * 80)
            print(f"Data: {len(prices)} points")

        # Determine which models to test
        models_to_test = self._get_models_to_test()

        if self.verbose:
            print(f"Testing {len(models_to_test)} models: {', '.join(models_to_test)}")
            print()

        # Run validation for each model
        validation_results = {}
        for model_name in models_to_test:
            if self.verbose:
                print(f"{'=' * 70}")
                print(f"Testing: {model_name}")
                print(f"{'=' * 70}")

            try:
                # Create model instance
                model = self.registry.create_model(model_name)

                # Adjust validation params based on data size
                val_params = self._adjust_validation_params(len(prices))

                # Run validation
                validator = WalkForwardValidator(
                    model=model,
                    **val_params
                )

                results = validator.run(prices, dates)
                validation_results[model_name] = results

                # Print quick metrics
                if self.verbose:
                    metrics = results['metrics']
                    print(f"  RMSE: {metrics['rmse']:.4f}")
                    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
                    if 'coverage' in metrics:
                        print(f"  Coverage: {metrics['coverage']:.2%}")

            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Failed: {e}")
                continue

        if not validation_results:
            raise ValueError("No models successfully validated")

        # Score all models
        if self.verbose:
            print("\n" + "=" * 80)
            print("SCORING MODELS")
            print("=" * 80)

        scorer = DualPenaltyScorer(**self.scoring_params)
        comparison = compare_models(validation_results, scorer)

        if self.verbose:
            print(comparison['comparison_table'])

        # Get best model
        best_model_name = comparison['best_model']
        best_score = comparison['scores'][best_model_name]['total_score']

        # Extract all scores
        all_scores = {
            name: score_dict['total_score']
            for name, score_dict in comparison['scores'].items()
        }

        # Fit best model on full data (optional)
        if fit_on_full_data:
            if self.verbose:
                print(f"\nRefitting best model ({best_model_name}) on full dataset...")

            best_model = self.registry.create_model(best_model_name)
            best_model.fit(prices, dates)

            if self.verbose:
                print("  ✓ Best model fitted on full data")
        else:
            # Use the model from validation (fitted on last window)
            best_model = self.registry.create_model(best_model_name)

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"SELECTION COMPLETE: {best_model_name}")
            print("=" * 80)

        return SelectionResult(
            best_model_name=best_model_name,
            best_model=best_model,
            best_score=best_score,
            all_scores=all_scores,
            validation_results=validation_results,
            comparison=comparison
        )

    def _get_models_to_test(self) -> List[str]:
        """Determine which models to test."""
        if self.models is not None:
            # User specified models
            return self.models
        else:
            # Test all available models
            return self.registry.list_models(available_only=True)

    def _adjust_validation_params(self, n_points: int) -> Dict[str, Any]:
        """
        Adjust validation parameters based on data size.

        Args:
            n_points: Number of data points

        Returns:
            Adjusted validation parameters
        """
        params = self.validation_params.copy()

        # Adjust initial window
        if 'initial_window' in params:
            # Use at most 60% of data for initial window
            max_window = int(n_points * 0.6)
            params['initial_window'] = min(params['initial_window'], max_window)

        # Adjust step size
        if 'step_size' in params:
            # For small datasets, use smaller steps
            if n_points < 200:
                params['step_size'] = max(1, n_points // 20)

        return params

    def select_by_characteristics(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        required_characteristics: Optional[List[str]] = None,
        fit_on_full_data: bool = True
    ) -> SelectionResult:
        """
        Select best model with specific characteristics.

        Args:
            prices: Historical prices
            dates: Optional dates
            required_characteristics: Required model characteristics
                (e.g., ['probabilistic', 'handles_seasonality'])
            fit_on_full_data: Whether to refit on full dataset

        Returns:
            SelectionResult with best model matching characteristics
        """
        if required_characteristics is None:
            required_characteristics = []

        # Get models with required characteristics
        matching_models = self.registry.get_by_characteristics(required_characteristics)

        if not matching_models:
            raise ValueError(
                f"No models found with characteristics: {required_characteristics}"
            )

        if self.verbose:
            print(f"Models matching characteristics {required_characteristics}:")
            print(f"  {', '.join(matching_models)}")

        # Temporarily override model list
        original_models = self.models
        self.models = matching_models

        # Run selection
        result = self.select_best(prices, dates, fit_on_full_data)

        # Restore original model list
        self.models = original_models

        return result

    def compare_categories(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> Dict[str, SelectionResult]:
        """
        Compare best model from each category.

        Args:
            prices: Historical prices
            dates: Optional dates

        Returns:
            Dictionary mapping category → SelectionResult
        """
        if self.verbose:
            print("=" * 80)
            print("COMPARING MODEL CATEGORIES")
            print("=" * 80)

        categories = ['fractal', 'baseline', 'bayesian']
        results = {}

        for category in categories:
            models = self.registry.get_by_category(category)

            if not models:
                if self.verbose:
                    print(f"\n{category.upper()}: No models available")
                continue

            if self.verbose:
                print(f"\n{category.upper()} MODELS: {', '.join(models)}")

            # Select best from this category
            original_models = self.models
            self.models = models

            try:
                result = self.select_best(prices, dates, fit_on_full_data=False)
                results[category] = result
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Failed: {e}")

            self.models = original_models

        # Compare category winners
        if self.verbose and results:
            print("\n" + "=" * 80)
            print("CATEGORY WINNERS")
            print("=" * 80)

            for category, result in sorted(results.items(), key=lambda x: -x[1].best_score):
                print(f"{category:15s} → {result.best_model_name:20s} (score: {result.best_score:.4f})")

        return results
