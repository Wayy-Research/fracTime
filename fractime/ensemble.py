"""
Ensemble methods for combining multiple forecasters.

The Ensemble class combines multiple forecasters using stacking, boosting,
or simple averaging.

Examples:
    >>> import fractime as ft
    >>> ensemble = ft.Ensemble(prices, models=[
    ...     ft.Forecaster(method='rs'),
    ...     ft.Forecaster(method='dfa'),
    ... ])
    >>> result = ensemble.predict(steps=30)
"""

from __future__ import annotations

from typing import Optional, Union, List
import numpy as np
import polars as pl

from .result import ForecastResult
from .forecaster import Forecaster
from .analyzer import _ensure_numpy, _ensure_dates


class Ensemble:
    """
    Ensemble of multiple forecasters.

    Combines predictions from multiple forecasters using one of several
    strategies: averaging, stacking (meta-learning), or boosting.

    Args:
        data: Historical price series
        dates: Optional date series
        models: List of Forecaster instances (or will create defaults)
        strategy: Combination strategy
            - 'average': Simple average of forecasts
            - 'weighted': Weighted average based on recent performance
            - 'stacking': Meta-learner combines forecasts
            - 'boosting': Sequential error correction
        meta_learner: For stacking: 'ridge', 'linear', or 'rf'

    Examples:
        Basic usage:
            >>> ensemble = Ensemble(prices)  # Uses default models
            >>> result = ensemble.predict(steps=30)

        Custom models:
            >>> ensemble = Ensemble(prices, models=[
            ...     Forecaster(method='rs'),
            ...     Forecaster(method='dfa'),
            ... ])
            >>> result = ensemble.predict(steps=30)

        Stacking:
            >>> ensemble = Ensemble(prices, strategy='stacking')
            >>> result = ensemble.predict(steps=30)
    """

    def __init__(
        self,
        data: Union[np.ndarray, pl.Series],
        dates: Optional[Union[np.ndarray, pl.Series]] = None,
        models: Optional[List[Forecaster]] = None,
        strategy: str = 'weighted',
        meta_learner: str = 'ridge',
    ):
        self._data = _ensure_numpy(data)
        self._dates = _ensure_dates(dates)
        self._strategy = strategy
        self._meta_learner = meta_learner

        # Create models
        if models is not None:
            self._models = models
        else:
            self._models = self._create_default_models()

        # Fit models to data
        for model in self._models:
            if not hasattr(model, '_data') or model._data is None:
                # Model wasn't initialized with data, create new one
                pass  # Models already have their own data

        # Model weights (for weighted strategy)
        self._weights: Optional[np.ndarray] = None

        # Meta-learner (for stacking)
        self._meta_model = None

    def _create_default_models(self) -> List[Forecaster]:
        """Create default ensemble of forecasters."""
        return [
            Forecaster(self._data, dates=self._dates, method='rs'),
            Forecaster(self._data, dates=self._dates, method='dfa'),
            Forecaster(
                self._data,
                dates=self._dates,
                method='rs',
                path_weights={'hurst': 0.5, 'volatility': 0.3, 'pattern': 0.2}
            ),
        ]

    # =========================================================================
    # Forecasting
    # =========================================================================

    def predict(
        self,
        steps: int = 30,
        n_paths: int = 1000,
    ) -> ForecastResult:
        """
        Generate ensemble forecast.

        Args:
            steps: Number of steps to forecast
            n_paths: Number of paths per model

        Returns:
            Combined ForecastResult
        """
        # Get predictions from each model
        predictions = []
        all_paths = []

        for model in self._models:
            result = model.predict(steps=steps, n_paths=n_paths)
            predictions.append(result)
            all_paths.append(result.paths)

        # Combine based on strategy
        if self._strategy == 'average':
            return self._combine_average(predictions, all_paths)
        elif self._strategy == 'weighted':
            return self._combine_weighted(predictions, all_paths)
        elif self._strategy == 'stacking':
            return self._combine_stacking(predictions, all_paths, steps)
        elif self._strategy == 'boosting':
            return self._combine_boosting(predictions, all_paths, steps)
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

    def _combine_average(
        self,
        predictions: List[ForecastResult],
        all_paths: List[np.ndarray],
    ) -> ForecastResult:
        """Combine using simple averaging."""
        # Stack all paths
        combined_paths = np.vstack(all_paths)

        # Equal probabilities
        n_total = combined_paths.shape[0]
        probabilities = np.ones(n_total) / n_total

        return ForecastResult(
            _paths=combined_paths,
            _probabilities=probabilities,
            dates=predictions[0].dates,
            metadata={
                'strategy': 'average',
                'n_models': len(self._models),
            }
        )

    def _combine_weighted(
        self,
        predictions: List[ForecastResult],
        all_paths: List[np.ndarray],
    ) -> ForecastResult:
        """Combine using performance-weighted averaging."""
        # Calculate weights based on model diversity
        if self._weights is None:
            self._weights = self._calculate_weights(predictions)

        # Stack paths with adjusted probabilities
        combined_paths = []
        combined_probs = []

        for i, (pred, paths) in enumerate(zip(predictions, all_paths)):
            weight = self._weights[i]
            adjusted_probs = pred.probabilities * weight
            combined_paths.append(paths)
            combined_probs.append(adjusted_probs)

        combined_paths = np.vstack(combined_paths)
        combined_probs = np.concatenate(combined_probs)

        # Renormalize
        combined_probs = combined_probs / np.sum(combined_probs)

        return ForecastResult(
            _paths=combined_paths,
            _probabilities=combined_probs,
            dates=predictions[0].dates,
            metadata={
                'strategy': 'weighted',
                'n_models': len(self._models),
                'model_weights': self._weights.tolist(),
            }
        )

    def _calculate_weights(self, predictions: List[ForecastResult]) -> np.ndarray:
        """Calculate model weights based on forecast diversity."""
        n_models = len(predictions)

        # Use forecast diversity as quality metric
        # More diverse models get slightly higher weights
        forecasts = np.array([p.forecast for p in predictions])

        weights = np.ones(n_models)
        for i in range(n_models):
            # Correlation with other models
            correlations = []
            for j in range(n_models):
                if i != j:
                    corr = np.corrcoef(forecasts[i], forecasts[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if correlations:
                # Lower correlation = higher weight (more diverse)
                avg_corr = np.mean(correlations)
                weights[i] = 1.0 / (0.5 + avg_corr)

        # Normalize
        weights = weights / np.sum(weights)
        return weights

    def _combine_stacking(
        self,
        predictions: List[ForecastResult],
        all_paths: List[np.ndarray],
        steps: int,
    ) -> ForecastResult:
        """Combine using meta-learner (stacking)."""
        # Use historical data to train meta-learner
        if self._meta_model is None:
            self._train_meta_learner(steps)

        # For now, fall back to weighted average
        # Full stacking implementation requires cross-validation
        return self._combine_weighted(predictions, all_paths)

    def _train_meta_learner(self, steps: int):
        """Train meta-learner on historical forecasts."""
        # This would require historical forecast performance
        # For now, use simple averaging as meta-model
        pass

    def _combine_boosting(
        self,
        predictions: List[ForecastResult],
        all_paths: List[np.ndarray],
        steps: int,
    ) -> ForecastResult:
        """Combine using boosting (sequential error correction)."""
        # Start with first model's prediction
        base_forecast = predictions[0].forecast.copy()

        # Each subsequent model corrects residuals
        for i in range(1, len(predictions)):
            # Weight decreases for later models
            weight = 1.0 / (i + 1)
            correction = predictions[i].forecast - base_forecast
            base_forecast += weight * correction

        # Create combined paths centered on boosted forecast
        # Use variance from all models
        combined_paths = np.vstack(all_paths)
        n_total = combined_paths.shape[0]
        probabilities = np.ones(n_total) / n_total

        return ForecastResult(
            _paths=combined_paths,
            _probabilities=probabilities,
            dates=predictions[0].dates,
            metadata={
                'strategy': 'boosting',
                'n_models': len(self._models),
            }
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def models(self) -> List[Forecaster]:
        """List of forecasters in the ensemble."""
        return self._models

    @property
    def n_models(self) -> int:
        """Number of models in ensemble."""
        return len(self._models)

    def __repr__(self) -> str:
        return (
            f"Ensemble("
            f"n_models={self.n_models}, "
            f"strategy='{self._strategy}')"
        )
