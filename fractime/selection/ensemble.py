"""
Ensemble forecasting methods.

Combine multiple models to create robust forecasts that often
outperform individual models.
"""

import numpy as np
from typing import Dict, List, Optional, Any


class EnsembleForecaster:
    """
    Base class for ensemble forecasting.

    Combines forecasts from multiple models using various weighting schemes.

    Parameters:
        models: Dictionary mapping names to model instances
        weighting: 'equal', 'performance', or custom weights
        metric: Metric for performance weighting ('rmse', 'mae', 'direction')

    Example:
        >>> from fractime.selection import EnsembleForecaster
        >>> import fractime as ft
        >>>
        >>> models = {
        >>>     'fractal': ft.FractalForecaster(),
        >>>     'arima': ARIMAForecaster()
        >>> }
        >>>
        >>> ensemble = EnsembleForecaster(models, weighting='equal')
        >>> ensemble.fit(prices, dates)
        >>> forecast = ensemble.predict(n_steps=10)
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weighting: str = 'equal',
        metric: str = 'rmse',
        weights: Optional[Dict[str, float]] = None
    ):
        self.models = models
        self.weighting = weighting
        self.metric = metric
        self.custom_weights = weights

        # Computed weights (after fitting)
        self.weights = None
        self.is_fitted = False

    def fit(self, prices: np.ndarray, dates: Optional[np.ndarray] = None) -> None:
        """
        Fit all component models.

        Args:
            prices: Historical prices
            dates: Optional dates
        """
        # Fit each model
        for name, model in self.models.items():
            try:
                model.fit(prices, dates)
            except TypeError:
                # Some models don't accept dates
                model.fit(prices)

        # Compute weights
        if self.weighting == 'equal':
            n = len(self.models)
            self.weights = {name: 1.0 / n for name in self.models.keys()}
        elif self.weighting == 'custom':
            if self.custom_weights is None:
                raise ValueError("Custom weights must be provided")
            self.weights = self.custom_weights
        elif self.weighting == 'performance':
            # Performance weighting requires validation (implement below)
            # For now, use equal weights
            n = len(self.models)
            self.weights = {name: 1.0 / n for name in self.models.keys()}
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

        self.is_fitted = True

    def predict(
        self,
        n_steps: Optional[int] = None,
        end_date: Optional[Any] = None,
        confidence: float = 0.95,
        n_paths: int = 1000
    ) -> Dict:
        """
        Generate ensemble forecast.

        Args:
            n_steps: Number of steps ahead
            end_date: Alternative to n_steps
            confidence: Confidence level
            n_paths: Number of paths (for compatible models)

        Returns:
            Dictionary with:
                - forecast: Weighted ensemble forecast
                - lower: Lower confidence bound
                - upper: Upper confidence bound
                - individual_forecasts: Forecasts from each model
                - weights: Weight given to each model
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get forecasts from all models
        individual_forecasts = {}
        individual_lower = {}
        individual_upper = {}

        for name, model in self.models.items():
            try:
                result = model.predict(
                    n_steps=n_steps,
                    end_date=end_date,
                    confidence=confidence,
                    n_paths=n_paths
                )

                # Extract forecast
                if isinstance(result, dict):
                    forecast = result.get('forecast', result.get('mean'))
                    lower = result.get('lower')
                    upper = result.get('upper')
                else:
                    forecast = result
                    lower = None
                    upper = None

                individual_forecasts[name] = forecast
                if lower is not None:
                    individual_lower[name] = lower
                if upper is not None:
                    individual_upper[name] = upper

            except Exception as e:
                print(f"Warning: Model {name} failed to predict: {e}")
                continue

        if not individual_forecasts:
            raise ValueError("No models successfully generated forecasts")

        # Combine forecasts using weights
        ensemble_forecast = self._combine_forecasts(individual_forecasts)

        # Combine confidence intervals (if available)
        if individual_lower and individual_upper:
            ensemble_lower = self._combine_forecasts(individual_lower)
            ensemble_upper = self._combine_forecasts(individual_upper)
        else:
            ensemble_lower = None
            ensemble_upper = None

        return {
            'forecast': ensemble_forecast,
            'mean': ensemble_forecast,
            'lower': ensemble_lower,
            'upper': ensemble_upper,
            'individual_forecasts': individual_forecasts,
            'weights': self.weights
        }

    def _combine_forecasts(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine individual forecasts using weights.

        Args:
            forecasts: Dictionary mapping model names to forecasts

        Returns:
            Weighted ensemble forecast
        """
        # Ensure all forecasts have same length
        lengths = [len(f) for f in forecasts.values()]
        if len(set(lengths)) > 1:
            # Truncate to minimum length
            min_len = min(lengths)
            forecasts = {name: f[:min_len] for name, f in forecasts.items()}

        # Weighted average
        ensemble = np.zeros(len(next(iter(forecasts.values()))))

        for name, forecast in forecasts.items():
            weight = self.weights.get(name, 0.0)
            ensemble += weight * forecast

        return ensemble

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        return self.weights


class WeightedEnsemble(EnsembleForecaster):
    """
    Ensemble with performance-based weighting.

    Automatically computes optimal weights based on historical performance.

    Parameters:
        models: Dictionary mapping names to model instances
        metric: Metric for weighting ('rmse', 'mae', 'direction')
        validation_window: Window size for computing performance weights

    Example:
        >>> ensemble = WeightedEnsemble(models, metric='rmse')
        >>> ensemble.fit(prices, dates)
        >>> forecast = ensemble.predict(n_steps=10)
    """

    def __init__(
        self,
        models: Dict[str, Any],
        metric: str = 'rmse',
        validation_window: int = 100
    ):
        super().__init__(models, weighting='performance', metric=metric)
        self.validation_window = validation_window

    def fit(self, prices: np.ndarray, dates: Optional[np.ndarray] = None) -> None:
        """
        Fit models and compute performance-based weights.

        Uses a held-out validation window to assess model performance
        and compute inverse-performance weights (better models get higher weight).

        Args:
            prices: Historical prices
            dates: Optional dates
        """
        if len(prices) < self.validation_window + 50:
            # Not enough data for validation, use equal weights
            super().fit(prices, dates)
            return

        # Split into training and validation
        split_point = len(prices) - self.validation_window
        train_prices = prices[:split_point]
        val_prices = prices[split_point:]

        train_dates = dates[:split_point] if dates is not None else None

        # Fit models on training data
        fitted_models = {}
        for name, model in self.models.items():
            try:
                model.fit(train_prices, train_dates)
                fitted_models[name] = model
            except Exception as e:
                print(f"Warning: Model {name} failed to fit: {e}")

        # Evaluate on validation data
        performances = {}
        for name, model in fitted_models.items():
            try:
                # Generate forecasts for validation window
                forecasts = []
                for i in range(self.validation_window):
                    result = model.predict(n_steps=1)
                    if isinstance(result, dict):
                        forecast = result.get('forecast', result.get('mean'))
                    else:
                        forecast = result
                    forecasts.append(forecast[0] if len(forecast) > 0 else forecast)

                forecasts = np.array(forecasts)
                actuals = val_prices[:len(forecasts)]

                # Compute performance metric
                if self.metric == 'rmse':
                    perf = np.sqrt(np.mean((forecasts - actuals) ** 2))
                elif self.metric == 'mae':
                    perf = np.mean(np.abs(forecasts - actuals))
                elif self.metric == 'direction':
                    # For direction, lower is worse, so invert
                    correct = np.sum(np.sign(forecasts[1:] - actuals[:-1]) ==
                                   np.sign(actuals[1:] - actuals[:-1]))
                    perf = 1.0 - (correct / (len(actuals) - 1))
                else:
                    perf = np.sqrt(np.mean((forecasts - actuals) ** 2))

                performances[name] = perf

            except Exception as e:
                print(f"Warning: Could not evaluate {name}: {e}")

        # Compute weights (inverse performance)
        if performances:
            # Inverse weights: better models (lower error) get higher weight
            inverse_perfs = {name: 1.0 / (perf + 1e-10) for name, perf in performances.items()}
            total = sum(inverse_perfs.values())
            self.weights = {name: inv_perf / total for name, inv_perf in inverse_perfs.items()}
        else:
            # Fallback to equal weights
            n = len(self.models)
            self.weights = {name: 1.0 / n for name in self.models.keys()}

        # Refit all models on full data
        for name, model in self.models.items():
            try:
                model.fit(prices, dates)
            except TypeError:
                model.fit(prices)

        self.is_fitted = True


def create_ensemble(
    model_names: List[str],
    weighting: str = 'equal',
    registry: Optional[Any] = None
) -> EnsembleForecaster:
    """
    Create an ensemble from model names.

    Args:
        model_names: List of model names to include
        weighting: Weighting scheme ('equal' or 'performance')
        registry: ModelRegistry to use

    Returns:
        Configured EnsembleForecaster

    Example:
        >>> from fractime.selection import create_ensemble
        >>>
        >>> ensemble = create_ensemble(['Fractal', 'ARIMA', 'GARCH'])
        >>> ensemble.fit(prices, dates)
        >>> forecast = ensemble.predict(n_steps=10)
    """
    from .registry import get_global_registry

    if registry is None:
        registry = get_global_registry()

    # Create model instances
    models = {}
    for name in model_names:
        try:
            models[name] = registry.create_model(name)
        except Exception as e:
            print(f"Warning: Could not create model {name}: {e}")

    if not models:
        raise ValueError("No models successfully created")

    # Create ensemble
    if weighting == 'performance':
        return WeightedEnsemble(models)
    else:
        return EnsembleForecaster(models, weighting=weighting)
