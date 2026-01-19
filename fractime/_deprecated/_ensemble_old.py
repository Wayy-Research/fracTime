"""
Ensemble forecasting methods for combining multiple models.

Implements advanced ensemble techniques:
- Stacking: Meta-learning approach using cross-validation
- Boosting: Sequential error correction approach
"""

import numpy as np
import warnings
import copy
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

# Try to import scikit-learn for meta-learners
try:
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    Ridge = None
    LinearRegression = None
    RandomForestRegressor = None
    TimeSeriesSplit = None


class StackingForecaster:
    """
    Stacking ensemble forecaster using meta-learning.

    Combines predictions from multiple base models using a meta-learner.
    Uses time series cross-validation to generate out-of-fold predictions
    for training the meta-learner without data leakage.

    Args:
        base_models: List of fitted forecaster objects
        meta_learner: Meta-learner type ('ridge', 'linear', 'rf')
        meta_learner_params: Parameters for the meta-learner
        n_splits: Number of CV splits for training meta-learner
        use_original_features: If True, include original prices as meta-features

    Example:
        >>> from fractime import FractalForecaster
        >>> from fractime.baselines import ARIMAForecaster, LSTMForecaster
        >>>
        >>> # Fit base models
        >>> models = [
        ...     FractalForecaster().fit(prices),
        ...     ARIMAForecaster().fit(prices),
        ...     LSTMForecaster().fit(prices)
        ... ]
        >>>
        >>> # Create stacking ensemble
        >>> stacker = StackingForecaster(models, meta_learner='ridge')
        >>> stacker.fit(prices)
        >>> forecast = stacker.predict(n_steps=10)
    """

    def __init__(
        self,
        base_models: Optional[List[Any]] = None,
        meta_learner: str = 'ridge',
        meta_learner_params: Optional[Dict] = None,
        n_splits: int = 5,
        use_original_features: bool = True
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn not installed. Install with: uv pip install scikit-learn"
            )

        self.base_models = base_models or []
        self.meta_learner_type = meta_learner
        self.meta_learner_params = meta_learner_params or {}
        self.n_splits = n_splits
        self.use_original_features = use_original_features

        self.meta_model = None
        self.prices = None
        self.model_names = []

    def _create_meta_learner(self):
        """Create the meta-learner model."""
        if self.meta_learner_type == 'ridge':
            return Ridge(alpha=1.0, **self.meta_learner_params)
        elif self.meta_learner_type == 'linear':
            return LinearRegression(**self.meta_learner_params)
        elif self.meta_learner_type == 'rf':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                **self.meta_learner_params
            )
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")

    def add_model(self, model: Any, name: Optional[str] = None):
        """
        Add a base model to the ensemble.

        Args:
            model: Fitted forecaster object with predict() method
            name: Optional name for the model
        """
        self.base_models.append(model)
        if name:
            self.model_names.append(name)
        else:
            model_name = getattr(model, '__class__', type(model)).__name__
            self.model_names.append(model_name)

    def fit(self, prices: np.ndarray, **kwargs) -> 'StackingForecaster':
        """
        Fit the stacking ensemble.

        Uses time series cross-validation to generate meta-features without leakage.

        Args:
            prices: Historical price series
            **kwargs: Additional arguments

        Returns:
            self: Fitted ensemble
        """
        self.prices = np.asarray(prices).flatten()

        if len(self.base_models) == 0:
            raise ValueError("No base models added. Use add_model() first.")

        # Ensure we have names for all models
        if len(self.model_names) < len(self.base_models):
            for i in range(len(self.model_names), len(self.base_models)):
                model = self.base_models[i]
                model_name = getattr(model, '__class__', type(model)).__name__
                self.model_names.append(f"{model_name}_{i}")

        # Generate out-of-fold predictions for meta-learner training
        # Use time series split to respect temporal order
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        meta_features_list = []
        meta_targets_list = []

        min_train_size = max(100, len(self.prices) // (self.n_splits + 1))

        for train_idx, test_idx in tscv.split(self.prices):
            if len(train_idx) < min_train_size:
                continue  # Skip if training set too small

            train_prices = self.prices[train_idx]
            test_prices = self.prices[test_idx]

            # Get predictions from each base model
            fold_predictions = []

            for model in self.base_models:
                try:
                    # Refit model on training data
                    if hasattr(model, 'fit'):
                        # Use deepcopy to properly clone the model
                        model_copy = copy.deepcopy(model)
                        model_copy.fit(train_prices)
                    else:
                        model_copy = model

                    # Predict for test period
                    n_test = len(test_idx)
                    result = model_copy.predict(n_steps=n_test)

                    if isinstance(result, dict):
                        pred = result.get('forecast', result.get('mean', np.zeros(n_test)))
                    else:
                        pred = result

                    # Ensure correct length
                    if len(pred) != n_test:
                        pred = np.resize(pred, n_test)

                    fold_predictions.append(pred)

                except Exception as e:
                    warnings.warn(f"Model prediction failed: {e}. Using zeros.")
                    fold_predictions.append(np.zeros(len(test_idx)))

            # Stack predictions as features
            fold_predictions = np.column_stack(fold_predictions)

            # Add original features if requested
            if self.use_original_features:
                # Use last k prices as features
                lookback = min(5, len(train_idx) // 2)
                original_features = []

                for i in range(len(test_idx)):
                    if i == 0:
                        feat = train_prices[-lookback:]
                    else:
                        feat = test_prices[max(0, i-lookback):i]

                    if len(feat) < lookback:
                        feat = np.pad(feat, (lookback - len(feat), 0), mode='edge')

                    original_features.append(feat)

                original_features = np.array(original_features)
                fold_predictions = np.hstack([fold_predictions, original_features])

            meta_features_list.append(fold_predictions)
            meta_targets_list.append(test_prices)

        if len(meta_features_list) == 0:
            raise ValueError("Could not generate meta-features. Data may be too short.")

        # Combine all folds
        meta_features = np.vstack(meta_features_list)
        meta_targets = np.concatenate(meta_targets_list)

        # Remove NaN/Inf values that might occur with constant prices or errors
        valid_mask = np.isfinite(meta_features).all(axis=1) & np.isfinite(meta_targets)
        if not valid_mask.any():
            raise ValueError("No valid samples after filtering NaN/Inf values")

        meta_features = meta_features[valid_mask]
        meta_targets = meta_targets[valid_mask]

        # Train meta-learner
        self.meta_model = self._create_meta_learner()
        self.meta_model.fit(meta_features, meta_targets)

        return self

    def predict(self, n_steps: int = 10, **kwargs) -> Dict:
        """
        Generate ensemble forecast.

        Args:
            n_steps: Number of steps to forecast
            **kwargs: Additional arguments

        Returns:
            Dictionary with forecast results
        """
        if self.meta_model is None:
            raise ValueError("Ensemble must be fitted before prediction. Call fit() first.")

        # Get predictions from all base models
        base_predictions = []
        base_uncertainties = []

        for model in self.base_models:
            try:
                result = model.predict(n_steps=n_steps, **kwargs)

                if isinstance(result, dict):
                    pred = result.get('forecast', result.get('mean'))
                    std = result.get('std', np.ones(n_steps) * np.std(self.prices) * 0.1)
                else:
                    pred = result
                    std = np.ones(n_steps) * np.std(self.prices) * 0.1

                base_predictions.append(pred)
                base_uncertainties.append(std)

            except Exception as e:
                warnings.warn(f"Model prediction failed: {e}. Using fallback.")
                # Fallback: last value
                base_predictions.append(np.full(n_steps, self.prices[-1]))
                base_uncertainties.append(np.ones(n_steps) * np.std(self.prices) * 0.2)

        base_predictions = np.column_stack(base_predictions)
        base_uncertainties = np.column_stack(base_uncertainties)

        # Add original features if used during training
        if self.use_original_features:
            lookback = min(5, len(self.prices) // 2)
            recent_prices = self.prices[-lookback:]

            # For multi-step ahead, we need to simulate features
            # For simplicity, use last known prices
            original_features = np.tile(recent_prices, (n_steps, 1))
            meta_features = np.hstack([base_predictions, original_features])
        else:
            meta_features = base_predictions

        # Generate meta-learner prediction
        forecast = self.meta_model.predict(meta_features)

        # Combine uncertainties (inverse variance weighting)
        # More weight to models with lower uncertainty
        weights = 1.0 / (base_uncertainties ** 2 + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted average of uncertainties
        combined_std = np.sqrt(np.sum(weights * base_uncertainties ** 2, axis=1))

        # Prediction intervals
        lower = forecast - 1.96 * combined_std
        upper = forecast + 1.96 * combined_std

        return {
            'forecast': forecast,
            'mean': forecast,
            'std': combined_std,
            'lower': lower,
            'upper': upper,
            'model_name': 'Stacking',
            'params': {
                'n_models': len(self.base_models),
                'meta_learner': self.meta_learner_type,
                'model_names': self.model_names
            },
            'base_predictions': base_predictions  # Include for analysis
        }

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get the importance/weights of each base model.

        Returns:
            Dictionary mapping model names to importance scores
        """
        if self.meta_model is None:
            return {}

        # For linear models, use coefficients
        if hasattr(self.meta_model, 'coef_'):
            coefs = self.meta_model.coef_[:len(self.base_models)]
            # Normalize to sum to 1
            weights = np.abs(coefs) / np.abs(coefs).sum()
            return dict(zip(self.model_names, weights))

        # For tree-based models, use feature importances
        elif hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_[:len(self.base_models)]
            weights = importances / importances.sum()
            return dict(zip(self.model_names, weights))

        return {}


class BoostingForecaster:
    """
    Boosting ensemble forecaster using sequential error correction.

    Trains models sequentially, where each new model focuses on
    correcting errors made by the previous ensemble.

    Args:
        base_model_configs: List of (model_class, model_params) tuples
        n_estimators: Number of boosting iterations
        learning_rate: Shrinkage factor for each model's contribution
        loss: Loss function ('mse' or 'mae')

    Example:
        >>> from fractime import FractalForecaster
        >>> from fractime.baselines import ARIMAForecaster
        >>>
        >>> configs = [
        ...     (FractalForecaster, {}),
        ...     (ARIMAForecaster, {'order': (1,1,1)})
        ... ]
        >>>
        >>> booster = BoostingForecaster(configs, n_estimators=5)
        >>> booster.fit(prices)
        >>> forecast = booster.predict(n_steps=10)
    """

    def __init__(
        self,
        base_model_configs: Optional[List[tuple]] = None,
        n_estimators: int = 5,
        learning_rate: float = 0.1,
        loss: str = 'mse'
    ):
        self.base_model_configs = base_model_configs or []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss

        self.models = []
        self.model_weights = []
        self.prices = None
        self.initial_prediction = None

    def add_model_config(self, model_class, model_params: Optional[Dict] = None):
        """
        Add a model configuration to the boosting sequence.

        Args:
            model_class: Model class to instantiate
            model_params: Parameters to pass to model constructor
        """
        self.base_model_configs.append((model_class, model_params or {}))

    def fit(self, prices: np.ndarray, **kwargs) -> 'BoostingForecaster':
        """
        Fit the boosting ensemble.

        Args:
            prices: Historical price series
            **kwargs: Additional arguments

        Returns:
            self: Fitted ensemble
        """
        self.prices = np.asarray(prices).flatten()

        if len(self.base_model_configs) == 0:
            raise ValueError("No model configurations added. Use add_model_config() first.")

        # Initial prediction (mean)
        self.initial_prediction = np.mean(self.prices)
        residuals = self.prices - self.initial_prediction

        self.models = []
        self.model_weights = []

        # Boosting iterations
        for i in range(self.n_estimators):
            # Select model configuration (cycle through if needed)
            model_class, model_params = self.base_model_configs[i % len(self.base_model_configs)]

            try:
                # Create and fit model on residuals
                model = model_class(**model_params)

                # For models that expect prices, we give them original prices
                # but will use residual-based weighting
                model.fit(self.prices)

                # Get in-sample predictions
                n_test = min(20, len(self.prices) // 4)
                result = model.predict(n_steps=n_test)

                if isinstance(result, dict):
                    pred = result.get('forecast', result.get('mean'))
                else:
                    pred = result

                # Calculate model weight based on performance on residuals
                # Use last n_test points for evaluation
                if len(residuals) >= n_test:
                    actual_residuals = residuals[-n_test:]

                    if self.loss == 'mse':
                        error = np.mean((actual_residuals - (pred - self.prices[-n_test:])) ** 2)
                    else:  # mae
                        error = np.mean(np.abs(actual_residuals - (pred - self.prices[-n_test:])))

                    # Weight inversely proportional to error
                    weight = 1.0 / (error + 1e-8)
                else:
                    weight = 1.0

                # Apply learning rate
                weight *= self.learning_rate

                self.models.append(model)
                self.model_weights.append(weight)

                # Update residuals
                # This is simplified - in practice would need proper residual updates
                if len(residuals) >= n_test:
                    residual_update = pred - self.prices[-n_test:]
                    residuals[-n_test:] -= weight * residual_update

            except Exception as e:
                warnings.warn(f"Boosting iteration {i} failed: {e}. Stopping early.")
                break

        # Normalize weights
        if len(self.model_weights) > 0:
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]

        return self

    def predict(self, n_steps: int = 10, **kwargs) -> Dict:
        """
        Generate boosted ensemble forecast.

        Args:
            n_steps: Number of steps to forecast
            **kwargs: Additional arguments

        Returns:
            Dictionary with forecast results
        """
        if len(self.models) == 0:
            raise ValueError("Ensemble must be fitted before prediction. Call fit() first.")

        # Start with initial prediction
        forecast = np.full(n_steps, self.initial_prediction)
        all_stds = []

        # Add weighted predictions from each model
        for model, weight in zip(self.models, self.model_weights):
            try:
                result = model.predict(n_steps=n_steps, **kwargs)

                if isinstance(result, dict):
                    pred = result.get('forecast', result.get('mean'))
                    std = result.get('std', np.ones(n_steps) * np.std(self.prices) * 0.1)
                else:
                    pred = result
                    std = np.ones(n_steps) * np.std(self.prices) * 0.1

                # Add weighted prediction
                forecast += weight * (pred - self.initial_prediction)
                all_stds.append(std * weight)

            except Exception as e:
                warnings.warn(f"Model prediction failed: {e}")

        # Combine uncertainties
        if all_stds:
            combined_std = np.sqrt(np.sum(np.array(all_stds) ** 2, axis=0))
        else:
            combined_std = np.ones(n_steps) * np.std(self.prices) * 0.1

        # Prediction intervals
        lower = forecast - 1.96 * combined_std
        upper = forecast + 1.96 * combined_std

        return {
            'forecast': forecast,
            'mean': forecast,
            'std': combined_std,
            'lower': lower,
            'upper': upper,
            'model_name': 'Boosting',
            'params': {
                'n_estimators': len(self.models),
                'learning_rate': self.learning_rate,
                'model_weights': self.model_weights
            }
        }

    def get_model_weights(self) -> List[float]:
        """Get the weights for each model in the ensemble."""
        return self.model_weights
