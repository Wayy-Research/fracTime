"""
Exogenous predictors module for fractal time series forecasting.

This module provides functionality for incorporating exogenous variables
(external predictors) into the fractal forecasting framework, enabling
the model to condition forecasts on external factors like economic indicators,
market indices, or other correlated time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy import stats
import warnings


class ExogenousHandler:
    """
    Handles exogenous variables for fractal forecasting.

    This class manages the preprocessing, analysis, and integration of
    external predictors into the fractal simulation framework.

    Features:
        - Automatic lag selection for each exogenous variable
        - Cross-correlation analysis with target series
        - Fractal coherence between exogenous and target
        - Regime-conditional effects
    """

    def __init__(
        self,
        max_lags: int = 10,
        min_correlation: float = 0.1,
        use_differences: bool = True,
        scale_features: bool = True
    ):
        """
        Initialize the exogenous handler.

        Args:
            max_lags: Maximum number of lags to consider for each variable
            min_correlation: Minimum correlation to include a variable
            use_differences: Whether to use differenced/returns of exogenous vars
            scale_features: Whether to standardize exogenous features
        """
        self.max_lags = max_lags
        self.min_correlation = min_correlation
        self.use_differences = use_differences
        self.scale_features = scale_features

        self.scaler = StandardScaler() if scale_features else None
        self.exog_data = None
        self.exog_names = None
        self.lag_structure = {}
        self.correlations = {}
        self.fitted = False

    def fit(
        self,
        target: np.ndarray,
        exogenous: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]
    ) -> 'ExogenousHandler':
        """
        Fit the exogenous handler to target and exogenous data.

        Args:
            target: Target time series (prices)
            exogenous: Exogenous variables as array, DataFrame, or dict

        Returns:
            Self for method chaining
        """
        # Convert exogenous to standardized format
        self.exog_data, self.exog_names = self._preprocess_exogenous(exogenous)

        # Ensure same length as target
        min_len = min(len(target), self.exog_data.shape[0])
        target = target[-min_len:]
        self.exog_data = self.exog_data[-min_len:]

        # Compute target returns
        target_returns = np.diff(np.log(target))

        # Analyze each exogenous variable
        for i, name in enumerate(self.exog_names):
            exog_series = self.exog_data[:, i]

            # Compute returns/differences if requested
            if self.use_differences:
                # Use log returns if all positive, else simple differences
                if np.all(exog_series > 0):
                    exog_changes = np.diff(np.log(exog_series))
                else:
                    exog_changes = np.diff(exog_series)
            else:
                exog_changes = exog_series[1:]  # Just align length

            # Find optimal lag structure
            lag_correlations = []
            for lag in range(self.max_lags + 1):
                if lag == 0:
                    corr = np.corrcoef(
                        target_returns[:len(exog_changes)],
                        exog_changes[:len(target_returns)]
                    )[0, 1]
                else:
                    # Lagged correlation (exogenous leads target)
                    min_len_corr = min(len(target_returns) - lag, len(exog_changes) - lag)
                    if min_len_corr < 10:
                        corr = 0
                    else:
                        corr = np.corrcoef(
                            target_returns[lag:lag+min_len_corr],
                            exog_changes[:min_len_corr]
                        )[0, 1]

                if np.isnan(corr):
                    corr = 0
                lag_correlations.append((lag, corr))

            # Select best lag (highest absolute correlation)
            best_lag, best_corr = max(lag_correlations, key=lambda x: abs(x[1]))

            self.lag_structure[name] = {
                'best_lag': best_lag,
                'correlation': best_corr,
                'all_correlations': lag_correlations,
                'include': abs(best_corr) >= self.min_correlation
            }
            self.correlations[name] = best_corr

        # Scale features if requested
        if self.scale_features:
            self.scaler.fit(self.exog_data)

        self.fitted = True
        return self

    def _preprocess_exogenous(
        self,
        exogenous: Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert exogenous data to standardized format.

        Args:
            exogenous: Input exogenous data in various formats

        Returns:
            Tuple of (data array, column names)
        """
        if isinstance(exogenous, pd.DataFrame):
            names = list(exogenous.columns)
            data = exogenous.values
        elif isinstance(exogenous, dict):
            names = list(exogenous.keys())
            data = np.column_stack([exogenous[name] for name in names])
        elif isinstance(exogenous, np.ndarray):
            if exogenous.ndim == 1:
                data = exogenous.reshape(-1, 1)
                names = ['exog_0']
            else:
                data = exogenous
                names = [f'exog_{i}' for i in range(data.shape[1])]
        else:
            raise ValueError(f"Unsupported exogenous type: {type(exogenous)}")

        return data.astype(np.float64), names

    def get_feature_matrix(
        self,
        target: np.ndarray,
        exogenous: Optional[Union[np.ndarray, pd.DataFrame, Dict]] = None,
        include_lags: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix from exogenous variables.

        Args:
            target: Target time series
            exogenous: Optional new exogenous data (uses fitted if None)
            include_lags: Whether to include lagged features

        Returns:
            Tuple of (feature_matrix, aligned_target)
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")

        if exogenous is not None:
            exog_data, _ = self._preprocess_exogenous(exogenous)
        else:
            exog_data = self.exog_data

        # Build feature matrix with optimal lags
        features = []
        max_lag_used = 0

        for i, name in enumerate(self.exog_names):
            if not self.lag_structure[name]['include']:
                continue

            best_lag = self.lag_structure[name]['best_lag']
            max_lag_used = max(max_lag_used, best_lag)

            exog_series = exog_data[:, i]

            # Add level feature
            if include_lags and best_lag > 0:
                # Lagged feature
                lagged = np.zeros(len(exog_series))
                lagged[best_lag:] = exog_series[:-best_lag]
                features.append(lagged)
            else:
                features.append(exog_series)

            # Add change feature
            if self.use_differences:
                if np.all(exog_series > 0):
                    changes = np.concatenate([[0], np.diff(np.log(exog_series))])
                else:
                    changes = np.concatenate([[0], np.diff(exog_series)])

                if include_lags and best_lag > 0:
                    lagged_changes = np.zeros(len(changes))
                    lagged_changes[best_lag:] = changes[:-best_lag]
                    features.append(lagged_changes)
                else:
                    features.append(changes)

        if not features:
            # No exogenous features meet threshold
            return np.zeros((len(target), 0)), target

        X = np.column_stack(features)

        # Align with target
        if max_lag_used > 0:
            X = X[max_lag_used:]
            target_aligned = target[max_lag_used:]
        else:
            target_aligned = target

        # Scale if requested
        if self.scale_features and self.scaler is not None:
            # Only scale fitted features
            X_scaled = np.zeros_like(X)
            for i in range(X.shape[1]):
                X_scaled[:, i] = (X[:, i] - np.mean(X[:, i])) / (np.std(X[:, i]) + 1e-8)
            X = X_scaled

        return X, target_aligned

    def get_summary(self) -> Dict:
        """
        Get summary of exogenous variable analysis.

        Returns:
            Dictionary with analysis results
        """
        if not self.fitted:
            return {'fitted': False}

        summary = {
            'fitted': True,
            'n_variables': len(self.exog_names),
            'n_included': sum(1 for v in self.lag_structure.values() if v['include']),
            'variables': {}
        }

        for name in self.exog_names:
            lag_info = self.lag_structure[name]
            summary['variables'][name] = {
                'best_lag': lag_info['best_lag'],
                'correlation': lag_info['correlation'],
                'included': lag_info['include']
            }

        return summary


class ExogenousRegimeModifier:
    """
    Modifies fractal path probabilities based on exogenous regime conditions.

    This class learns how different exogenous conditions affect the
    probability distribution of future paths, allowing the forecaster
    to condition on current exogenous state.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        model_type: str = 'ridge'
    ):
        """
        Initialize the regime modifier.

        Args:
            n_regimes: Number of exogenous regimes to identify
            model_type: Type of model to use ('ridge', 'elasticnet', 'rf', 'gbm')
        """
        self.n_regimes = n_regimes
        self.model_type = model_type

        # Initialize model
        if model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'elasticnet':
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=50, max_depth=5)
        elif model_type == 'gbm':
            self.model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.regime_labels = None
        self.regime_centers = None
        self.fitted = False

    def fit(
        self,
        target_returns: np.ndarray,
        exog_features: np.ndarray
    ) -> 'ExogenousRegimeModifier':
        """
        Fit the regime modifier.

        Args:
            target_returns: Historical target returns
            exog_features: Feature matrix from ExogenousHandler

        Returns:
            Self for method chaining
        """
        if exog_features.shape[0] != len(target_returns):
            min_len = min(exog_features.shape[0], len(target_returns))
            exog_features = exog_features[-min_len:]
            target_returns = target_returns[-min_len:]

        # Cluster exogenous states into regimes
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.regime_labels = kmeans.fit_predict(exog_features)
        self.regime_centers = kmeans.cluster_centers_

        # Fit return prediction model
        self.model.fit(exog_features, target_returns)

        # Store regime statistics
        self.regime_stats = {}
        for i in range(self.n_regimes):
            regime_mask = self.regime_labels == i
            regime_returns = target_returns[regime_mask]

            self.regime_stats[i] = {
                'mean': np.mean(regime_returns),
                'std': np.std(regime_returns),
                'skew': stats.skew(regime_returns) if len(regime_returns) > 2 else 0,
                'count': np.sum(regime_mask)
            }

        self.fitted = True
        return self

    def predict_return_adjustment(
        self,
        exog_features: np.ndarray
    ) -> Tuple[float, int]:
        """
        Predict return adjustment based on current exogenous state.

        Args:
            exog_features: Current exogenous feature vector

        Returns:
            Tuple of (expected_return_adjustment, current_regime)
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")

        if exog_features.ndim == 1:
            exog_features = exog_features.reshape(1, -1)

        # Predict expected return
        expected_return = self.model.predict(exog_features)[0]

        # Identify current regime
        distances = np.linalg.norm(
            self.regime_centers - exog_features,
            axis=1
        )
        current_regime = np.argmin(distances)

        return expected_return, current_regime

    def adjust_path_probabilities(
        self,
        paths: np.ndarray,
        probabilities: np.ndarray,
        exog_features: np.ndarray,
        adjustment_strength: float = 0.5
    ) -> np.ndarray:
        """
        Adjust path probabilities based on exogenous conditions.

        Args:
            paths: Simulated price paths
            probabilities: Original path probabilities
            exog_features: Current exogenous features
            adjustment_strength: How strongly to apply adjustment (0-1)

        Returns:
            Adjusted probabilities
        """
        if not self.fitted:
            return probabilities

        expected_return, current_regime = self.predict_return_adjustment(exog_features)

        # Get regime statistics
        regime_stats = self.regime_stats[current_regime]
        expected_vol = regime_stats['std']

        # Compute path returns and volatilities
        path_returns = np.log(paths[:, -1] / paths[:, 0]) / paths.shape[1]

        # Compute similarity to expected return
        return_distances = np.abs(path_returns - expected_return)
        return_scores = np.exp(-return_distances / (expected_vol + 1e-8))

        # Blend with original probabilities
        adjusted_probs = (1 - adjustment_strength) * probabilities + adjustment_strength * return_scores
        adjusted_probs = adjusted_probs / adjusted_probs.sum()

        return adjusted_probs


class ExogenousForecastAdjuster:
    """
    Adjusts fractal forecasts based on exogenous variable forecasts.

    This enables incorporating external forecasts (e.g., from other models
    or analyst estimates) into the fractal forecast.
    """

    def __init__(self):
        """Initialize the forecast adjuster."""
        self.exog_forecasts = {}
        self.adjustment_weights = {}

    def add_exog_forecast(
        self,
        name: str,
        forecast: np.ndarray,
        weight: float = 1.0
    ):
        """
        Add an exogenous forecast.

        Args:
            name: Name of the exogenous variable
            forecast: Forecast values
            weight: Weight for this forecast's influence
        """
        self.exog_forecasts[name] = forecast
        self.adjustment_weights[name] = weight

    def adjust_paths(
        self,
        paths: np.ndarray,
        exog_handler: ExogenousHandler,
        regime_modifier: Optional[ExogenousRegimeModifier] = None,
        adjustment_strength: float = 0.3
    ) -> np.ndarray:
        """
        Adjust simulated paths based on exogenous forecasts.

        Args:
            paths: Original simulated paths
            exog_handler: Fitted ExogenousHandler
            regime_modifier: Optional fitted regime modifier
            adjustment_strength: Strength of adjustment (0-1)

        Returns:
            Adjusted paths
        """
        if not self.exog_forecasts:
            return paths

        n_paths, n_steps = paths.shape
        adjusted_paths = paths.copy()

        # Build expected adjustments from exogenous forecasts
        total_adjustment = np.zeros(n_steps)
        total_weight = 0

        for name, forecast in self.exog_forecasts.items():
            if name not in exog_handler.correlations:
                continue

            correlation = exog_handler.correlations[name]
            weight = self.adjustment_weights.get(name, 1.0)

            # Compute expected impact based on correlation
            forecast_changes = np.diff(forecast, prepend=forecast[0])
            expected_impact = correlation * forecast_changes * weight

            # Adjust to match path length
            if len(expected_impact) != n_steps:
                # Interpolate or truncate
                if len(expected_impact) > n_steps:
                    expected_impact = expected_impact[:n_steps]
                else:
                    # Extend with last value
                    expected_impact = np.concatenate([
                        expected_impact,
                        np.full(n_steps - len(expected_impact), expected_impact[-1])
                    ])

            total_adjustment += expected_impact
            total_weight += abs(weight)

        if total_weight > 0:
            total_adjustment /= total_weight

        # Apply adjustment to paths
        for i in range(n_paths):
            # Scale adjustment by path volatility to maintain character
            path_vol = np.std(np.diff(np.log(paths[i])))
            scaled_adjustment = total_adjustment * path_vol * adjustment_strength

            # Apply as multiplicative adjustment
            cumulative_adjustment = np.cumsum(scaled_adjustment)
            adjusted_paths[i] = paths[i] * np.exp(cumulative_adjustment)

        return adjusted_paths


def compute_exogenous_fractal_coherence(
    target: np.ndarray,
    exogenous: np.ndarray,
    window_sizes: List[int] = [21, 63, 126]
) -> Dict:
    """
    Compute fractal coherence between target and exogenous series.

    This measures how well the fractal properties (Hurst exponent,
    volatility clustering) of the exogenous series align with the target.

    Args:
        target: Target time series
        exogenous: Exogenous time series
        window_sizes: Windows for multi-scale analysis

    Returns:
        Dictionary with coherence metrics
    """
    from .analysis import FractalAnalyzer

    analyzer = FractalAnalyzer()

    results = {
        'overall_coherence': 0,
        'scale_coherence': {},
        'hurst_correlation': 0,
        'volatility_correlation': 0
    }

    # Compute rolling Hurst exponents
    target_hursts = []
    exog_hursts = []

    for window in window_sizes:
        t_hursts = []
        e_hursts = []

        for i in range(window, len(target)):
            t_segment = target[i-window:i]
            e_segment = exogenous[i-window:i]

            try:
                t_h = analyzer.compute_hurst(t_segment)
                e_h = analyzer.compute_hurst(e_segment)
                t_hursts.append(t_h)
                e_hursts.append(e_h)
            except:
                continue

        if len(t_hursts) > 10:
            hurst_corr = np.corrcoef(t_hursts, e_hursts)[0, 1]
            if not np.isnan(hurst_corr):
                results['scale_coherence'][window] = hurst_corr
                target_hursts.extend(t_hursts)
                exog_hursts.extend(e_hursts)

    # Overall Hurst correlation
    if len(target_hursts) > 10:
        results['hurst_correlation'] = np.corrcoef(target_hursts, exog_hursts)[0, 1]

    # Volatility clustering correlation
    target_returns = np.diff(np.log(target))
    exog_returns = np.diff(np.log(exogenous)) if np.all(exogenous > 0) else np.diff(exogenous)

    target_vol = np.abs(target_returns)
    exog_vol = np.abs(exog_returns[:len(target_vol)])

    if len(target_vol) > 20:
        # Compute volatility autocorrelations
        target_vol_ac = np.corrcoef(target_vol[:-5], target_vol[5:])[0, 1]
        exog_vol_ac = np.corrcoef(exog_vol[:-5], exog_vol[5:])[0, 1]

        # Cross volatility correlation
        vol_cross_corr = np.corrcoef(target_vol, exog_vol)[0, 1]

        if not np.isnan(vol_cross_corr):
            results['volatility_correlation'] = vol_cross_corr

    # Compute overall coherence
    coherence_values = list(results['scale_coherence'].values())
    if coherence_values:
        results['overall_coherence'] = np.mean(np.abs(coherence_values))

    return results
