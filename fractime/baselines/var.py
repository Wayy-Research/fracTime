"""
Vector Autoregression (VAR) baseline model.

Implements multivariate time series forecasting using statsmodels VAR.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Union

# Try to import statsmodels
try:
    from statsmodels.tsa.api import VAR as StatsVAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    StatsVAR = None


class VARForecaster:
    """
    Vector Autoregression (VAR) forecasting model.

    VAR models capture linear interdependencies among multiple time series.
    Each variable is modeled as a linear function of past values of itself
    and past values of other variables.

    Args:
        maxlags: Maximum number of lags to consider (None for automatic selection)
        ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
        trend: Trend specification ('c'=constant, 'ct'=constant+trend, 'ctt'=constant+linear+quadratic, 'n'=none)

    Example:
        >>> # For multivariate forecasting (e.g., price + volume)
        >>> data = np.column_stack([prices, volumes])
        >>> model = VARForecaster(maxlags=5, ic='aic')
        >>> model.fit(data)
        >>> forecast = model.predict(n_steps=10)
    """

    def __init__(
        self,
        maxlags: Optional[int] = None,
        ic: str = 'aic',
        trend: str = 'c'
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels not installed. Install with: uv pip install statsmodels"
            )

        self.maxlags = maxlags
        self.ic = ic
        self.trend = trend
        self.model = None
        self.fitted_model = None
        self.data = None
        self.n_vars = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'VARForecaster':
        """
        Fit the VAR model to multivariate time series data.

        Args:
            data: Multivariate time series (T x K) where T is time steps and K is number of variables
            **kwargs: Additional arguments passed to VAR.fit()

        Returns:
            self: Fitted model
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                # Univariate case - convert to 2D
                data = data.reshape(-1, 1)
            self.data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        else:
            self.data = data

        self.n_vars = self.data.shape[1]

        # Handle edge cases
        if len(self.data) < 20:
            warnings.warn(
                f"Very short time series ({len(self.data)} points). "
                "VAR may not perform well."
            )

        # Automatically determine maxlags if not specified
        if self.maxlags is None:
            # Rule of thumb: maxlags = min(10, T/4)
            self.maxlags = min(10, len(self.data) // 4)

        try:
            # Create the model
            self.model = StatsVAR(self.data)

            # Fit with automatic lag selection
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.fitted_model = self.model.fit(
                    maxlags=self.maxlags,
                    ic=self.ic,
                    trend=self.trend,
                    **kwargs
                )

        except Exception as e:
            warnings.warn(f"VAR fitting failed: {e}. Using fallback.")
            # Fallback to simpler model with fewer lags
            try:
                self.fitted_model = self.model.fit(
                    maxlags=min(2, len(self.data) // 10),
                    trend='c'
                )
            except:
                # Ultimate fallback: lag=1, no trend
                self.fitted_model = self.model.fit(maxlags=1, trend='n')

        return self

    def predict(self, n_steps: int = 10, return_all_vars: bool = False, **kwargs) -> Dict:
        """
        Generate forecast for n_steps ahead.

        Args:
            n_steps: Number of steps to forecast
            return_all_vars: If True, return forecasts for all variables; if False, only first variable
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Dictionary with forecast results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Generate forecast
        forecast_result = self.fitted_model.forecast(self.data.values[-self.fitted_model.k_ar:], steps=n_steps)

        # Get forecast for first variable (typically price)
        if self.n_vars == 1 or not return_all_vars:
            forecast = forecast_result[:, 0]
        else:
            # Return all variables
            forecast = forecast_result

        # Generate prediction intervals using simulation
        try:
            # Simulate from the VAR model to get distribution
            n_sim = 1000
            simulations = []

            for _ in range(n_sim):
                # Use the model's simulate method
                sim = self.fitted_model.simulate_var(nsimulations=n_steps)
                simulations.append(sim[:, 0] if not return_all_vars else sim)

            simulations = np.array(simulations)

            # Calculate intervals
            if simulations.ndim == 2:
                # Univariate forecast
                lower = np.percentile(simulations, 2.5, axis=0)
                upper = np.percentile(simulations, 97.5, axis=0)
                mean = np.mean(simulations, axis=0)
                std = np.std(simulations, axis=0)
            else:
                # Multivariate forecast
                lower = np.percentile(simulations, 2.5, axis=0)
                upper = np.percentile(simulations, 97.5, axis=0)
                mean = np.mean(simulations, axis=0)
                std = np.std(simulations, axis=0)

        except:
            # Fallback: use expanding standard errors
            residuals = self.fitted_model.resid.values
            if residuals.ndim == 1:
                residual_std = np.std(residuals)
            else:
                residual_std = np.std(residuals[:, 0])

            expanding_std = residual_std * np.sqrt(np.arange(1, n_steps + 1))

            if forecast.ndim == 1:
                lower = forecast - 1.96 * expanding_std
                upper = forecast + 1.96 * expanding_std
                mean = forecast
                std = expanding_std
            else:
                # Multivariate case
                lower = forecast - 1.96 * expanding_std[:, np.newaxis]
                upper = forecast + 1.96 * expanding_std[:, np.newaxis]
                mean = forecast
                std = np.tile(expanding_std[:, np.newaxis], (1, forecast.shape[1]))

        return {
            'forecast': forecast,
            'mean': mean,
            'std': std,
            'lower': lower,
            'upper': upper,
            'model_name': 'VAR',
            'params': {
                'lags': self.fitted_model.k_ar,
                'n_vars': self.n_vars,
                'trend': self.trend
            }
        }

    def get_model_params(self) -> Dict:
        """Get fitted model parameters."""
        if self.fitted_model is None:
            return {}

        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'fpe': self.fitted_model.fpe,
            'hqic': self.fitted_model.hqic,
            'lags': self.fitted_model.k_ar,
            'n_vars': self.n_vars,
            'n_obs': self.fitted_model.nobs
        }

    def impulse_response(self, periods: int = 10, impulse_var: int = 0, response_var: int = 0) -> np.ndarray:
        """
        Compute impulse response function.

        Shows how a shock to one variable affects another variable over time.

        Args:
            periods: Number of periods to compute
            impulse_var: Index of variable receiving the shock
            response_var: Index of variable whose response we track

        Returns:
            Array of impulse responses
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first.")

        irf = self.fitted_model.irf(periods=periods)
        return irf.irfs[:, response_var, impulse_var]

    def forecast_error_variance_decomposition(self, periods: int = 10, var_index: int = 0) -> pd.DataFrame:
        """
        Decompose forecast error variance.

        Shows what proportion of the forecast error variance for a variable
        is due to shocks in each variable.

        Args:
            periods: Number of periods ahead
            var_index: Index of variable to analyze

        Returns:
            DataFrame with variance decomposition
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first.")

        fevd = self.fitted_model.fevd(periods=periods)
        return fevd.decomp[:, var_index, :]
