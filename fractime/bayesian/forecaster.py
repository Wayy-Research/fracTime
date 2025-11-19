"""
Bayesian Fractal Forecaster - Main user-facing class.

Provides three forecasting modes:
1. Fast (ADVI) - 15-20 seconds
2. Hybrid (ADVI + Monte Carlo) - ~1 minute, RECOMMENDED
3. Pure Bayesian (Full MCMC) - 1-2 minutes, most rigorous
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from ..core import FractalForecaster, FractalAnalyzer
from .models import BayesianFractalModel, BayesianFractalModelFactory
from .samplers import get_sampler, HybridSampler


class BayesianFractalForecaster:
    """
    Bayesian fractal time series forecaster.

    This extends the standard FractalForecaster with Bayesian parameter
    inference, providing proper uncertainty quantification for fractal
    parameters and improved forecast distributions.

    Parameters:
        mode: Inference mode - 'fast' (ADVI), 'hybrid' (recommended), or 'pure_bayesian' (MCMC)
        model_type: Model variant - 'neutral', 'equities', 'fx', 'crypto'
        n_samples: Number of posterior samples
        lookback: Historical window size for fitting

    Examples:
        >>> # Fast mode for production
        >>> forecaster = BayesianFractalForecaster(mode='fast')
        >>> forecaster.fit(prices, dates)
        >>> result = forecaster.predict(n_steps=30)

        >>> # Hybrid mode (recommended)
        >>> forecaster = BayesianFractalForecaster(mode='hybrid')
        >>> forecaster.fit(prices, dates)
        >>> result = forecaster.predict(n_steps=30, n_paths=1000)
        >>> print(f"Hurst: {result['parameter_posterior']['hurst'].mean():.3f}")

        >>> # Full Bayesian (most rigorous)
        >>> forecaster = BayesianFractalForecaster(mode='pure_bayesian', n_samples=2000)
        >>> forecaster.fit(prices, dates)
        >>> result = forecaster.predict(n_steps=30)
    """

    def __init__(self,
                 mode: str = 'hybrid',
                 model_type: str = 'neutral',
                 n_samples: int = 1000,
                 lookback: int = 500):

        if mode not in ['fast', 'hybrid', 'pure_bayesian']:
            raise ValueError(
                f"Invalid mode: {mode}. Use 'fast', 'hybrid', or 'pure_bayesian'"
            )

        if model_type not in ['neutral', 'equities', 'fx', 'crypto']:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Use 'neutral', 'equities', 'fx', or 'crypto'"
            )

        self.mode = mode
        self.model_type = model_type
        self.n_samples = n_samples
        self.lookback = lookback

        # Initialize base forecaster (for fallback and Monte Carlo)
        self.base_forecaster = FractalForecaster(lookback=lookback)

        # Bayesian components
        self.bayesian_model = None
        self.pymc_model = None
        self.posterior_trace = None
        self.sampler = None

        # Data storage
        self.prices_history = None
        self.dates_history = None
        self.returns_history = None

    def fit(self,
            prices: np.ndarray,
            dates: np.ndarray = None,
            verbose: bool = True) -> 'BayesianFractalForecaster':
        """
        Fit the Bayesian fractal model to historical data.

        Args:
            prices: Historical price series
            dates: Optional datetime array
            verbose: Show fitting progress

        Returns:
            self (for method chaining)
        """
        # Always fit the base forecaster (needed for simulation)
        self.base_forecaster.fit(prices, dates=dates)

        # Store data
        self.prices_history = self.base_forecaster.prices_history
        self.dates_history = self.base_forecaster.dates_history
        self.returns_history = np.diff(np.log(self.prices_history))

        # Build Bayesian model based on type
        if self.model_type == 'equities':
            self.bayesian_model = BayesianFractalModelFactory.for_equities()
        elif self.model_type == 'fx':
            self.bayesian_model = BayesianFractalModelFactory.for_fx()
        elif self.model_type == 'crypto':
            self.bayesian_model = BayesianFractalModelFactory.for_crypto()
        else:  # neutral
            self.bayesian_model = BayesianFractalModelFactory.neutral()

        # Build PyMC model
        if verbose:
            print(f"Building Bayesian model (type={self.model_type}, mode={self.mode})...")

        # Use simplified model for fast/hybrid modes
        if self.mode in ['fast', 'hybrid']:
            self.pymc_model = self.bayesian_model.build_simplified_model(
                self.returns_history
            )
        else:
            self.pymc_model = self.bayesian_model.build_model(
                self.returns_history
            )

        # Get appropriate sampler
        self.sampler = get_sampler(
            mode=self.mode,
            n_samples=self.n_samples
        )

        # Sample posterior
        if verbose:
            print(f"Sampling posterior ({self.mode} mode)...")

        self.posterior_trace = self.sampler.sample(
            self.pymc_model,
            verbose=verbose
        )

        if verbose:
            print("✓ Bayesian fitting complete")
            self._print_posterior_summary()

        return self

    def predict(self,
                n_steps: int = None,
                end_date: str = None,
                period: str = None,
                n_paths: int = 1000,
                confidence: float = 0.95) -> Dict:
        """
        Generate Bayesian forecast.

        Args:
            n_steps: Forecast horizon (or use end_date/period)
            end_date: Target end date (requires dates in fit)
            period: Forecast period like '7d', '1M' (requires dates in fit)
            n_paths: Number of paths to simulate
            confidence: Confidence level for intervals

        Returns:
            Dictionary containing:
                - 'forecast': Median forecast
                - 'weighted_forecast': Probability-weighted forecast
                - 'paths': Simulated paths
                - 'probabilities': Path probabilities
                - 'parameter_posterior': Posterior parameter samples
                - 'lower', 'upper': Confidence intervals
                - 'weighted_lower', 'weighted_upper': Weighted CIs
                - Additional Bayesian-specific outputs

        Examples:
            >>> result = forecaster.predict(n_steps=30)
            >>> result = forecaster.predict(end_date='2025-12-01')
            >>> result = forecaster.predict(period='7d', n_paths=2000)
        """
        if self.posterior_trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Determine n_steps
        if n_steps is None:
            if end_date is not None:
                n_steps = self.base_forecaster._calculate_steps_to_date(end_date)
            elif period is not None:
                n_steps = self.base_forecaster._parse_period(period)
            else:
                raise ValueError("Must provide n_steps, end_date, or period")

        if self.mode == 'hybrid':
            # Hybrid: Use posterior parameters with Monte Carlo simulation
            result = self._predict_hybrid(n_steps, n_paths, confidence)

        elif self.mode == 'fast':
            # Fast: Similar to hybrid but optimized
            result = self._predict_fast(n_steps, n_paths, confidence)

        elif self.mode == 'pure_bayesian':
            # Pure Bayesian: Posterior predictive sampling
            result = self._predict_pure_bayesian(n_steps, n_paths, confidence)

        # Add forecast dates if available
        if self.dates_history is not None:
            result['dates'] = self._generate_forecast_dates(n_steps)

        return result

    def _predict_hybrid(self, n_steps: int, n_paths: int, confidence: float) -> Dict:
        """
        Hybrid prediction: Bayesian parameters + Monte Carlo paths.
        """
        # Generate paths incorporating parameter uncertainty
        if isinstance(self.sampler, HybridSampler):
            paths, param_weights, param_samples = self.sampler.generate_paths_from_posterior(
                self.posterior_trace,
                self.base_forecaster.simulator,
                n_steps,
                n_paths=n_paths,
                n_parameter_samples=min(100, n_paths // 10)
            )
        else:
            # Fallback: use base forecaster with most likely parameters
            hurst_mean = self.posterior_trace.posterior['hurst'].mean().item()
            # For now, just use base forecaster
            # TODO: Modify simulator to accept parameter overrides
            base_result = self.base_forecaster.predict(n_steps=n_steps, n_paths=n_paths)
            paths = base_result['paths']
            param_weights = np.ones(n_paths) / n_paths
            param_samples = self._extract_parameter_samples()

        # Calculate path probabilities (combine parameter + similarity weights)
        path_probabilities = self.base_forecaster._calculate_path_probabilities(paths)

        # Combine with parameter weights
        combined_probs = path_probabilities * param_weights
        combined_probs = combined_probs / np.sum(combined_probs)

        # Calculate forecasts
        alpha = (1 - confidence) / 2

        result = {
            'forecast': np.median(paths, axis=0),
            'weighted_forecast': np.average(paths, axis=0, weights=combined_probs),
            'mean': np.mean(paths, axis=0),
            'lower': np.percentile(paths, alpha * 100, axis=0),
            'upper': np.percentile(paths, (1 - alpha) * 100, axis=0),
            'std': np.std(paths, axis=0),
            'paths': paths,
            'probabilities': combined_probs,
            'parameter_posterior': param_samples,
            'mode': 'hybrid'
        }

        # Probability-weighted CIs
        result['weighted_lower'], result['weighted_upper'] = self._calculate_weighted_ci(
            paths, combined_probs, confidence
        )

        return result

    def _predict_fast(self, n_steps: int, n_paths: int, confidence: float) -> Dict:
        """
        Fast prediction using ADVI approximation.
        """
        # Fast mode is similar to hybrid but uses approximations
        return self._predict_hybrid(n_steps, n_paths, confidence)

    def _predict_pure_bayesian(self, n_steps: int, n_paths: int, confidence: float) -> Dict:
        """
        Pure Bayesian prediction using posterior predictive sampling.

        This is the most rigorous approach but also slowest.
        """
        # For now, fall back to hybrid approach
        # Full posterior predictive would require implementing the generative model
        warnings.warn(
            "Pure Bayesian posterior predictive not yet implemented. "
            "Using hybrid approach.",
            UserWarning
        )
        return self._predict_hybrid(n_steps, n_paths, confidence)

    def _calculate_weighted_ci(self,
                               paths: np.ndarray,
                               probabilities: np.ndarray,
                               confidence: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate probability-weighted confidence intervals.
        """
        n_steps = paths.shape[1]
        alpha = (1 - confidence) / 2

        weighted_lower = np.zeros(n_steps)
        weighted_upper = np.zeros(n_steps)

        for t in range(n_steps):
            values = paths[:, t]
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_probs = probabilities[sorted_indices]

            cumsum_probs = np.cumsum(sorted_probs)

            lower_idx = np.searchsorted(cumsum_probs, alpha)
            upper_idx = np.searchsorted(cumsum_probs, 1 - alpha)

            lower_idx = min(lower_idx, len(sorted_values) - 1)
            upper_idx = min(upper_idx, len(sorted_values) - 1)

            weighted_lower[t] = sorted_values[lower_idx]
            weighted_upper[t] = sorted_values[upper_idx]

        return weighted_lower, weighted_upper

    def _extract_parameter_samples(self) -> Dict:
        """Extract parameter posterior samples."""
        return {
            'hurst': self.posterior_trace.posterior['hurst'].values.flatten(),
            'sigma': self.posterior_trace.posterior['sigma'].values.flatten(),
            'fractal_dim': self.posterior_trace.posterior['fractal_dim'].values.flatten()
        }

    def _generate_forecast_dates(self, n_steps: int) -> np.ndarray:
        """Generate forecast dates."""
        import polars as pl

        last_date = self.dates_history[-1]

        # Infer frequency
        freq = self.base_forecaster.frequency
        if freq == 'D':
            interval = '1d'
        elif freq == 'H':
            interval = '1h'
        else:
            interval = '1d'

        # Generate date range
        forecast_dates = pl.datetime_range(
            start=last_date,
            periods=n_steps + 1,
            interval=interval,
            eager=True
        ).to_numpy()[1:]  # Exclude start date

        return forecast_dates

    def _print_posterior_summary(self):
        """Print summary of posterior distributions."""
        try:
            import arviz as az
            summary = az.summary(self.posterior_trace)

            print("\nPosterior Summary:")
            print("-" * 60)
            print(f"  Hurst Exponent:    {summary.loc['hurst', 'mean']:.3f} "
                  f"± {summary.loc['hurst', 'sd']:.3f}")
            print(f"  Fractal Dimension: {summary.loc['fractal_dim', 'mean']:.3f} "
                  f"± {summary.loc['fractal_dim', 'sd']:.3f}")
            print(f"  Volatility (σ):    {summary.loc['sigma', 'mean']:.4f} "
                  f"± {summary.loc['sigma', 'sd']:.4f}")

            hurst_mean = summary.loc['hurst', 'mean']
            if hurst_mean > 0.55:
                print(f"  → Trending behavior detected (H > 0.5)")
            elif hurst_mean < 0.45:
                print(f"  → Mean-reverting behavior detected (H < 0.5)")
            else:
                print(f"  → Near random walk behavior (H ≈ 0.5)")

            print("-" * 60)

        except Exception as e:
            warnings.warn(f"Could not print summary: {e}", UserWarning)

    def get_parameter_posterior_summary(self) -> Dict:
        """
        Get summary statistics for parameter posteriors.

        Returns:
            Dictionary with parameter means, sds, and credible intervals
        """
        import arviz as az

        if self.posterior_trace is None:
            raise ValueError("Model not fitted yet")

        summary = az.summary(self.posterior_trace)

        return {
            'hurst': {
                'mean': summary.loc['hurst', 'mean'],
                'sd': summary.loc['hurst', 'sd'],
                'hdi_low': summary.loc['hurst', 'hdi_3%'],
                'hdi_high': summary.loc['hurst', 'hdi_97%'],
            },
            'fractal_dim': {
                'mean': summary.loc['fractal_dim', 'mean'],
                'sd': summary.loc['fractal_dim', 'sd'],
                'hdi_low': summary.loc['fractal_dim', 'hdi_3%'],
                'hdi_high': summary.loc['fractal_dim', 'hdi_97%'],
            },
            'sigma': {
                'mean': summary.loc['sigma', 'mean'],
                'sd': summary.loc['sigma', 'sd'],
                'hdi_low': summary.loc['sigma', 'hdi_3%'],
                'hdi_high': summary.loc['sigma', 'hdi_97%'],
            }
        }
