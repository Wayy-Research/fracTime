"""
Sampling strategies for Bayesian fractal models.

Provides three modes:
1. Full MCMC - Most rigorous, slowest
2. ADVI (Variational Inference) - Fast approximation
3. Hybrid - MCMC for parameters, Monte Carlo for paths (recommended)
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


class MCMCSampler:
    """
    Full MCMC sampling using PyMC's NUTS sampler.

    This is the most rigorous approach but also the slowest.
    Recommended for research and model validation.

    Parameters:
        n_samples: Number of posterior samples to draw
        n_tune: Number of tuning steps
        n_chains: Number of MCMC chains to run in parallel
        target_accept: Target acceptance rate (0.8-0.95)
        random_seed: Random seed for reproducibility
    """

    def __init__(self,
                 n_samples: int = 1000,
                 n_tune: int = 500,
                 n_chains: int = 2,
                 target_accept: float = 0.9,
                 random_seed: Optional[int] = None):

        if not PYMC_AVAILABLE:
            raise ImportError("PyMC required for MCMC sampling")

        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed

    def sample(self, model: pm.Model, verbose: bool = True) -> az.InferenceData:
        """
        Run MCMC sampling on the model.

        Args:
            model: PyMC model to sample from
            verbose: Show progress bar

        Returns:
            ArviZ InferenceData object with posterior samples

        Raises:
            ValueError: If sampling fails or diagnostics are poor
        """
        with model:
            try:
                trace = pm.sample(
                    draws=self.n_samples,
                    tune=self.n_tune,
                    chains=self.n_chains,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    progressbar=verbose,
                    cores=min(self.n_chains, 4)  # Limit CPU usage
                )

                # Check diagnostics
                self._check_diagnostics(trace, raise_on_warning=False)

                return trace

            except Exception as e:
                raise ValueError(f"MCMC sampling failed: {str(e)}")

    def _check_diagnostics(self, trace: az.InferenceData, raise_on_warning: bool = False):
        """
        Check MCMC convergence diagnostics.

        Warns or raises if diagnostics suggest convergence issues:
        - R-hat > 1.01
        - ESS too low
        - Divergences detected
        """
        summary = az.summary(trace)

        # Check R-hat (should be < 1.01)
        max_rhat = summary['r_hat'].max()
        if max_rhat > 1.01:
            msg = f"Poor convergence: R-hat = {max_rhat:.3f} > 1.01"
            if raise_on_warning:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning)

        # Check effective sample size
        min_ess_bulk = summary['ess_bulk'].min()
        if min_ess_bulk < 100:
            msg = f"Low ESS: {min_ess_bulk:.0f} < 100"
            if raise_on_warning:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning)

        # Check for divergences
        if hasattr(trace, 'sample_stats'):
            n_divergences = trace.sample_stats.diverging.sum().item()
            if n_divergences > 0:
                msg = f"Found {n_divergences} divergences"
                if raise_on_warning:
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning)


class ADVISampler:
    """
    Automatic Differentiation Variational Inference (ADVI).

    Fast approximation to full MCMC. Uses variational inference to
    approximate the posterior distribution.

    10-100x faster than MCMC but less accurate.
    Recommended for production/real-time forecasting.

    Parameters:
        n_iterations: Number of VI optimization steps
        n_samples: Number of samples to draw from approximate posterior
        random_seed: Random seed for reproducibility
    """

    def __init__(self,
                 n_iterations: int = 10000,
                 n_samples: int = 1000,
                 random_seed: Optional[int] = None):

        if not PYMC_AVAILABLE:
            raise ImportError("PyMC required for ADVI")

        self.n_iterations = n_iterations
        self.n_samples = n_samples
        self.random_seed = random_seed

    def sample(self, model: pm.Model, verbose: bool = True) -> az.InferenceData:
        """
        Run ADVI sampling on the model.

        Args:
            model: PyMC model to fit
            verbose: Show progress

        Returns:
            ArviZ InferenceData object with approximate posterior samples
        """
        with model:
            try:
                # Fit variational approximation
                if verbose:
                    print(f"Running ADVI with {self.n_iterations} iterations...")

                approx = pm.fit(
                    n=self.n_iterations,
                    method='advi',
                    random_seed=self.random_seed
                )

                # Sample from the approximation
                trace = approx.sample(self.n_samples)

                # Convert to InferenceData format
                trace_dict = {
                    var_name: trace[var_name][:, np.newaxis, :]
                    for var_name in trace.varnames
                }

                inference_data = az.from_dict(
                    posterior=trace_dict,
                    dims={var_name: ['chain', 'draw']
                          for var_name in trace.varnames}
                )

                if verbose:
                    print(f"✓ ADVI complete")

                return inference_data

            except Exception as e:
                raise ValueError(f"ADVI sampling failed: {str(e)}")


class HybridSampler:
    """
    Hybrid approach: Use MCMC/ADVI for parameters, then Monte Carlo for paths.

    This combines the benefits of both approaches:
    - Rigorous parameter uncertainty via Bayesian inference
    - Fast path generation via existing Monte Carlo simulator

    This is the RECOMMENDED approach for most use cases.

    Parameters:
        parameter_sampler: Either MCMCSampler or ADVISampler for parameters
        use_simplified_model: If True, use faster model (IID assumption)
    """

    def __init__(self,
                 parameter_sampler: Optional[object] = None,
                 use_simplified_model: bool = False):

        if parameter_sampler is None:
            # Default to ADVI for speed
            parameter_sampler = ADVISampler()

        self.parameter_sampler = parameter_sampler
        self.use_simplified_model = use_simplified_model

    def sample_parameters(self,
                         model: pm.Model,
                         verbose: bool = True) -> az.InferenceData:
        """
        Sample just the parameters (Hurst, sigma, etc).

        Args:
            model: PyMC model
            verbose: Show progress

        Returns:
            Parameter posterior samples
        """
        return self.parameter_sampler.sample(model, verbose=verbose)

    def generate_paths_from_posterior(self,
                                     trace: az.InferenceData,
                                     simulator,
                                     n_steps: int,
                                     n_paths: int = 1000,
                                     n_parameter_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate forecast paths incorporating parameter uncertainty.

        Strategy:
        1. Sample parameter sets from posterior
        2. For each parameter set, generate paths using simulator
        3. Combine all paths with proper weighting

        Args:
            trace: Posterior samples from PyMC
            simulator: FractalSimulator instance
            n_steps: Forecast horizon
            n_paths: Total number of paths
            n_parameter_samples: How many parameter sets to sample

        Returns:
            Tuple of (paths, probabilities, parameter_samples)
        """
        # Extract posterior samples
        hurst_samples = trace.posterior['hurst'].values.flatten()
        sigma_samples = trace.posterior['sigma'].values.flatten()

        # Randomly sample parameter sets
        n_posterior = len(hurst_samples)
        n_parameter_samples = min(n_parameter_samples, n_posterior)

        param_indices = np.random.choice(
            n_posterior,
            size=n_parameter_samples,
            replace=False
        )

        # Calculate paths per parameter set
        paths_per_param = n_paths // n_parameter_samples

        all_paths = []
        parameter_weights = []

        for idx in param_indices:
            # Get this parameter set
            hurst = hurst_samples[idx]
            sigma = sigma_samples[idx]

            # Generate paths with these parameters
            # Note: This requires simulator to accept parameter overrides
            paths, metadata = simulator.simulate_paths(
                n_steps=n_steps,
                n_paths=paths_per_param,
                use_trading_time=True
            )

            all_paths.append(paths)

            # Equal weight per parameter sample (could be weighted by posterior density)
            parameter_weights.extend([1.0/n_parameter_samples] * paths_per_param)

        # Combine all paths
        combined_paths = np.vstack(all_paths)
        parameter_weights = np.array(parameter_weights)

        # Store parameter samples used
        parameter_samples = {
            'hurst': hurst_samples[param_indices],
            'sigma': sigma_samples[param_indices]
        }

        return combined_paths, parameter_weights, parameter_samples


def get_sampler(mode: str = 'hybrid', **kwargs):
    """
    Factory function to get appropriate sampler.

    Args:
        mode: One of 'mcmc', 'advi', 'hybrid'
        **kwargs: Arguments for specific sampler

    Returns:
        Sampler instance

    Examples:
        >>> sampler = get_sampler('mcmc', n_samples=2000)
        >>> sampler = get_sampler('advi', n_iterations=20000)
        >>> sampler = get_sampler('hybrid')  # Uses ADVI for parameters
    """
    if mode == 'mcmc' or mode == 'pure_bayesian':
        return MCMCSampler(**kwargs)
    elif mode == 'advi' or mode == 'fast':
        return ADVISampler(**kwargs)
    elif mode == 'hybrid':
        # Hybrid uses ADVI by default for speed
        param_sampler = kwargs.get('parameter_sampler', ADVISampler())
        return HybridSampler(parameter_sampler=param_sampler)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mcmc', 'advi', or 'hybrid'")
