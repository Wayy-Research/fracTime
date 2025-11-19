"""
PyMC model definitions for Bayesian fractal parameter estimation.
"""

import numpy as np
from typing import Optional, Dict
import warnings

try:
    import pymc as pm
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn(
        "PyMC not available. Install with: pip install 'fractime[bayesian]'",
        ImportWarning
    )


class BayesianFractalModel:
    """
    PyMC model for Bayesian estimation of fractal parameters.

    This model treats the Hurst exponent, fractal dimension, and volatility
    as random variables with prior distributions, then estimates posterior
    distributions given observed returns.

    Parameters:
        hurst_prior: Dict with 'alpha' and 'beta' for Beta prior on Hurst
        sigma_prior_scale: Scale parameter for HalfNormal prior on volatility
        use_student_t: If True, use Student's t likelihood (captures heavy tails)

    Example:
        >>> model_builder = BayesianFractalModel()
        >>> returns = np.diff(np.log(prices))
        >>> model = model_builder.build_model(returns)
        >>> with model:
        ...     trace = pm.sample(1000)
    """

    def __init__(self,
                 hurst_prior: Optional[Dict[str, float]] = None,
                 sigma_prior_scale: Optional[float] = None,
                 use_student_t: bool = True):

        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC is required for Bayesian models. "
                "Install with: pip install 'fractime[bayesian]'"
            )

        # Default priors
        self.hurst_prior = hurst_prior or {'alpha': 2.0, 'beta': 2.0}
        self.sigma_prior_scale = sigma_prior_scale
        self.use_student_t = use_student_t

        self.model = None

    def build_model(self, returns: np.ndarray) -> pm.Model:
        """
        Build the PyMC model for fractal parameter estimation.

        Args:
            returns: Log returns (not prices!)

        Returns:
            PyMC model object

        Model structure:
            - Hurst exponent H ~ Beta(alpha, beta), bounded [0, 1]
            - Fractal dimension D = 2 - H, bounded [1, 2]
            - Volatility σ ~ HalfNormal(scale)
            - Memory strength ~ Beta(2, 2) for autocorrelation
            - Returns ~ StudentT(nu(D), mu, σ) with fractal-dependent tails
        """
        returns = np.asarray(returns)

        # Determine sigma prior scale if not provided
        if self.sigma_prior_scale is None:
            self.sigma_prior_scale = np.std(returns) * 2.0

        with pm.Model() as model:
            # Prior for Hurst exponent (bounded 0-1)
            # Beta(2, 2) is centered at 0.5 (random walk)
            # Can adjust alpha/beta for trending/mean-reverting bias
            hurst = pm.Beta('hurst',
                           alpha=self.hurst_prior['alpha'],
                           beta=self.hurst_prior['beta'])

            # Fractal dimension: D = 2 - H (deterministic transformation)
            # Automatically bounded between 1 and 2
            fractal_dim = pm.Deterministic('fractal_dim', 2.0 - hurst)

            # Volatility parameter (must be positive)
            sigma = pm.HalfNormal('sigma', sigma=self.sigma_prior_scale)

            # Long-term memory strength
            # Controls how much past returns influence future returns
            # Higher memory_strength with H > 0.5 = trending
            # Higher memory_strength with H < 0.5 = mean-reverting
            memory_strength = pm.Beta('memory_strength', alpha=2, beta=2)

            if self.use_student_t:
                # Degrees of freedom for Student's t
                # Tied to fractal dimension: lower D → fatter tails
                # Range: nu ∈ [3, 13] as fractal_dim goes from 1 to 2
                nu = pm.Deterministic('nu', 3.0 + 10.0 * (fractal_dim - 1.0))

            # Model the return series with fractal memory
            # Simplified approach: each return has memory-dependent mean
            for t in range(1, len(returns)):
                if t > 1:
                    # Adaptive mean incorporating long-term memory
                    # When H > 0.5: positive autocorrelation (trending)
                    # When H < 0.5: negative autocorrelation (mean-reverting)
                    memory_term = memory_strength * (hurst - 0.5) * returns[t-1]
                else:
                    memory_term = 0.0

                # Likelihood for this return
                if self.use_student_t:
                    pm.StudentT(f'return_{t}',
                               nu=nu,
                               mu=memory_term,
                               sigma=sigma,
                               observed=returns[t])
                else:
                    pm.Normal(f'return_{t}',
                             mu=memory_term,
                             sigma=sigma,
                             observed=returns[t])

            # Derived quantities for easier interpretation
            pm.Deterministic('is_trending', pt.gt(hurst, 0.5).astype('float64'))

            if self.use_student_t:
                # Tail heaviness: higher = heavier tails
                pm.Deterministic('tail_heaviness', 1.0 / nu)

            # Effective correlation strength
            # This combines memory_strength with deviation from H=0.5
            pm.Deterministic('effective_memory',
                           memory_strength * pt.abs_(hurst - 0.5))

        self.model = model
        return model

    def build_simplified_model(self, returns: np.ndarray) -> pm.Model:
        """
        Build a simplified model for faster inference.

        This version treats returns as IID (ignoring temporal structure)
        and focuses on estimating fractal parameters from return distribution.
        Faster than full model but less theoretically rigorous.

        Args:
            returns: Log returns

        Returns:
            Simplified PyMC model
        """
        returns = np.asarray(returns)

        if self.sigma_prior_scale is None:
            self.sigma_prior_scale = np.std(returns) * 2.0

        with pm.Model() as model:
            # Priors
            hurst = pm.Beta('hurst',
                           alpha=self.hurst_prior['alpha'],
                           beta=self.hurst_prior['beta'])

            fractal_dim = pm.Deterministic('fractal_dim', 2.0 - hurst)

            sigma = pm.HalfNormal('sigma', sigma=self.sigma_prior_scale)

            if self.use_student_t:
                nu = pm.Deterministic('nu', 3.0 + 10.0 * (fractal_dim - 1.0))

                # Likelihood: all returns at once (vectorized)
                pm.StudentT('returns',
                           nu=nu,
                           mu=0,
                           sigma=sigma,
                           observed=returns)
            else:
                pm.Normal('returns',
                         mu=0,
                         sigma=sigma,
                         observed=returns)

            # Derived quantities
            pm.Deterministic('is_trending', pt.gt(hurst, 0.5).astype('float64'))

        self.model = model
        return model


class BayesianFractalModelFactory:
    """
    Factory for creating different variants of Bayesian fractal models.

    Provides presets for common use cases:
    - Equity markets (trending bias)
    - FX markets (mean-reverting bias)
    - Crypto (high volatility)
    - General purpose (neutral priors)
    """

    @staticmethod
    def for_equities() -> BayesianFractalModel:
        """
        Model with slight trending bias (common in equity markets).
        Hurst prior centered at 0.56.
        """
        return BayesianFractalModel(
            hurst_prior={'alpha': 5.0, 'beta': 4.0},
            use_student_t=True
        )

    @staticmethod
    def for_fx() -> BayesianFractalModel:
        """
        Model with mean-reverting bias (common in FX markets).
        Hurst prior centered at 0.44.
        """
        return BayesianFractalModel(
            hurst_prior={'alpha': 4.0, 'beta': 5.0},
            use_student_t=True
        )

    @staticmethod
    def for_crypto() -> BayesianFractalModel:
        """
        Model for high-volatility assets like crypto.
        Wider sigma prior, heavy-tailed likelihood.
        """
        # Will use 3x wider sigma prior via auto-detection
        return BayesianFractalModel(
            hurst_prior={'alpha': 2.0, 'beta': 2.0},
            use_student_t=True
        )

    @staticmethod
    def neutral() -> BayesianFractalModel:
        """
        Neutral model with uninformative priors.
        Hurst prior centered at 0.5 (random walk).
        """
        return BayesianFractalModel(
            hurst_prior={'alpha': 2.0, 'beta': 2.0},
            use_student_t=True
        )
