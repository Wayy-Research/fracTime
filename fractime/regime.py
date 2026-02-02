"""
HMM-based market regime detection.

This module implements Hidden Markov Model (HMM) based regime detection
for financial time series. It uses the Baum-Welch algorithm (via hmmlearn)
to identify market regimes from return data.

The key insight is that market behavior is not constant - it switches between
distinct regimes (Bull/Bear, or Bull/Bear/Sideways) with different return
distributions. HMMs provide a principled way to:
1. Identify these latent regimes
2. Estimate transition probabilities between regimes
3. Compute the probability of being in each regime at any time

Examples:
    Basic usage with 2 regimes (Bull/Bear):
        >>> import numpy as np
        >>> from fractime.regime import RegimeDetector
        >>> returns = np.random.randn(500) * 0.02
        >>> detector = RegimeDetector(n_regimes=2)
        >>> detector.fit(returns)
        >>> regimes = detector.predict(returns)
        >>> current_regime, prob = detector.get_current_regime(returns)

    With 3 regimes (Bull/Bear/Sideways):
        >>> detector = RegimeDetector(n_regimes=3)
        >>> detector.fit(returns)
        >>> probs = detector.predict_proba(returns)

References:
    - Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle"
    - Ang, A., & Bekaert, G. (2002). "Regime Switches in Interest Rates"
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    GaussianHMM = None


class RegimeDetector:
    """
    HMM-based market regime detection.

    Uses Gaussian Hidden Markov Models to identify latent market regimes
    from return data. Supports both 2-regime (Bull/Bear) and 3-regime
    (Bull/Bear/Sideways) configurations.

    The model is trained using the Baum-Welch algorithm, which is an
    Expectation-Maximization (EM) algorithm for HMMs. After training,
    the Viterbi algorithm is used for state decoding.

    Regime identification is based on the learned Gaussian emission
    distributions:
        - Bull: High positive mean, lower variance
        - Bear: Negative mean, higher variance
        - Sideways (3-regime only): Near-zero mean, low variance

    Attributes:
        n_regimes: Number of regimes (2 or 3)
        random_state: Random seed for reproducibility
        model: The fitted GaussianHMM model (None before fit)

    Example:
        >>> detector = RegimeDetector(n_regimes=2)
        >>> detector.fit(returns)
        >>> current_regime, probability = detector.get_current_regime(returns)
        >>> print(f"Current regime: {current_regime} ({probability:.1%})")
    """

    def __init__(
        self,
        n_regimes: int = 2,
        random_state: int = 42,
        n_iter: int = 100,
        tol: float = 1e-4,
        covariance_type: str = "full",
    ) -> None:
        """
        Initialize the RegimeDetector.

        Args:
            n_regimes: Number of regimes to detect (2 or 3).
                - 2: Bull/Bear regimes
                - 3: Bull/Bear/Sideways regimes
            random_state: Random seed for reproducibility.
            n_iter: Maximum number of EM iterations.
            tol: Convergence tolerance for EM algorithm.
            covariance_type: Type of covariance parameters to use.
                Must be one of 'spherical', 'diag', 'full', 'tied'.
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for RegimeDetector. "
                "Install with: uv pip install hmmlearn"
            )

        if n_regimes not in (2, 3):
            raise ValueError(
                f"n_regimes must be 2 or 3, got {n_regimes}. "
                "Use 2 for Bull/Bear, 3 for Bull/Bear/Sideways."
            )

        self.n_regimes = n_regimes
        self.random_state = random_state
        self._n_iter = n_iter
        self._tol = tol
        self._covariance_type = covariance_type
        self._model: Optional[GaussianHMM] = None
        self._state_to_regime: Optional[Dict[int, str]] = None

    @property
    def model(self) -> Optional[GaussianHMM]:
        """The underlying GaussianHMM model."""
        return self._model

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._model is not None

    def _check_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                "RegimeDetector has not been fitted. Call fit() first."
            )

    def _prepare_data(self, returns: NDArray[np.floating]) -> NDArray[np.floating]:
        """Validate and reshape returns for HMM."""
        returns = np.asarray(returns)
        if returns.ndim != 1:
            raise ValueError(f"Expected 1D returns array, got shape {returns.shape}")
        if len(returns) < self.n_regimes * 10:
            raise ValueError(
                f"Insufficient data: got {len(returns)} observations, "
                f"need at least {self.n_regimes * 10} for {self.n_regimes} regimes."
            )
        if np.any(~np.isfinite(returns)):
            raise ValueError("Returns contain NaN or Inf values.")
        return returns.reshape(-1, 1)

    def _identify_regimes(self) -> Dict[int, str]:
        """
        Map HMM states to regime names based on learned parameters.

        States are ordered by mean return:
        - Highest mean -> Bull
        - Lowest mean -> Bear
        - Middle (if 3 regimes) -> Sideways
        """
        self._check_fitted()
        means = self._model.means_.flatten()
        sorted_by_mean = np.argsort(means)

        if self.n_regimes == 2:
            return {
                sorted_by_mean[1]: "bull",
                sorted_by_mean[0]: "bear",
            }
        else:
            return {
                sorted_by_mean[2]: "bull",
                sorted_by_mean[0]: "bear",
                sorted_by_mean[1]: "sideways",
            }

    def fit(self, returns: NDArray[np.floating]) -> "RegimeDetector":
        """
        Fit the HMM to return data.

        Uses the Baum-Welch algorithm (EM) to estimate model parameters.

        Args:
            returns: 1D array of return values (e.g., log returns)

        Returns:
            self: The fitted RegimeDetector
        """
        X = self._prepare_data(returns)
        np.random.seed(self.random_state)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            tol=self._tol,
            random_state=self.random_state,
        )
        self._model.fit(X)
        self._state_to_regime = self._identify_regimes()
        return self

    def predict(self, returns: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict regime for each observation using Viterbi algorithm.

        Args:
            returns: 1D array of return values

        Returns:
            Array of regime states (integers 0 to n_regimes-1)
        """
        self._check_fitted()
        X = self._prepare_data(returns)
        return self._model.predict(X)

    def predict_proba(self, returns: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Get regime probabilities for each observation.

        Args:
            returns: 1D array of return values

        Returns:
            Array of shape (n_observations, n_regimes) with probabilities
        """
        self._check_fitted()
        X = self._prepare_data(returns)
        return self._model.predict_proba(X)

    @property
    def transition_matrix(self) -> NDArray[np.floating]:
        """
        Get the fitted transition probability matrix.

        Returns:
            Array of shape (n_regimes, n_regimes) where entry [i,j]
            is the probability of transitioning from state i to state j.
        """
        self._check_fitted()
        return self._model.transmat_

    @property
    def regime_means(self) -> NDArray[np.floating]:
        """
        Get the mean return for each regime.

        Returns:
            Array of shape (n_regimes,) with mean returns
        """
        self._check_fitted()
        return self._model.means_.flatten()

    @property
    def regime_vars(self) -> NDArray[np.floating]:
        """
        Get the variance for each regime.

        Returns:
            Array of shape (n_regimes,) with variances
        """
        self._check_fitted()
        covars = self._model.covars_
        if covars.ndim == 3:
            return covars[:, 0, 0]
        return covars.flatten()

    @property
    def stationary_distribution(self) -> NDArray[np.floating]:
        """
        Get the stationary distribution over regimes.

        This represents the long-run probability of being in each regime.

        Returns:
            Array of shape (n_regimes,) with stationary probabilities
        """
        self._check_fitted()
        return self._model.get_stationary_distribution()

    def get_regime_name(self, state: int) -> str:
        """
        Get the regime name for a given state index.

        Args:
            state: State index (0 to n_regimes-1)

        Returns:
            Regime name ('bull', 'bear', or 'sideways')
        """
        self._check_fitted()
        if state not in self._state_to_regime:
            raise ValueError(f"Invalid state {state}. Must be 0 to {self.n_regimes-1}")
        return self._state_to_regime[state]

    def get_current_regime(
        self, returns: NDArray[np.floating]
    ) -> Tuple[str, float]:
        """
        Get the current regime name and its probability.

        Args:
            returns: 1D array of return values (uses last observation)

        Returns:
            Tuple of (regime_name, probability)
        """
        self._check_fitted()
        probs = self.predict_proba(returns)
        last_probs = probs[-1]
        most_likely_state = int(np.argmax(last_probs))
        regime_name = self._state_to_regime[most_likely_state]
        probability = float(last_probs[most_likely_state])
        return regime_name, probability

    def get_regime_probabilities_dict(
        self, returns: NDArray[np.floating]
    ) -> Dict[str, float]:
        """
        Get current regime probabilities as a dictionary.

        Args:
            returns: 1D array of return values

        Returns:
            Dict mapping regime names to probabilities
        """
        self._check_fitted()
        probs = self.predict_proba(returns)
        last_probs = probs[-1]
        return {
            self._state_to_regime[i]: float(last_probs[i])
            for i in range(self.n_regimes)
        }

    def score(self, returns: NDArray[np.floating]) -> float:
        """
        Compute the log-likelihood of the returns under the model.

        Args:
            returns: 1D array of return values

        Returns:
            Log-likelihood score
        """
        self._check_fitted()
        X = self._prepare_data(returns)
        return self._model.score(X)

    def expected_duration(self) -> NDArray[np.floating]:
        """
        Compute expected duration in each regime.

        Based on the self-transition probabilities.

        Returns:
            Array of expected durations (in number of observations)
        """
        self._check_fitted()
        self_probs = np.diag(self._model.transmat_)
        return 1.0 / (1.0 - self_probs + 1e-10)

    def summary(self) -> str:
        """
        Generate a text summary of the fitted model.

        Returns:
            Formatted string with regime parameters and transition matrix
        """
        self._check_fitted()
        lines = [
            "HMM Regime Detection Summary",
            "=" * 40,
            f"Number of regimes: {self.n_regimes}",
            "",
            "Regime Parameters:",
            "-" * 40,
        ]
        durations = self.expected_duration()
        stationary = self.stationary_distribution

        for state in range(self.n_regimes):
            regime_name = self._state_to_regime[state]
            mean = self.regime_means[state]
            var = self.regime_vars[state]
            std = np.sqrt(var)
            duration = durations[state]
            prob = stationary[state]
            lines.extend([
                f"  {regime_name.upper()} (state {state}):",
                f"    Mean return:       {mean:+.6f}",
                f"    Volatility (std):  {std:.6f}",
                f"    Expected duration: {duration:.1f} periods",
                f"    Stationary prob:   {prob:.1%}",
                "",
            ])

        lines.extend(["Transition Matrix:", "-" * 40])
        header = "    " + "".join(
            f"{self._state_to_regime[j]:>10}" for j in range(self.n_regimes)
        )
        lines.append(header)
        for i in range(self.n_regimes):
            row = f"{self._state_to_regime[i]:>4}"
            for j in range(self.n_regimes):
                row += f"{self.transition_matrix[i, j]:>10.3f}"
            lines.append(row)

        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"RegimeDetector(n_regimes={self.n_regimes}, "
                f"fitted=True, random_state={self.random_state})"
            )
        return (
            f"RegimeDetector(n_regimes={self.n_regimes}, "
            f"fitted=False, random_state={self.random_state})"
        )
