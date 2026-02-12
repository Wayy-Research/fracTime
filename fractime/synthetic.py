"""
Synthetic data generators for controlled simulation studies.

Generates price series with known statistical properties (Hurst exponent,
ARIMA structure, regime switches, fat tails) for evaluating FracTime vs
baseline models under controlled conditions.

Note on reproducibility: numba's RNG is separate from NumPy's global RNG.
Functions that call generate_fbm use _seeded_fbm (which seeds inside @njit)
for deterministic output. Non-numba randomness uses save/restore of NumPy
state to avoid contaminating global RNG across calls.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit
from scipy import stats

from ._numba import generate_fbm

if TYPE_CHECKING:
    from .research.data_collector import AssetData

__all__ = [
    "generate_fbm_prices",
    "generate_arima_prices",
    "generate_regime_switching_prices",
    "generate_volatility_shift_prices",
    "generate_fat_tail_prices",
    "create_synthetic_asset",
]


@njit
def _seeded_fbm(n: int, hurst: float, seed: int) -> np.ndarray:
    """Generate fBm with reproducible numba-internal seed."""
    np.random.seed(seed)
    return generate_fbm(n, hurst)


def _sub_seed(base_seed: int, index: int) -> int:
    """Derive a deterministic sub-seed that avoids collisions."""
    # Knuth multiplicative hash
    return (base_seed * 2654435761 + index) % (2**31 - 1)


@contextlib.contextmanager
def _numpy_seed_context(seed: Optional[int]):
    """Temporarily set NumPy global seed, restore state after."""
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)
    else:
        yield


def generate_fbm_prices(
    n: int,
    hurst: float,
    start_price: float = 100.0,
    volatility: float = 0.20,
    drift: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate price series from fractional Brownian motion.

    Args:
        n: Number of price observations (including start price).
        hurst: Hurst exponent (0, 1). H > 0.5 trending, H < 0.5 mean-reverting.
        start_price: Initial price level.
        volatility: Annualized volatility (scaled by sqrt(252)).
        drift: Annualized drift (scaled by 252).
        seed: Random seed for reproducibility.

    Returns:
        Array of prices with length n.
    """
    # generate_fbm returns a PATH: B_H(0)=0, B_H(1), ..., B_H(n-2)
    # We need INCREMENTS (fractional Gaussian noise) for returns
    if seed is not None:
        fbm_path = _seeded_fbm(n - 1, hurst, seed)
    else:
        fbm_path = generate_fbm(n - 1, hurst)

    # Diff the path to get fractional Gaussian noise (fGn) increments
    # For H>0.5: positively correlated increments (persistence)
    # For H<0.5: negatively correlated increments (anti-persistence)
    # For H=0.5: uncorrelated increments (random walk)
    fbm_increments = np.diff(np.concatenate([[0.0], fbm_path]))

    # Scale to daily returns: vol / sqrt(252) per day
    daily_vol = volatility / np.sqrt(252)
    daily_drift = drift / 252

    # Normalize fBm increments to unit variance, then rescale
    std = np.std(fbm_increments)
    if std > 0:
        returns = fbm_increments / std * daily_vol + daily_drift
    else:
        with _numpy_seed_context(seed):
            returns = np.random.normal(daily_drift, daily_vol, n - 1)

    # Convert to prices via geometric returns
    log_prices = np.zeros(n)
    log_prices[0] = np.log(start_price)
    log_prices[1:] = log_prices[0] + np.cumsum(returns)
    prices = np.exp(log_prices)

    return prices


def generate_arima_prices(
    n: int,
    order: Tuple[int, int, int] = (1, 1, 0),
    ar_params: Optional[Sequence[float]] = None,
    ma_params: Optional[Sequence[float]] = None,
    start_price: float = 100.0,
    volatility: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate price series from an ARIMA process.

    The ARMA process generates returns (or differenced log-prices),
    which are then integrated and exponentiated to produce prices.

    Args:
        n: Number of price observations.
        order: (p, d, q) ARIMA order. d is applied as integration.
        ar_params: AR coefficients [phi_1, ..., phi_p]. Defaults based on p.
            All roots of the AR polynomial must lie outside the unit circle.
        ma_params: MA coefficients [theta_1, ..., theta_q]. Defaults based on q.
        start_price: Initial price level.
        volatility: Innovation standard deviation for the ARMA process.
        seed: Random seed for reproducibility.

    Returns:
        Array of prices with length n.
    """
    from statsmodels.tsa.arima_process import arma_generate_sample

    p, d, q = order

    # Build AR polynomial: [1, -phi_1, -phi_2, ...]
    if ar_params is not None:
        ar = np.r_[1, [-c for c in ar_params]]
    elif p > 0:
        ar = np.r_[1, [-0.5 / i for i in range(1, p + 1)]]
    else:
        ar = np.array([1.0])

    # Check AR stability: the lag polynomial roots must lie outside the unit
    # circle.  np.roots([1, -phi_1, ...]) returns the *reciprocal* roots,
    # so stability requires np.abs(reciprocal_roots) < 1.
    if len(ar) > 1:
        reciprocal_roots = np.roots(ar)
        if np.any(np.abs(reciprocal_roots) >= 1.0):
            raise ValueError(
                f"AR process is unstable: reciprocal roots {reciprocal_roots} "
                "have modulus >= 1. Reduce AR coefficient magnitudes."
            )

    # Build MA polynomial: [1, theta_1, theta_2, ...]
    if ma_params is not None:
        ma = np.r_[1, list(ma_params)]
    elif q > 0:
        ma = np.r_[1, [0.5 / i for i in range(1, q + 1)]]
    else:
        ma = np.array([1.0])

    # Generate ARMA sample with burn-in handled by statsmodels
    with _numpy_seed_context(seed):
        arma_sample = arma_generate_sample(
            ar=ar, ma=ma, nsample=n - 1, scale=volatility, burnin=200
        )
    returns = arma_sample

    # Apply integration (d times cumsum)
    series = returns.copy()
    for _ in range(d):
        series = np.cumsum(series)

    # Convert to prices
    log_prices = np.zeros(n)
    log_prices[0] = np.log(start_price)
    log_prices[1:] = log_prices[0] + series
    prices = np.exp(log_prices)

    return prices


def generate_regime_switching_prices(
    n: int,
    regime_hursts: Sequence[float] = (0.3, 0.7),
    regime_durations: Sequence[int] = (126, 126),
    start_price: float = 100.0,
    volatility: float = 0.20,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate price series with regime-switching Hurst exponents.

    Concatenates fBm segments with different Hurst exponents, cycling
    through regimes until n observations are reached.

    Args:
        n: Number of price observations.
        regime_hursts: Hurst exponent for each regime.
        regime_durations: Duration (in days) for each regime.
        start_price: Initial price level.
        volatility: Annualized volatility (same for all regimes).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (prices, regime_labels) where regime_labels[i] is
        the regime index at time i.
    """
    n_regimes = len(regime_hursts)
    all_returns: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    daily_vol = volatility / np.sqrt(252)

    remaining = n - 1  # we need n-1 returns for n prices
    regime_idx = 0

    while remaining > 0:
        h = regime_hursts[regime_idx % n_regimes]
        dur = min(regime_durations[regime_idx % n_regimes], remaining)

        # Generate fBm segment and diff to get fGn increments
        if seed is not None:
            fbm_path = _seeded_fbm(dur, h, _sub_seed(seed, regime_idx))
        else:
            fbm_path = generate_fbm(dur, h)
        fbm_inc = np.diff(np.concatenate([[0.0], fbm_path]))
        std = np.std(fbm_inc)
        if std > 0:
            segment_returns = fbm_inc / std * daily_vol
        else:
            with _numpy_seed_context(_sub_seed(seed, regime_idx) if seed else None):
                segment_returns = np.random.normal(0, daily_vol, dur)

        all_returns.append(segment_returns)
        all_labels.append(np.full(dur, regime_idx % n_regimes, dtype=int))

        remaining -= dur
        regime_idx += 1

    returns = np.concatenate(all_returns)[: n - 1]
    labels_returns = np.concatenate(all_labels)[: n - 1]

    # Build prices
    log_prices = np.zeros(n)
    log_prices[0] = np.log(start_price)
    log_prices[1:] = log_prices[0] + np.cumsum(returns)
    prices = np.exp(log_prices)

    # Label for each price point (first point gets regime 0)
    regime_labels = np.zeros(n, dtype=int)
    regime_labels[1:] = labels_returns

    return prices, regime_labels


def generate_volatility_shift_prices(
    n: int,
    hurst: float = 0.5,
    volatility_sequence: Optional[Sequence[Tuple[int, float]]] = None,
    start_price: float = 100.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate price series with structural volatility breaks.

    Args:
        n: Number of price observations.
        hurst: Hurst exponent (constant across regimes).
        volatility_sequence: List of (duration, annualized_vol) tuples.
            Durations must sum to at least n-1.
            Defaults to calm-crisis-calm split evenly.
        start_price: Initial price level.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (prices, vol_regime) where vol_regime[i] is the
        annualized volatility at time i.
    """
    if volatility_sequence is None:
        third = (n - 1) // 3
        remainder = (n - 1) - 2 * third
        volatility_sequence = [
            (third, 0.15),
            (third, 0.60),
            (remainder, 0.15),
        ]

    total_dur = sum(dur for dur, _ in volatility_sequence)
    if total_dur < n - 1:
        raise ValueError(
            f"volatility_sequence durations sum to {total_dur}, "
            f"but need at least {n - 1} for {n} price observations."
        )

    all_returns: List[np.ndarray] = []
    all_vols: List[np.ndarray] = []

    for seg_idx, (dur, ann_vol) in enumerate(volatility_sequence):
        daily_vol = ann_vol / np.sqrt(252)
        if seed is not None:
            fbm_path = _seeded_fbm(dur, hurst, _sub_seed(seed, seg_idx))
        else:
            fbm_path = generate_fbm(dur, hurst)
        fbm_inc = np.diff(np.concatenate([[0.0], fbm_path]))
        std = np.std(fbm_inc)
        if std > 0:
            segment_returns = fbm_inc / std * daily_vol
        else:
            with _numpy_seed_context(_sub_seed(seed, seg_idx) if seed else None):
                segment_returns = np.random.normal(0, daily_vol, dur)

        all_returns.append(segment_returns)
        all_vols.append(np.full(dur, ann_vol))

    returns = np.concatenate(all_returns)[: n - 1]
    vol_labels = np.concatenate(all_vols)[: n - 1]

    log_prices = np.zeros(n)
    log_prices[0] = np.log(start_price)
    log_prices[1:] = log_prices[0] + np.cumsum(returns)
    prices = np.exp(log_prices)

    vol_regime = np.zeros(n)
    vol_regime[0] = vol_labels[0] if len(vol_labels) > 0 else 0.15
    vol_regime[1:] = vol_labels

    return prices, vol_regime


def generate_fat_tail_prices(
    n: int,
    hurst: float = 0.5,
    df: float = 5.0,
    start_price: float = 100.0,
    volatility: float = 0.20,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate price series with fat-tailed (Student-t) innovations.

    Uses fBm correlation structure but replaces Gaussian innovations
    with t-distributed innovations for heavier tails.

    Args:
        n: Number of price observations.
        hurst: Hurst exponent for correlation structure.
        df: Degrees of freedom for t-distribution (must be > 2 for finite
            variance). Lower = fatter tails.
        start_price: Initial price level.
        volatility: Annualized volatility.
        seed: Random seed for reproducibility.

    Returns:
        Array of prices with length n.
    """
    if df <= 2.0:
        raise ValueError(
            f"Degrees of freedom must be > 2 for finite variance, got {df}. "
            "Use df > 2 (e.g., df=3 for very heavy tails)."
        )

    daily_vol = volatility / np.sqrt(252)

    # Generate fBm path and diff to get fGn increments for correlation structure
    if seed is not None:
        fbm_path = _seeded_fbm(n - 1, hurst, seed)
    else:
        fbm_path = generate_fbm(n - 1, hurst)
    fbm_inc = np.diff(np.concatenate([[0.0], fbm_path]))

    # Rank-transform to uniform, then inverse-CDF to t-distribution
    # This preserves the autocorrelation structure while changing marginals
    ranks = stats.rankdata(fbm_inc) / (len(fbm_inc) + 1)
    t_innovations = stats.t.ppf(ranks, df=df)

    # Scale to desired volatility
    t_std = np.std(t_innovations)
    if t_std > 0:
        returns = t_innovations / t_std * daily_vol
    else:
        with _numpy_seed_context(seed):
            returns = stats.t.rvs(df=df, size=n - 1) * daily_vol

    log_prices = np.zeros(n)
    log_prices[0] = np.log(start_price)
    log_prices[1:] = log_prices[0] + np.cumsum(returns)
    prices = np.exp(log_prices)

    return prices


def create_synthetic_asset(
    prices: np.ndarray,
    ticker: str,
    name: str,
    start_date: str = "2020-01-01",
    category: str = "synthetic",
    subcategory: str = "simulation",
) -> AssetData:
    """Build an AssetData container from a synthetic price series.

    Args:
        prices: Price array.
        ticker: Ticker symbol (e.g. "SYN_FBM_H07").
        name: Human-readable name.
        start_date: ISO-format start date for generating business days.
        category: Asset category for metadata.
        subcategory: Asset subcategory for metadata.

    Returns:
        AssetData compatible with ExperimentRunner.
    """
    from .research.data_collector import AssetData

    n = len(prices)
    start = datetime.fromisoformat(start_date)

    # Generate business-day dates
    dates = []
    current = start
    while len(dates) < n:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += timedelta(days=1)
    dates_arr = np.array(dates)

    # Compute log returns (first entry is 0)
    log_returns = np.zeros(n)
    log_returns[1:] = np.diff(np.log(prices))

    return AssetData(
        ticker=ticker,
        name=name,
        category=category,
        subcategory=subcategory,
        frequency="daily",
        dates=dates_arr,
        prices=prices,
        returns=log_returns,
        metadata={"source": "synthetic", "generator": "fractime.synthetic"},
    )
