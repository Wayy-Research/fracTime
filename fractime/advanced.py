"""
Advanced fractal analytics for FracTime.

New methods added for the evolved framework:
- Wavelet Hurst (Db4): Hurst estimation via wavelet variance decay
- MF-ADCCA: Multifractal Asymmetric Detrended Cross-Correlation Analysis
- Quantum Price Levels (QPL): Schrodinger-equation support/resistance
- Fractal Coherence: Cross-dimensional Hurst consistency
- Fractal Trend Dimension (FTD): Asymmetric box-counting for up/down moves
- DTW Alignment: Dynamic Time Warping for beta correction
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d

from ._numba import compute_hurst_rs, compute_hurst_dfa, linear_regression


# =============================================================================
# Wavelet Hurst (Db4) — Section 8
# =============================================================================

def compute_wavelet_hurst(
    prices: np.ndarray,
    wavelet: str = "db4",
    max_level: Optional[int] = None,
) -> float:
    """Estimate Hurst exponent from wavelet variance decay.

    Uses the discrete wavelet transform to decompose the series into
    scales. The Hurst exponent is estimated from the log-log regression
    of wavelet variance vs scale.

    Args:
        prices: Price series (positive values).
        wavelet: Wavelet family (default ``'db4'``).
        max_level: Maximum decomposition level. ``None`` for automatic.

    Returns:
        Hurst exponent estimate (0–1).
    """
    import pywt

    returns = np.diff(np.log(np.asarray(prices, dtype=np.float64)))
    if len(returns) < 16:
        return 0.5

    if max_level is None:
        max_level = pywt.dwt_max_level(len(returns), pywt.Wavelet(wavelet).dec_len)
    max_level = min(max_level, pywt.dwt_max_level(len(returns), pywt.Wavelet(wavelet).dec_len))

    if max_level < 2:
        return 0.5

    coeffs = pywt.wavedec(returns, wavelet, level=max_level)

    # Wavelet variance at each scale j = 1..max_level
    log_scales = []
    log_vars = []
    for j in range(1, max_level + 1):
        detail = coeffs[max_level - j + 1]
        if len(detail) < 2:
            continue
        var_j = np.var(detail)
        if var_j > 0:
            log_scales.append(np.log(2.0**j))
            log_vars.append(np.log(var_j))

    if len(log_scales) < 2:
        return 0.5

    log_scales_arr = np.array(log_scales)
    log_vars_arr = np.array(log_vars)

    # H = (slope + 1) / 2  from wavelet variance ~ 2^{j(2H+1)}
    slope = float(linear_regression(log_scales_arr, log_vars_arr))
    hurst = (slope + 1.0) / 2.0
    return float(np.clip(hurst, 0.01, 0.99))


def compute_rolling_wavelet_hurst(
    prices: np.ndarray,
    window: int = 252,
    step: int = 1,
    wavelet: str = "db4",
) -> np.ndarray:
    """Compute rolling wavelet Hurst exponent.

    Args:
        prices: Price series.
        window: Rolling window size.
        step: Step size between estimates.
        wavelet: Wavelet family.

    Returns:
        Array of shape ``(n_windows,)`` with Hurst estimates.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    results = []
    for i in range(window, n, step):
        segment = prices[i - window : i]
        results.append(compute_wavelet_hurst(segment, wavelet=wavelet))
    return np.array(results, dtype=np.float64)


# =============================================================================
# MF-ADCCA — Section 2.3
# =============================================================================

def compute_mf_adcca(
    series_x: np.ndarray,
    series_y: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    min_scale: int = 10,
    max_scale: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multifractal Asymmetric Detrended Cross-Correlation Analysis.

    Computes the detrended cross-correlation function at multiple
    scales and moment orders, yielding the cross-correlation scaling
    exponent ``h_xy(q)`` and the ADCCA coefficient ``rho_q``.

    Args:
        series_x: First series (e.g. prices of asset X).
        series_y: Second series (e.g. prices of asset Y).
        q_values: Array of moment orders.  Default ``linspace(-5, 5, 21)``.
        min_scale: Minimum window size.
        max_scale: Maximum window size.

    Returns:
        Tuple of ``(rho_q, h_xy)`` where both have shape ``(len(q_values),)``.
    """
    x = np.asarray(series_x, dtype=np.float64)
    y = np.asarray(series_y, dtype=np.float64)

    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    q_values = np.asarray(q_values, dtype=np.float64)

    # Integrate
    rx = np.diff(np.log(x)) if np.all(x > 0) else np.diff(x)
    ry = np.diff(np.log(y)) if np.all(y > 0) else np.diff(y)

    n = min(len(rx), len(ry))
    rx, ry = rx[:n], ry[:n]

    cum_x = np.cumsum(rx - rx.mean())
    cum_y = np.cumsum(ry - ry.mean())

    max_scale = min(max_scale, n // 4)
    if max_scale <= min_scale:
        return np.full(len(q_values), np.nan), np.full(len(q_values), np.nan)

    # Log-spaced scales
    n_scales = min(20, max_scale - min_scale)
    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_scale), n_scales
    ).astype(int))
    scales = scales[scales >= 4]

    h_xy = np.zeros(len(q_values))
    rho_q = np.zeros(len(q_values))

    for qi, q in enumerate(q_values):
        log_s = []
        log_f = []

        for s in scales:
            n_seg = n // s
            if n_seg < 1:
                continue

            f2_xy = np.zeros(n_seg)
            f2_xx = np.zeros(n_seg)
            f2_yy = np.zeros(n_seg)

            for k in range(n_seg):
                start = k * s
                end = start + s
                seg_x = cum_x[start:end]
                seg_y = cum_y[start:end]

                t = np.arange(s, dtype=np.float64)
                # Linear detrend
                sx = float(linear_regression(t, seg_x))
                ix = np.mean(seg_x) - sx * np.mean(t)
                res_x = seg_x - (sx * t + ix)

                sy = float(linear_regression(t, seg_y))
                iy = np.mean(seg_y) - sy * np.mean(t)
                res_y = seg_y - (sy * t + iy)

                f2_xy[k] = np.mean(res_x * res_y)
                f2_xx[k] = np.mean(res_x**2)
                f2_yy[k] = np.mean(res_y**2)

            # Sign-preserving power for cross-covariance
            if abs(q) < 1e-10:
                f_val = np.exp(0.5 * np.mean(np.log(np.abs(f2_xy) + 1e-30)))
            else:
                sign_f2 = np.sign(f2_xy)
                abs_f2 = np.abs(f2_xy) + 1e-30
                exponent = np.clip(q / 2.0, -50.0, 50.0)
                powered = sign_f2 * np.power(abs_f2, exponent)
                powered = np.nan_to_num(powered, nan=0.0, posinf=0.0, neginf=0.0)
                mean_powered = np.mean(powered)
                inv_q = np.clip(1.0 / q, -50.0, 50.0)
                f_val = np.sign(mean_powered) * np.power(
                    np.abs(mean_powered) + 1e-30, inv_q
                )

            if f_val > 0:
                log_s.append(np.log(float(s)))
                log_f.append(np.log(f_val))

        if len(log_s) >= 2:
            h_xy[qi] = float(linear_regression(
                np.array(log_s), np.array(log_f)
            ))
        else:
            h_xy[qi] = np.nan

        # Rho_q = F_xy / sqrt(F_xx * F_yy)  (at largest scale)
        valid_xx = f2_xx[f2_xx > 0]
        valid_yy = f2_yy[f2_yy > 0]
        if len(valid_xx) > 0 and len(valid_yy) > 0:
            denom = np.sqrt(np.mean(valid_xx) * np.mean(valid_yy))
            if denom > 1e-30:
                rho_q[qi] = np.mean(f2_xy) / denom
            else:
                rho_q[qi] = np.nan
        else:
            rho_q[qi] = np.nan

    return rho_q, h_xy


# =============================================================================
# Quantum Price Levels (QPL) — Section 7
# =============================================================================

def compute_qpl(
    prices: np.ndarray,
    n_levels: int = 5,
    n_bins: int = 100,
    potential_type: str = "gaussian",
) -> np.ndarray:
    """Compute Quantum Price Levels via discretised Schrodinger equation.

    Constructs a potential ``V(x)`` from the negative log probability density
    of the price series, then solves the 1-D time-independent Schrodinger
    equation on a finite-difference grid.  Eigenvalues map to QPL
    support/resistance levels.

    Args:
        prices: Price series.
        n_levels: Number of quantum levels to extract.
        n_bins: Grid resolution for finite-difference.
        potential_type: ``'gaussian'`` KDE or ``'histogram'`` for V(x).

    Returns:
        Array of ``n_levels`` price levels sorted ascending.
    """
    prices = np.asarray(prices, dtype=np.float64)
    p_min, p_max = float(prices.min()), float(prices.max())
    margin = 0.05 * (p_max - p_min)
    if p_max - p_min < 1e-10:
        # Constant price series — return that price for all levels
        return np.full(n_levels, p_min)

    x = np.linspace(p_min - margin, p_max + margin, n_bins)
    dx = x[1] - x[0]

    # Build potential V(x) from price density
    if potential_type == "gaussian":
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(prices, bw_method="silverman")
        density = kde(x)
    else:
        counts, edges = np.histogram(prices, bins=n_bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        f = interp1d(centres, counts, kind="linear", fill_value=0, bounds_error=False)
        density = f(x)

    density = np.maximum(density, 1e-30)
    V = -np.log(density)
    V -= V.min()  # shift so minimum is 0

    # Finite-difference Hamiltonian: H = -1/2 d^2/dx^2 + V(x)
    diag = 1.0 / dx**2 + V
    off = -0.5 / dx**2 * np.ones(n_bins - 1)
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

    try:
        eigenvalues, eigenvectors = linalg.eigh(H)
    except linalg.LinAlgError:
        # Fallback: return evenly spaced levels within the price range
        return np.linspace(p_min, p_max, n_levels)

    # Map first n_levels eigenvalues back to price space
    n_levels = min(n_levels, len(eigenvalues))
    qpl_levels = np.zeros(n_levels)

    for i in range(n_levels):
        psi = eigenvectors[:, i] ** 2
        psi /= psi.sum()  # normalise to probability
        qpl_levels[i] = np.dot(psi, x)

    return np.sort(qpl_levels)


# =============================================================================
# Fractal Coherence — Section 10
# =============================================================================

def compute_fractal_coherence(
    price: np.ndarray,
    volume: np.ndarray,
    volatility: Optional[np.ndarray] = None,
    window: int = 63,
    method: str = "rs",
) -> float:
    """Cross-dimensional fractal coherence.

    Computes the Hurst exponent for each provided dimension (price,
    volume, and optionally volatility) over the trailing *window*,
    then returns coherence as ``1 - normalised_spread``.

    Args:
        price: Price series.
        volume: Volume series (same length as price).
        volatility: Optional volatility series.
        window: Look-back window for Hurst estimation.
        method: ``'rs'`` or ``'dfa'``.

    Returns:
        Coherence score in [0, 1].  Higher = more consistent fractal
        behaviour across dimensions.
    """
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    n = min(len(price), len(volume))
    price = price[-window:] if n > window else price
    volume = volume[-window:] if n > window else volume

    hurst_fn = compute_hurst_dfa if method == "dfa" else compute_hurst_rs
    min_scale, max_scale = 10, min(50, len(price) // 4)

    hursts = []
    for series in [price, volume]:
        if len(series) >= 2 * min_scale:
            h = float(hurst_fn(series, min_scale, max_scale))
            hursts.append(h)

    if volatility is not None:
        vol = np.asarray(volatility, dtype=np.float64)
        vol = vol[-window:] if len(vol) > window else vol
        if len(vol) >= 2 * min_scale:
            h = float(hurst_fn(vol, min_scale, max_scale))
            hursts.append(h)

    if len(hursts) < 2:
        return 0.0

    hursts_arr = np.array(hursts)
    spread = hursts_arr.max() - hursts_arr.min()
    # Normalise by theoretical max spread (0–1)
    coherence = 1.0 - min(spread, 1.0)
    return float(coherence)


def compute_rolling_fractal_coherence(
    price: np.ndarray,
    volume: np.ndarray,
    volatility: Optional[np.ndarray] = None,
    window: int = 63,
    step: int = 1,
    method: str = "rs",
) -> np.ndarray:
    """Compute rolling fractal coherence.

    Args:
        price: Price series.
        volume: Volume series.
        volatility: Optional volatility series.
        window: Look-back window.
        step: Step size.
        method: Hurst method.

    Returns:
        Array of coherence values.
    """
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    n = min(len(price), len(volume))
    results = []
    for i in range(window, n, step):
        p = price[i - window : i]
        v = volume[i - window : i]
        vol = volatility[i - window : i] if volatility is not None else None
        results.append(compute_fractal_coherence(p, v, vol, window=window, method=method))
    return np.array(results, dtype=np.float64)


# =============================================================================
# Fractal Trend Dimension (FTD) — Section 4.3
# =============================================================================

def compute_ftd(
    prices: np.ndarray,
    window: int = 63,
) -> Tuple[float, float]:
    """Asymmetric Fractal Trend Dimension.

    Separately estimates box-counting dimension for upward and downward
    movements, capturing directional asymmetry in fractal complexity.

    Args:
        prices: Price series.
        window: Look-back window (uses last *window* points).

    Returns:
        ``(D_plus, D_minus)`` — fractal dimension for up and down moves.
    """
    prices = np.asarray(prices, dtype=np.float64)
    prices = prices[-window:] if len(prices) > window else prices
    returns = np.diff(np.log(prices))

    if len(returns) < 10:
        return (1.5, 1.5)

    up_mask = returns > 0
    down_mask = returns < 0

    def _box_dim(series: np.ndarray) -> float:
        if len(series) < 4:
            return 1.5
        # Normalize to [0, 1]
        s_min, s_max = series.min(), series.max()
        if s_max - s_min < 1e-10:
            return 1.5
        normed = (series - s_min) / (s_max - s_min)

        sizes = []
        counts = []
        for k in range(2, min(len(normed) // 2, 30)):
            n_boxes = int(np.ceil(len(normed) / k))
            boxes = set()
            for i in range(len(normed)):
                bx = i // k
                by = int(normed[i] * k)
                boxes.add((bx, by))
            if len(boxes) > 0:
                sizes.append(np.log(1.0 / k))
                counts.append(np.log(len(boxes)))

        if len(sizes) < 2:
            return 1.5

        slope = float(linear_regression(np.array(sizes), np.array(counts)))
        return float(np.clip(slope, 1.0, 2.0))

    # Build cumulative price-like series for each direction
    up_returns = returns.copy()
    up_returns[~up_mask] = 0
    up_series = np.cumsum(up_returns)

    down_returns = returns.copy()
    down_returns[~down_mask] = 0
    down_series = np.cumsum(np.abs(down_returns))

    d_plus = _box_dim(up_series)
    d_minus = _box_dim(down_series)

    return (d_plus, d_minus)


# =============================================================================
# DTW Alignment — Section 6
# =============================================================================

def compute_dtw_alignment(
    series_x: np.ndarray,
    series_y: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Dynamic Time Warping alignment between two series.

    Args:
        series_x: First series (length M).
        series_y: Second series (length N).

    Returns:
        ``(path, distance)`` where *path* has shape ``(K, 2)`` with aligned
        index pairs and *distance* is the normalised DTW distance.
    """
    import warnings

    x = np.asarray(series_x, dtype=np.float64)
    y = np.asarray(series_y, dtype=np.float64)
    m, n = len(x), len(y)

    if m * n > 25_000_000:
        warnings.warn(
            f"DTW on series of length {m} x {n} = {m * n} cells. "
            f"Consider downsampling for series longer than ~5000 points.",
            stacklevel=2,
        )

    # Cost matrix
    cost = np.full((m + 1, n + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d = (x[i - 1] - y[j - 1]) ** 2
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    distance = float(np.sqrt(cost[m, n] / (m + n)))

    # Backtrack optimal path
    path = []
    i, j = m, n
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda c: c[0])
    path.reverse()

    return np.array(path, dtype=np.int64), distance


def compute_dtw_beta(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
) -> float:
    """Compute DTW-corrected beta.

    Standard OLS beta assumes synchronous returns.  This function uses
    Dynamic Time Warping to align asynchronous return series before
    computing the regression beta, correcting for lead/lag effects.

    Args:
        stock_returns: Stock return series.
        market_returns: Market return series.

    Returns:
        DTW-corrected beta coefficient.
    """
    stock = np.asarray(stock_returns, dtype=np.float64)
    market = np.asarray(market_returns, dtype=np.float64)

    path, _ = compute_dtw_alignment(stock, market)

    aligned_stock = stock[path[:, 0]]
    aligned_market = market[path[:, 1]]

    if len(aligned_market) < 2 or np.std(aligned_market) < 1e-10:
        return 1.0

    # OLS beta on aligned series
    cov = np.mean(
        (aligned_stock - aligned_stock.mean())
        * (aligned_market - aligned_market.mean())
    )
    var_m = np.var(aligned_market)
    if var_m < 1e-10:
        return 1.0

    return float(cov / var_m)
