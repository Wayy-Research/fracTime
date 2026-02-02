"""
Transaction Cost Analysis (TCA) for FracTime research.

This module implements transaction cost modeling for realistic backtest evaluation:
- Corwin-Schultz spread estimator (from High/Low prices)
- Almgren-Chriss market impact model (simplified)
- Total Implementation Cost computation

Reference:
    Corwin, S.A. & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads
    from Daily High and Low Prices." The Journal of Finance, 67(2), 719-760.

    Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio transactions."
    Journal of Risk, 3(2), 5-40.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numba import njit


# =============================================================================
# Numba-Accelerated Corwin-Schultz Implementation
# =============================================================================


@njit
def _corwin_schultz_spread_numba(
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated Corwin-Schultz spread estimator.

    The Corwin-Schultz estimator uses the high-low range over consecutive
    days to infer the bid-ask spread. The key insight is that daily high
    prices are typically buyer-initiated (at the ask) and daily lows are
    seller-initiated (at the bid).

    Formula:
        beta = sum of squared log(H/L) over 2-day windows
        gamma = squared log(H_2day / L_2day)
        alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

    Args:
        high: High prices array
        low: Low prices array

    Returns:
        Spread estimates as fraction of price (not basis points)
    """
    n = len(high)
    spreads = np.empty(n)
    spreads[:] = np.nan

    if n < 2:
        return spreads

    # Constants
    sqrt2 = np.sqrt(2.0)
    k = 3.0 - 2.0 * sqrt2

    for i in range(1, n):
        # Single-day log ranges
        h1, l1 = high[i - 1], low[i - 1]
        h2, l2 = high[i], low[i]

        # Avoid log of zero or negative
        if h1 <= 0 or l1 <= 0 or h2 <= 0 or l2 <= 0:
            spreads[i] = 0.0
            continue

        if l1 <= 0 or l2 <= 0:
            spreads[i] = 0.0
            continue

        log_hl1 = np.log(h1 / l1)
        log_hl2 = np.log(h2 / l2)

        # Beta: sum of squared single-day log ranges
        beta = log_hl1 ** 2 + log_hl2 ** 2

        # 2-day high and low
        h_2day = max(h1, h2)
        l_2day = min(l1, l2)

        if l_2day <= 0:
            spreads[i] = 0.0
            continue

        # Gamma: squared 2-day log range
        gamma = np.log(h_2day / l_2day) ** 2

        # Alpha calculation
        sqrt_beta = np.sqrt(beta)
        sqrt_2beta = np.sqrt(2.0 * beta)
        sqrt_gamma_k = np.sqrt(gamma / k) if k != 0 else 0.0

        alpha = (sqrt_2beta - sqrt_beta) / k - sqrt_gamma_k

        # Spread estimate
        if alpha <= -100:
            # Prevent overflow in exp
            spread = 0.0
        elif alpha >= 100:
            spread = 2.0  # Cap at 200% (unrealistic but prevents overflow)
        else:
            exp_alpha = np.exp(alpha)
            spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)

        # Spread should be non-negative; negative estimates indicate
        # noise or violations of assumptions
        spreads[i] = max(0.0, spread)

    return spreads


@njit
def _compute_rolling_volatility_numba(
    returns: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute rolling annualized volatility (numba-accelerated).

    Args:
        returns: Log returns array
        window: Rolling window size

    Returns:
        Annualized volatility (assuming 252 trading days)
    """
    n = len(returns)
    volatility = np.empty(n)
    volatility[:] = np.nan

    if n < window:
        return volatility

    for i in range(window - 1, n):
        window_returns = returns[i - window + 1 : i + 1]
        std = np.std(window_returns)
        volatility[i] = std * np.sqrt(252.0)

    return volatility


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TransactionCosts:
    """Container for transaction cost components.

    All costs are stored in basis points (bps) for consistency.
    1 basis point = 0.01% = 0.0001

    Attributes:
        commission: Commission/brokerage costs (bps)
        spread_cost: Estimated bid-ask spread cost (bps)
        market_impact: Estimated market impact cost (bps)
        total_cost: Total round-trip cost (bps)
    """

    commission: float
    spread_cost: float
    market_impact: float
    total_cost: float

    @property
    def total_bps(self) -> float:
        """Total cost in basis points (alias for total_cost)."""
        return self.total_cost

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "commission_bps": self.commission,
            "spread_cost_bps": self.spread_cost,
            "market_impact_bps": self.market_impact,
            "total_cost_bps": self.total_cost,
        }

    def __repr__(self) -> str:
        return (
            f"TransactionCosts(commission={self.commission:.1f}bps, "
            f"spread={self.spread_cost:.1f}bps, "
            f"impact={self.market_impact:.1f}bps, "
            f"total={self.total_cost:.1f}bps)"
        )


# =============================================================================
# TCA Calculator
# =============================================================================


class TCACalculator:
    """
    Transaction Cost Analysis calculator.

    Uses Corwin-Schultz spread estimation and simplified Almgren-Chriss
    market impact modeling to estimate realistic trading costs.

    This is critical for evaluating whether a forecasting model's alpha
    survives after accounting for execution costs.

    Example:
        >>> tca = TCACalculator(commission_bps=5.0, impact_coefficient=0.1)
        >>> spread = tca.estimate_spread_corwin_schultz(high, low)
        >>> impact = tca.estimate_market_impact(volatility)
        >>> costs = tca.compute_costs(prices, high, low, volatility, signals)

    Args:
        commission_bps: Commission per trade in basis points (default 10 bps)
        impact_coefficient: Market impact coefficient (eta in Almgren-Chriss)
            Higher values = more market impact. Default 0.1.
        volatility_window: Window for rolling volatility calculation (default 21)

    Note:
        The default commission of 10 bps is conservative for institutional trading.
        Retail traders may face higher costs; HFT may face lower.
    """

    def __init__(
        self,
        commission_bps: float = 10.0,
        impact_coefficient: float = 0.1,
        volatility_window: int = 21,
    ) -> None:
        """Initialize TCA calculator with cost parameters."""
        if commission_bps < 0:
            raise ValueError("commission_bps must be non-negative")
        if impact_coefficient < 0:
            raise ValueError("impact_coefficient must be non-negative")
        if volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")

        self.commission_bps = commission_bps
        self.impact_coefficient = impact_coefficient
        self.volatility_window = volatility_window

    def estimate_spread_corwin_schultz(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate bid-ask spread using Corwin-Schultz method.

        The Corwin-Schultz estimator exploits the fact that daily high prices
        reflect buyer-initiated trades (at ask) while lows reflect
        seller-initiated trades (at bid). The spread can be inferred from
        the 2-day high-low range relative to 1-day ranges.

        Args:
            high: High prices array
            low: Low prices array

        Returns:
            Estimated spread as fraction of price (multiply by 10000 for bps)

        Note:
            - Returns NaN for first observation (needs 2-day window)
            - Negative estimates are floored to 0 (indicates noise)
            - Very high spreads may indicate data quality issues or illiquidity
        """
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)

        if len(high) != len(low):
            raise ValueError("high and low arrays must have same length")
        if len(high) < 2:
            return np.full(len(high), np.nan)

        return _corwin_schultz_spread_numba(high, low)

    def estimate_market_impact(
        self,
        volatility: np.ndarray,
        order_size_pct: float = 0.01,
    ) -> np.ndarray:
        """
        Estimate market impact using simplified Almgren-Chriss model.

        The Almgren-Chriss model estimates market impact as:
            Impact = eta * sigma * sqrt(V_order / V_daily)

        We simplify this to:
            Impact_bps = eta * sigma * sqrt(order_size_pct) * 10000

        Args:
            volatility: Annualized volatility series (as decimal, e.g., 0.20 for 20%)
            order_size_pct: Order size as fraction of daily volume (default 1%)

        Returns:
            Estimated market impact in basis points

        Note:
            This is a simplified model. Real impact depends on:
            - Execution algorithm (TWAP, VWAP, IS, etc.)
            - Time of day
            - Market conditions
            - Stock-specific liquidity
        """
        volatility = np.asarray(volatility, dtype=np.float64)

        if order_size_pct <= 0:
            raise ValueError("order_size_pct must be positive")

        # Impact = eta * sigma * sqrt(participation_rate)
        # Convert to basis points (* 10000)
        impact_bps = (
            self.impact_coefficient
            * volatility
            * np.sqrt(order_size_pct)
            * 10000
        )

        return np.maximum(impact_bps, 0.0)

    def compute_volatility(
        self,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute rolling annualized volatility from prices.

        Args:
            prices: Price series

        Returns:
            Annualized volatility (assuming 252 trading days)
        """
        prices = np.asarray(prices, dtype=np.float64)

        if len(prices) < 2:
            return np.full(len(prices), np.nan)

        # Log returns
        returns = np.diff(np.log(prices))
        returns = np.concatenate([[np.nan], returns])

        return _compute_rolling_volatility_numba(returns, self.volatility_window)

    def compute_costs(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        trade_signals: Optional[np.ndarray] = None,
        order_size_pct: float = 0.01,
    ) -> TransactionCosts:
        """
        Compute total transaction costs.

        Total Implementation Cost (TIC) = Commission + Spread + Market Impact

        This is a round-trip cost estimate for entering and exiting positions.

        Args:
            prices: Price series
            high: High prices (for Corwin-Schultz spread estimation)
            low: Low prices (for Corwin-Schultz spread estimation)
            volatility: Pre-computed volatility (if None, computed from prices)
            trade_signals: Trading signals (-1, 0, +1) to identify trade points
            order_size_pct: Order size as fraction of daily volume

        Returns:
            TransactionCosts with all cost components in basis points

        Note:
            If trade_signals is provided, costs are computed only at trade points
            and then averaged. Otherwise, costs are averaged across all periods.
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        # Commission is fixed
        commission = self.commission_bps

        # Estimate spread
        if high is not None and low is not None:
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            spread_frac = self.estimate_spread_corwin_schultz(high, low)
            # Convert fraction to bps
            spread_bps = spread_frac * 10000
        else:
            # Default spread estimate based on typical liquid stocks
            spread_bps = np.full(n, 5.0)  # 5 bps default

        # Compute volatility if not provided
        if volatility is None:
            volatility = self.compute_volatility(prices)
        else:
            volatility = np.asarray(volatility, dtype=np.float64)

        # Estimate market impact
        impact_bps = self.estimate_market_impact(volatility, order_size_pct)

        # Filter to trade points if signals provided
        if trade_signals is not None:
            trade_signals = np.asarray(trade_signals)
            # Trade occurs when signal changes (entry/exit)
            signal_changes = np.abs(np.diff(trade_signals)) > 0
            signal_changes = np.concatenate([[False], signal_changes])
            trade_mask = signal_changes

            # Also include first and last if position exists
            if len(trade_signals) > 0 and trade_signals[0] != 0:
                trade_mask[0] = True
            if len(trade_signals) > 1 and trade_signals[-1] != 0:
                trade_mask[-1] = True

            if np.sum(trade_mask) > 0:
                spread_bps = spread_bps[trade_mask]
                impact_bps = impact_bps[trade_mask]

        # Average costs (ignoring NaN)
        avg_spread = float(np.nanmean(spread_bps)) if len(spread_bps) > 0 else 0.0
        avg_impact = float(np.nanmean(impact_bps)) if len(impact_bps) > 0 else 0.0

        # Handle NaN
        if np.isnan(avg_spread):
            avg_spread = 5.0  # Default to 5 bps
        if np.isnan(avg_impact):
            avg_impact = 0.0

        # Total round-trip cost (2x for entry + exit)
        # Commission and spread apply to both legs
        total = 2 * commission + 2 * avg_spread + 2 * avg_impact

        return TransactionCosts(
            commission=2 * commission,
            spread_cost=2 * avg_spread,
            market_impact=2 * avg_impact,
            total_cost=total,
        )

    def compute_net_sharpe(
        self,
        gross_returns: np.ndarray,
        trade_signals: np.ndarray,
        costs: Optional[TransactionCosts] = None,
        prices: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Sharpe ratio after transaction costs.

        Args:
            gross_returns: Strategy returns before costs
            trade_signals: Trading signals (-1, 0, +1)
            costs: Pre-computed costs (if None, computed from prices)
            prices: Price series (used if costs is None)
            high: High prices (used if costs is None)
            low: Low prices (used if costs is None)

        Returns:
            Annualized Sharpe ratio after costs
        """
        gross_returns = np.asarray(gross_returns)
        trade_signals = np.asarray(trade_signals)

        if len(gross_returns) < 2:
            return np.nan

        # Compute costs if not provided
        if costs is None:
            if prices is None:
                raise ValueError("Either costs or prices must be provided")
            costs = self.compute_costs(
                prices=prices,
                high=high,
                low=low,
                trade_signals=trade_signals,
            )

        # Identify trade points
        signal_changes = np.abs(np.diff(trade_signals)) > 0
        signal_changes = np.concatenate([[False], signal_changes])

        # Cost per trade in decimal form
        cost_per_trade = costs.total_bps / 10000

        # Deduct costs at trade points
        net_returns = gross_returns.copy()
        net_returns[signal_changes] -= cost_per_trade

        # Compute Sharpe
        if np.std(net_returns) == 0:
            return np.nan

        sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)
        return float(sharpe)


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_net_metrics(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tca: Optional[TCACalculator] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Convenience function to compute metrics with TCA.

    Computes both gross and net Sharpe ratios, along with
    transaction cost estimates and trade frequency.

    Args:
        forecasts: Point forecasts
        actuals: Actual values
        current_prices: Prices at forecast time
        high: High prices for spread estimation
        low: Low prices for spread estimation
        tca: TCACalculator instance (uses defaults if None)
        risk_free_rate: Annualized risk-free rate (default 0)

    Returns:
        Dictionary containing:
        - sharpe_gross: Sharpe ratio before costs
        - sharpe_net: Sharpe ratio after costs
        - total_costs_bps: Average round-trip transaction cost
        - trade_frequency: Fraction of periods with position changes
        - spread_bps: Average estimated spread
        - impact_bps: Average estimated market impact

    Example:
        >>> metrics = compute_net_metrics(
        ...     forecasts=preds,
        ...     actuals=actual_prices,
        ...     current_prices=prices,
        ...     high=high_prices,
        ...     low=low_prices,
        ... )
        >>> print(f"Net Sharpe: {metrics['sharpe_net']:.2f}")
    """
    forecasts = np.asarray(forecasts)
    actuals = np.asarray(actuals)
    current_prices = np.asarray(current_prices)
    high = np.asarray(high)
    low = np.asarray(low)

    n = min(len(forecasts), len(actuals), len(current_prices))
    if n < 2:
        return {
            "sharpe_gross": np.nan,
            "sharpe_net": np.nan,
            "total_costs_bps": np.nan,
            "trade_frequency": np.nan,
            "spread_bps": np.nan,
            "impact_bps": np.nan,
        }

    forecasts = forecasts[:n]
    actuals = actuals[:n]
    current_prices = current_prices[:n]
    high = high[:n]
    low = low[:n]

    # Create TCA calculator if not provided
    if tca is None:
        tca = TCACalculator()

    # Generate directional trading signals
    signals = np.sign(forecasts - current_prices)

    # Compute actual returns
    actual_returns = (actuals - current_prices) / current_prices

    # Strategy returns (signal * actual return)
    strategy_returns = signals * actual_returns

    # Compute gross Sharpe
    if np.std(strategy_returns) == 0:
        sharpe_gross = np.nan
    else:
        excess_returns = strategy_returns - risk_free_rate / 252
        sharpe_gross = float(
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        )

    # Compute transaction costs
    costs = tca.compute_costs(
        prices=current_prices,
        high=high,
        low=low,
        trade_signals=signals,
    )

    # Compute net Sharpe
    sharpe_net = tca.compute_net_sharpe(
        gross_returns=strategy_returns,
        trade_signals=signals,
        costs=costs,
    )

    # Trade frequency
    signal_changes = np.sum(np.abs(np.diff(signals)) > 0)
    trade_frequency = signal_changes / (n - 1) if n > 1 else 0.0

    return {
        "sharpe_gross": sharpe_gross,
        "sharpe_net": sharpe_net,
        "total_costs_bps": costs.total_bps,
        "trade_frequency": float(trade_frequency),
        "spread_bps": costs.spread_cost / 2,  # One-way spread
        "impact_bps": costs.market_impact / 2,  # One-way impact
    }
