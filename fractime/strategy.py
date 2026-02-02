"""
Regime-based trading strategy framework.

This module provides a practical framework for using FracTime's regime detection
and fractal analysis for risk management and position sizing.

The key insight from empirical testing is that fractal features are valuable
for RISK MANAGEMENT, not price prediction. This module implements that insight.

Example:
    >>> from fractime.strategy import RegimeStrategy
    >>> from fractime.regime import RegimeDetector
    >>>
    >>> # Create strategy with HMM regime detection
    >>> strategy = RegimeStrategy(
    ...     regime_detector=RegimeDetector(n_regimes=2),
    ...     bull_allocation=1.0,
    ...     bear_allocation=0.3,
    ... )
    >>>
    >>> # Backtest on historical data
    >>> results = strategy.backtest(prices, returns)
    >>> print(f"Sharpe: {results['sharpe']:.2f}")
    >>> print(f"Max Drawdown: {results['max_drawdown']:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from .regime import RegimeDetector


@dataclass
class BacktestResult:
    """Results from strategy backtest.

    Attributes:
        total_return: Cumulative return over period
        annualized_return: Annualized return
        sharpe: Annualized Sharpe ratio (assuming 0 risk-free rate)
        sortino: Sortino ratio (downside deviation)
        max_drawdown: Maximum drawdown (negative value)
        calmar: Calmar ratio (annualized return / max drawdown)
        volatility: Annualized volatility
        win_rate: Fraction of positive return days
        avg_position: Average position size
        regime_accuracy: How often regime matched subsequent return sign
        returns: Daily returns series
        positions: Daily position sizes
        regimes: Detected regimes
        equity_curve: Cumulative equity curve
    """
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    volatility: float
    win_rate: float
    avg_position: float
    regime_accuracy: float
    returns: NDArray[np.floating]
    positions: NDArray[np.floating]
    regimes: NDArray[np.int_]
    equity_curve: NDArray[np.floating]

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe': self.sharpe,
            'sortino': self.sortino,
            'max_drawdown': self.max_drawdown,
            'calmar': self.calmar,
            'volatility': self.volatility,
            'win_rate': self.win_rate,
            'avg_position': self.avg_position,
            'regime_accuracy': self.regime_accuracy,
        }

    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Backtest Results
================
Total Return:      {self.total_return:>8.1%}
Annualized Return: {self.annualized_return:>8.1%}
Sharpe Ratio:      {self.sharpe:>8.2f}
Sortino Ratio:     {self.sortino:>8.2f}
Max Drawdown:      {self.max_drawdown:>8.1%}
Calmar Ratio:      {self.calmar:>8.2f}
Volatility:        {self.volatility:>8.1%}
Win Rate:          {self.win_rate:>8.1%}
Avg Position:      {self.avg_position:>8.1%}
Regime Accuracy:   {self.regime_accuracy:>8.1%}
"""


class RegimeStrategy:
    """
    Regime-based position sizing strategy.

    Uses HMM regime detection to adjust position sizes based on market state.
    This implements the key finding that fractal features are valuable for
    risk management, not price prediction.

    The strategy:
    - Detects Bull/Bear/Sideways regimes using HMM
    - Adjusts position size based on detected regime
    - Optionally incorporates volatility scaling

    Args:
        regime_detector: Fitted or unfitted RegimeDetector instance
        bull_allocation: Position size in Bull regime (default 1.0 = 100%)
        bear_allocation: Position size in Bear regime (default 0.3 = 30%)
        sideways_allocation: Position size in Sideways regime (default 0.5 = 50%)
        vol_scale: Whether to scale positions by inverse volatility
        vol_target: Target annualized volatility when vol_scale=True
        vol_lookback: Lookback for volatility calculation
        min_allocation: Minimum position size (default 0.0)
        max_allocation: Maximum position size (default 1.0)

    Example:
        >>> detector = RegimeDetector(n_regimes=2)
        >>> strategy = RegimeStrategy(
        ...     regime_detector=detector,
        ...     bull_allocation=1.0,
        ...     bear_allocation=0.3,
        ... )
        >>> results = strategy.backtest(prices)
        >>> print(results.summary())
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        bull_allocation: float = 1.0,
        bear_allocation: float = 0.3,
        sideways_allocation: float = 0.5,
        vol_scale: bool = False,
        vol_target: float = 0.15,
        vol_lookback: int = 21,
        min_allocation: float = 0.0,
        max_allocation: float = 1.0,
    ) -> None:
        """Initialize the regime strategy."""
        self.regime_detector = regime_detector or RegimeDetector(n_regimes=2)
        self.bull_allocation = bull_allocation
        self.bear_allocation = bear_allocation
        self.sideways_allocation = sideways_allocation
        self.vol_scale = vol_scale
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

        self._is_fitted = False

    def fit(self, returns: NDArray[np.floating]) -> "RegimeStrategy":
        """
        Fit the regime detector on historical returns.

        Args:
            returns: Array of log returns

        Returns:
            self: Fitted strategy
        """
        self.regime_detector.fit(returns)
        self._is_fitted = True
        return self

    def get_position(
        self,
        returns: NDArray[np.floating],
    ) -> Tuple[float, str, float]:
        """
        Get current position size based on regime.

        Args:
            returns: Historical returns up to current point

        Returns:
            Tuple of (position_size, regime_name, regime_probability)
        """
        if not self._is_fitted:
            self.fit(returns)

        regime_name, prob = self.regime_detector.get_current_regime(returns)

        # Base allocation from regime
        if regime_name == 'bull':
            base_alloc = self.bull_allocation
        elif regime_name == 'bear':
            base_alloc = self.bear_allocation
        else:  # sideways
            base_alloc = self.sideways_allocation

        # Optional volatility scaling
        if self.vol_scale and len(returns) >= self.vol_lookback:
            recent_vol = np.std(returns[-self.vol_lookback:]) * np.sqrt(252)
            vol_scalar = self.vol_target / (recent_vol + 1e-10)
            base_alloc = base_alloc * vol_scalar

        # Apply bounds
        position = np.clip(base_alloc, self.min_allocation, self.max_allocation)

        return float(position), regime_name, prob

    def backtest(
        self,
        prices: Optional[NDArray[np.floating]] = None,
        returns: Optional[NDArray[np.floating]] = None,
        warmup: int = 252,
        transaction_cost_bps: float = 10.0,
    ) -> BacktestResult:
        """
        Backtest the strategy on historical data.

        Args:
            prices: Price series (will compute returns if provided)
            returns: Return series (used if prices not provided)
            warmup: Number of observations for initial regime fitting
            transaction_cost_bps: Transaction cost in basis points

        Returns:
            BacktestResult with performance metrics
        """
        # Compute returns from prices if needed
        if prices is not None:
            prices = np.asarray(prices)
            returns = np.diff(np.log(prices))
        elif returns is not None:
            returns = np.asarray(returns)
        else:
            raise ValueError("Either prices or returns must be provided")

        n = len(returns)
        if n < warmup + 10:
            raise ValueError(f"Need at least {warmup + 10} observations, got {n}")

        # Initialize arrays
        positions = np.zeros(n)
        regimes = np.zeros(n, dtype=np.int_)
        strategy_returns = np.zeros(n)

        # Fit on warmup period
        self.fit(returns[:warmup])

        # Walk forward
        prev_position = 0.0
        for i in range(warmup, n):
            # Get position for this period (using data up to but not including i)
            position, regime_name, _ = self.get_position(returns[:i])

            # Map regime name to int
            if regime_name == 'bull':
                regimes[i] = 0
            elif regime_name == 'bear':
                regimes[i] = 1
            else:
                regimes[i] = 2

            positions[i] = position

            # Compute return (position from yesterday * today's return)
            strategy_returns[i] = position * returns[i]

            # Deduct transaction costs for position changes
            position_change = abs(position - prev_position)
            if position_change > 0.01:  # Only if meaningful change
                cost = position_change * transaction_cost_bps / 10000
                strategy_returns[i] -= cost

            prev_position = position

        # Compute metrics (only on trading period)
        trading_returns = strategy_returns[warmup:]
        trading_positions = positions[warmup:]
        trading_regimes = regimes[warmup:]
        actual_returns = returns[warmup:]

        # Basic metrics
        total_return = np.exp(np.sum(trading_returns)) - 1
        n_years = len(trading_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        volatility = np.std(trading_returns) * np.sqrt(252)
        sharpe = (annualized_return / volatility) if volatility > 0 else 0

        # Sortino (downside deviation)
        downside_returns = trading_returns[trading_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino = (annualized_return / downside_vol) if downside_vol > 0 else 0

        # Max drawdown
        equity_curve = np.exp(np.cumsum(trading_returns))
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / running_max - 1
        max_drawdown = float(np.min(drawdowns))

        # Calmar
        calmar = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0

        # Win rate
        win_rate = np.mean(trading_returns > 0)

        # Average position
        avg_position = np.mean(trading_positions)

        # Regime accuracy: did regime correctly predict return sign?
        bull_mask = trading_regimes == 0
        bear_mask = trading_regimes == 1

        correct = 0
        total = 0
        if bull_mask.sum() > 0:
            correct += np.sum(actual_returns[bull_mask] > 0)
            total += bull_mask.sum()
        if bear_mask.sum() > 0:
            correct += np.sum(actual_returns[bear_mask] < 0)
            total += bear_mask.sum()

        regime_accuracy = correct / total if total > 0 else 0.5

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_drawdown,
            calmar=calmar,
            volatility=volatility,
            win_rate=win_rate,
            avg_position=avg_position,
            regime_accuracy=regime_accuracy,
            returns=strategy_returns,
            positions=positions,
            regimes=regimes,
            equity_curve=np.exp(np.cumsum(strategy_returns)),
        )

    def compare_to_benchmark(
        self,
        prices: NDArray[np.floating],
        warmup: int = 252,
    ) -> Dict[str, BacktestResult]:
        """
        Compare strategy to buy-and-hold benchmark.

        Args:
            prices: Price series
            warmup: Warmup period

        Returns:
            Dictionary with 'strategy' and 'benchmark' BacktestResults
        """
        # Strategy results
        strategy_result = self.backtest(prices=prices, warmup=warmup)

        # Benchmark: buy and hold
        returns = np.diff(np.log(prices))
        trading_returns = returns[warmup:]

        total_return = np.exp(np.sum(trading_returns)) - 1
        n_years = len(trading_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = np.std(trading_returns) * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        downside_returns = trading_returns[trading_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0

        equity_curve = np.exp(np.cumsum(trading_returns))
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / running_max - 1
        max_drawdown = float(np.min(drawdowns))
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        benchmark_result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_drawdown,
            calmar=calmar,
            volatility=volatility,
            win_rate=np.mean(trading_returns > 0),
            avg_position=1.0,
            regime_accuracy=0.5,
            returns=np.concatenate([np.zeros(warmup), trading_returns]),
            positions=np.ones(len(returns)),
            regimes=np.zeros(len(returns), dtype=np.int_),
            equity_curve=np.concatenate([np.ones(warmup), equity_curve]),
        )

        return {
            'strategy': strategy_result,
            'benchmark': benchmark_result,
        }

    def __repr__(self) -> str:
        return (
            f"RegimeStrategy(bull={self.bull_allocation:.0%}, "
            f"bear={self.bear_allocation:.0%}, "
            f"sideways={self.sideways_allocation:.0%}, "
            f"vol_scale={self.vol_scale})"
        )


def quick_backtest(
    prices: NDArray[np.floating],
    bull_alloc: float = 1.0,
    bear_alloc: float = 0.3,
    n_regimes: int = 2,
) -> BacktestResult:
    """
    Quick backtest with sensible defaults.

    Args:
        prices: Price series
        bull_alloc: Allocation in bull regime
        bear_alloc: Allocation in bear regime
        n_regimes: Number of regimes (2 or 3)

    Returns:
        BacktestResult

    Example:
        >>> from fractime.strategy import quick_backtest
        >>> result = quick_backtest(prices, bull_alloc=1.0, bear_alloc=0.3)
        >>> print(f"Sharpe: {result.sharpe:.2f}")
    """
    strategy = RegimeStrategy(
        regime_detector=RegimeDetector(n_regimes=n_regimes),
        bull_allocation=bull_alloc,
        bear_allocation=bear_alloc,
    )
    return strategy.backtest(prices=prices)
