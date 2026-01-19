"""
Walk-forward validation for time series forecasting models.

The core idea: Fit = Backtest
- For each time step, fit on data up to that point
- Generate forecast for next step(s)
- Observe actual values
- Record metrics

For Bayesian models: Sequential updating (online learning)
For classical models: Refit on expanding window
"""

import numpy as np
from typing import Dict, Optional, Any, List
from .metrics import ForecastMetrics


class WalkForwardValidator:
    """
    Walk-forward validation with expanding or rolling windows.

    This is the backtesting framework that works with ANY model.
    The key innovation: For Bayesian models, posterior(t) becomes prior(t+1).

    Parameters:
        model: Any model with fit() and predict() methods
        initial_window: Minimum data points before first forecast
        step_size: How many steps between refits (1 = refit every step)
        forecast_horizon: How many steps ahead to forecast
        window_type: 'expanding' (use all history) or 'rolling' (fixed window)
        rolling_window_size: Size of rolling window (if window_type='rolling')

    Example:
        >>> from fractime import FractalForecaster
        >>> from fractime.backtesting import WalkForwardValidator
        >>>
        >>> validator = WalkForwardValidator(FractalForecaster())
        >>> results = validator.run(prices, dates)
        >>>
        >>> print(f"RMSE: {results['metrics']['rmse']:.4f}")
    """

    def __init__(
        self,
        model: Any,
        initial_window: int = 252,  # 1 year of daily data
        step_size: int = 1,          # Refit every step
        forecast_horizon: int = 1,   # 1-step-ahead forecast
        window_type: str = 'expanding',  # or 'rolling'
        rolling_window_size: Optional[int] = None,
        verbose: bool = True
    ):
        self.model = model
        self.initial_window = initial_window
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon
        self.window_type = window_type
        self.rolling_window_size = rolling_window_size or initial_window
        self.verbose = verbose

        # Storage for results
        self.forecasts = []
        self.actuals = []
        self.forecast_times = []
        self.parameter_history = []
        self.ci_lower = []
        self.ci_upper = []

    def run(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run walk-forward validation.

        At each step t:
        1. Fit model on data[0:t] (or data[t-window:t] for rolling)
        2. Forecast data[t:t+horizon]
        3. Observe actual data[t:t+horizon]
        4. Record metrics and parameters

        Args:
            prices: Price time series
            dates: Optional date array

        Returns:
            Dictionary with:
                - forecasts: All forecasts made
                - actuals: All actual values
                - metrics: Accuracy metrics
                - parameter_history: Parameter evolution (if available)
                - forecast_times: When each forecast was made
        """
        n = len(prices)

        if n < self.initial_window + self.forecast_horizon:
            raise ValueError(
                f"Need at least {self.initial_window + self.forecast_horizon} data points, "
                f"got {n}"
            )

        if self.verbose:
            print("=" * 70)
            print("WALK-FORWARD VALIDATION")
            print("=" * 70)
            print(f"Total data points:     {n}")
            print(f"Initial window:        {self.initial_window}")
            print(f"Forecast horizon:      {self.forecast_horizon}")
            print(f"Step size:             {self.step_size}")
            print(f"Window type:           {self.window_type}")

            n_forecasts = (n - self.initial_window - self.forecast_horizon) // self.step_size + 1
            print(f"Number of forecasts:   {n_forecasts}")
            print()

        # Walk forward through time
        for t in range(self.initial_window, n - self.forecast_horizon, self.step_size):
            if self.verbose and (t - self.initial_window) % 50 == 0:
                progress = (t - self.initial_window) / (n - self.initial_window - self.forecast_horizon)
                print(f"Progress: {progress:.1%} (t={t}/{n})")

            # Determine training window
            if self.window_type == 'expanding':
                train_start = 0
                train_end = t
            else:  # rolling
                train_start = max(0, t - self.rolling_window_size)
                train_end = t

            # Training data
            train_prices = prices[train_start:train_end]
            train_dates = dates[train_start:train_end] if dates is not None else None

            # Fit model
            # For Bayesian models with sequential updating, this might use previous posterior
            try:
                self.model.fit(train_prices, dates=train_dates)
            except TypeError:
                # Some models don't accept dates
                self.model.fit(train_prices)

            # Generate forecast
            try:
                result = self.model.predict(n_steps=self.forecast_horizon)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Forecast failed at t={t}: {e}")
                continue

            # Extract forecast (handle different result formats)
            if isinstance(result, dict):
                # Use weighted forecast if available, otherwise regular forecast
                if 'weighted_forecast' in result:
                    forecast = result['weighted_forecast']
                elif 'forecast' in result:
                    forecast = result['forecast']
                else:
                    forecast = result.get('mean', np.zeros(self.forecast_horizon))

                # Extract confidence intervals if available
                lower = result.get('weighted_lower', result.get('lower', None))
                upper = result.get('weighted_upper', result.get('upper', None))
            else:
                # Result is just an array
                forecast = result
                lower = None
                upper = None

            # Observe actual
            actual = prices[t:t+self.forecast_horizon]

            # Store results
            self.forecasts.append(forecast)
            self.actuals.append(actual)
            self.forecast_times.append(t)

            if lower is not None and upper is not None:
                self.ci_lower.append(lower)
                self.ci_upper.append(upper)

            # Track parameters if model provides them
            if hasattr(self.model, 'get_parameter_posterior_summary'):
                try:
                    params = self.model.get_parameter_posterior_summary()
                    self.parameter_history.append({
                        'time': t,
                        'hurst_mean': params['hurst']['mean'],
                        'hurst_sd': params['hurst']['sd']
                    })
                except:
                    pass
            elif hasattr(self.model, 'hurst'):
                # Classical model with point estimate
                self.parameter_history.append({
                    'time': t,
                    'hurst': self.model.hurst
                })

        # Compute metrics
        metrics = self._compute_metrics()

        if self.verbose:
            print("\n" + "=" * 70)
            print("VALIDATION COMPLETE")
            print("=" * 70)
            ForecastMetrics.print_summary(metrics)

        return {
            'forecasts': self.forecasts,
            'actuals': self.actuals,
            'forecast_times': self.forecast_times,
            'metrics': metrics,
            'parameter_history': self.parameter_history,
            'ci_lower': self.ci_lower if self.ci_lower else None,
            'ci_upper': self.ci_upper if self.ci_upper else None,
        }

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute accuracy and calibration metrics from stored results."""
        if not self.forecasts:
            return {}

        # Convert lists to arrays (handle variable-length forecasts)
        # Use only the first step of multi-step forecasts for simplicity
        forecasts_array = np.array([f[0] if len(f) > 0 else np.nan for f in self.forecasts])
        actuals_array = np.array([a[0] if len(a) > 0 else np.nan for a in self.actuals])

        # Remove NaN values
        valid_mask = ~(np.isnan(forecasts_array) | np.isnan(actuals_array))
        forecasts_array = forecasts_array[valid_mask]
        actuals_array = actuals_array[valid_mask]

        if len(forecasts_array) == 0:
            return {}

        # Get current prices (for directional accuracy)
        # Current price is the actual value right before the forecast
        current_prices = np.array([
            self.actuals[i-1][-1] if i > 0 else actuals_array[i]
            for i in range(len(actuals_array))
        ])

        # Confidence intervals (if available)
        if self.ci_lower and self.ci_upper:
            lower_array = np.array([l[0] for l in self.ci_lower])
            upper_array = np.array([u[0] for u in self.ci_upper])
            lower_array = lower_array[valid_mask]
            upper_array = upper_array[valid_mask]
        else:
            lower_array = None
            upper_array = None

        # Compute all metrics
        metrics = ForecastMetrics.compute_all(
            forecasts=forecasts_array,
            actuals=actuals_array,
            current_prices=current_prices,
            lower=lower_array,
            upper=upper_array
        )

        return metrics

    def get_parameter_stability(self, window: int = 20) -> float:
        """
        Measure parameter stability (overfitting metric).

        High variance in parameter estimates suggests overfitting.

        Args:
            window: Window size for measuring stability

        Returns:
            Parameter variance (lower is better)
        """
        if not self.parameter_history:
            return 0.0

        # Extract Hurst values
        if 'hurst_mean' in self.parameter_history[0]:
            # Bayesian model
            hurst_values = [p['hurst_mean'] for p in self.parameter_history[-window:]]
        elif 'hurst' in self.parameter_history[0]:
            # Classical model
            hurst_values = [p['hurst'] for p in self.parameter_history[-window:]]
        else:
            return 0.0

        return np.std(hurst_values)

    def plot_results(self):
        """
        Plot forecast vs actual (requires matplotlib).

        Shows:
        - Forecasts vs actuals over time
        - Confidence intervals (if available)
        - Parameter evolution (if available)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, cannot plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Forecasts vs Actuals
        ax1 = axes[0]
        forecasts_plot = [f[0] for f in self.forecasts]
        actuals_plot = [a[0] for a in self.actuals]

        ax1.plot(self.forecast_times, actuals_plot, 'b-', label='Actual', alpha=0.7)
        ax1.plot(self.forecast_times, forecasts_plot, 'r--', label='Forecast', alpha=0.7)

        # CI if available
        if self.ci_lower and self.ci_upper:
            lower_plot = [l[0] for l in self.ci_lower]
            upper_plot = [u[0] for u in self.ci_upper]
            ax1.fill_between(self.forecast_times, lower_plot, upper_plot,
                            alpha=0.2, color='red', label='95% CI')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.set_title('Forecast vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Parameter Evolution
        ax2 = axes[1]
        if self.parameter_history:
            times = [p['time'] for p in self.parameter_history]

            if 'hurst_mean' in self.parameter_history[0]:
                # Bayesian with uncertainty
                hurst = [p['hurst_mean'] for p in self.parameter_history]
                hurst_sd = [p['hurst_sd'] for p in self.parameter_history]
                ax2.plot(times, hurst, 'g-', label='Hurst Exponent')
                ax2.fill_between(times,
                                np.array(hurst) - np.array(hurst_sd),
                                np.array(hurst) + np.array(hurst_sd),
                                alpha=0.2, color='green')
            elif 'hurst' in self.parameter_history[0]:
                # Classical point estimate
                hurst = [p['hurst'] for p in self.parameter_history]
                ax2.plot(times, hurst, 'g-', label='Hurst Exponent')

            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Walk (H=0.5)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Hurst Exponent')
            ax2.set_title('Parameter Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No parameter history available',
                    ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        plt.show()

        return fig
