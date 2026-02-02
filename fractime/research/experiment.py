"""
Walk-forward experiment runner for FracTime research.

This module implements walk-forward validation for comparing FracTime
against baseline models across multiple assets and horizons.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig, MODELS
from .data_collector import AssetData
from .metrics import MetricsCalculator, ForecastMetrics

logger = logging.getLogger(__name__)


@dataclass
class ForecastRecord:
    """Record of a single forecast."""

    forecast_date: datetime
    horizon: int
    model_name: str
    ticker: str
    current_price: float
    forecast_mean: float
    forecast_median: float
    lower_50: float
    upper_50: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    actual: Optional[float] = None
    paths: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "forecast_date": self.forecast_date.isoformat() if hasattr(self.forecast_date, "isoformat") else str(self.forecast_date),
            "horizon": self.horizon,
            "model_name": self.model_name,
            "ticker": self.ticker,
            "current_price": float(self.current_price),
            "forecast_mean": float(self.forecast_mean),
            "forecast_median": float(self.forecast_median),
            "lower_50": float(self.lower_50),
            "upper_50": float(self.upper_50),
            "lower_80": float(self.lower_80),
            "upper_80": float(self.upper_80),
            "lower_95": float(self.lower_95),
            "upper_95": float(self.upper_95),
            "actual": float(self.actual) if self.actual is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentResult:
    """Results from a complete experiment run."""

    config: ExperimentConfig
    forecasts: List[ForecastRecord] = field(default_factory=list)
    metrics: Dict[str, Dict[str, ForecastMetrics]] = field(default_factory=dict)
    run_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    def add_forecast(self, record: ForecastRecord) -> None:
        """Add a forecast record."""
        self.forecasts.append(record)

    def get_forecasts_by_model(self, model_name: str) -> List[ForecastRecord]:
        """Get all forecasts for a specific model."""
        return [f for f in self.forecasts if f.model_name == model_name]

    def get_forecasts_by_ticker(self, ticker: str) -> List[ForecastRecord]:
        """Get all forecasts for a specific ticker."""
        return [f for f in self.forecasts if f.ticker == ticker]

    def get_forecasts_by_horizon(self, horizon: int) -> List[ForecastRecord]:
        """Get all forecasts for a specific horizon."""
        return [f for f in self.forecasts if f.horizon == horizon]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "forecasts": [f.to_dict() for f in self.forecasts],
            "metrics": {
                model: {
                    key: metrics.to_dict() if hasattr(metrics, "to_dict") else str(metrics)
                    for key, metrics in model_metrics.items()
                }
                for model, model_metrics in self.metrics.items()
            },
            "run_time": self.run_time,
            "errors": self.errors,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save results to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.config.output_dir / "experiments" / f"{self.config.name}_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved results to {path}")
        return path


class ModelWrapper:
    """Wrapper for unified model interface."""

    def __init__(self, model_name: str, model_config: Dict[str, Any]) -> None:
        self.model_name = model_name
        self.model_config = model_config
        self.model = None
        self._fitted = False

    def fit(self, prices: np.ndarray, dates: Optional[np.ndarray] = None) -> "ModelWrapper":
        """Fit the model to historical data."""
        model_class = self.model_config["class"]
        module_name = self.model_config["module"]
        params = self.model_config.get("params", {})

        try:
            if module_name == "fractime":
                if model_class == "Forecaster":
                    from fractime import Forecaster
                    self.model = Forecaster(prices, dates=dates, **params)
                elif model_class == "Ensemble":
                    from fractime import Ensemble
                    self.model = Ensemble(prices, **params)
            elif module_name == "fractime.baselines":
                from fractime import baselines
                forecaster_class = getattr(baselines, model_class)
                self.model = forecaster_class(**params)
                self.model.fit(prices, dates=dates)
            else:
                raise ValueError(f"Unknown module: {module_name}")

            self._fitted = True
        except Exception as e:
            logger.error(f"Failed to fit {self.model_name}: {e}")
            raise

        return self

    def predict(
        self,
        steps: int,
        n_paths: int = 1000,
        confidence_levels: List[float] = None,
    ) -> Dict[str, Any]:
        """Generate forecast with confidence intervals."""
        if not self._fitted:
            raise ValueError("Model must be fitted before predicting")

        if confidence_levels is None:
            confidence_levels = [0.50, 0.80, 0.95]

        result = {
            "mean": None,
            "median": None,
            "paths": None,
            "intervals": {},
        }

        try:
            if hasattr(self.model, "predict"):
                # FracTime models use 'steps', baseline models use 'n_steps'
                if self.model_config["module"] == "fractime":
                    pred = self.model.predict(steps=steps, n_paths=n_paths)
                else:
                    # Baseline models have different parameter conventions
                    model_class = self.model_config["class"]
                    if model_class == "LSTMForecaster":
                        # LSTM uses n_simulations instead of n_paths
                        pred = self.model.predict(n_steps=steps, n_simulations=min(n_paths, 100))
                    elif model_class in ("ARIMAForecaster", "GARCHForecaster", "ProphetForecaster"):
                        # These accept n_paths parameter
                        pred = self.model.predict(n_steps=steps, n_paths=n_paths)
                    else:
                        # ETS and others use **kwargs
                        pred = self.model.predict(n_steps=steps)

                if hasattr(pred, "mean"):
                    result["mean"] = pred.mean[-1] if hasattr(pred.mean, "__len__") else pred.mean
                elif hasattr(pred, "forecast"):
                    result["mean"] = pred.forecast[-1] if hasattr(pred.forecast, "__len__") else pred.forecast

                if hasattr(pred, "forecast"):
                    result["median"] = pred.forecast[-1] if hasattr(pred.forecast, "__len__") else pred.forecast
                else:
                    result["median"] = result["mean"]

                if hasattr(pred, "_paths"):
                    result["paths"] = pred._paths[:, -1] if pred._paths.ndim > 1 else pred._paths

                for level in confidence_levels:
                    if hasattr(pred, "ci"):
                        lower, upper = pred.ci(level)
                        result["intervals"][level] = {
                            "lower": lower[-1] if hasattr(lower, "__len__") else lower,
                            "upper": upper[-1] if hasattr(upper, "__len__") else upper,
                        }
                    elif hasattr(pred, "lower") and hasattr(pred, "upper"):
                        result["intervals"][level] = {
                            "lower": pred["lower"][-1] if isinstance(pred, dict) else pred.lower[-1],
                            "upper": pred["upper"][-1] if isinstance(pred, dict) else pred.upper[-1],
                        }
                    elif isinstance(pred, dict):
                        result["mean"] = pred.get("mean", pred.get("forecast"))
                        if result["mean"] is not None and hasattr(result["mean"], "__len__"):
                            result["mean"] = result["mean"][-1]
                        result["median"] = result["mean"]
                        if "lower" in pred and "upper" in pred:
                            lower = pred["lower"]
                            upper = pred["upper"]
                            result["intervals"][level] = {
                                "lower": lower[-1] if hasattr(lower, "__len__") else lower,
                                "upper": upper[-1] if hasattr(upper, "__len__") else upper,
                            }

                if hasattr(pred, "metadata"):
                    result["metadata"] = pred.metadata

        except Exception as e:
            logger.error(f"Prediction failed for {self.model_name}: {e}")
            raise

        return result


class ExperimentRunner:
    """Run walk-forward validation experiments.

    This class implements walk-forward validation for comparing FracTime
    against baseline models. It supports expanding and rolling windows,
    multiple forecast horizons, and stores all paths for probabilistic
    evaluation.

    Example:
        >>> config = ExperimentConfig.quick_test()
        >>> runner = ExperimentRunner(config)
        >>> data = {"AAPL": asset_data}
        >>> results = runner.run_all(data)
    """

    def __init__(self, config: ExperimentConfig, verbose: bool = True) -> None:
        """Initialize the experiment runner.

        Args:
            config: Experiment configuration
            verbose: Print progress information
        """
        self.config = config
        self.verbose = verbose
        self.metrics_calculator = MetricsCalculator()

    def _get_train_slice(self, t: int, n: int) -> Tuple[int, int]:
        """Get training data slice indices.

        Args:
            t: Current time index
            n: Total length of data

        Returns:
            Tuple of (start_index, end_index)
        """
        if self.config.window_type == "expanding":
            return (0, t)
        else:
            window_size = self.config.rolling_window_size or self.config.train_window
            start = max(0, t - window_size)
            return (start, t)

    def run_single_asset(
        self,
        asset_data: AssetData,
        models: Optional[List[str]] = None,
    ) -> List[ForecastRecord]:
        """Run experiment for a single asset.

        Args:
            asset_data: Asset price data
            models: List of model names to test (default: config.models)

        Returns:
            List of ForecastRecord objects
        """
        if models is None:
            models = self.config.models

        records: List[ForecastRecord] = []
        prices = asset_data.prices
        dates = asset_data.dates
        n = len(prices)

        min_train = self.config.train_window
        max_horizon = max(self.config.horizons.values())

        for t in range(min_train, n - max_horizon, self.config.step_size):
            train_start, train_end = self._get_train_slice(t, n)
            train_prices = prices[train_start:train_end]
            train_dates = dates[train_start:train_end] if dates is not None else None
            current_price = prices[train_end - 1]
            forecast_date = dates[train_end - 1] if dates is not None else t

            for model_name in models:
                if model_name not in MODELS:
                    logger.warning(f"Unknown model: {model_name}")
                    continue

                model_config = MODELS[model_name]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wrapper = ModelWrapper(model_name, model_config)
                        wrapper.fit(train_prices, train_dates)

                    for horizon_name, horizon in self.config.horizons.items():
                        if train_end + horizon > n:
                            continue

                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pred = wrapper.predict(
                                    steps=horizon,
                                    n_paths=self.config.n_paths,
                                    confidence_levels=self.config.confidence_levels,
                                )

                            actual = prices[train_end + horizon - 1]

                            intervals = pred.get("intervals", {})
                            lower_50 = intervals.get(0.50, {}).get("lower", pred["mean"])
                            upper_50 = intervals.get(0.50, {}).get("upper", pred["mean"])
                            lower_80 = intervals.get(0.80, {}).get("lower", pred["mean"])
                            upper_80 = intervals.get(0.80, {}).get("upper", pred["mean"])
                            lower_95 = intervals.get(0.95, {}).get("lower", pred["mean"])
                            upper_95 = intervals.get(0.95, {}).get("upper", pred["mean"])

                            record = ForecastRecord(
                                forecast_date=forecast_date,
                                horizon=horizon,
                                model_name=model_name,
                                ticker=asset_data.ticker,
                                current_price=current_price,
                                forecast_mean=pred["mean"] if pred["mean"] is not None else current_price,
                                forecast_median=pred["median"] if pred["median"] is not None else pred["mean"],
                                lower_50=lower_50 if lower_50 is not None else pred["mean"],
                                upper_50=upper_50 if upper_50 is not None else pred["mean"],
                                lower_80=lower_80 if lower_80 is not None else pred["mean"],
                                upper_80=upper_80 if upper_80 is not None else pred["mean"],
                                lower_95=lower_95 if lower_95 is not None else pred["mean"],
                                upper_95=upper_95 if upper_95 is not None else pred["mean"],
                                actual=actual,
                                paths=pred.get("paths"),
                                metadata=pred.get("metadata"),
                            )
                            records.append(record)

                        except Exception as e:
                            logger.debug(f"Forecast failed for {model_name} at t={t}, h={horizon}: {e}")

                except Exception as e:
                    logger.debug(f"Model fitting failed for {model_name} at t={t}: {e}")

            if self.verbose and t % (self.config.step_size * 10) == 0:
                progress = (t - min_train) / (n - max_horizon - min_train) * 100
                print(f"  {asset_data.ticker}: {progress:.0f}% complete ({len(records)} forecasts)")

        return records

    def run_all(
        self,
        data: Dict[str, AssetData],
        models: Optional[List[str]] = None,
    ) -> ExperimentResult:
        """Run experiment for all assets in the data dictionary.

        Args:
            data: Dictionary mapping ticker to AssetData
            models: List of model names to test (default: config.models)

        Returns:
            ExperimentResult with all forecasts and metrics
        """
        import time
        start_time = time.time()

        result = ExperimentResult(config=self.config)

        if models is None:
            models = self.config.models

        for i, (ticker, asset_data) in enumerate(data.items()):
            if self.verbose:
                print(f"\n[{i + 1}/{len(data)}] Processing {ticker}...")

            try:
                records = self.run_single_asset(asset_data, models)
                for record in records:
                    result.add_forecast(record)

                if self.verbose:
                    print(f"  Generated {len(records)} forecasts for {ticker}")

            except Exception as e:
                error_msg = f"Failed to process {ticker}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        result.run_time = time.time() - start_time

        if self.verbose:
            print(f"\nExperiment completed in {result.run_time:.1f} seconds")
            print(f"Total forecasts: {len(result.forecasts)}")
            if result.errors:
                print(f"Errors: {len(result.errors)}")

        result.metrics = self._compute_all_metrics(result)

        return result

    def _compute_all_metrics(self, result: ExperimentResult) -> Dict[str, Dict[str, ForecastMetrics]]:
        """Compute metrics for all model/horizon combinations."""
        metrics: Dict[str, Dict[str, ForecastMetrics]] = {}

        for model_name in self.config.models:
            model_forecasts = result.get_forecasts_by_model(model_name)
            if not model_forecasts:
                continue

            metrics[model_name] = {}

            for horizon_name, horizon in self.config.horizons.items():
                horizon_forecasts = [f for f in model_forecasts if f.horizon == horizon]
                if not horizon_forecasts:
                    continue

                forecasts = np.array([f.forecast_mean for f in horizon_forecasts])
                actuals = np.array([f.actual for f in horizon_forecasts if f.actual is not None])
                current_prices = np.array([f.current_price for f in horizon_forecasts])

                if len(actuals) < len(forecasts):
                    forecasts = forecasts[:len(actuals)]
                    current_prices = current_prices[:len(actuals)]

                lowers_95 = np.array([f.lower_95 for f in horizon_forecasts[:len(actuals)]])
                uppers_95 = np.array([f.upper_95 for f in horizon_forecasts[:len(actuals)]])

                paths_list = [f.paths for f in horizon_forecasts[:len(actuals)] if f.paths is not None]
                paths = np.array(paths_list) if paths_list else None

                try:
                    m = self.metrics_calculator.compute_all(
                        forecasts=forecasts,
                        actuals=actuals,
                        current_prices=current_prices,
                        lower=lowers_95,
                        upper=uppers_95,
                        paths=paths,
                        confidence_level=0.95,
                    )
                    metrics[model_name][horizon_name] = m
                except Exception as e:
                    logger.warning(f"Metrics computation failed for {model_name}/{horizon_name}: {e}")

            all_forecasts = np.array([f.forecast_mean for f in model_forecasts if f.actual is not None])
            all_actuals = np.array([f.actual for f in model_forecasts if f.actual is not None])
            all_current = np.array([f.current_price for f in model_forecasts if f.actual is not None])

            if len(all_forecasts) > 0:
                try:
                    all_lowers = np.array([f.lower_95 for f in model_forecasts if f.actual is not None])
                    all_uppers = np.array([f.upper_95 for f in model_forecasts if f.actual is not None])

                    m = self.metrics_calculator.compute_all(
                        forecasts=all_forecasts,
                        actuals=all_actuals,
                        current_prices=all_current,
                        lower=all_lowers,
                        upper=all_uppers,
                        paths=None,
                        confidence_level=0.95,
                    )
                    metrics[model_name]["overall"] = m
                except Exception as e:
                    logger.warning(f"Overall metrics failed for {model_name}: {e}")

        return metrics
