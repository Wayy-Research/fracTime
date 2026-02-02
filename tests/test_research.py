"""
Unit tests for FracTime research framework.

Tests cover:
- Configuration classes
- Metrics computation
- Statistical tests
- Data structures
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import tempfile

from fractime.research.config import (
    ExperimentConfig,
    AssetConfig,
    TEST_UNIVERSE,
    HORIZONS,
    MODELS,
)
from fractime.research.metrics import (
    MetricsCalculator,
    ForecastMetrics,
    compute_rmse,
    compute_mae,
    compute_mape,
    compute_crps,
    compute_direction_accuracy,
    compute_coverage,
    compute_sharpe_ratio,
)
from fractime.research.statistical_tests import (
    diebold_mariano_test,
    model_confidence_set,
    bonferroni_correction,
    holm_correction,
    pairwise_dm_tests,
    compute_loss_matrix_from_forecasts,
)
from fractime.research.experiment import ForecastRecord, ExperimentResult


class TestConfig:
    """Tests for experiment configuration."""

    def test_asset_config_defaults(self):
        """Test AssetConfig default values."""
        asset = AssetConfig("AAPL", "Apple Inc.", "equity", "tech")
        assert asset.ticker == "AAPL"
        assert asset.frequency == "daily"
        assert asset.start_date == "2015-01-01"
        assert asset.end_date == "2024-12-31"

    def test_asset_config_crypto_dates(self):
        """Test that crypto assets get different start date."""
        asset = AssetConfig("BTC-USD", "Bitcoin", "crypto", "major")
        assert asset.start_date == "2017-01-01"

    def test_experiment_config_default(self):
        """Test default experiment configuration."""
        config = ExperimentConfig.default()
        assert config.name == "default_experiment"
        assert config.train_window == 252
        assert len(config.assets) > 0
        assert len(config.models) > 0

    def test_experiment_config_quick_test(self):
        """Test quick test configuration."""
        config = ExperimentConfig.quick_test()
        assert config.name == "quick_test"
        assert len(config.assets) == 3
        assert config.step_size == 63

    def test_experiment_config_to_dict(self):
        """Test configuration serialization."""
        config = ExperimentConfig.quick_test()
        d = config.to_dict()
        assert "name" in d
        assert "assets" in d
        assert "models" in d
        assert "timestamp" in d

    def test_horizons_defined(self):
        """Test that standard horizons are defined."""
        assert "1d" in HORIZONS
        assert "5d" in HORIZONS
        assert "21d" in HORIZONS
        assert "63d" in HORIZONS

    def test_models_defined(self):
        """Test that required models are defined."""
        assert "fractime_rs" in MODELS
        assert "fractime_dfa" in MODELS
        assert "arima" in MODELS
        assert "garch" in MODELS

    def test_test_universe_not_empty(self):
        """Test that test universe has assets."""
        assert len(TEST_UNIVERSE) > 20


class TestMetrics:
    """Tests for metrics computation."""

    def test_rmse_perfect(self):
        """Test RMSE with perfect predictions."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(forecasts, actuals) == 0.0

    def test_rmse_known_value(self):
        """Test RMSE with known value."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([2.0, 3.0, 4.0])
        assert compute_rmse(forecasts, actuals) == 1.0

    def test_mae_perfect(self):
        """Test MAE with perfect predictions."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])
        assert compute_mae(forecasts, actuals) == 0.0

    def test_mae_known_value(self):
        """Test MAE with known value."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([2.0, 3.0, 4.0])
        assert compute_mae(forecasts, actuals) == 1.0

    def test_mape_perfect(self):
        """Test MAPE with perfect predictions."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])
        assert compute_mape(forecasts, actuals) == 0.0

    def test_mape_handles_zero(self):
        """Test MAPE handles zero actuals."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([0.0, 2.0, 3.0])
        result = compute_mape(forecasts, actuals)
        assert not np.isnan(result)

    def test_direction_accuracy_perfect(self):
        """Test direction accuracy with perfect predictions."""
        forecasts = np.array([110.0, 90.0, 120.0])
        actuals = np.array([115.0, 85.0, 125.0])
        current = np.array([100.0, 100.0, 100.0])
        assert compute_direction_accuracy(forecasts, actuals, current) == 1.0

    def test_direction_accuracy_wrong(self):
        """Test direction accuracy with wrong predictions."""
        forecasts = np.array([110.0, 90.0])
        actuals = np.array([90.0, 110.0])
        current = np.array([100.0, 100.0])
        assert compute_direction_accuracy(forecasts, actuals, current) == 0.0

    def test_coverage_perfect(self):
        """Test coverage with perfect intervals."""
        actuals = np.array([10.0, 20.0, 30.0])
        lower = np.array([5.0, 15.0, 25.0])
        upper = np.array([15.0, 25.0, 35.0])
        assert compute_coverage(actuals, lower, upper) == 1.0

    def test_coverage_none(self):
        """Test coverage with no containment."""
        actuals = np.array([10.0, 20.0, 30.0])
        lower = np.array([20.0, 30.0, 40.0])
        upper = np.array([25.0, 35.0, 45.0])
        assert compute_coverage(actuals, lower, upper) == 0.0

    def test_crps_basic(self):
        """Test CRPS computation."""
        forecasts = np.array([10.0, 20.0])
        actuals = np.array([11.0, 19.0])
        crps = compute_crps(forecasts, actuals, paths=None)
        assert crps > 0

    def test_crps_with_paths(self):
        """Test CRPS with Monte Carlo paths."""
        np.random.seed(42)
        forecasts = np.array([10.0, 20.0])
        actuals = np.array([10.5, 19.5])
        paths = np.random.normal(forecasts.reshape(-1, 1), 2, (2, 100))
        crps = compute_crps(forecasts, actuals, paths=paths)
        assert crps > 0

    def test_sharpe_ratio(self):
        """Test Sharpe ratio computation."""
        forecasts = np.array([105.0, 95.0, 110.0, 90.0])
        actuals = np.array([110.0, 90.0, 115.0, 85.0])
        current = np.array([100.0, 100.0, 100.0, 100.0])
        sharpe = compute_sharpe_ratio(forecasts, actuals, current)
        assert sharpe > 0

    def test_metrics_calculator_all(self):
        """Test MetricsCalculator.compute_all()."""
        calc = MetricsCalculator()

        np.random.seed(42)
        forecasts = np.random.normal(100, 5, 50)
        actuals = forecasts + np.random.normal(0, 2, 50)
        current = np.full(50, 100.0)
        lower = forecasts - 5
        upper = forecasts + 5

        metrics = calc.compute_all(
            forecasts=forecasts,
            actuals=actuals,
            current_prices=current,
            lower=lower,
            upper=upper,
        )

        assert isinstance(metrics, ForecastMetrics)
        assert metrics.n_forecasts == 50
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert 0 <= metrics.direction_accuracy <= 1
        assert 0 <= metrics.coverage_95 <= 1

    def test_forecast_metrics_to_dict(self):
        """Test ForecastMetrics serialization."""
        metrics = ForecastMetrics(
            rmse=1.5,
            mae=1.2,
            direction_accuracy=0.65,
            n_forecasts=100,
        )
        d = metrics.to_dict()
        assert d["rmse"] == 1.5
        assert d["mae"] == 1.2
        assert d["n_forecasts"] == 100


class TestStatisticalTests:
    """Tests for statistical significance tests."""

    def test_dm_test_same_errors(self):
        """Test DM test with identical errors."""
        errors = np.random.normal(0, 1, 100)
        result = diebold_mariano_test(errors, errors, "Model1", "Model2")
        assert result.p_value > 0.05

    def test_dm_test_different_errors(self):
        """Test DM test with clearly different errors."""
        np.random.seed(42)
        errors1 = np.random.normal(0, 1, 200)
        errors2 = np.random.normal(0, 3, 200)
        result = diebold_mariano_test(errors1, errors2, "Better", "Worse")
        assert result.mean_loss_diff < 0

    def test_dm_test_result_structure(self):
        """Test DM test result structure."""
        np.random.seed(42)
        errors1 = np.random.normal(0, 1, 100)
        errors2 = np.random.normal(0, 1, 100)
        result = diebold_mariano_test(errors1, errors2)

        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "conclusion")
        assert hasattr(result, "significant_at_05")

    def test_dm_test_to_dict(self):
        """Test DM test result serialization."""
        np.random.seed(42)
        errors1 = np.random.normal(0, 1, 100)
        errors2 = np.random.normal(0, 1, 100)
        result = diebold_mariano_test(errors1, errors2)
        d = result.to_dict()

        assert "statistic" in d
        assert "p_value" in d
        assert "conclusion" in d

    def test_mcs_single_model(self):
        """Test MCS with single model."""
        losses = {"model1": np.random.uniform(0, 1, 100)}
        result = model_confidence_set(losses)
        assert "model1" in result.superior_models

    def test_mcs_multiple_models(self):
        """Test MCS with multiple models."""
        np.random.seed(42)
        losses = {
            "best": np.random.uniform(0, 0.5, 100),
            "medium": np.random.uniform(0.3, 0.8, 100),
            "worst": np.random.uniform(0.6, 1.2, 100),
        }
        result = model_confidence_set(losses, alpha=0.1)

        assert len(result.superior_models) >= 1
        assert "best" in result.superior_models

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.005, 0.02, 0.03, 0.04, 0.05]
        rejections, adjusted = bonferroni_correction(p_values, alpha=0.05)

        assert adjusted == 0.01
        assert rejections[0] is True  # 0.005 < 0.01
        assert rejections[-1] is False  # 0.05 > 0.01

    def test_holm_correction(self):
        """Test Holm-Bonferroni correction."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.05]
        rejections, adjusted_p = holm_correction(p_values, alpha=0.05)

        assert rejections[0] is True
        assert len(adjusted_p) == len(p_values)

    def test_pairwise_dm_tests(self):
        """Test pairwise DM tests against baseline."""
        np.random.seed(42)
        model_errors = {
            "baseline": np.random.normal(0, 1, 100),
            "model1": np.random.normal(0, 1.2, 100),
            "model2": np.random.normal(0, 0.8, 100),
        }
        results = pairwise_dm_tests(model_errors, "baseline")

        assert "model1" in results
        assert "model2" in results
        assert "baseline" not in results

    def test_compute_loss_matrix(self):
        """Test loss matrix computation."""
        forecasts = {
            "model1": np.array([10.0, 20.0, 30.0]),
            "model2": np.array([11.0, 19.0, 31.0]),
        }
        actuals = np.array([10.0, 20.0, 30.0])

        losses = compute_loss_matrix_from_forecasts(forecasts, actuals, "squared")

        assert "model1" in losses
        assert "model2" in losses
        assert losses["model1"][0] == 0.0
        assert losses["model2"][0] == 1.0


class TestDataStructures:
    """Tests for research data structures."""

    def test_forecast_record_basic(self):
        """Test ForecastRecord creation."""
        from datetime import datetime

        record = ForecastRecord(
            forecast_date=datetime.now(),
            horizon=5,
            model_name="fractime_rs",
            ticker="AAPL",
            current_price=150.0,
            forecast_mean=155.0,
            forecast_median=154.5,
            lower_50=152.0,
            upper_50=158.0,
            lower_80=148.0,
            upper_80=162.0,
            lower_95=145.0,
            upper_95=165.0,
            actual=157.0,
        )

        assert record.ticker == "AAPL"
        assert record.horizon == 5
        assert record.actual == 157.0

    def test_forecast_record_to_dict(self):
        """Test ForecastRecord serialization."""
        from datetime import datetime

        record = ForecastRecord(
            forecast_date=datetime(2024, 1, 1),
            horizon=5,
            model_name="test",
            ticker="TEST",
            current_price=100.0,
            forecast_mean=105.0,
            forecast_median=104.0,
            lower_50=102.0,
            upper_50=108.0,
            lower_80=100.0,
            upper_80=110.0,
            lower_95=98.0,
            upper_95=112.0,
        )

        d = record.to_dict()
        assert d["ticker"] == "TEST"
        assert d["horizon"] == 5
        assert d["forecast_mean"] == 105.0

    def test_experiment_result_add_forecast(self):
        """Test ExperimentResult forecast management."""
        from datetime import datetime

        config = ExperimentConfig.quick_test()
        result = ExperimentResult(config=config)

        record = ForecastRecord(
            forecast_date=datetime.now(),
            horizon=1,
            model_name="test",
            ticker="TEST",
            current_price=100.0,
            forecast_mean=101.0,
            forecast_median=101.0,
            lower_50=100.0,
            upper_50=102.0,
            lower_80=99.0,
            upper_80=103.0,
            lower_95=98.0,
            upper_95=104.0,
        )

        result.add_forecast(record)
        assert len(result.forecasts) == 1

    def test_experiment_result_filtering(self):
        """Test ExperimentResult filtering methods."""
        from datetime import datetime

        config = ExperimentConfig.quick_test()
        result = ExperimentResult(config=config)

        for model in ["model1", "model2"]:
            for ticker in ["AAPL", "MSFT"]:
                for horizon in [1, 5]:
                    record = ForecastRecord(
                        forecast_date=datetime.now(),
                        horizon=horizon,
                        model_name=model,
                        ticker=ticker,
                        current_price=100.0,
                        forecast_mean=101.0,
                        forecast_median=101.0,
                        lower_50=100.0,
                        upper_50=102.0,
                        lower_80=99.0,
                        upper_80=103.0,
                        lower_95=98.0,
                        upper_95=104.0,
                    )
                    result.add_forecast(record)

        assert len(result.forecasts) == 8
        assert len(result.get_forecasts_by_model("model1")) == 4
        assert len(result.get_forecasts_by_ticker("AAPL")) == 4
        assert len(result.get_forecasts_by_horizon(1)) == 4

    def test_experiment_result_save(self):
        """Test ExperimentResult serialization."""
        config = ExperimentConfig.quick_test()
        result = ExperimentResult(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_results.json"
            saved_path = result.save(path)

            assert saved_path.exists()
            assert saved_path.suffix == ".json"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_metrics_empty_arrays(self):
        """Test metrics with empty arrays."""
        assert np.isnan(compute_rmse(np.array([]), np.array([])))
        assert np.isnan(compute_mae(np.array([]), np.array([])))
        assert np.isnan(compute_mape(np.array([]), np.array([])))

    def test_metrics_single_value(self):
        """Test metrics with single value."""
        assert compute_rmse(np.array([1.0]), np.array([2.0])) == 1.0
        assert compute_mae(np.array([1.0]), np.array([2.0])) == 1.0

    def test_dm_test_insufficient_data(self):
        """Test DM test with insufficient data."""
        errors = np.array([1.0, 2.0])
        result = diebold_mariano_test(errors, errors)
        assert np.isnan(result.statistic)

    def test_direction_accuracy_no_movement(self):
        """Test direction accuracy when price doesn't move."""
        forecasts = np.array([100.0, 100.0])
        actuals = np.array([100.0, 100.0])
        current = np.array([100.0, 100.0])
        result = compute_direction_accuracy(forecasts, actuals, current)
        assert np.isnan(result)

    def test_config_output_dir_creation(self):
        """Test that config creates output directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output"
            config = ExperimentConfig(output_dir=output_path)

            assert output_path.exists()
            assert (output_path / "data").exists()
            assert (output_path / "experiments").exists()
            assert (output_path / "figures").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
