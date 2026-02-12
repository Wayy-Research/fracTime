#!/usr/bin/env python3
"""
FracTime vs ARIMA Simulation Study.

Runs 6 controlled scenarios with synthetic and real data to characterize
when fractal-based forecasting outperforms ARIMA and vice versa.

Usage:
    python scripts/run_simulation_study.py          # Full study (20 seeds)
    python scripts/run_simulation_study.py --quick   # Quick run (5 seeds, 2 scenarios)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fractime.research.config import ExperimentConfig
from fractime.research.experiment import ExperimentRunner
from fractime.research.metrics import ForecastMetrics
from fractime.synthetic import (
    create_synthetic_asset,
    generate_arima_prices,
    generate_fat_tail_prices,
    generate_fbm_prices,
    generate_regime_switching_prices,
    generate_volatility_shift_prices,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "paper_results" / "simulation_study"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for the simulation study."""

    n_seeds: int = 20
    n_obs: int = 1000
    train_window: int = 252
    step_size: int = 63
    n_paths: int = 500
    horizons: Dict[str, int] = field(default_factory=lambda: {"5d": 5, "21d": 21})
    models: List[str] = field(default_factory=lambda: ["fractime_rs", "arima"])
    scenarios: Optional[List[int]] = None  # None = all

    @classmethod
    def quick(cls) -> "SimulationConfig":
        return cls(
            n_seeds=5,
            n_obs=500,
            train_window=252,
            step_size=63,
            n_paths=200,
            scenarios=[1, 2],
        )


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def _make_experiment_config(sim: SimulationConfig, name: str) -> ExperimentConfig:
    """Create an ExperimentConfig matching simulation parameters."""
    return ExperimentConfig(
        name=name,
        description=name,
        assets=[],
        horizons=sim.horizons,
        models=sim.models,
        train_window=sim.train_window,
        step_size=sim.step_size,
        window_type="rolling",
        rolling_window_size=sim.train_window,
        n_paths=sim.n_paths,
        output_dir=OUTPUT_DIR,
    )


def _run_walk_forward(
    asset: AssetData,
    sim: SimulationConfig,
    scenario_name: str,
) -> Dict[str, Dict[str, ForecastMetrics]]:
    """Run walk-forward experiment on a single synthetic asset."""
    config = _make_experiment_config(sim, scenario_name)
    runner = ExperimentRunner(config, verbose=False)
    result = runner.run_all({asset.ticker: asset}, models=sim.models)
    return result.metrics


def _aggregate_metrics(
    all_metrics: List[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate metrics across seeds: compute mean and std for each metric."""
    if not all_metrics:
        return {}

    # Collect metric names from first seed
    first = all_metrics[0]
    models = list(first.keys())
    if not models:
        return {}

    horizons = list(first[models[0]].keys())
    metric_fields = [
        "rmse", "mae", "direction_accuracy", "coverage_95",
        "sharpe_ratio", "crps", "n_forecasts",
    ]

    agg: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model in models:
        agg[model] = {}
        for horizon in horizons:
            values: Dict[str, List[float]] = {f: [] for f in metric_fields}
            for seed_metrics in all_metrics:
                m = seed_metrics.get(model, {}).get(horizon)
                if m is None:
                    continue
                for f in metric_fields:
                    val = getattr(m, f, np.nan) if hasattr(m, f) else np.nan
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        values[f].append(float(val))

            summary: Dict[str, float] = {}
            for f in metric_fields:
                vals = values[f]
                if vals:
                    summary[f"{f}_mean"] = float(np.mean(vals))
                    summary[f"{f}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    summary[f"{f}_n"] = len(vals)
            agg[model][horizon] = summary

    return agg


def run_scenario_1(sim: SimulationConfig) -> Dict[str, Any]:
    """Scenario 1: Varying Hurst exponent.

    Hypothesis: FracTime wins at H far from 0.5, ARIMA wins at H ~ 0.5.
    """
    print("\n=== Scenario 1: Varying Hurst ===")
    hurst_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    results: Dict[str, Any] = {"hurst_values": hurst_values, "by_hurst": {}}

    for h in hurst_values:
        print(f"  H={h:.2f}: ", end="", flush=True)
        seed_metrics: List[Dict[str, Dict[str, ForecastMetrics]]] = []

        for seed in range(sim.n_seeds):
            prices = generate_fbm_prices(
                sim.n_obs, hurst=h, volatility=0.20, seed=seed * 100 + int(h * 100)
            )
            asset = create_synthetic_asset(
                prices, f"FBM_H{h:.2f}", f"fBm H={h:.2f}"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = _run_walk_forward(asset, sim, f"scenario1_H{h:.2f}")
            seed_metrics.append(metrics)
            print(".", end="", flush=True)

        agg = _aggregate_metrics(seed_metrics)
        results["by_hurst"][str(h)] = agg
        print(" done")

    return results


def run_scenario_2(sim: SimulationConfig) -> Dict[str, Any]:
    """Scenario 2: ARIMA-native data.

    Hypothesis: ARIMA dominates on data that matches its assumptions.
    """
    print("\n=== Scenario 2: ARIMA-Native Data ===")
    arima_configs = [
        {"name": "AR1_phi05", "order": (1, 1, 0), "ar_params": [0.5], "ma_params": None},
        {"name": "MA1_theta05", "order": (0, 1, 1), "ar_params": None, "ma_params": [0.5]},
        {"name": "ARMA21", "order": (2, 1, 1), "ar_params": [0.3, 0.2], "ma_params": [0.4]},
    ]
    results: Dict[str, Any] = {"configs": [c["name"] for c in arima_configs], "by_config": {}}

    for cfg in arima_configs:
        print(f"  {cfg['name']}: ", end="", flush=True)
        seed_metrics: List[Dict[str, Dict[str, ForecastMetrics]]] = []

        for seed in range(sim.n_seeds):
            prices = generate_arima_prices(
                sim.n_obs,
                order=cfg["order"],
                ar_params=cfg["ar_params"],
                ma_params=cfg["ma_params"],
                volatility=0.005,
                seed=seed * 100,
            )
            asset = create_synthetic_asset(prices, cfg["name"], cfg["name"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = _run_walk_forward(asset, sim, f"scenario2_{cfg['name']}")
            seed_metrics.append(metrics)
            print(".", end="", flush=True)

        agg = _aggregate_metrics(seed_metrics)
        results["by_config"][cfg["name"]] = agg
        print(" done")

    return results


def run_scenario_3(sim: SimulationConfig) -> Dict[str, Any]:
    """Scenario 3: Regime switching (H alternates 0.3 <-> 0.7 every 126d).

    Hypothesis: FracTime adapts to Hurst shifts better than ARIMA.
    """
    print("\n=== Scenario 3: Regime Switching ===")
    seed_metrics: List[Dict[str, Dict[str, ForecastMetrics]]] = []

    for seed in range(sim.n_seeds):
        prices, labels = generate_regime_switching_prices(
            sim.n_obs,
            regime_hursts=(0.3, 0.7),
            regime_durations=(126, 126),
            seed=seed * 100,
        )
        asset = create_synthetic_asset(prices, "REGIME_SWITCH", "Regime H=0.3/0.7")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = _run_walk_forward(asset, sim, "scenario3_regime")
        seed_metrics.append(metrics)
        print(".", end="", flush=True)

    print(" done")
    agg = _aggregate_metrics(seed_metrics)
    return {"aggregated": agg}


def run_scenario_4(sim: SimulationConfig) -> Dict[str, Any]:
    """Scenario 4: Structural volatility breaks (calm -> crisis -> calm).

    Hypothesis: FracTime's MC paths adapt faster to vol regime changes.
    """
    print("\n=== Scenario 4: Structural Breaks ===")
    seed_metrics: List[Dict[str, Dict[str, ForecastMetrics]]] = []

    for seed in range(sim.n_seeds):
        prices, vol_regime = generate_volatility_shift_prices(
            sim.n_obs,
            hurst=0.5,
            volatility_sequence=[(333, 0.15), (334, 0.60), (333, 0.15)],
            seed=seed * 100,
        )
        asset = create_synthetic_asset(prices, "VOL_BREAK", "Calm-Crisis-Calm")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = _run_walk_forward(asset, sim, "scenario4_volbreak")
        seed_metrics.append(metrics)
        print(".", end="", flush=True)

    print(" done")
    agg = _aggregate_metrics(seed_metrics)
    return {"aggregated": agg}


def run_scenario_5(sim: SimulationConfig) -> Dict[str, Any]:
    """Scenario 5: Fat tails (t-distributed innovations, df=5).

    Hypothesis: FracTime's empirical intervals beat ARIMA's Gaussian intervals.
    """
    print("\n=== Scenario 5: Fat Tails ===")
    df_values = [5.0, 10.0]
    results: Dict[str, Any] = {"df_values": df_values, "by_df": {}}

    for df in df_values:
        print(f"  df={df:.0f}: ", end="", flush=True)
        seed_metrics: List[Dict[str, Dict[str, ForecastMetrics]]] = []

        for seed in range(sim.n_seeds):
            prices = generate_fat_tail_prices(
                sim.n_obs, hurst=0.5, df=df, volatility=0.20, seed=seed * 100
            )
            asset = create_synthetic_asset(
                prices, f"FAT_DF{df:.0f}", f"t-dist df={df:.0f}"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = _run_walk_forward(asset, sim, f"scenario5_df{df:.0f}")
            seed_metrics.append(metrics)
            print(".", end="", flush=True)

        agg = _aggregate_metrics(seed_metrics)
        results["by_df"][str(df)] = agg
        print(" done")

    return results


def run_scenario_6(sim: SimulationConfig) -> Dict[str, Any]:
    """Scenario 6: Model simulation comparison on real S&P 500 data.

    Fits both models to real data, then compares statistical properties
    of simulated paths against the real data.
    """
    print("\n=== Scenario 6: Model Simulation Comparison ===")

    try:
        import yfinance as yf
    except ImportError:
        print("  SKIPPED: yfinance not installed")
        return {"skipped": True, "reason": "yfinance not installed"}

    # Fetch S&P 500 last ~2 years
    print("  Fetching S&P 500 data...", end="", flush=True)
    try:
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(period="2y", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 504:
            print(" insufficient data")
            return {"skipped": True, "reason": "insufficient data"}
        prices = df["Close"].to_numpy()[-504:]
    except Exception as e:
        print(f" failed: {e}")
        return {"skipped": True, "reason": str(e)}
    print(" done")

    train_prices = prices[:252]
    test_prices = prices[252:]
    test_returns = np.diff(np.log(test_prices))

    # Real data statistics
    real_stats = _compute_path_stats(test_returns)

    # Fit FracTime and generate simulations via Simulator (public API)
    print("  Fitting FracTime...", end="", flush=True)
    fractime_sim_stats = []
    try:
        from fractime import Simulator

        sim = Simulator(train_prices)
        paths = sim.simulate(n_steps=252, n_paths=100)
        for i in range(paths.shape[0]):
            sim_returns = np.diff(np.log(paths[i])) if paths.shape[1] > 1 else np.array([0.0])
            fractime_sim_stats.append(_compute_path_stats(sim_returns))
    except Exception as e:
        print(f" failed: {e}")
        fractime_sim_stats = []
    print(" done")

    # Fit ARIMA and generate simulations
    print("  Fitting ARIMA...", end="", flush=True)
    arima_sim_stats = []
    try:
        from fractime.baselines import ARIMAForecaster

        for seed in range(100):
            np.random.seed(seed)
            arima = ARIMAForecaster()
            arima.fit(train_prices)
            pred = arima.predict(n_steps=252, n_paths=1)
            if isinstance(pred, dict):
                sim_prices = pred.get("forecast", pred.get("mean", np.array([train_prices[-1]])))
            elif hasattr(pred, "forecast"):
                sim_prices = pred.forecast
            else:
                sim_prices = np.array([train_prices[-1]])
            sim_prices = np.atleast_1d(sim_prices)
            sim_returns = np.diff(np.log(sim_prices)) if len(sim_prices) > 1 else np.array([0.0])
            arima_sim_stats.append(_compute_path_stats(sim_returns))
    except Exception as e:
        print(f" failed: {e}")
        arima_sim_stats = []
    print(" done")

    results: Dict[str, Any] = {"real_stats": real_stats}

    if fractime_sim_stats:
        results["fractime_sim"] = _aggregate_path_stats(fractime_sim_stats)
    if arima_sim_stats:
        results["arima_sim"] = _aggregate_path_stats(arima_sim_stats)

    return results


def _compute_path_stats(returns: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a return series."""
    from scipy import stats as sp_stats

    if len(returns) < 5:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0, "hurst": 0.5}

    # Hurst from prices reconstructed from returns
    prices = np.exp(np.concatenate([[0], np.cumsum(returns)])) * 100
    try:
        from fractime._numba import compute_hurst_rs
        h = compute_hurst_rs(prices, min_lag=10, max_lag=min(100, len(prices) // 3))
    except Exception:
        h = 0.5

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "skew": float(sp_stats.skew(returns)),
        "kurtosis": float(sp_stats.kurtosis(returns, fisher=True)),
        "hurst": float(h),
    }


def _aggregate_path_stats(
    stats_list: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate path statistics across simulations."""
    if not stats_list:
        return {}
    keys = list(stats_list[0].keys())
    agg: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [s[k] for s in stats_list if k in s]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary(
    all_results: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate a markdown summary of all scenario results."""
    lines: List[str] = []
    lines.append("# FracTime vs ARIMA Simulation Study — Summary")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}\n")

    # Scenario 1
    if "scenario_1" in all_results:
        lines.append("## Scenario 1: Varying Hurst Exponent")
        s1 = all_results["scenario_1"]
        lines.append("\n| H | FracTime RMSE | ARIMA RMSE | Ratio | FracTime Dir% | ARIMA Dir% | FracTime Cov95 | ARIMA Cov95 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for h_str, agg in s1.get("by_hurst", {}).items():
            ft = agg.get("fractime_rs", {}).get("overall", {})
            ar = agg.get("arima", {}).get("overall", {})
            ft_rmse = ft.get("rmse_mean", float("nan"))
            ar_rmse = ar.get("rmse_mean", float("nan"))
            ratio = ft_rmse / ar_rmse if ar_rmse and ar_rmse > 0 else float("nan")
            ft_dir = ft.get("direction_accuracy_mean", float("nan"))
            ar_dir = ar.get("direction_accuracy_mean", float("nan"))
            ft_cov = ft.get("coverage_95_mean", float("nan"))
            ar_cov = ar.get("coverage_95_mean", float("nan"))
            lines.append(
                f"| {h_str} | {ft_rmse:.2f} | {ar_rmse:.2f} | {ratio:.3f} "
                f"| {ft_dir:.1%} | {ar_dir:.1%} | {ft_cov:.1%} | {ar_cov:.1%} |"
            )
        lines.append("")

    # Scenario 2
    if "scenario_2" in all_results:
        lines.append("## Scenario 2: ARIMA-Native Data")
        s2 = all_results["scenario_2"]
        lines.append("\n| Config | FracTime RMSE | ARIMA RMSE | FracTime Dir% | ARIMA Dir% |")
        lines.append("|---|---|---|---|---|")
        for cfg_name, agg in s2.get("by_config", {}).items():
            ft = agg.get("fractime_rs", {}).get("overall", {})
            ar = agg.get("arima", {}).get("overall", {})
            lines.append(
                f"| {cfg_name} | {ft.get('rmse_mean', float('nan')):.4f} "
                f"| {ar.get('rmse_mean', float('nan')):.4f} "
                f"| {ft.get('direction_accuracy_mean', float('nan')):.1%} "
                f"| {ar.get('direction_accuracy_mean', float('nan')):.1%} |"
            )
        lines.append("")

    # Scenario 3
    if "scenario_3" in all_results:
        lines.append("## Scenario 3: Regime Switching (H=0.3/0.7)")
        _append_aggregated_table(lines, all_results["scenario_3"])

    # Scenario 4
    if "scenario_4" in all_results:
        lines.append("## Scenario 4: Structural Volatility Breaks")
        _append_aggregated_table(lines, all_results["scenario_4"])

    # Scenario 5
    if "scenario_5" in all_results:
        lines.append("## Scenario 5: Fat Tails")
        s5 = all_results["scenario_5"]
        for df_str, agg in s5.get("by_df", {}).items():
            lines.append(f"\n### df = {df_str}")
            lines.append("\n| Model | RMSE | Dir% | Cov95 | CRPS | Sharpe |")
            lines.append("|---|---|---|---|---|---|")
            for model, horizons in agg.items():
                ov = horizons.get("overall", {})
                lines.append(
                    f"| {model} "
                    f"| {ov.get('rmse_mean', float('nan')):.4f} "
                    f"| {ov.get('direction_accuracy_mean', float('nan')):.1%} "
                    f"| {ov.get('coverage_95_mean', float('nan')):.1%} "
                    f"| {ov.get('crps_mean', float('nan')):.4f} "
                    f"| {ov.get('sharpe_ratio_mean', float('nan')):.2f} |"
                )
        lines.append("")

    # Scenario 6
    if "scenario_6" in all_results:
        s6 = all_results["scenario_6"]
        if not s6.get("skipped"):
            lines.append("## Scenario 6: Model Simulation Comparison (S&P 500)")
            real = s6.get("real_stats", {})
            lines.append(f"\nReal data: mean={real.get('mean', 0):.6f}, "
                         f"std={real.get('std', 0):.6f}, "
                         f"skew={real.get('skew', 0):.3f}, "
                         f"kurtosis={real.get('kurtosis', 0):.3f}, "
                         f"hurst={real.get('hurst', 0):.3f}")
            lines.append("\n| Stat | Real | FracTime (mean +/- std) | ARIMA (mean +/- std) |")
            lines.append("|---|---|---|---|")
            for stat in ["mean", "std", "skew", "kurtosis", "hurst"]:
                r_val = real.get(stat, float("nan"))
                ft_agg = s6.get("fractime_sim", {}).get(stat, {})
                ar_agg = s6.get("arima_sim", {}).get(stat, {})
                ft_str = f"{ft_agg.get('mean', float('nan')):.4f} +/- {ft_agg.get('std', float('nan')):.4f}" if ft_agg else "N/A"
                ar_str = f"{ar_agg.get('mean', float('nan')):.4f} +/- {ar_agg.get('std', float('nan')):.4f}" if ar_agg else "N/A"
                lines.append(f"| {stat} | {r_val:.4f} | {ft_str} | {ar_str} |")
            lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "summary.md"
    report_path.write_text(report)
    print(f"\nSummary written to {report_path}")
    return report


def _append_aggregated_table(lines: List[str], scenario_data: Dict[str, Any]) -> None:
    """Append a simple aggregated metrics table for a scenario."""
    agg = scenario_data.get("aggregated", {})
    if not agg:
        lines.append("\n*No results.*\n")
        return
    lines.append("\n| Model | RMSE | MAE | Dir% | Cov95 | CRPS | Sharpe |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, horizons in agg.items():
        ov = horizons.get("overall", {})
        lines.append(
            f"| {model} "
            f"| {ov.get('rmse_mean', float('nan')):.4f} "
            f"| {ov.get('mae_mean', float('nan')):.4f} "
            f"| {ov.get('direction_accuracy_mean', float('nan')):.1%} "
            f"| {ov.get('coverage_95_mean', float('nan')):.1%} "
            f"| {ov.get('crps_mean', float('nan')):.4f} "
            f"| {ov.get('sharpe_ratio_mean', float('nan')):.2f} |"
        )
    lines.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FracTime vs ARIMA Simulation Study")
    parser.add_argument("--quick", action="store_true", help="Quick run: 5 seeds, 2 scenarios, n=500")
    parser.add_argument("--scenarios", type=int, nargs="+", help="Run specific scenarios (1-6)")
    args = parser.parse_args()

    if args.quick:
        sim = SimulationConfig.quick()
        print("=== QUICK MODE: 5 seeds, n=500, scenarios 1+2 ===")
    else:
        sim = SimulationConfig()
        print(f"=== FULL STUDY: {sim.n_seeds} seeds, n={sim.n_obs} ===")

    if args.scenarios:
        sim.scenarios = args.scenarios

    scenarios_to_run = sim.scenarios or [1, 2, 3, 4, 5, 6]

    # Ensure output directory
    scenarios_dir = OUTPUT_DIR / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {
        "config": {
            "n_seeds": sim.n_seeds,
            "n_obs": sim.n_obs,
            "train_window": sim.train_window,
            "step_size": sim.step_size,
            "n_paths": sim.n_paths,
            "horizons": sim.horizons,
            "models": sim.models,
            "timestamp": datetime.now().isoformat(),
        }
    }

    start = time.time()

    scenario_funcs = {
        1: ("scenario_1", run_scenario_1),
        2: ("scenario_2", run_scenario_2),
        3: ("scenario_3", run_scenario_3),
        4: ("scenario_4", run_scenario_4),
        5: ("scenario_5", run_scenario_5),
        6: ("scenario_6", run_scenario_6),
    }

    for s_num in scenarios_to_run:
        if s_num not in scenario_funcs:
            print(f"Unknown scenario {s_num}, skipping")
            continue

        key, func = scenario_funcs[s_num]
        t0 = time.time()
        result = func(sim)
        elapsed = time.time() - t0

        # Save individual scenario
        result["elapsed_seconds"] = elapsed
        all_results[key] = result

        out_path = scenarios_dir / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved to {out_path} ({elapsed:.1f}s)")

    total_time = time.time() - start
    all_results["total_elapsed_seconds"] = total_time

    # Save combined results
    combined_path = OUTPUT_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")

    # Generate summary report
    generate_summary(all_results, OUTPUT_DIR)

    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
