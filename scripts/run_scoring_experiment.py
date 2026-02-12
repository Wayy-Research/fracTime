#!/usr/bin/env python3
"""Compare v1 (symmetric) vs v2 (directional tilt) path scoring.

Runs walk-forward on synthetic data with known Hurst exponents to isolate
whether the H + momentum directional tilt improves forecasting.

Usage:
    python scripts/run_scoring_experiment.py
    python scripts/run_scoring_experiment.py --seeds 20 --n 1000
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress convergence warnings from ARIMA
warnings.filterwarnings("ignore")

from fractime import Forecaster
from fractime.baselines.arima import ARIMAForecaster
from fractime.synthetic import generate_fbm_prices

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "paper_results" / "scoring_experiment"


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward run."""
    model: str
    hurst: float
    seed: int
    forecasts: List[float]
    actuals: List[float]
    current_prices: List[float]
    lowers: List[float]
    uppers: List[float]


def run_walk_forward(
    prices: np.ndarray,
    model_name: str,
    train_window: int = 252,
    horizon: int = 5,
    step_size: int = 21,
    n_paths: int = 500,
) -> WalkForwardResult:
    """Run walk-forward for a single model on a single price series."""
    n = len(prices)
    forecasts, actuals, currents, lowers, uppers = [], [], [], [], []

    for t in range(train_window, n - horizon, step_size):
        train = prices[:t]
        actual = prices[t + horizon - 1]
        current = prices[t - 1]

        try:
            if model_name == "arima":
                model = ARIMAForecaster()
                model.fit(train)
                pred = model.predict(n_steps=horizon)
                fc = pred["forecast"][-1]
                lo = pred["lower"][-1]
                hi = pred["upper"][-1]
            elif model_name in ("fractime_v1", "fractime_v2", "fractime_v3"):
                scoring_map = {"fractime_v1": "v1", "fractime_v2": "v2", "fractime_v3": "v3"}
                model = Forecaster(train, method="rs", scoring=scoring_map[model_name])
                pred = model.predict(steps=horizon, n_paths=n_paths)
                fc = pred.mean[-1]
                lo, hi = pred.ci(0.95)
                lo, hi = lo[-1], hi[-1]
            else:
                raise ValueError(f"Unknown model: {model_name}")

            forecasts.append(float(fc))
            actuals.append(float(actual))
            currents.append(float(current))
            lowers.append(float(lo))
            uppers.append(float(hi))

        except Exception as e:
            # Skip failed forecasts
            continue

    return WalkForwardResult(
        model=model_name,
        hurst=0.0,  # filled by caller
        seed=0,
        forecasts=forecasts,
        actuals=actuals,
        current_prices=currents,
        lowers=lowers,
        uppers=uppers,
    )


def compute_metrics(result: WalkForwardResult) -> Dict[str, float]:
    """Compute evaluation metrics from walk-forward results."""
    fc = np.array(result.forecasts)
    ac = np.array(result.actuals)
    cp = np.array(result.current_prices)
    lo = np.array(result.lowers)
    hi = np.array(result.uppers)

    if len(fc) < 2:
        return {"rmse": np.nan, "dir_acc": np.nan, "dir_info": np.nan,
                "sharpe": np.nan, "coverage_95": np.nan, "n": len(fc)}

    # RMSE
    rmse = float(np.sqrt(np.mean((fc - ac) ** 2)))

    # Direction accuracy
    fc_dir = np.sign(fc - cp)
    ac_dir = np.sign(ac - cp)
    mask = ac_dir != 0
    dir_acc = float(np.mean(fc_dir[mask] == ac_dir[mask])) if mask.any() else np.nan

    # Directional information content: |accuracy - 0.5|
    dir_info = abs(dir_acc - 0.5) if not np.isnan(dir_acc) else np.nan

    # Sharpe ratio of directional signals
    # Forecasts occur every step_size=21 days → ~12 periods/year
    signals = np.sign(fc - cp)
    returns = (ac - cp) / cp
    strat_returns = signals * returns
    if np.std(strat_returns) > 0:
        sharpe = float(np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(12))
    else:
        sharpe = np.nan

    # Coverage
    within = (ac >= lo) & (ac <= hi)
    coverage = float(np.mean(within))

    return {
        "rmse": rmse,
        "dir_acc": dir_acc,
        "dir_info": dir_info,
        "sharpe": sharpe,
        "coverage_95": coverage,
        "n": len(fc),
    }


def run_experiment(
    hurst_values: List[float],
    n_seeds: int = 5,
    n_points: int = 500,
    train_window: int = 252,
    horizon: int = 5,
    step_size: int = 21,
    n_paths: int = 500,
) -> Dict:
    """Run full experiment across Hurst values and seeds."""
    models = ["fractime_v1", "fractime_v2", "fractime_v3", "arima"]
    all_results = {}

    for h in hurst_values:
        print(f"\n{'='*60}")
        print(f"  Hurst = {h}")
        print(f"{'='*60}")

        h_results = {m: [] for m in models}

        for seed in range(n_seeds):
            prices = generate_fbm_prices(
                n=n_points, hurst=h, volatility=0.20, seed=42 + seed
            )
            print(f"  Seed {seed + 1}/{n_seeds}", end="", flush=True)

            for model_name in models:
                wf = run_walk_forward(
                    prices, model_name,
                    train_window=train_window,
                    horizon=horizon,
                    step_size=step_size,
                    n_paths=n_paths,
                )
                wf.hurst = h
                wf.seed = seed
                metrics = compute_metrics(wf)
                h_results[model_name].append(metrics)
                print(f"  [{model_name}: dir={metrics['dir_acc']:.1%}]", end="", flush=True)

            print()

        # Aggregate across seeds
        all_results[str(h)] = {}
        for model_name in models:
            seed_metrics = h_results[model_name]
            agg = {}
            for key in ["rmse", "dir_acc", "dir_info", "sharpe", "coverage_95"]:
                vals = [m[key] for m in seed_metrics if not np.isnan(m[key])]
                if vals:
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                else:
                    agg[f"{key}_mean"] = np.nan
                    agg[f"{key}_std"] = np.nan
            agg["n_seeds"] = len(seed_metrics)
            all_results[str(h)][model_name] = agg

    return all_results


def print_summary(results: Dict) -> None:
    """Print a comparison table."""
    print(f"\n{'='*80}")
    print("  SCORING EXPERIMENT RESULTS")
    print(f"{'='*80}")

    # Direction accuracy comparison
    print(f"\n--- Direction Accuracy (higher = better) ---")
    print(f"{'H':>6} | {'v1':>10} | {'v2':>10} | {'v3':>10} | {'ARIMA':>10} | {'v3-v1':>8} | {'v3-ARIMA':>10}")
    print("-" * 80)
    for h, mr in sorted(results.items(), key=lambda x: float(x[0])):
        v1 = mr.get("fractime_v1", {}).get("dir_acc_mean", np.nan)
        v2 = mr.get("fractime_v2", {}).get("dir_acc_mean", np.nan)
        v3 = mr.get("fractime_v3", {}).get("dir_acc_mean", np.nan)
        ar = mr.get("arima", {}).get("dir_acc_mean", np.nan)
        d1 = v3 - v1 if not (np.isnan(v3) or np.isnan(v1)) else np.nan
        da = v3 - ar if not (np.isnan(v3) or np.isnan(ar)) else np.nan
        print(f"{h:>6} | {v1:>9.1%} | {v2:>9.1%} | {v3:>9.1%} | {ar:>9.1%} | {d1:>+7.1%} | {da:>+9.1%}")

    # Directional information content
    print(f"\n--- Directional Info |acc - 50%| (higher = more signal) ---")
    print(f"{'H':>6} | {'v1':>10} | {'v2':>10} | {'v3':>10} | {'ARIMA':>10}")
    print("-" * 55)
    for h, mr in sorted(results.items(), key=lambda x: float(x[0])):
        v1 = mr.get("fractime_v1", {}).get("dir_info_mean", np.nan)
        v2 = mr.get("fractime_v2", {}).get("dir_info_mean", np.nan)
        v3 = mr.get("fractime_v3", {}).get("dir_info_mean", np.nan)
        ar = mr.get("arima", {}).get("dir_info_mean", np.nan)
        print(f"{h:>6} | {v1:>9.1%} | {v2:>9.1%} | {v3:>9.1%} | {ar:>9.1%}")

    # RMSE comparison
    print(f"\n--- RMSE (lower = better) ---")
    print(f"{'H':>6} | {'v1':>10} | {'v2':>10} | {'v3':>10} | {'ARIMA':>10} | {'v3/ARIMA':>10}")
    print("-" * 65)
    for h, mr in sorted(results.items(), key=lambda x: float(x[0])):
        v1 = mr.get("fractime_v1", {}).get("rmse_mean", np.nan)
        v2 = mr.get("fractime_v2", {}).get("rmse_mean", np.nan)
        v3 = mr.get("fractime_v3", {}).get("rmse_mean", np.nan)
        ar = mr.get("arima", {}).get("rmse_mean", np.nan)
        ratio = v3 / ar if ar > 0 and not np.isnan(v3) else np.nan
        print(f"{h:>6} | {v1:>9.4f} | {v2:>9.4f} | {v3:>9.4f} | {ar:>9.4f} | {ratio:>9.3f}")

    # Sharpe comparison
    print(f"\n--- Sharpe Ratio (higher = better) ---")
    print(f"{'H':>6} | {'v1':>10} | {'v2':>10} | {'v3':>10} | {'ARIMA':>10}")
    print("-" * 55)
    for h, mr in sorted(results.items(), key=lambda x: float(x[0])):
        v1 = mr.get("fractime_v1", {}).get("sharpe_mean", np.nan)
        v2 = mr.get("fractime_v2", {}).get("sharpe_mean", np.nan)
        v3 = mr.get("fractime_v3", {}).get("sharpe_mean", np.nan)
        ar = mr.get("arima", {}).get("sharpe_mean", np.nan)
        print(f"{h:>6} | {v1:>9.2f} | {v2:>9.2f} | {v3:>9.2f} | {ar:>9.2f}")

    # Coverage comparison
    print(f"\n--- Coverage 95% (target: 95%) ---")
    print(f"{'H':>6} | {'v1':>10} | {'v2':>10} | {'v3':>10} | {'ARIMA':>10}")
    print("-" * 55)
    for h, mr in sorted(results.items(), key=lambda x: float(x[0])):
        v1 = mr.get("fractime_v1", {}).get("coverage_95_mean", np.nan)
        v2 = mr.get("fractime_v2", {}).get("coverage_95_mean", np.nan)
        v3 = mr.get("fractime_v3", {}).get("coverage_95_mean", np.nan)
        ar = mr.get("arima", {}).get("coverage_95_mean", np.nan)
        print(f"{h:>6} | {v1:>9.1%} | {v2:>9.1%} | {v3:>9.1%} | {ar:>9.1%}")


def main():
    parser = argparse.ArgumentParser(description="V1 vs V2 scoring experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per Hurst value")
    parser.add_argument("--n", type=int, default=500, help="Price series length")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon")
    parser.add_argument("--n-paths", type=int, default=500, help="MC paths per forecast")
    args = parser.parse_args()

    hurst_values = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"Scoring Experiment: v1 vs v2 vs ARIMA")
    print(f"  Seeds: {args.seeds}, Series length: {args.n}, Horizon: {args.horizon}d")
    print(f"  Hurst values: {hurst_values}")

    t0 = time.time()
    results = run_experiment(
        hurst_values=hurst_values,
        n_seeds=args.seeds,
        n_points=args.n,
        horizon=args.horizon,
        n_paths=args.n_paths,
    )
    elapsed = time.time() - t0

    print_summary(results)
    print(f"\nCompleted in {elapsed:.0f}s")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
