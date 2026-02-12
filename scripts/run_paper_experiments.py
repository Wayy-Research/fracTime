#!/usr/bin/env python3
"""
Run experiments for the FracTime paper.

Walk-forward comparison of FracTime models versus baselines on a small
asset universe, plus analytical demonstrations of new v0.6 features.

Outputs:
  - paper_results/experiments/*.json   — raw experiment results
  - paper_results/reports/*.tex        — LaTeX tables
  - paper_results/reports/*.md         — Markdown summary
  - paper_results/analysis/            — analytical demonstrations (JSON)

Usage:
    python scripts/run_paper_experiments.py
    python scripts/run_paper_experiments.py --quick   # fast subset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fractime.research.config import AssetConfig, ExperimentConfig, MODELS
from fractime.research.data_collector import DataCollector
from fractime.research.experiment import ExperimentRunner
from fractime.research.metrics import MetricsCalculator
from fractime.research.report_generator import ReportGenerator
from fractime.research.statistical_tests import pairwise_dm_tests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

PAPER_ASSETS = [
    AssetConfig("^GSPC", "S&P 500", "index", "broad",
                start_date="2015-01-01", end_date="2024-12-31"),
    AssetConfig("AAPL", "Apple Inc.", "equity", "tech",
                start_date="2015-01-01", end_date="2024-12-31"),
    AssetConfig("BTC-USD", "Bitcoin", "crypto", "major",
                start_date="2017-01-01", end_date="2024-12-31"),
    AssetConfig("ETH-USD", "Ethereum", "crypto", "major",
                start_date="2017-06-01", end_date="2024-12-31"),
]

PAPER_MODELS = [
    "fractime_rs",
    "fractime_dfa",
    "fractime_ensemble",
    "arima",
    "garch",
    "ets",
]

PAPER_HORIZONS = {"1d": 1, "5d": 5, "21d": 21}


def build_config(quick: bool = False) -> ExperimentConfig:
    """Build experiment configuration for the paper."""
    if quick:
        return ExperimentConfig(
            name="paper_quick",
            description="Quick paper experiment",
            assets=PAPER_ASSETS[:2],
            horizons={"1d": 1, "5d": 5},
            models=["fractime_rs", "arima", "ets"],
            train_window=252,
            step_size=63,
            n_paths=200,
            output_dir=Path("paper_results"),
        )
    return ExperimentConfig(
        name="paper_experiments",
        description="FracTime paper: walk-forward model comparison",
        assets=PAPER_ASSETS,
        horizons=PAPER_HORIZONS,
        models=PAPER_MODELS,
        train_window=252,
        step_size=63,
        n_paths=500,
        output_dir=Path("paper_results"),
    )


# ── Walk-forward experiment ──────────────────────────────────────────────────

def run_walk_forward(config: ExperimentConfig) -> None:
    """Run the walk-forward comparison experiment and save reports."""
    print("\n" + "=" * 60)
    print("  Phase 1: Walk-forward model comparison")
    print("=" * 60)

    # Fetch data
    collector = DataCollector(config)
    data = collector.fetch_all(verbose=True)

    if not data:
        print("ERROR: No data fetched. Check your network connection.")
        return

    # Run experiments
    runner = ExperimentRunner(config, verbose=True)
    results = runner.run_all(data)

    # Save raw results
    results_path = results.save()
    print(f"\nResults saved → {results_path}")

    # Generate reports
    report_gen = ReportGenerator(results, output_dir=config.output_dir / "reports")
    report_paths = report_gen.generate_all()
    report_gen.export_data_json()

    print("\nGenerated reports:")
    for name, path in report_paths.items():
        print(f"  {name}: {path}")

    # Print summary to console
    _print_summary(results)


def _print_summary(results) -> None:
    """Print a concise results summary."""
    metrics = results.metrics
    if not metrics:
        print("\nNo metrics computed.")
        return

    print("\n" + "-" * 60)
    print("  Results Summary")
    print("-" * 60)

    for model, model_metrics in metrics.items():
        if "overall" in model_metrics:
            m = model_metrics["overall"]
            rmse = f"{m.rmse:.4f}" if not np.isnan(m.rmse) else "--"
            mae = f"{m.mae:.4f}" if not np.isnan(m.mae) else "--"
            da = f"{m.direction_accuracy * 100:.1f}%" if not np.isnan(m.direction_accuracy) else "--"
            sr = f"{m.sharpe_ratio:.2f}" if not np.isnan(m.sharpe_ratio) else "--"
            print(f"  {model:20s}  RMSE={rmse:>8s}  MAE={mae:>8s}  Dir={da:>6s}  Sharpe={sr:>6s}")


# ── Analytical demonstrations (new v0.6 features) ───────────────────────────

def run_analytical_demos(config: ExperimentConfig) -> None:
    """Run analytical demonstrations for paper sections 7–10."""
    print("\n" + "=" * 60)
    print("  Phase 2: Analytical demonstrations (v0.6 features)")
    print("=" * 60)

    from fractime.advanced import (
        compute_wavelet_hurst,
        compute_rolling_wavelet_hurst,
        compute_qpl,
        compute_fractal_coherence,
        compute_ftd,
        compute_mf_adcca,
        compute_dtw_beta,
    )

    analysis_dir = config.output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Fetch S&P 500 for demonstrations
    collector = DataCollector(config)
    sp500_asset = AssetConfig(
        "^GSPC", "S&P 500", "index", "broad",
        start_date="2015-01-01", end_date="2024-12-31",
    )
    sp500 = collector.fetch_asset(sp500_asset)
    if sp500 is None:
        print("  WARNING: Could not fetch S&P 500 data. Skipping demos.")
        return

    prices = sp500.prices
    volume = sp500.volume if sp500.volume is not None else np.ones_like(prices)

    results = {}

    # 1. Wavelet Hurst
    print("  [1/6] Wavelet Hurst (Db4)...")
    wh = compute_wavelet_hurst(prices)
    rolling_wh = compute_rolling_wavelet_hurst(prices, window=252, step=21)
    results["wavelet_hurst"] = {
        "point_estimate": float(wh),
        "rolling_mean": float(np.mean(rolling_wh)),
        "rolling_std": float(np.std(rolling_wh)),
        "rolling_min": float(np.min(rolling_wh)),
        "rolling_max": float(np.max(rolling_wh)),
        "n_windows": len(rolling_wh),
    }
    print(f"    H_wavelet = {wh:.4f}  (rolling mean={np.mean(rolling_wh):.4f})")

    # 2. QPL
    print("  [2/6] Quantum Price Levels...")
    qpl_levels = compute_qpl(prices, n_levels=5)
    results["qpl"] = {
        "levels": qpl_levels.tolist(),
        "n_levels": len(qpl_levels),
        "price_range": [float(prices.min()), float(prices.max())],
    }
    print(f"    QPL levels: {[f'{l:.1f}' for l in qpl_levels]}")

    # 3. Fractal coherence
    print("  [3/6] Fractal coherence (price + volume)...")
    coh = compute_fractal_coherence(prices, volume, window=252)
    results["fractal_coherence"] = {
        "coherence": float(coh),
    }
    print(f"    Coherence = {coh:.4f}")

    # 4. FTD
    print("  [4/6] Fractal Trend Dimension...")
    d_plus, d_minus = compute_ftd(prices, window=252)
    results["ftd"] = {
        "d_plus": float(d_plus),
        "d_minus": float(d_minus),
        "asymmetry": float(abs(d_plus - d_minus)),
    }
    print(f"    D+ = {d_plus:.4f}, D- = {d_minus:.4f}")

    # 5. MF-ADCCA (S&P 500 vs AAPL if available)
    print("  [5/6] MF-ADCCA cross-correlation...")
    aapl_asset = AssetConfig(
        "AAPL", "Apple Inc.", "equity", "tech",
        start_date="2015-01-01", end_date="2024-12-31",
    )
    aapl = collector.fetch_asset(aapl_asset)
    if aapl is not None:
        min_len = min(len(prices), len(aapl.prices))
        rho_q, h_xy = compute_mf_adcca(
            prices[-min_len:], aapl.prices[-min_len:],
            q_values=np.array([-2.0, 0.0, 2.0, 4.0]),
        )
        results["mf_adcca"] = {
            "rho_q": rho_q.tolist(),
            "h_xy": h_xy.tolist(),
            "q_values": [-2.0, 0.0, 2.0, 4.0],
            "assets": ["^GSPC", "AAPL"],
        }
        print(f"    rho_q(2) = {rho_q[2]:.4f}, h_xy(2) = {h_xy[2]:.4f}")
    else:
        print("    Skipped (AAPL data unavailable)")

    # 6. DTW Beta
    print("  [6/6] DTW-corrected beta...")
    if aapl is not None:
        min_len = min(len(sp500.returns), len(aapl.returns))
        stock_ret = aapl.returns[-min_len:]
        market_ret = sp500.returns[-min_len:]
        # Remove first element (=0 from cumsum construction)
        stock_ret = stock_ret[1:]
        market_ret = market_ret[1:]

        dtw_beta = compute_dtw_beta(stock_ret, market_ret)

        # OLS beta for comparison
        cov = np.cov(stock_ret, market_ret)
        ols_beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0

        results["dtw_beta"] = {
            "dtw_beta": float(dtw_beta),
            "ols_beta": float(ols_beta),
            "difference": float(abs(dtw_beta - ols_beta)),
            "assets": ["AAPL", "^GSPC"],
        }
        print(f"    OLS beta = {ols_beta:.4f}, DTW beta = {dtw_beta:.4f}")
    else:
        print("    Skipped (AAPL data unavailable)")

    # Save
    out_path = analysis_dir / "analytical_demos.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Analysis saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FracTime paper experiments")
    parser.add_argument("--quick", action="store_true", help="Quick subset run")
    args = parser.parse_args()

    config = build_config(quick=args.quick)
    t0 = time.time()

    run_walk_forward(config)
    run_analytical_demos(config)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  All done in {elapsed:.0f}s  ({elapsed / 60:.1f} min)")
    print(f"  Output directory: {config.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
