#!/usr/bin/env python3
"""
FracTime Research Study Runner.

This script runs the complete research study pipeline:
1. Collect data for all test assets
2. Run experiments across all models and horizons
3. Compute metrics and statistical tests
4. Generate reports and figures
5. Build interactive visualization site

Usage:
    # Run quick test
    python scripts/run_research_study.py --mode quick

    # Run full study
    python scripts/run_research_study.py --mode full

    # Run with custom configuration
    python scripts/run_research_study.py --assets AAPL,MSFT,^GSPC --models fractime_rs,arima
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fractime.research import (
    ExperimentConfig,
    DataCollector,
    ExperimentRunner,
    ReportGenerator,
)
from fractime.research.config import AssetConfig, MODELS
from fractime.research.visualization import ResearchPlotter
from fractime.research.visualization.threejs import ThreeJSSiteGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment configuration from command line arguments."""
    if args.mode == "quick":
        return ExperimentConfig.quick_test()
    elif args.mode == "full":
        return ExperimentConfig.full_study()
    elif args.mode == "custom":
        assets = []
        for ticker in args.assets.split(","):
            ticker = ticker.strip()
            assets.append(
                AssetConfig(
                    ticker=ticker,
                    name=ticker,
                    category="custom",
                    subcategory="custom",
                )
            )

        models = [m.strip() for m in args.models.split(",")]
        for m in models:
            if m not in MODELS:
                raise ValueError(f"Unknown model: {m}. Available: {list(MODELS.keys())}")

        return ExperimentConfig(
            name=args.name or "custom_experiment",
            description="Custom experiment configuration",
            assets=assets,
            models=models,
            step_size=args.step_size,
            n_paths=args.n_paths,
            output_dir=Path(args.output_dir) if args.output_dir else Path("research_results"),
        )
    else:
        return ExperimentConfig.default()


def run_data_collection(config: ExperimentConfig, force_refresh: bool = False) -> dict:
    """Run data collection phase."""
    print("\n" + "=" * 60)
    print("PHASE 1: Data Collection")
    print("=" * 60)

    collector = DataCollector(config, force_refresh=force_refresh)
    data = collector.fetch_all(verbose=True)

    print(f"\nData collection complete. {len(data)} assets loaded.")
    print(collector.summary())

    return data


def run_experiments(config: ExperimentConfig, data: dict) -> "ExperimentResult":
    """Run experiment phase."""
    print("\n" + "=" * 60)
    print("PHASE 2: Running Experiments")
    print("=" * 60)

    runner = ExperimentRunner(config, verbose=True)
    results = runner.run_all(data)

    print(f"\nExperiments complete.")
    print(f"Total forecasts: {len(results.forecasts)}")
    print(f"Run time: {results.run_time:.1f} seconds")

    return results


def run_report_generation(results: "ExperimentResult") -> dict:
    """Run report generation phase."""
    print("\n" + "=" * 60)
    print("PHASE 3: Generating Reports")
    print("=" * 60)

    generator = ReportGenerator(results)
    paths = generator.generate_all()

    print("\nGenerated reports:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")

    json_path = generator.export_data_json()
    print(f"  - data_json: {json_path}")

    return paths


def run_visualization(results: "ExperimentResult") -> dict:
    """Run visualization generation phase."""
    print("\n" + "=" * 60)
    print("PHASE 4: Generating Visualizations")
    print("=" * 60)

    paths = {}

    try:
        plotter = ResearchPlotter(results)
        figure_paths = plotter.generate_all()
        paths.update(figure_paths)
        print(f"\nGenerated {len(figure_paths)} static figures.")
    except ImportError as e:
        print(f"\nSkipping static plots: {e}")

    try:
        site_generator = ThreeJSSiteGenerator(results)
        site_path = site_generator.generate()
        paths["interactive_site"] = site_path
        print(f"\nGenerated interactive site: {site_path}")
    except Exception as e:
        print(f"\nFailed to generate interactive site: {e}")

    return paths


def save_results(results: "ExperimentResult") -> Path:
    """Save experiment results."""
    path = results.save()
    print(f"\nSaved results to: {path}")
    return path


def main():
    """Main entry point for research study."""
    parser = argparse.ArgumentParser(
        description="Run FracTime research study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick test with 3 assets
    python scripts/run_research_study.py --mode quick

    # Run full study (all assets, all models)
    python scripts/run_research_study.py --mode full

    # Run custom configuration
    python scripts/run_research_study.py --mode custom \\
        --assets "AAPL,MSFT,^GSPC" \\
        --models "fractime_rs,arima,garch"
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "default", "full", "custom"],
        default="default",
        help="Experiment mode (default: default)",
    )

    parser.add_argument(
        "--assets",
        type=str,
        help="Comma-separated list of assets (for custom mode)",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="fractime_rs,arima,garch",
        help="Comma-separated list of models (for custom mode)",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name",
    )

    parser.add_argument(
        "--step-size",
        type=int,
        default=21,
        help="Step size between forecasts (default: 21)",
    )

    parser.add_argument(
        "--n-paths",
        type=int,
        default=1000,
        help="Number of Monte Carlo paths (default: 1000)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: research_results)",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh data (ignore cache)",
    )

    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("FracTime Research Study")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        config = create_config_from_args(args)
        print(f"\nConfiguration: {config.name}")
        print(f"  Assets: {len(config.assets)}")
        print(f"  Models: {', '.join(config.models)}")
        print(f"  Horizons: {', '.join(config.horizons.keys())}")
        print(f"  Output: {config.output_dir}")

        data = run_data_collection(config, force_refresh=args.force_refresh)

        if not data:
            print("\nNo data collected. Exiting.")
            return 1

        results = run_experiments(config, data)

        if not results.forecasts:
            print("\nNo forecasts generated. Exiting.")
            return 1

        run_report_generation(results)

        if not args.skip_viz:
            run_visualization(results)

        results_path = save_results(results)

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Study Complete")
        print("=" * 60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Results saved to: {results_path}")
        print(f"Output directory: {config.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n\nStudy interrupted by user.")
        return 130

    except Exception as e:
        logger.exception("Study failed with error")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
