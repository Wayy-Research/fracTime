"""
Report generation for FracTime research.

This module generates publication-quality LaTeX tables and Markdown
summaries from experiment results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ExperimentConfig
from .experiment import ExperimentResult
from .metrics import ForecastMetrics
from .statistical_tests import (
    DMTestResult,
    MCSResult,
    pairwise_dm_tests,
    compute_loss_matrix_from_forecasts,
    model_confidence_set,
)


class ReportGenerator:
    """Generate reports from experiment results.

    This class creates publication-quality LaTeX tables and Markdown
    summaries from FracTime research experiment results.

    Example:
        >>> generator = ReportGenerator(results, output_dir=Path("reports"))
        >>> generator.generate_all()
    """

    def __init__(
        self,
        results: ExperimentResult,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the report generator.

        Args:
            results: Experiment results to generate reports from
            output_dir: Directory for output files
        """
        self.results = results
        self.output_dir = output_dir or results.config.output_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> Dict[str, Path]:
        """Generate all reports.

        Returns:
            Dictionary mapping report name to file path
        """
        paths: Dict[str, Path] = {}

        paths["main_results_latex"] = self.generate_main_results_table_latex()
        paths["main_results_md"] = self.generate_main_results_table_markdown()
        paths["dm_tests_latex"] = self.generate_dm_tests_table_latex()
        paths["calibration_latex"] = self.generate_calibration_table_latex()
        paths["summary_md"] = self.generate_summary_markdown()

        return paths

    def _format_metric(
        self,
        value: float,
        precision: int = 4,
        percentage: bool = False,
        best: bool = False,
    ) -> str:
        """Format a metric value for display.

        Args:
            value: Metric value
            precision: Decimal precision
            percentage: Format as percentage
            best: Bold the value (best in column)

        Returns:
            Formatted string
        """
        if np.isnan(value):
            return "--"

        if percentage:
            formatted = f"{value * 100:.{precision - 2}f}\\%"
        else:
            formatted = f"{value:.{precision}f}"

        if best:
            formatted = f"\\textbf{{{formatted}}}"

        return formatted

    def _get_best_models(
        self,
        metrics: Dict[str, Dict[str, ForecastMetrics]],
        metric_name: str,
        lower_is_better: bool = True,
    ) -> Dict[str, str]:
        """Find the best model for each horizon based on a metric.

        Args:
            metrics: Nested dict of model -> horizon -> metrics
            metric_name: Name of metric to compare
            lower_is_better: Whether lower values are better

        Returns:
            Dictionary mapping horizon to best model name
        """
        best_models: Dict[str, str] = {}

        horizons = set()
        for model_metrics in metrics.values():
            horizons.update(model_metrics.keys())

        for horizon in horizons:
            best_val = float("inf") if lower_is_better else float("-inf")
            best_model = None

            for model_name, model_metrics in metrics.items():
                if horizon not in model_metrics:
                    continue

                val = getattr(model_metrics[horizon], metric_name, np.nan)
                if np.isnan(val):
                    continue

                if lower_is_better and val < best_val:
                    best_val = val
                    best_model = model_name
                elif not lower_is_better and val > best_val:
                    best_val = val
                    best_model = model_name

            if best_model:
                best_models[horizon] = best_model

        return best_models

    def generate_main_results_table_latex(self) -> Path:
        """Generate main results table in LaTeX format.

        Returns:
            Path to generated file
        """
        metrics = self.results.metrics
        if not metrics:
            return self._write_empty_table("main_results.tex")

        horizons = ["1d", "5d", "21d", "63d", "overall"]
        models = list(metrics.keys())

        best_rmse = self._get_best_models(metrics, "rmse", lower_is_better=True)
        best_mae = self._get_best_models(metrics, "mae", lower_is_better=True)
        best_dir = self._get_best_models(metrics, "direction_accuracy", lower_is_better=False)

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Forecast Accuracy by Model and Horizon}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{l" + "ccc" * len(horizons) + "}",
            "\\toprule",
        ]

        header = "Model"
        for h in horizons:
            header += f" & \\multicolumn{{3}}{{c}}{{{h.upper()}}}"
        header += " \\\\"
        lines.append(header)

        subheader = ""
        for _ in horizons:
            subheader += " & RMSE & MAE & Dir."
        subheader += " \\\\"
        lines.append("\\cmidrule(lr){2-4}" * len(horizons))
        lines.append(subheader)
        lines.append("\\midrule")

        for model in models:
            model_display = model.replace("_", "\\_")
            row = model_display

            for h in horizons:
                if h in metrics[model]:
                    m = metrics[model][h]

                    rmse = self._format_metric(
                        m.rmse, precision=4,
                        best=(best_rmse.get(h) == model)
                    )
                    mae = self._format_metric(
                        m.mae, precision=4,
                        best=(best_mae.get(h) == model)
                    )
                    dir_acc = self._format_metric(
                        m.direction_accuracy, precision=2, percentage=True,
                        best=(best_dir.get(h) == model)
                    )
                    row += f" & {rmse} & {mae} & {dir_acc}"
                else:
                    row += " & -- & -- & --"

            row += " \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item Note: Best values in each column are in bold. Dir. = Directional Accuracy.",
            "\\end{tablenotes}",
            "\\end{table}",
        ])

        path = self.output_dir / "main_results.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def generate_main_results_table_markdown(self) -> Path:
        """Generate main results table in Markdown format.

        Returns:
            Path to generated file
        """
        metrics = self.results.metrics
        if not metrics:
            return self._write_empty_table("main_results.md", latex=False)

        horizons = ["1d", "5d", "21d", "63d", "overall"]
        models = list(metrics.keys())

        lines = ["# Main Results\n"]
        lines.append("## Forecast Accuracy by Model and Horizon\n")

        header = "| Model |"
        separator = "| --- |"
        for h in horizons:
            header += f" RMSE ({h}) | MAE ({h}) | Dir ({h}) |"
            separator += " --- | --- | --- |"

        lines.append(header)
        lines.append(separator)

        for model in models:
            row = f"| {model} |"

            for h in horizons:
                if h in metrics[model]:
                    m = metrics[model][h]
                    rmse = f"{m.rmse:.4f}" if not np.isnan(m.rmse) else "--"
                    mae = f"{m.mae:.4f}" if not np.isnan(m.mae) else "--"
                    dir_acc = f"{m.direction_accuracy * 100:.1f}%" if not np.isnan(m.direction_accuracy) else "--"
                    row += f" {rmse} | {mae} | {dir_acc} |"
                else:
                    row += " -- | -- | -- |"

            lines.append(row)

        path = self.output_dir / "main_results.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def generate_dm_tests_table_latex(self) -> Path:
        """Generate Diebold-Mariano test results table in LaTeX.

        Returns:
            Path to generated file
        """
        metrics = self.results.metrics
        forecasts = self.results.forecasts

        if not forecasts or "fractime_rs" not in metrics:
            return self._write_empty_table("dm_tests.tex")

        baseline = "fractime_rs"
        models = [m for m in metrics.keys() if m != baseline]

        model_forecasts: Dict[str, List[float]] = {baseline: []}
        model_errors: Dict[str, np.ndarray] = {}

        for model in [baseline] + models:
            preds = []
            acts = []
            for f in forecasts:
                if f.model_name == model and f.actual is not None:
                    preds.append(f.forecast_mean)
                    acts.append(f.actual)

            if preds:
                model_errors[model] = np.array(preds) - np.array(acts)

        if len(model_errors) < 2:
            return self._write_empty_table("dm_tests.tex")

        dm_results = pairwise_dm_tests(
            model_errors,
            baseline_model=baseline,
            loss_function="squared",
            h=1,
        )

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Diebold-Mariano Tests (FracTime R/S as Baseline)}",
            "\\label{tab:dm_tests}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Model & DM Statistic & p-value & Conclusion \\\\",
            "\\midrule",
        ]

        for model, result in dm_results.items():
            model_display = model.replace("_", "\\_")
            stat = f"{result.statistic:.3f}" if not np.isnan(result.statistic) else "--"
            pval = f"{result.p_value:.4f}" if not np.isnan(result.p_value) else "--"

            if result.p_value < 0.01:
                sig = "***"
            elif result.p_value < 0.05:
                sig = "**"
            elif result.p_value < 0.10:
                sig = "*"
            else:
                sig = ""

            if result.mean_loss_diff < 0:
                conclusion = "FracTime better"
            elif result.mean_loss_diff > 0:
                conclusion = f"{model_display} better"
            else:
                conclusion = "No diff."

            lines.append(f"{model_display} & {stat} & {pval}{sig} & {conclusion} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item *p<0.10, **p<0.05, ***p<0.01",
            "\\item Negative statistic indicates FracTime R/S has lower loss.",
            "\\end{tablenotes}",
            "\\end{table}",
        ])

        path = self.output_dir / "dm_tests.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def generate_calibration_table_latex(self) -> Path:
        """Generate calibration table in LaTeX format.

        Returns:
            Path to generated file
        """
        metrics = self.results.metrics
        if not metrics:
            return self._write_empty_table("calibration.tex")

        models = list(metrics.keys())

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Probabilistic Calibration by Model}",
            "\\label{tab:calibration}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Model & Coverage-50\\% & Coverage-80\\% & Coverage-95\\% & CRPS & Cal. Error & Interval Width \\\\",
            "\\midrule",
        ]

        for model in models:
            if "overall" not in metrics[model]:
                continue

            m = metrics[model]["overall"]
            model_display = model.replace("_", "\\_")

            cov_50 = self._format_metric(m.coverage_50, precision=2, percentage=True)
            cov_80 = self._format_metric(m.coverage_80, precision=2, percentage=True)
            cov_95 = self._format_metric(m.coverage_95, precision=2, percentage=True)
            crps = self._format_metric(m.crps, precision=4)
            cal_err = self._format_metric(m.calibration_error, precision=4)
            width = self._format_metric(m.mean_interval_width, precision=4)

            lines.append(
                f"{model_display} & {cov_50} & {cov_80} & {cov_95} & {crps} & {cal_err} & {width} \\\\"
            )

        lines.extend([
            "\\midrule",
            "Expected & 50\\% & 80\\% & 95\\% & -- & 0 & -- \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item Coverage should match expected levels for well-calibrated forecasts.",
            "\\item CRPS = Continuous Ranked Probability Score (lower is better).",
            "\\end{tablenotes}",
            "\\end{table}",
        ])

        path = self.output_dir / "calibration.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def generate_summary_markdown(self) -> Path:
        """Generate comprehensive Markdown summary.

        Returns:
            Path to generated file
        """
        config = self.results.config
        metrics = self.results.metrics

        lines = [
            "# FracTime Research Study Summary\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Experiment Configuration\n",
            f"- **Name:** {config.name}",
            f"- **Description:** {config.description}",
            f"- **Assets:** {len(config.assets)}",
            f"- **Models:** {', '.join(config.models)}",
            f"- **Horizons:** {', '.join(config.horizons.keys())}",
            f"- **Training Window:** {config.train_window}",
            f"- **Window Type:** {config.window_type}",
            f"- **Monte Carlo Paths:** {config.n_paths}",
            "",
            "## Experiment Statistics\n",
            f"- **Total Forecasts:** {len(self.results.forecasts)}",
            f"- **Run Time:** {self.results.run_time:.1f} seconds" if self.results.run_time else "- **Run Time:** N/A",
            f"- **Errors:** {len(self.results.errors)}",
            "",
        ]

        if metrics:
            lines.append("## Key Findings\n")

            if "overall" in list(metrics.values())[0]:
                lines.append("### Overall Performance (All Horizons)\n")
                lines.append("| Model | RMSE | MAE | Direction | Sharpe | Coverage-95% |")
                lines.append("| --- | --- | --- | --- | --- | --- |")

                for model, model_metrics in metrics.items():
                    if "overall" in model_metrics:
                        m = model_metrics["overall"]
                        rmse = f"{m.rmse:.4f}" if not np.isnan(m.rmse) else "--"
                        mae = f"{m.mae:.4f}" if not np.isnan(m.mae) else "--"
                        dir_acc = f"{m.direction_accuracy * 100:.1f}%" if not np.isnan(m.direction_accuracy) else "--"
                        sharpe = f"{m.sharpe_ratio:.2f}" if not np.isnan(m.sharpe_ratio) else "--"
                        cov = f"{m.coverage_95 * 100:.1f}%" if not np.isnan(m.coverage_95) else "--"
                        lines.append(f"| {model} | {rmse} | {mae} | {dir_acc} | {sharpe} | {cov} |")

                lines.append("")

            lines.append("### Best Model by Metric\n")

            metric_names = [
                ("rmse", "RMSE (lower is better)", True),
                ("mae", "MAE (lower is better)", True),
                ("direction_accuracy", "Direction Accuracy (higher is better)", False),
                ("sharpe_ratio", "Sharpe Ratio (higher is better)", False),
            ]

            for metric_name, description, lower_is_better in metric_names:
                best_val = float("inf") if lower_is_better else float("-inf")
                best_model = None

                for model, model_metrics in metrics.items():
                    if "overall" in model_metrics:
                        val = getattr(model_metrics["overall"], metric_name, np.nan)
                        if not np.isnan(val):
                            if lower_is_better and val < best_val:
                                best_val = val
                                best_model = model
                            elif not lower_is_better and val > best_val:
                                best_val = val
                                best_model = model

                if best_model:
                    lines.append(f"- **{description}:** {best_model} ({best_val:.4f})")

            lines.append("")

        if self.results.errors:
            lines.append("## Errors\n")
            for error in self.results.errors[:10]:
                lines.append(f"- {error}")
            if len(self.results.errors) > 10:
                lines.append(f"- ... and {len(self.results.errors) - 10} more")
            lines.append("")

        lines.extend([
            "## Generated Files\n",
            "- `main_results.tex` - LaTeX table with main results",
            "- `main_results.md` - Markdown table with main results",
            "- `dm_tests.tex` - Diebold-Mariano test results",
            "- `calibration.tex` - Probabilistic calibration metrics",
            "",
            "---",
            "*Generated by FracTime Research Framework*",
        ])

        path = self.output_dir / "summary.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def _write_empty_table(self, filename: str, latex: bool = True) -> Path:
        """Write an empty table placeholder.

        Args:
            filename: Output filename
            latex: Whether to write LaTeX format

        Returns:
            Path to file
        """
        path = self.output_dir / filename

        if latex:
            content = "% No data available for this table\n"
        else:
            content = "# No data available\n"

        with open(path, "w") as f:
            f.write(content)

        return path

    def export_data_json(self) -> Path:
        """Export all metrics data to JSON for visualization.

        Returns:
            Path to JSON file
        """
        data = {
            "config": self.results.config.to_dict(),
            "metrics": {},
            "forecasts_summary": {},
        }

        for model, model_metrics in self.results.metrics.items():
            data["metrics"][model] = {
                horizon: m.to_dict() for horizon, m in model_metrics.items()
            }

        for model in self.results.config.models:
            model_forecasts = self.results.get_forecasts_by_model(model)
            data["forecasts_summary"][model] = {
                "n_forecasts": len(model_forecasts),
                "assets": list(set(f.ticker for f in model_forecasts)),
            }

        path = self.output_dir / "results_data.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return path
