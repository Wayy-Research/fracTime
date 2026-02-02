"""
Three.js interactive visualization site generator.

This module generates a self-contained HTML site with interactive 3D
visualizations of FracTime research results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...experiment import ExperimentResult, ForecastRecord
from ...metrics import ForecastMetrics


class ThreeJSSiteGenerator:
    """Generate interactive Three.js visualization site.

    Creates a self-contained HTML site with embedded JavaScript for
    interactive 3D visualizations of research results.

    Example:
        >>> generator = ThreeJSSiteGenerator(results)
        >>> generator.generate()
    """

    def __init__(
        self,
        results: ExperimentResult,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the site generator.

        Args:
            results: Experiment results to visualize
            output_dir: Output directory for the site
        """
        self.results = results
        self.output_dir = output_dir or results.config.output_dir / "site"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_metrics_data(self) -> Dict[str, Any]:
        """Prepare metrics data for JSON export."""
        data = {}

        for model, model_metrics in self.results.metrics.items():
            data[model] = {}
            for horizon, metrics in model_metrics.items():
                data[model][horizon] = {
                    "rmse": float(metrics.rmse) if not np.isnan(metrics.rmse) else None,
                    "mae": float(metrics.mae) if not np.isnan(metrics.mae) else None,
                    "mape": float(metrics.mape) if not np.isnan(metrics.mape) else None,
                    "crps": float(metrics.crps) if not np.isnan(metrics.crps) else None,
                    "direction_accuracy": float(metrics.direction_accuracy) if not np.isnan(metrics.direction_accuracy) else None,
                    "coverage_95": float(metrics.coverage_95) if not np.isnan(metrics.coverage_95) else None,
                    "sharpe_ratio": float(metrics.sharpe_ratio) if not np.isnan(metrics.sharpe_ratio) else None,
                    "n_forecasts": metrics.n_forecasts,
                }

        return data

    def _prepare_forecast_data(self, max_forecasts: int = 500) -> List[Dict[str, Any]]:
        """Prepare forecast data for visualization."""
        forecasts = []

        sampled = self.results.forecasts[:max_forecasts]

        for f in sampled:
            forecasts.append({
                "date": str(f.forecast_date)[:10],
                "horizon": f.horizon,
                "model": f.model_name,
                "ticker": f.ticker,
                "current": float(f.current_price),
                "forecast": float(f.forecast_mean),
                "actual": float(f.actual) if f.actual else None,
                "lower_95": float(f.lower_95),
                "upper_95": float(f.upper_95),
                "hurst": f.metadata.get("hurst") if f.metadata else None,
            })

        return forecasts

    def _prepare_path_samples(self, n_samples: int = 10) -> List[Dict[str, Any]]:
        """Prepare sample path data for 3D visualization."""
        path_samples = []

        forecasts_with_paths = [
            f for f in self.results.forecasts
            if f.paths is not None and len(f.paths) > 0
        ][:n_samples]

        for f in forecasts_with_paths:
            paths = f.paths[:50].tolist() if len(f.paths) > 50 else f.paths.tolist()
            path_samples.append({
                "ticker": f.ticker,
                "model": f.model_name,
                "horizon": f.horizon,
                "current": float(f.current_price),
                "actual": float(f.actual) if f.actual else None,
                "paths": paths,
            })

        return path_samples

    def _generate_data_js(self) -> str:
        """Generate embedded data JavaScript."""
        metrics = self._prepare_metrics_data()
        forecasts = self._prepare_forecast_data()
        path_samples = self._prepare_path_samples()

        config_data = {
            "name": self.results.config.name,
            "models": self.results.config.models,
            "horizons": self.results.config.horizons,
            "n_forecasts": len(self.results.forecasts),
        }

        return f"""
const FRACTIME_DATA = {{
    config: {json.dumps(config_data)},
    metrics: {json.dumps(metrics)},
    forecasts: {json.dumps(forecasts)},
    pathSamples: {json.dumps(path_samples)}
}};
"""

    def generate(self) -> Path:
        """Generate the complete visualization site.

        Returns:
            Path to the generated index.html
        """
        data_js = self._generate_data_js()
        html_content = self._generate_html(data_js)

        index_path = self.output_dir / "index.html"
        with open(index_path, "w") as f:
            f.write(html_content)

        return index_path

    def _generate_html(self, data_js: str) -> str:
        """Generate the complete HTML page."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FracTime Research Results</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --primary: #1a73e8;
            --secondary: #34a853;
            --accent: #ea4335;
            --bg: #f8f9fa;
            --card: #ffffff;
            --text: #202124;
            --text-light: #5f6368;
            --border: #e0e0e0;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}

        .header {{
            background: var(--primary);
            color: white;
            padding: 2rem;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .header p {{
            opacity: 0.9;
        }}

        .nav {{
            background: var(--card);
            border-bottom: 1px solid var(--border);
            padding: 1rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .nav ul {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            list-style: none;
        }}

        .nav a {{
            color: var(--text);
            text-decoration: none;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background 0.2s;
        }}

        .nav a:hover {{
            background: var(--bg);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        section {{
            margin-bottom: 3rem;
        }}

        .section-title {{
            font-size: 1.75rem;
            margin-bottom: 1.5rem;
            color: var(--text);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
        }}

        .card {{
            background: var(--card);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }}

        .card h3 {{
            margin-bottom: 1rem;
            color: var(--text);
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}

        .metric-card {{
            text-align: center;
            padding: 1.5rem;
        }}

        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
        }}

        .metric-label {{
            color: var(--text-light);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}

        #pathVisualization {{
            width: 100%;
            height: 500px;
            background: #1a1a2e;
            border-radius: 8px;
        }}

        .chart-container {{
            position: relative;
            height: 400px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--bg);
            font-weight: 600;
        }}

        tr:hover {{
            background: var(--bg);
        }}

        .highlight {{
            background: #e8f5e9 !important;
            font-weight: 600;
        }}

        .controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}

        select, button {{
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
        }}

        button {{
            background: var(--primary);
            color: white;
            border: none;
        }}

        button:hover {{
            opacity: 0.9;
        }}

        .legend {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-light);
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }}
    </style>
</head>
<body>
    <header class="header">
        <h1>FracTime Research Results</h1>
        <p>Interactive visualization of fractal-based time series forecasting</p>
    </header>

    <nav class="nav">
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#comparison">Model Comparison</a></li>
            <li><a href="#paths">Path Visualization</a></li>
            <li><a href="#calibration">Calibration</a></li>
            <li><a href="#results">Detailed Results</a></li>
        </ul>
    </nav>

    <main class="container">
        <section id="overview">
            <h2 class="section-title">Overview</h2>
            <div class="grid">
                <div class="card metric-card">
                    <div class="metric-value" id="totalForecasts">-</div>
                    <div class="metric-label">Total Forecasts</div>
                </div>
                <div class="card metric-card">
                    <div class="metric-value" id="numModels">-</div>
                    <div class="metric-label">Models Compared</div>
                </div>
                <div class="card metric-card">
                    <div class="metric-value" id="bestModel">-</div>
                    <div class="metric-label">Best Model (RMSE)</div>
                </div>
                <div class="card metric-card">
                    <div class="metric-value" id="avgDirection">-</div>
                    <div class="metric-label">Avg. Direction Accuracy</div>
                </div>
            </div>
        </section>

        <section id="comparison">
            <h2 class="section-title">Model Comparison</h2>
            <div class="card">
                <div class="controls">
                    <select id="metricSelect">
                        <option value="rmse">RMSE</option>
                        <option value="mae">MAE</option>
                        <option value="direction_accuracy">Direction Accuracy</option>
                        <option value="coverage_95">Coverage (95%)</option>
                        <option value="sharpe_ratio">Sharpe Ratio</option>
                    </select>
                    <select id="horizonSelect">
                        <option value="all">All Horizons</option>
                    </select>
                </div>
                <div class="chart-container">
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>
        </section>

        <section id="paths">
            <h2 class="section-title">Monte Carlo Path Visualization</h2>
            <div class="card">
                <div class="controls">
                    <select id="pathSampleSelect"></select>
                    <button id="animatePaths">Animate</button>
                </div>
                <div id="pathVisualization"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4fc3f7;"></div>
                        <span>Simulated Paths</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff5722;"></div>
                        <span>Actual Outcome</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4caf50;"></div>
                        <span>Forecast Mean</span>
                    </div>
                </div>
            </div>
        </section>

        <section id="calibration">
            <h2 class="section-title">Probabilistic Calibration</h2>
            <div class="grid">
                <div class="card">
                    <h3>Calibration Diagram</h3>
                    <div class="chart-container">
                        <canvas id="calibrationChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Direction Accuracy by Model</h3>
                    <div class="chart-container">
                        <canvas id="directionChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <section id="results">
            <h2 class="section-title">Detailed Results</h2>
            <div class="card">
                <h3>Model Performance Table</h3>
                <div style="overflow-x: auto;">
                    <table id="resultsTable">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Horizon</th>
                                <th>RMSE</th>
                                <th>MAE</th>
                                <th>Direction</th>
                                <th>Coverage-95%</th>
                                <th>Sharpe</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>
    </main>

    <footer>
        <p>Generated by FracTime Research Framework</p>
    </footer>

    <script>
{data_js}

// Color palette for models
const MODEL_COLORS = {{
    'fractime_rs': '#1f77b4',
    'fractime_dfa': '#2ca02c',
    'fractime_ensemble': '#17becf',
    'arima': '#ff7f0e',
    'garch': '#d62728',
    'prophet': '#9467bd',
    'ets': '#8c564b',
    'lstm': '#e377c2'
}};

// Initialize overview stats
function initOverview() {{
    document.getElementById('totalForecasts').textContent =
        FRACTIME_DATA.config.n_forecasts.toLocaleString();
    document.getElementById('numModels').textContent =
        FRACTIME_DATA.config.models.length;

    // Find best model by RMSE
    let bestModel = null;
    let bestRmse = Infinity;
    for (const [model, horizons] of Object.entries(FRACTIME_DATA.metrics)) {{
        if (horizons.overall && horizons.overall.rmse < bestRmse) {{
            bestRmse = horizons.overall.rmse;
            bestModel = model;
        }}
    }}
    document.getElementById('bestModel').textContent = bestModel || '-';

    // Average direction accuracy
    let totalDir = 0, countDir = 0;
    for (const horizons of Object.values(FRACTIME_DATA.metrics)) {{
        if (horizons.overall && horizons.overall.direction_accuracy) {{
            totalDir += horizons.overall.direction_accuracy;
            countDir++;
        }}
    }}
    const avgDir = countDir > 0 ? (totalDir / countDir * 100).toFixed(1) + '%' : '-';
    document.getElementById('avgDirection').textContent = avgDir;
}}

// Model comparison chart
let comparisonChart;
function initComparisonChart() {{
    const ctx = document.getElementById('comparisonChart').getContext('2d');

    // Populate horizon select
    const horizonSelect = document.getElementById('horizonSelect');
    for (const h of Object.keys(FRACTIME_DATA.config.horizons)) {{
        const opt = document.createElement('option');
        opt.value = h;
        opt.textContent = h.toUpperCase();
        horizonSelect.appendChild(opt);
    }}

    updateComparisonChart();

    document.getElementById('metricSelect').addEventListener('change', updateComparisonChart);
    document.getElementById('horizonSelect').addEventListener('change', updateComparisonChart);
}}

function updateComparisonChart() {{
    const metric = document.getElementById('metricSelect').value;
    const horizon = document.getElementById('horizonSelect').value;

    const models = FRACTIME_DATA.config.models;
    const data = [];
    const colors = [];

    for (const model of models) {{
        const modelData = FRACTIME_DATA.metrics[model];
        if (!modelData) continue;

        let value = null;
        if (horizon === 'all' && modelData.overall) {{
            value = modelData.overall[metric];
        }} else if (modelData[horizon]) {{
            value = modelData[horizon][metric];
        }}

        data.push(value || 0);
        colors.push(MODEL_COLORS[model] || '#999');
    }}

    if (comparisonChart) {{
        comparisonChart.destroy();
    }}

    const ctx = document.getElementById('comparisonChart').getContext('2d');
    comparisonChart = new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: models,
            datasets: [{{
                label: metric.toUpperCase(),
                data: data,
                backgroundColor: colors,
                borderWidth: 0
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }}
            }},
            scales: {{
                y: {{ beginAtZero: true }}
            }}
        }}
    }});
}}

// Three.js path visualization
let scene, camera, renderer, pathLines = [];

function initPathVisualization() {{
    const container = document.getElementById('pathVisualization');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 100);

    renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // Populate sample select
    const select = document.getElementById('pathSampleSelect');
    FRACTIME_DATA.pathSamples.forEach((sample, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `${{sample.ticker}} - ${{sample.model}} (h=${{sample.horizon}})`;
        select.appendChild(opt);
    }});

    if (FRACTIME_DATA.pathSamples.length > 0) {{
        drawPaths(0);
    }}

    select.addEventListener('change', (e) => drawPaths(parseInt(e.target.value)));
    document.getElementById('animatePaths').addEventListener('click', animatePaths);

    animate();
}}

function drawPaths(sampleIndex) {{
    // Clear existing paths
    pathLines.forEach(line => scene.remove(line));
    pathLines = [];

    if (sampleIndex >= FRACTIME_DATA.pathSamples.length) return;

    const sample = FRACTIME_DATA.pathSamples[sampleIndex];
    const paths = sample.paths;
    const current = sample.current;
    const actual = sample.actual;

    // Normalize prices for visualization
    const allValues = [current, ...paths, actual].filter(v => v !== null);
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const range = maxVal - minVal || 1;

    const normalize = (v) => ((v - minVal) / range - 0.5) * 60;

    // Draw paths
    const pathMaterial = new THREE.LineBasicMaterial({{
        color: 0x4fc3f7,
        transparent: true,
        opacity: 0.3
    }});

    paths.forEach(endPrice => {{
        const points = [
            new THREE.Vector3(-30, normalize(current), 0),
            new THREE.Vector3(30, normalize(endPrice), 0)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, pathMaterial);
        scene.add(line);
        pathLines.push(line);
    }});

    // Draw actual
    if (actual !== null) {{
        const actualMaterial = new THREE.LineBasicMaterial({{ color: 0xff5722, linewidth: 3 }});
        const actualPoints = [
            new THREE.Vector3(-30, normalize(current), 0),
            new THREE.Vector3(30, normalize(actual), 0)
        ];
        const actualGeometry = new THREE.BufferGeometry().setFromPoints(actualPoints);
        const actualLine = new THREE.Line(actualGeometry, actualMaterial);
        scene.add(actualLine);
        pathLines.push(actualLine);
    }}

    // Draw mean forecast
    const meanEnd = paths.reduce((a, b) => a + b, 0) / paths.length;
    const meanMaterial = new THREE.LineBasicMaterial({{ color: 0x4caf50, linewidth: 2 }});
    const meanPoints = [
        new THREE.Vector3(-30, normalize(current), 0),
        new THREE.Vector3(30, normalize(meanEnd), 0)
    ];
    const meanGeometry = new THREE.BufferGeometry().setFromPoints(meanPoints);
    const meanLine = new THREE.Line(meanGeometry, meanMaterial);
    scene.add(meanLine);
    pathLines.push(meanLine);
}}

function animatePaths() {{
    // Simple rotation animation
    let angle = 0;
    const animateStep = () => {{
        angle += 0.02;
        camera.position.x = 100 * Math.sin(angle);
        camera.position.z = 100 * Math.cos(angle);
        camera.lookAt(0, 0, 0);

        if (angle < Math.PI * 2) {{
            requestAnimationFrame(animateStep);
        }} else {{
            camera.position.set(0, 0, 100);
            camera.lookAt(0, 0, 0);
        }}
    }};
    animateStep();
}}

function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}}

// Calibration chart
function initCalibrationChart() {{
    const ctx = document.getElementById('calibrationChart').getContext('2d');

    const expected = [0.5, 0.8, 0.95];
    const datasets = [];

    for (const [model, horizons] of Object.entries(FRACTIME_DATA.metrics)) {{
        if (!horizons.overall) continue;

        const observed = [
            horizons.overall.coverage_50 || 0.5,
            horizons.overall.coverage_80 || 0.8,
            horizons.overall.coverage_95 || 0.95
        ];

        datasets.push({{
            label: model,
            data: expected.map((e, i) => ({{ x: e, y: observed[i] }})),
            borderColor: MODEL_COLORS[model] || '#999',
            backgroundColor: MODEL_COLORS[model] || '#999',
            pointRadius: 6,
            showLine: true,
            tension: 0
        }});
    }}

    // Perfect calibration line
    datasets.unshift({{
        label: 'Perfect',
        data: [{{ x: 0.4, y: 0.4 }}, {{ x: 1, y: 1 }}],
        borderColor: '#999',
        borderDash: [5, 5],
        pointRadius: 0,
        showLine: true
    }});

    new Chart(ctx, {{
        type: 'scatter',
        data: {{ datasets }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{
                    title: {{ display: true, text: 'Expected Coverage' }},
                    min: 0.4, max: 1
                }},
                y: {{
                    title: {{ display: true, text: 'Observed Coverage' }},
                    min: 0.4, max: 1
                }}
            }},
            plugins: {{
                legend: {{ position: 'bottom' }}
            }}
        }}
    }});
}}

// Direction accuracy chart
function initDirectionChart() {{
    const ctx = document.getElementById('directionChart').getContext('2d');

    const models = [];
    const directions = [];
    const colors = [];

    for (const [model, horizons] of Object.entries(FRACTIME_DATA.metrics)) {{
        if (horizons.overall && horizons.overall.direction_accuracy) {{
            models.push(model);
            directions.push(horizons.overall.direction_accuracy * 100);
            colors.push(MODEL_COLORS[model] || '#999');
        }}
    }}

    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: models,
            datasets: [{{
                label: 'Direction Accuracy (%)',
                data: directions,
                backgroundColor: colors
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {{
                legend: {{ display: false }}
            }},
            scales: {{
                x: {{
                    min: 0, max: 100,
                    title: {{ display: true, text: 'Accuracy (%)' }}
                }}
            }}
        }}
    }});
}}

// Results table
function initResultsTable() {{
    const tbody = document.querySelector('#resultsTable tbody');

    // Find best values for highlighting
    const bestValues = {{}};
    for (const [model, horizons] of Object.entries(FRACTIME_DATA.metrics)) {{
        for (const [horizon, metrics] of Object.entries(horizons)) {{
            const key = horizon;
            if (!bestValues[key]) bestValues[key] = {{}};

            for (const [metric, value] of Object.entries(metrics)) {{
                if (value === null) continue;
                const lower = ['rmse', 'mae'].includes(metric);
                if (!bestValues[key][metric] ||
                    (lower && value < bestValues[key][metric]) ||
                    (!lower && value > bestValues[key][metric])) {{
                    bestValues[key][metric] = value;
                }}
            }}
        }}
    }}

    for (const [model, horizons] of Object.entries(FRACTIME_DATA.metrics)) {{
        for (const [horizon, metrics] of Object.entries(horizons)) {{
            const row = document.createElement('tr');

            const format = (v, digits=4) => v !== null ? v.toFixed(digits) : '-';
            const formatPct = (v) => v !== null ? (v * 100).toFixed(1) + '%' : '-';

            const cells = [
                model,
                horizon.toUpperCase(),
                format(metrics.rmse),
                format(metrics.mae),
                formatPct(metrics.direction_accuracy),
                formatPct(metrics.coverage_95),
                format(metrics.sharpe_ratio, 2)
            ];

            cells.forEach((cell, i) => {{
                const td = document.createElement('td');
                td.textContent = cell;
                row.appendChild(td);
            }});

            tbody.appendChild(row);
        }}
    }}
}}

// Initialize everything
document.addEventListener('DOMContentLoaded', () => {{
    initOverview();
    initComparisonChart();
    initPathVisualization();
    initCalibrationChart();
    initDirectionChart();
    initResultsTable();
}});
    </script>
</body>
</html>
"""
