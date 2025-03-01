"""
FracTime Compute API Server

This server provides high-performance computational endpoints for the FracTime application.
It uses Flask for the API endpoints and handles heavy workloads separately from the UI.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Import FracTime modules
from fractime.core import FractalAnalyzer, FractalSimulator
from fractime.quantum import QuantumPriceLevels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fractime-compute")

# Initialize Flask app
app = Flask(__name__)

# Create a thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Cache for simulation results to avoid redundant computation
simulation_cache = {}


@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, str]:
    """Health check endpoint for Kubernetes probes."""
    return {"status": "healthy"}


@app.route("/api/analyze", methods=["POST"])
def analyze_data() -> Dict[str, Any]:
    """
    Analyze time series data using fractal methods.
    
    Expected JSON payload:
    {
        "data": [...],  # Time series data as a list of floats
        "timestamps": [...],  # Optional timestamps as ISO format strings
        "params": {  # Optional parameters
            "window_size": 20,
            "min_window": 10,
            "max_window": 100
        }
    }
    """
    try:
        payload = request.json
        if not payload or "data" not in payload:
            return jsonify({"error": "Missing data in request"}), 400
            
        data = np.array(payload["data"])
        timestamps = payload.get("timestamps")
        params = payload.get("params", {})
        
        # Create analyzer and run analysis
        analyzer = FractalAnalyzer(
            window_size=params.get("window_size", 20),
            min_window=params.get("min_window", 10),
            max_window=params.get("max_window", 100),
        )
        
        # Prepare the data
        if timestamps:
            df = pd.DataFrame({"price": data, "timestamp": pd.to_datetime(timestamps)})
            df.set_index("timestamp", inplace=True)
        else:
            df = pd.DataFrame({"price": data})
        
        # Run analysis
        results = analyzer.analyze(df)
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                results[key] = value.to_dict(orient="records")
        
        return jsonify({"results": results, "status": "success"})
    
    except Exception as e:
        logger.exception("Error in analyze_data endpoint")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate_paths() -> Dict[str, Any]:
    """
    Simulate future price paths using fractal methods.
    
    Expected JSON payload:
    {
        "data": [...],  # Historical time series data
        "params": {  # Optional parameters
            "n_paths": 1000,
            "horizon": 30,
            "use_gpu": true
        }
    }
    """
    try:
        payload = request.json
        if not payload or "data" not in payload:
            return jsonify({"error": "Missing data in request"}), 400
            
        data = np.array(payload["data"])
        params = payload.get("params", {})
        
        # Create cache key
        cache_key = f"{hash(data.tobytes())}_{params.get('n_paths')}_{params.get('horizon')}"
        
        # Check cache
        if cache_key in simulation_cache:
            logger.info(f"Cache hit for {cache_key}")
            return jsonify({"results": simulation_cache[cache_key], "status": "success"})
        
        # Create simulator
        simulator = FractalSimulator()
        
        # Run simulation
        use_gpu = params.get("use_gpu", True)
        n_paths = params.get("n_paths", 1000)
        horizon = params.get("horizon", 30)
        
        if use_gpu:
            try:
                paths = simulator.simulate_paths_gpu(data, n_paths=n_paths, horizon=horizon)
            except Exception as e:
                logger.warning(f"GPU simulation failed, falling back to CPU: {e}")
                paths = simulator.simulate_paths_fast(data, n_paths=n_paths, horizon=horizon)
        else:
            paths = simulator.simulate_paths_fast(data, n_paths=n_paths, horizon=horizon)
        
        # Process results
        results = {
            "paths": paths.tolist(),
            "summary": {
                "mean": np.mean(paths, axis=0).tolist(),
                "median": np.median(paths, axis=0).tolist(),
                "min": np.min(paths, axis=0).tolist(),
                "max": np.max(paths, axis=0).tolist(),
                "std": np.std(paths, axis=0).tolist(),
                "percentile_5": np.percentile(paths, 5, axis=0).tolist(),
                "percentile_95": np.percentile(paths, 95, axis=0).tolist(),
            }
        }
        
        # Cache results
        simulation_cache[cache_key] = results
        
        return jsonify({"results": results, "status": "success"})
    
    except Exception as e:
        logger.exception("Error in simulate_paths endpoint")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/quantum_levels", methods=["POST"])
def quantum_price_levels() -> Dict[str, Any]:
    """
    Calculate quantum price levels for the given time series.
    
    Expected JSON payload:
    {
        "data": [...],  # Time series data
        "params": {  # Optional parameters
            "num_levels": 7,
            "period": 30
        }
    }
    """
    try:
        payload = request.json
        if not payload or "data" not in payload:
            return jsonify({"error": "Missing data in request"}), 400
            
        data = np.array(payload["data"])
        params = payload.get("params", {})
        
        # Create quantum price level calculator
        qpl = QuantumPriceLevels(
            num_levels=params.get("num_levels", 7),
            period=params.get("period", 30)
        )
        
        # Calculate levels
        levels = qpl.calculate_levels(data)
        
        return jsonify({
            "levels": levels.tolist() if isinstance(levels, np.ndarray) else levels,
            "status": "success"
        })
    
    except Exception as e:
        logger.exception("Error in quantum_price_levels endpoint")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/backtest", methods=["POST"])
def run_backtest() -> Dict[str, Any]:
    """
    Run a backtest on the fractal strategy.
    
    Expected JSON payload:
    {
        "data": {...},  # Dictionary of symbol -> price data
        "params": {  # Optional parameters
            "window_size": 20,
            "threshold": 0.7
        }
    }
    """
    try:
        payload = request.json
        if not payload or "data" not in payload:
            return jsonify({"error": "Missing data in request"}), 400
            
        data_dict = payload["data"]
        params = payload.get("params", {})
        
        # Create analyzer
        analyzer = FractalAnalyzer(
            window_size=params.get("window_size", 20),
            threshold=params.get("threshold", 0.7)
        )
        
        # Convert data to DataFrames
        dataframes = {}
        for symbol, prices in data_dict.items():
            dataframes[symbol] = pd.DataFrame({"price": prices})
        
        # Run backtest
        results = analyzer.run_backtest(dataframes)
        
        # Process results for JSON
        processed_results = {}
        for symbol, result in results.items():
            if isinstance(result, pd.DataFrame):
                processed_results[symbol] = result.to_dict(orient="records")
            elif isinstance(result, dict):
                processed_results[symbol] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in result.items()
                }
            else:
                processed_results[symbol] = result
        
        return jsonify({"results": processed_results, "status": "success"})
    
    except Exception as e:
        logger.exception("Error in run_backtest endpoint")
        return jsonify({"error": str(e), "status": "error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)