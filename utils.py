# Utility functions and shared code for the FracTime app

# Standard library imports
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import warnings
import logging
import os
import sys

# Third-party library imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st
import yfinance as yf

# Local application/library imports
from fractime import (
    FractalAnalyzer,
    FractalSimulator,
    get_yahoo_data,
    PathAnalyzer,
    FractalVisualizer,
    run_backtest,
    MultidimensionalFractalAnalyzer,
    QuantumPriceLevelGenerator,
    demo_quantum_fractal_analysis
)

# Function to analyze a single symbol
def analyze_symbol(
    symbol: str, 
    start_date: str, 
    n_paths: int = 1000, 
    n_steps: int = 30,
    use_trading_time: bool = True,
    warping_alpha: float = 0.5,
    enable_time_forecast: bool = True,
    use_cross_dim: bool = True,
    use_quantum_levels: bool = True,
    quantum_influence: float = 0.5
) -> dict:
    """Analyze a single symbol."""
    try:
        print(f"Running simulation for {symbol} with {n_paths} paths")
        data = get_yahoo_data(symbol, start_date)
        prices = data['Close'].to_numpy()
        dates = data['Date'].to_numpy()
        volumes = data['Volume'].to_numpy() if 'Volume' in data else None
        
        analyzer = FractalAnalyzer()
        simulator = FractalSimulator(prices, analyzer, volumes=volumes)
        
        # Print before and time the simulation
        import time
        start_time = time.time()
        
        # Run simulation with trading time parameters
        paths, path_analysis = simulator.simulate_paths(
            n_steps=n_steps, 
            n_paths=n_paths,
            use_trading_time=use_trading_time,
            warping_alpha=warping_alpha,
            enable_time_forecast=enable_time_forecast
        )
        
        end_time = time.time()
        
        # Verify the actual number of paths generated
        print(f"Generated {paths.shape[0]} paths in {end_time - start_time:.2f} seconds")
        
        # Get analysis results
        analysis_results = analyzer.analyze_patterns(prices)
        
        # Cross-dimensional analysis if enabled
        cross_dim_results = None
        if use_cross_dim and volumes is not None:
            try:
                # Create data series (price and volume)
                price_series = prices
                volume_series = volumes
                
                # Normalize volume to be on similar scale as price
                norm_volume = volume_series / np.mean(volume_series) * np.mean(price_series)
                
                # Run multidimensional analysis
                multi_analyzer = MultidimensionalFractalAnalyzer()
                cross_dim_results = multi_analyzer.analyze([price_series, norm_volume])
                print("Cross-dimensional analysis completed")
            except Exception as e:
                print(f"Error in cross-dimensional analysis: {e}")
        
        # Quantum price levels if enabled
        quantum_levels = None
        quantum_weights = None
        if use_quantum_levels:
            try:
                # Generate quantum price levels
                qpl_generator = QuantumPriceLevelGenerator()
                quantum_levels = qpl_generator.generate_price_levels(prices)
                
                # Weight paths based on quantum levels
                quantum_weights = qpl_generator.filter_paths_by_levels(
                    paths, influence_strength=quantum_influence
                )
                
                # Update path_analysis with quantum weights
                path_analysis['quantum_weights'] = quantum_weights.tolist()
                
                # Find most likely path
                most_likely_idx = np.argmax(quantum_weights)
                path_analysis['most_likely_path'] = paths[most_likely_idx]
                
                print("Quantum price level analysis completed")
            except Exception as e:
                print(f"Error in quantum price level analysis: {e}")
        
        return {
            'symbol': symbol,
            'prices': prices,
            'dates': dates,  # Include dates in results
            'volumes': volumes,  # Include volumes
            'analysis_results': analysis_results,
            'simulation_results': (paths, path_analysis),  # Store as tuple
            'time_map': simulator.time_map,  # Include time mapping for visualization
            'cross_dim_results': cross_dim_results,  # Add cross-dimensional results
            'quantum_levels': quantum_levels,  # Add quantum price levels
            'status': 'success',
            'trading_time_settings': {
                'use_trading_time': use_trading_time,
                'warping_alpha': warping_alpha,
                'enable_time_forecast': enable_time_forecast
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        }

# Function to generate a comprehensive report for multiple symbols
def generate_report(
    symbols: list, 
    start_date: str, 
    n_paths: int, 
    n_steps: int,
    use_trading_time: bool = True,
    use_cross_dim: bool = True, 
    use_quantum_levels: bool = True
):
    """Generate a comprehensive analysis report for each symbol."""
    
    # Create status indicators
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Store all results
    results = []
    
    # Process each symbol
    for i, symbol in enumerate(symbols):
        status.info(f"Analyzing {symbol}... This may take a minute.")
        
        # Run the analysis
        result = analyze_symbol(
            symbol, 
            start_date, 
            n_paths=n_paths, 
            n_steps=n_steps,
            use_trading_time=use_trading_time,
            warping_alpha=0.5,  # Default value
            enable_time_forecast=True,  # Always enabled
            use_cross_dim=use_cross_dim,
            use_quantum_levels=use_quantum_levels,
            quantum_influence=0.5  # Default value
        )
        
        results.append(result)
        progress_bar.progress((i + 1) / len(symbols))
    
    # Clear status once done
    status.empty()
    progress_bar.empty()
    
    return results