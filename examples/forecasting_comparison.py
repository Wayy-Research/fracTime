#!/usr/bin/env python
"""
FracTime - Forecasting Comparison Example

This script demonstrates how to use the FracTime backtesting framework to
compare different forecasting methods on time series data.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import logging
import os
import time

# Import FracTime modules
from fractime.data_loader_polars import TimeSeriesDataLoader
from fractime.backtester_polars import TimeSeriesBacktester

# Import forecasters
from fractime.forecasting import (
    # Statistical forecasters
    ARIMAForecaster,
    SARIMAForecaster,
    ExponentialSmoothingForecaster,
    
    # Fractal forecasters
    StateTransitionFRSRForecaster,
    FractalProjectionForecaster,
    FractalClassificationForecaster,
    RescaledRangeForecaster,
    FractalInterpolationForecaster,
    FractalReductionForecaster,
    
    # Machine learning forecasters
    RandomForestForecaster,
    XGBoostForecaster,
    SVRForecaster,
    KNNForecaster
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data(data: pl.DataFrame, target_col: str, n_lags: int = 5) -> Tuple[pl.DataFrame, str]:
    """
    Prepare time series data for forecasting by creating lag features.
    
    Args:
        data: Time series data
        target_col: Name of the target column
        n_lags: Number of lag features to create
        
    Returns:
        Tuple of prepared data and target column name
    """
    # Make sure the data is sorted
    if 'Date' in data.columns:
        data = data.sort('Date')
    
    # Create lag features
    for i in range(1, n_lags + 1):
        data = data.with_columns(
            pl.col(target_col).shift(i).alias(f'lag_{i}')
        )
    
    # Drop rows with NaN values
    data = data.drop_nulls()
    
    return data, target_col


def run_forecasting_comparison(
    data: pl.DataFrame,
    target_col: str,
    date_col: str = None,
    window_size: int = 60,
    step_size: int = 5,
    forecast_horizon: int = 5,
    expanding_window: bool = False,
    n_lags: int = 5,
    output_dir: str = None
) -> Dict:
    """
    Run a comparison of different forecasting methods on time series data.
    
    Args:
        data: Time series data
        target_col: Name of the target column
        date_col: Name of the date column (optional)
        window_size: Size of the sliding window for backtesting
        step_size: Step size for moving the window
        forecast_horizon: Number of steps to forecast ahead
        expanding_window: Whether to use an expanding window or sliding window
        n_lags: Number of lag features to use
        output_dir: Directory to save output files (optional)
        
    Returns:
        Dictionary of backtesting results
    """
    # Prepare the data
    prepared_data, prepared_target = prepare_data(data, target_col, n_lags)
    
    # Define feature columns (lag columns)
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)]
    
    # Create forecasters
    forecasters = {
        # Statistical forecasters
        'ARIMA(1,1,1)': ARIMAForecaster(p=1, d=1, q=1),
        'ETS': ExponentialSmoothingForecaster(),
        
        # Fractal forecasters
        'ST-FRSR': StateTransitionFRSRForecaster(),
        'FractalProjection': FractalProjectionForecaster(),
        'FractalClassification': FractalClassificationForecaster(),
        'RescaledRange': RescaledRangeForecaster(),
        
        # Machine learning forecasters
        'RandomForest': RandomForestForecaster(n_estimators=100, random_state=42),
        'KNN': KNNForecaster(n_neighbors=3)
    }
    
    # Create the backtester
    backtester = TimeSeriesBacktester(forecasters)
    
    # Run the backtest
    logger.info("Starting backtesting...")
    start_time = time.time()
    
    results = backtester.run_backtest(
        data=prepared_data,
        target_col=prepared_target,
        feature_cols=feature_cols,
        window_size=window_size,
        step_size=step_size,
        forecast_horizon=forecast_horizon,
        expanding_window=expanding_window,
        date_col=date_col
    )
    
    logger.info(f"Backtesting completed in {time.time() - start_time:.2f} seconds")
    
    # Generate a report
    report = backtester.generate_report()
    logger.info("\n" + report)
    
    # Plot the results
    plot = backtester.plot_results()
    
    # Save the report and plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the report
        with open(os.path.join(output_dir, 'forecast_comparison_report.md'), 'w') as f:
            f.write(report)
        
        # Save the plot
        plot.savefig(os.path.join(output_dir, 'forecast_comparison_plot.png'), dpi=300, bbox_inches='tight')
    
    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FracTime Forecasting Comparison')
    
    # Data options
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Stock ticker symbol (for Yahoo Finance)')
    parser.add_argument('--target', type=str, default='Close', help='Target column name')
    parser.add_argument('--date-col', type=str, default='Date', help='Date column name')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date for the data')
    parser.add_argument('--end-date', type=str, default=None, help='End date for the data (default: today)')
    
    # Backtesting options
    parser.add_argument('--window-size', type=int, default=60, help='Size of the sliding window')
    parser.add_argument('--step-size', type=int, default=5, help='Step size for moving the window')
    parser.add_argument('--forecast-horizon', type=int, default=5, help='Number of steps to forecast ahead')
    parser.add_argument('--expanding-window', action='store_true', help='Use expanding window instead of sliding window')
    
    # Feature options
    parser.add_argument('--n-lags', type=int, default=5, help='Number of lag features to use')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output files')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load data
    logger.info(f"Loading data for {args.ticker} from {args.start_date} to {args.end_date or 'today'}")
    
    loader = TimeSeriesDataLoader()
    data = loader.get_yahoo_data(args.ticker, args.start_date, args.end_date)
    
    # Run the comparison
    results = run_forecasting_comparison(
        data=data,
        target_col=args.target,
        date_col=args.date_col,
        window_size=args.window_size,
        step_size=args.step_size,
        forecast_horizon=args.forecast_horizon,
        expanding_window=args.expanding_window,
        n_lags=args.n_lags,
        output_dir=args.output_dir
    )
    
    # Plot the results (in addition to the backtester plot)
    plt.figure(figsize=(12, 6))
    
    # Choose the best model based on RMSE
    best_model = min(
        [(model, results[model].get('avg_rmse', float('inf'))) for model in results],
        key=lambda x: x[1] if not np.isnan(x[1]) else float('inf')
    )[0]
    
    # Create a bar chart of average RMSE for each model
    models = list(results.keys())
    rmse_values = [results[model].get('avg_rmse', np.nan) for model in models]
    
    # Sort by RMSE
    sorted_indices = np.argsort(rmse_values)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_rmse = [rmse_values[i] for i in sorted_indices]
    
    # Create the plot
    plt.bar(sorted_models, sorted_rmse, color='skyblue')
    plt.axhline(y=sorted_rmse[0], color='r', linestyle='--', alpha=0.7, label='Best RMSE')
    
    plt.title(f'Average RMSE by Model (Best: {sorted_models[0]})')
    plt.ylabel('RMSE')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    if args.output_dir:
        plt.savefig(os.path.join(args.output_dir, 'rmse_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    main()