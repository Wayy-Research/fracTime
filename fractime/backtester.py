"""
Time Series Backtesting Framework for FracTime

This module provides a comprehensive backtesting framework for evaluating
and comparing different time series forecasting methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Union, Optional, Any, Tuple
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesBacktester:
    """
    A framework for backtesting time series forecasting models.
    
    This class allows for systematic evaluation of multiple forecasting models
    using various backtesting strategies such as expanding window, sliding window,
    or single train-test split.
    
    Attributes:
        models (dict): Dictionary of model name to model object
        metrics (dict): Dictionary of metric name to metric function
        results (dict): Results of the backtesting
        forecasts (dict): Forecasts from each model
        actuals (list): Actual values for comparison
    """
    
    def __init__(self, models: Optional[Dict[str, Any]] = None, 
                 metrics: Optional[Dict[str, Callable]] = None):
        """
        Initialize the backtester with models and metrics.
        
        Args:
            models: Dictionary of model name to model object
            metrics: Dictionary of metric name to metric function
        """
        self.models = models or {}
        self.metrics = metrics or {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "mape": lambda y, y_pred: mean_absolute_percentage_error(y, y_pred) if not np.any(y == 0) else np.nan,
            "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score
        }
        self.results = {}
        self.forecasts = {}
        self.actuals = []
        self.execution_times = {}
        self.forecast_horizon = 1
        
    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to the backtester.
        
        Args:
            name: Name of the model
            model: Model object with fit and predict methods
        """
        self.models[name] = model
        
    def add_metric(self, name: str, metric_func: Callable[[np.ndarray, np.ndarray], float]) -> None:
        """
        Add a custom metric function.
        
        Args:
            name: Name of the metric
            metric_func: Function that takes y_true and y_pred and returns a score
        """
        self.metrics[name] = metric_func
        
    def run_backtest(self, data: pd.DataFrame, target_col: str, 
                     feature_cols: Optional[List[str]] = None,
                     window_size: Optional[int] = None, step_size: int = 1, 
                     train_size: float = 0.7, expanding_window: bool = False,
                     forecast_horizon: int = 1,
                     recursive_forecast: bool = False,
                     date_col: Optional[str] = None) -> Dict[str, Dict[str, List[float]]]:
        """
        Run the backtest with expanding or sliding window.
        
        Args:
            data: Time series data
            target_col: Column name of the target variable
            feature_cols: List of feature column names. If None, all columns except target_col are used
            window_size: Size of the sliding window. If None, a single train-test split is used
            step_size: Number of steps to move the window forward
            train_size: Proportion of data to use for training if window_size is None
            expanding_window: If True, use expanding window, else use sliding window
            forecast_horizon: Number of steps to forecast ahead
            recursive_forecast: If True, use recursive forecasting for multi-step ahead
            date_col: Column name containing dates. If provided, will be used for time-based splitting
            
        Returns:
            Dictionary of model results
        """
        self._validate_data(data, target_col, feature_cols)
        
        self.forecast_horizon = forecast_horizon
        
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col and col != date_col]
        
        self.results = {model_name: {metric: [] for metric in self.metrics} 
                        for model_name in self.models}
        self.execution_times = {model_name: [] for model_name in self.models}
        self.forecasts = {model_name: [] for model_name in self.models}
        self.actuals = []
        
        # Create date index if date_col is provided
        if date_col is not None and date_col in data.columns:
            date_index = pd.DatetimeIndex(data[date_col])
        else:
            date_index = None
        
        if window_size is None:
            # Single train-test split
            self._run_single_split(data, target_col, feature_cols, train_size, 
                                  forecast_horizon, recursive_forecast, date_index)
        else:
            # Multiple window evaluation
            self._run_window_evaluation(data, target_col, feature_cols, window_size, 
                                       step_size, expanding_window, forecast_horizon, 
                                       recursive_forecast, date_index)
        
        # Calculate average metrics across all windows
        self._calculate_average_metrics()
                
        return self.results
    
    def _validate_data(self, data: pd.DataFrame, target_col: str, 
                      feature_cols: Optional[List[str]]) -> None:
        """
        Validate the input data for the backtest.
        
        Args:
            data: Time series data
            target_col: Column name of the target variable
            feature_cols: List of feature column names
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        if feature_cols is not None:
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns {missing_cols} not found in data")
        
        if data.isnull().values.any():
            logger.warning("Data contains NaN values, which may cause issues with some models")
    
    def _run_single_split(self, data: pd.DataFrame, target_col: str, 
                         feature_cols: List[str], train_size: float,
                         forecast_horizon: int, recursive_forecast: bool,
                         date_index: Optional[pd.DatetimeIndex]) -> None:
        """
        Run a single train-test split backtest.
        
        Args:
            data: Time series data
            target_col: Column name of the target variable
            feature_cols: List of feature column names
            train_size: Proportion of data to use for training
            forecast_horizon: Number of steps to forecast ahead
            recursive_forecast: If True, use recursive forecasting for multi-step ahead
            date_index: Optional date index for time-based splitting
        """
        # Determine split index
        train_idx = int(len(data) * train_size)
        
        # Split the data
        train_data = data.iloc[:train_idx]
        test_data = data.iloc[train_idx:]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        
        # For multi-step forecasting, we need a different approach
        if forecast_horizon > 1:
            self._evaluate_models_multistep(X_train, y_train, test_data, feature_cols, 
                                          target_col, forecast_horizon, recursive_forecast)
            self.actuals = test_data[target_col].values[:forecast_horizon]
        else:
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            self._evaluate_models(X_train, y_train, X_test, y_test)
            self.actuals = y_test.values
    
    def _run_window_evaluation(self, data: pd.DataFrame, target_col: str, 
                              feature_cols: List[str], window_size: int, 
                              step_size: int, expanding_window: bool,
                              forecast_horizon: int, recursive_forecast: bool,
                              date_index: Optional[pd.DatetimeIndex]) -> None:
        """
        Run a window-based backtest (expanding or sliding).
        
        Args:
            data: Time series data
            target_col: Column name of the target variable
            feature_cols: List of feature column names
            window_size: Size of the sliding window
            step_size: Number of steps to move the window forward
            expanding_window: If True, use expanding window, else use sliding window
            forecast_horizon: Number of steps to forecast ahead
            recursive_forecast: If True, use recursive forecasting for multi-step ahead
            date_index: Optional date index for time-based splitting
        """
        # Ensure window size is reasonable
        if window_size >= len(data):
            raise ValueError(f"Window size ({window_size}) must be less than data length ({len(data)})")
        
        # For each window
        for i in range(0, len(data) - window_size - forecast_horizon + 1, step_size):
            end_idx = i + window_size
            
            if expanding_window:
                # Expanding window: train data grows
                train_data = data.iloc[:end_idx]
            else:
                # Sliding window: fixed size train data
                train_data = data.iloc[i:end_idx]
                
            test_idx = end_idx
            
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            
            # For multi-step forecasting
            if forecast_horizon > 1:
                test_data = data.iloc[test_idx:test_idx+forecast_horizon]
                if len(test_data) < forecast_horizon:
                    continue
                
                # Evaluate models on this window
                self._evaluate_models_multistep(X_train, y_train, test_data, feature_cols, 
                                             target_col, forecast_horizon, recursive_forecast)
                self.actuals.extend(test_data[target_col].values)
            else:
                # Single-step forecasting
                test_data = data.iloc[test_idx:test_idx+1]
                if len(test_data) == 0:
                    continue
                    
                X_test = test_data[feature_cols]
                y_test = test_data[target_col]
                
                # Evaluate models on this window
                self._evaluate_models(X_train, y_train, X_test, y_test)
                self.actuals.extend(y_test.values)
    
    def _evaluate_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evaluate all models on the current window for single-step forecasting.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        """
        for model_name, model in self.models.items():
            try:
                # Time the execution
                start_time = time.time()
                
                # Clone the model to avoid data leakage if possible
                try:
                    m = clone(model)
                except:
                    # If model can't be cloned, use it directly
                    m = model
                
                # Fit the model
                m.fit(X_train, y_train)
                
                # Make predictions
                y_pred = m.predict(X_test)
                
                # Record execution time
                execution_time = time.time() - start_time
                self.execution_times[model_name].append(execution_time)
                
                # Store the predictions
                self.forecasts[model_name].extend(y_pred if isinstance(y_pred, list) else y_pred.tolist())
                
                # Calculate and store metrics
                for metric_name, metric_func in self.metrics.items():
                    try:
                        score = metric_func(y_test, y_pred)
                        self.results[model_name][metric_name].append(score)
                    except Exception as e:
                        logger.warning(f"Error calculating {metric_name} for {model_name}: {e}")
                        self.results[model_name][metric_name].append(np.nan)
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                # Add NaN values for metrics to maintain structure
                for metric_name in self.metrics:
                    self.results[model_name][metric_name].append(np.nan)
                self.execution_times[model_name].append(np.nan)
    
    def _evaluate_models_multistep(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                  test_data: pd.DataFrame, feature_cols: List[str], 
                                  target_col: str, forecast_horizon: int, 
                                  recursive_forecast: bool) -> None:
        """
        Evaluate all models on the current window for multi-step forecasting.
        
        Args:
            X_train: Training features
            y_train: Training target
            test_data: Test data including features and target
            feature_cols: List of feature column names
            target_col: Column name of the target variable
            forecast_horizon: Number of steps to forecast ahead
            recursive_forecast: If True, use recursive forecasting
        """
        for model_name, model in self.models.items():
            try:
                # Time the execution
                start_time = time.time()
                
                # Clone the model if possible
                try:
                    m = clone(model)
                except:
                    m = model
                
                # Fit the model
                m.fit(X_train, y_train)
                
                # Direct multi-step forecasting (if the model supports it)
                if hasattr(m, 'predict_many') and not recursive_forecast:
                    y_pred = m.predict_many(X_train.iloc[-1:], forecast_horizon)
                # Recursive forecasting
                elif recursive_forecast:
                    # Initialize with the last training point
                    current_features = X_train.iloc[-1:].copy()
                    y_pred = []
                    
                    # Recursively predict each step
                    for h in range(forecast_horizon):
                        # Make single-step prediction
                        pred = m.predict(current_features)
                        y_pred.append(pred[0])
                        
                        # Update features for next step prediction
                        # This requires knowledge of how features are created
                        # Here we make a simple assumption: shift the target 
                        # and keep other features the same
                        for i, col in enumerate(feature_cols):
                            if col.startswith('lag_') or col == target_col:
                                # Shift the lagged features
                                lag = int(col.replace('lag_', '')) if col.startswith('lag_') else 1
                                if lag > 1:
                                    # Update older lags
                                    current_features[col] = test_data.iloc[h:h+1][f'lag_{lag-1}'].values
                                else:
                                    # Update most recent lag with prediction
                                    current_features[col] = pred[0]
                else:
                    # Model doesn't support multi-step and we're not using recursive
                    # Just make direct predictions for each horizon
                    y_pred = []
                    for h in range(forecast_horizon):
                        # Use the appropriate row of test data for this horizon
                        if h < len(test_data):
                            test_features = test_data.iloc[h:h+1][feature_cols]
                            pred = m.predict(test_features)
                            y_pred.append(pred[0])
                
                # Record execution time
                execution_time = time.time() - start_time
                self.execution_times[model_name].append(execution_time)
                
                # Store the predictions
                self.forecasts[model_name].extend(y_pred if isinstance(y_pred, list) else y_pred.tolist())
                
                # Calculate and store metrics
                y_test = test_data[target_col].values[:forecast_horizon]
                if len(y_pred) < len(y_test):
                    # If we didn't generate enough predictions, pad with NaN
                    y_pred = np.append(y_pred, [np.nan] * (len(y_test) - len(y_pred)))
                elif len(y_pred) > len(y_test):
                    # If we generated too many predictions, truncate
                    y_pred = y_pred[:len(y_test)]
                
                for metric_name, metric_func in self.metrics.items():
                    try:
                        score = metric_func(y_test, y_pred)
                        self.results[model_name][metric_name].append(score)
                    except Exception as e:
                        logger.warning(f"Error calculating {metric_name} for {model_name}: {e}")
                        self.results[model_name][metric_name].append(np.nan)
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                # Add NaN values for metrics to maintain structure
                for metric_name in self.metrics:
                    self.results[model_name][metric_name].append(np.nan)
                self.execution_times[model_name].append(np.nan)
    
    def _calculate_average_metrics(self) -> None:
        """Calculate average metrics across all windows for each model."""
        for model_name in self.models:
            for metric in self.metrics:
                values = [v for v in self.results[model_name][metric] if not np.isnan(v)]
                if values:
                    self.results[model_name][f"avg_{metric}"] = np.mean(values)
                    self.results[model_name][f"std_{metric}"] = np.std(values)
                else:
                    self.results[model_name][f"avg_{metric}"] = np.nan
                    self.results[model_name][f"std_{metric}"] = np.nan
            
            # Add execution time metrics
            times = [t for t in self.execution_times[model_name] if not np.isnan(t)]
            if times:
                self.results[model_name]["avg_time"] = np.mean(times)
                self.results[model_name]["total_time"] = np.sum(times)
            else:
                self.results[model_name]["avg_time"] = np.nan
                self.results[model_name]["total_time"] = np.nan
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plot the backtesting results.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        if not self.actuals:
            logger.warning("No data to plot. Run backtest first.")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 1. Plot forecasts vs actuals
        ax = axes[0]
        
        # Convert all to numpy arrays for easier handling
        actuals = np.array(self.actuals)
        forecasts = {m: np.array(f) for m, f in self.forecasts.items()}
        
        # Create synthetic x-axis
        x = np.arange(len(actuals))
        
        # Plot actuals first
        ax.plot(x, actuals, label='Actual', color='black', linewidth=2)
        
        # Plot forecasts
        for model_name, forecast_values in forecasts.items():
            # Ensure lengths match
            if len(forecast_values) < len(actuals):
                forecast_values = np.pad(forecast_values, 
                                       (0, len(actuals) - len(forecast_values)), 
                                       'constant', 
                                       constant_values=np.nan)
            elif len(forecast_values) > len(actuals):
                forecast_values = forecast_values[:len(actuals)]
                
            ax.plot(x, forecast_values, label=f'{model_name}', alpha=0.7)
            
        ax.set_title('Forecasts vs Actuals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 2. Plot average metrics for each model
        ax = axes[1]
        
        # Get all metric names that start with "avg_" excluding "avg_time"
        metrics = [m for m in next(iter(self.results.values())).keys() 
                  if m.startswith('avg_') and m != 'avg_time']
        
        # Sort by model names
        models = sorted(self.models.keys())
        
        # Prepare data for grouped bar chart
        metric_data = []
        for metric in metrics:
            # Get values for this metric across all models
            values = [self.results[model].get(metric, np.nan) for model in models]
            metric_data.append((metric, values))
        
        # Sort metrics by name
        metric_data.sort(key=lambda x: x[0])
        
        # Create x positions for bars
        x = np.arange(len(models))
        width = 0.8 / len(metrics)
        
        # Plot each metric as a group of bars
        for i, (metric, values) in enumerate(metric_data):
            # Format the metric name for display
            metric_display = metric.replace('avg_', '').upper()
            
            # Position for this group of bars
            pos = x + i * width - 0.4 + width/2
            
            # Plot the bars
            bars = ax.bar(pos, values, width, label=metric_display)
            
            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{val:.3f}', ha='center', va='bottom', 
                           rotation=90, fontsize=8)
        
        # Customize the plot
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title('Average Metrics by Model')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 3. Plot execution times
        ax = axes[2]
        
        # Sort models by average execution time
        model_times = [(model, self.results[model].get('avg_time', np.nan)) 
                      for model in self.models]
        model_times.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
        
        models_sorted = [m[0] for m in model_times]
        times = [m[1] for m in model_times]
        
        # Create the bar chart
        bars = ax.bar(models_sorted, times, color='skyblue')
        
        # Add value labels
        for bar, val in zip(bars, times):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{val:.3f}s', ha='center', va='bottom', 
                       rotation=0, fontsize=8)
        
        # Customize the plot
        ax.set_title('Average Execution Time by Model')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticklabels(models_sorted, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of the backtesting results.
        
        Args:
            output_file: If provided, save the report to this file
            
        Returns:
            The report as a string
        """
        if not self.results:
            logger.warning("No results to report. Run backtest first.")
            return "No results available. Run backtest first."
        
        # Start building the report
        report = []
        report.append("# Time Series Backtesting Report")
        report.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Overall summary
        report.append("## 1. Overall Summary")
        report.append(f"* Number of models evaluated: {len(self.models)}")
        report.append(f"* Number of data points: {len(self.actuals)}")
        report.append(f"* Forecast horizon: {self.forecast_horizon}")
        report.append("")
        
        # 2. Model performance summary
        report.append("## 2. Model Performance Summary")
        
        # Create a summary table
        report.append("### Average Metrics")
        report.append("")
        
        # Table header
        metrics = [m for m in next(iter(self.results.values())).keys() 
                 if m.startswith('avg_')]
        header = ["Model"] + [m.replace('avg_', '').upper() for m in metrics]
        report.append(" | ".join(header))
        report.append(" | ".join(["---"] * len(header)))
        
        # Table rows
        for model_name in sorted(self.models.keys()):
            row = [model_name]
            for metric in metrics:
                value = self.results[model_name].get(metric, np.nan)
                row.append(f"{value:.4f}" if not np.isnan(value) else "N/A")
            report.append(" | ".join(row))
        
        report.append("")
        
        # 3. Detailed metrics for each model
        report.append("## 3. Detailed Model Performance")
        
        for model_name in sorted(self.models.keys()):
            report.append(f"### {model_name}")
            
            # Model type
            model_type = type(self.models[model_name]).__name__
            report.append(f"* Type: {model_type}")
            
            # Execution time
            avg_time = self.results[model_name].get('avg_time', np.nan)
            total_time = self.results[model_name].get('total_time', np.nan)
            
            if not np.isnan(avg_time):
                report.append(f"* Average execution time: {avg_time:.4f} seconds")
            if not np.isnan(total_time):
                report.append(f"* Total execution time: {total_time:.4f} seconds")
                
            # Average metrics
            report.append("\n#### Average Metrics")
            for metric in [m for m in self.results[model_name].keys() 
                         if m.startswith('avg_') and m != 'avg_time']:
                value = self.results[model_name].get(metric, np.nan)
                if not np.isnan(value):
                    report.append(f"* {metric.replace('avg_', '').upper()}: {value:.4f}")
            
            # Standard deviation of metrics
            report.append("\n#### Metric Stability (Standard Deviation)")
            for metric in [m for m in self.results[model_name].keys() 
                         if m.startswith('std_')]:
                value = self.results[model_name].get(metric, np.nan)
                if not np.isnan(value):
                    report.append(f"* {metric.replace('std_', '').upper()}: {value:.4f}")
            
            report.append("\n---\n")
        
        # 4. Conclusions and recommendations
        report.append("## 4. Conclusions and Recommendations")
        
        # Find the best model for each metric
        best_models = {}
        for metric in [m for m in metrics if m != 'avg_time']:
            metric_values = [(model, self.results[model].get(metric, np.nan)) 
                           for model in self.models]
            
            # Filter out NaN values
            metric_values = [(m, v) for m, v in metric_values if not np.isnan(v)]
            
            if metric_values:
                # For error metrics (MSE, MAE, RMSE, MAPE), lower is better
                if metric in ['avg_mse', 'avg_mae', 'avg_rmse', 'avg_mape']:
                    best_model = min(metric_values, key=lambda x: x[1])
                else:
                    # For other metrics (R2), higher is better
                    best_model = max(metric_values, key=lambda x: x[1])
                
                best_models[metric] = best_model
        
        # Report best models
        if best_models:
            report.append("### Best Performing Models")
            for metric, (model, value) in best_models.items():
                report.append(f"* {metric.replace('avg_', '').upper()}: **{model}** ({value:.4f})")
        
        # Add execution time comparison
        report.append("\n### Execution Time Comparison")
        fastest_model = min([(model, self.results[model].get('avg_time', float('inf'))) 
                           for model in self.models], 
                          key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
        
        report.append(f"* Fastest model: **{fastest_model[0]}** ({fastest_model[1]:.4f} seconds per prediction)")
        
        # Overall recommendation
        report.append("\n### Overall Recommendation")
        
        # Simple heuristic: recommend based on RMSE if available, otherwise based on another common metric
        recommend_metric = 'avg_rmse' if 'avg_rmse' in metrics else metrics[0]
        if recommend_metric in best_models:
            model, value = best_models[recommend_metric]
            report.append(f"Based on {recommend_metric.replace('avg_', '').upper()}, the recommended model is **{model}** with a score of {value:.4f}.")
        else:
            report.append("No clear recommendation can be made based on the available metrics.")
        
        # Join everything into a single string
        full_report = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(full_report)
            logger.info(f"Report saved to {output_file}")
        
        return full_report