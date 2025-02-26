# Standard library imports
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Third-party library imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st
import yfinance as yf

# Set page configuration
st.set_page_config(
    page_title="fractime",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local application/library imports
from fractime import (
    FractalAnalyzer,
    FractalSimulator,
    get_yahoo_data,
    PathAnalyzer,
    FractalVisualizer,
    run_backtest,
)
from fractime.quantum import (
    MultidimensionalFractalAnalyzer,
    QuantumPriceLevelGenerator,
)

# Remove quantum_finance imports and related code
# import quantum_finance as qf
# import inspect
# import importlib
# import sys

# Clear the cache to ensure we get fresh data
st.cache_data.clear()
st.cache_resource.clear()

def analyze_symbol(symbol: str, start_date: str, n_paths: int = 1000, n_steps: int = 30) -> dict:
    """Analyze a single symbol."""
    try:
        print(f"Running simulation for {symbol} with {n_paths} paths")
        data = get_yahoo_data(symbol, start_date)
        prices = data['Close'].to_numpy()
        dates = data['Date'].to_numpy()
        
        analyzer = FractalAnalyzer()
        simulator = FractalSimulator(prices, analyzer)
        
        # Print before and time the simulation
        import time
        start_time = time.time()
        paths, path_analysis = simulator.simulate_paths(n_steps=n_steps, n_paths=n_paths)
        end_time = time.time()
        
        # Verify the actual number of paths generated
        print(f"Generated {paths.shape[0]} paths in {end_time - start_time:.2f} seconds")
        
        # Get analysis results
        analysis_results = analyzer.analyze_patterns(prices)
        
        return {
            'symbol': symbol,
            'prices': prices,
            'dates': dates,  # Include dates in results
            'analysis_results': analysis_results,
            'simulation_results': (paths, path_analysis),  # Store as tuple
            'status': 'success'
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        }

def analyze_symbol_quantum(symbol: str, start_date: str) -> dict:
    """Analyze a single symbol using quantum methods."""
    try:
        data = get_yahoo_data(symbol, start_date)
        prices = data['Close'].to_numpy()
        volumes = data['Volume'].to_numpy() if 'Volume' in data else None
        dates = data['Date'].to_numpy()  
        
        # Create a list of data series to analyze
        data_series = [prices]
        if volumes is not None:
            # Normalize volumes to be on similar scale as prices
            norm_volumes = volumes / np.mean(volumes) * np.mean(prices)
            data_series.append(norm_volumes)
        
        # Use the MultidimensionalFractalAnalyzer for quantum analysis
        multi_analyzer = MultidimensionalFractalAnalyzer()
        multi_results = multi_analyzer.analyze(data_series)
        
        # Generate quantum price levels
        qpl_generator = QuantumPriceLevelGenerator()
        qpl_results = qpl_generator.generate_price_levels(prices)
        
        # Use standard FractalAnalyzer for traditional metrics
        standard_analyzer = FractalAnalyzer()
        standard_results = standard_analyzer.analyze_patterns(prices)
        
        # Combine results
        combined_results = {
            'quantum': {
                'multi_dimensional': multi_results,
                'price_levels': qpl_results
            },
            'standard': standard_results
        }
        
        # For visualization, use standard simulator
        simulator = FractalSimulator(prices, standard_analyzer)
        paths, path_analysis = simulator.simulate_paths(n_steps=30, n_paths=1000)
        
        return {
            'symbol': symbol,
            'prices': prices,
            'dates': dates,
            'analysis_results': combined_results,
            'simulation_results': (paths, path_analysis),
            'status': 'success'
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        }

def main():
    st.title("Fractal Time Series Analysis")
    
    # Add page navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Forecast Dashboard", "Backtest System", "Documentation", "Performance Benchmark"]
    )
    
    if page == "Forecast Dashboard":
        forecast_dashboard()
    elif page == "Backtest System":
        backtest_system()
    elif page == "Performance Benchmark":
        performance_benchmark_page()
    else:
        documentation_page()
    
# Rename your existing main content to this function
def forecast_dashboard():
    # Create status containers early
    status_container = st.container()
    progress_bar = st.progress(0)
    log_container = st.expander("Analysis Logs", expanded=True)
    
    # Sidebar controls
    st.sidebar.header("Analysis Settings")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Yahoo Finance Symbols", "CSV Upload"]
    )
    
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365)
    ).strftime('%Y-%m-%d')
    
    if input_method == "Yahoo Finance Symbols":
        symbols_input = st.sidebar.text_area(
            "Enter symbols (one per line)",
            "^GSPC\nAAPL\nMSFT"
        )
        symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV with columns: Symbol, Date, Close",
            type=['csv']
        )
        if uploaded_file:
            df = pl.read_csv(uploaded_file)
            symbols = df['Symbol'].unique().to_list()
        else:
            symbols = []
    
    # Analysis controls
    use_log_scale = st.sidebar.checkbox("Use logarithmic scale for paths", value=False)

    if use_log_scale:
        log_paths = st.sidebar.slider("Number of Simulation Paths (log10)", 2, 6, 3)
        n_paths = int(10**log_paths)
        st.sidebar.write(f"Actual paths: {n_paths:,}")
    else:
        n_paths = st.sidebar.slider("Number of Simulation Paths", 100, 100000, 1000)

    n_steps = st.sidebar.slider("Forecast Steps", 10, 100, 30)
    
    # In the sidebar, add a visualization option
    visualization_type = st.sidebar.radio(
        "Visualization Type", 
        ["Standard", "High Density (All Paths)"]
    )

    if st.sidebar.button("Run Analysis"):
        if not symbols:
            st.warning("Please enter symbols or upload data")
            return
        
        # Clear previous results
        status_container.empty()
        progress_bar.empty()
        log_container.empty()
        
        # Results containers
        results = []
        logs = []
        
        # Create a placeholder for status updates
        status_text = status_container.empty()
        log_text = log_container.empty()
        
        def update_status(msg: str):
            """Thread-safe status update."""
            logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
            status_text.text(f"Status: {msg}")
            log_text.text("\n".join(logs))
        
        update_status("Starting analysis...")
        
        # Process symbols sequentially with status updates
        for i, symbol in enumerate(symbols):
            update_status(f"Analyzing {symbol}...")
            result = analyze_symbol(symbol, start_date, n_paths=n_paths, n_steps=n_steps)
            results.append(result)
            
            if result['status'] == 'success':
                update_status(f"Successfully analyzed {symbol}")
            else:
                update_status(f"Error analyzing {symbol}: {result['error']}")
            
            progress_bar.progress((i + 1) / len(symbols))
        
        update_status("Analysis complete! Displaying results...")
        
        # Display results
        st.header("Analysis Results")
        
        # Summary statistics
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Symbols", len(symbols))
        col2.metric("Successful", len(successful))
        col3.metric("Failed", len(failed))
        
        # Detailed results
        st.subheader("Successful Analyses")
        for result in successful:
            with st.expander(f"Details for {result['symbol']}"):
                # Create visualization with selected approach
                try:
                    visualizer = FractalVisualizer()
                    
                    if visualization_type == "Standard":
                        fig = visualizer.plot_analysis_and_forecast(
                            result['prices'],
                            result['simulation_results'],
                            result['analysis_results'],
                            result['dates']
                        )
                    else:
                        fig = visualizer.plot_high_density_forecast(
                            result['prices'],
                            result['simulation_results'],
                            result['analysis_results'],
                            result['dates']
                        )
                        
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error visualizing {result['symbol']}: {e}")
                
                # Display metrics
                metrics = result['analysis_results']
                st.write(f"Hurst Exponent: {metrics['hurst']:.3f}")
                st.write(f"Fractal Dimension: {metrics['fractal_dim']:.3f}")
                st.write(f"Number of Patterns: {len(metrics['self_similar_patterns'])}")
        
        if failed:
            st.subheader("Failed Analyses")
            for result in failed:
                st.error(f"{result['symbol']}: {result['error']}")
        
        # Export results
        summary_data = [{
            'symbol': r['symbol'],
            'hurst': r['analysis_results']['hurst'] if r['status'] == 'success' else None,
            'fractal_dim': r['analysis_results']['fractal_dim'] if r['status'] == 'success' else None,
            'n_patterns': len(r['analysis_results']['self_similar_patterns']) if r['status'] == 'success' else None,
            'status': r['status'],
            'error': r.get('error', '')
        } for r in results]
        
        summary_df = pd.DataFrame(summary_data)
        st.download_button(
            "Download Summary CSV",
            summary_df.to_csv(index=False),
            "fractal_analysis_summary.csv",
            "text/csv"
        )

def backtest_system():
    """Page for backtesting the fractal model."""
    st.header("Fractal Model Backtesting")
    
    # Initialize session state for cancellation if not exists
    if 'cancel_requested' not in st.session_state:
        st.session_state.cancel_requested = False
    
    # Setup UI elements
    with st.form("backtest_form"):
        # Symbol selection
        symbols_input = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOG,AMZN,TSLA,^GSPC")
        symbols = [s.strip() for s in symbols_input.split(",")]
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_count = st.number_input("Samples per Symbol", min_value=1, value=30)
            forecast_horizon = st.number_input("Forecast Horizon (days)", min_value=5, value=20)
        
        with col2:
            start_date = st.date_input(
                "Start Date", 
                value=datetime.now() - timedelta(days=365*5)
            ).strftime("%Y-%m-%d")
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            ).strftime("%Y-%m-%d")
            
        # Metrics and benchmarks selection
        metrics = st.multiselect(
            "Metrics to evaluate",
            ["MAPE", "RMSE", "Direction Accuracy"],
            default=["MAPE", "Direction Accuracy"]
        )
        
        benchmarks = st.multiselect(
            "Benchmark models",
            ["Random Walk", "Simple Moving Average", "ARIMA"],
            default=["Random Walk", "Simple Moving Average"]
        )
        
        # Use a small, less prominent submit button
        submitted = st.form_submit_button("Apply Settings")
    
    # Execution and status section
    status_area = st.empty()
    progress_area = st.empty()
    results_area = st.container()
    
    # Create a placeholder for log messages
    log_container = st.expander("Backtest Logs", expanded=True)
    log_area = log_container.empty()
    
    # Store logs in session state
    if 'backtest_logs' not in st.session_state:
        st.session_state.backtest_logs = []
    
    # Add cancel button
    cancel_col1, cancel_col2 = st.columns(2)
    
    with cancel_col1:
        if st.button("Run Backtest", key="run_backtest_button"):
            # Reset cancellation flag
            st.session_state.cancel_requested = False
            st.session_state.backtest_logs = []
            
            with status_area:
                st.info("Backtest in progress...")
            
            # Update progress and status
            def update_progress(symbol, samples_completed):
                with progress_area:
                    st.progress(samples_completed / sample_count)
            
            def update_status(message):
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_message = f"{timestamp} - {message}"
                st.session_state.backtest_logs.append(log_message)
                
                with log_area:
                    st.text("\n".join(st.session_state.backtest_logs))
            
            # Run the backtest with callbacks
            try:
                backtest_results = run_backtest(
                    symbols=symbols,
                    sample_count=int(sample_count),
                    start_date=start_date,
                    end_date=end_date,
                    forecast_horizon=int(forecast_horizon),
                    metrics=metrics,
                    benchmarks=benchmarks,
                    status_callback=update_status,
                    progress_callback=update_progress,
                    cancellation_callback=lambda: st.session_state.cancel_requested,
                    parallel=True,
                    max_workers=4
                )
                
                # Store results in session state
                st.session_state.backtest_results = backtest_results
                
                # Check for valid results using the new structure
                valid_results = False
                if backtest_results and 'symbol_results' in backtest_results:
                    total_samples = sum(len(result.get('samples', [])) 
                                      for symbol, result in backtest_results['symbol_results'].items())
                    valid_results = total_samples > 0
                
                # Display success or warning
                with status_area:
                    if valid_results:
                        st.success(f"Backtest completed successfully with {total_samples} valid samples.")
                    else:
                        st.error("No valid samples were generated during the backtest.")
            
            except Exception as e:
                with status_area:
                    st.error(f"Error running backtest: {e}")
                import traceback
                with log_area:
                    st.text(traceback.format_exc())
    
    with cancel_col2:
        if st.button("Cancel Backtest"):
            st.session_state.cancel_requested = True
            with status_area:
                st.warning("Cancellation requested. Please wait...")
    
    # Display results if available
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        results = st.session_state.backtest_results
        
        # Check for valid results
        valid_results = False
        total_samples = 0
        if 'symbol_results' in results:
            for symbol, symbol_result in results['symbol_results'].items():
                if 'samples' in symbol_result:
                    total_samples += len(symbol_result['samples'])
            valid_results = total_samples > 0
        
        with results_area:
            if valid_results:
                display_backtest_results(results)
            else:
                st.error("No valid samples were generated during the backtest.")

def display_backtest_results(results):
    """Display the results of the backtest."""
    # Get the aggregated metrics
    if 'aggregate_metrics' in results:
        metrics = results['aggregate_metrics']
    else:
        metrics = {}
    
    # Count total samples
    total_samples = 0
    samples_by_symbol = {}
    
    if 'symbol_results' in results:
        for symbol, symbol_result in results['symbol_results'].items():
            if 'samples' in symbol_result:
                num_samples = len(symbol_result['samples'])
                samples_by_symbol[symbol] = num_samples
                total_samples += num_samples
    
    st.subheader(f"Backtest Results: {total_samples} Total Samples")
    
    # Display metrics table if we have data
    if metrics and 'fractal_model' in metrics:
        display_metrics_table(metrics)
        
        # Rest of the function to display charts/visualizations

def display_metrics_table(metrics):
    """Display the metrics table comparing fractal model to benchmarks."""
    # Create dataframe for metrics comparison
    fractal_metrics = metrics['fractal_model']
    benchmark_metrics = metrics.get('benchmarks', {})
    
    # Get list of all metrics
    all_metric_names = list(fractal_metrics.keys())
    
    # Get list of all benchmarks
    all_benchmarks = list(benchmark_metrics.keys())
    
    # Create a dataframe for comparison
    rows = []
    
    # Add header row
    header = ["Metric", "Fractal Model"]
    for benchmark in all_benchmarks:
        header.append(benchmark)
    rows.append(header)
    
    # Add rows for each metric
    for metric_name in all_metric_names:
        fractal_data = fractal_metrics[metric_name]
        
        # Create row for mean values
        mean_row = [f"{metric_name} (Mean)"]
        mean_row.append(f"{fractal_data.get('mean', 0):.4f}")
        
        for benchmark in all_benchmarks:
            if benchmark in benchmark_metrics and metric_name in benchmark_metrics[benchmark]:
                bench_data = benchmark_metrics[benchmark][metric_name]
                mean_row.append(f"{bench_data.get('mean', 0):.4f}")
            else:
                mean_row.append("N/A")
        
        rows.append(mean_row)
        
        # Create row for win rate
        if 'win_rate' in fractal_data:
            win_row = [f"{metric_name} (Win Rate %)"]
            for benchmark in all_benchmarks:
                if benchmark in benchmark_metrics and metric_name in benchmark_metrics[benchmark]:
                    win_rate = fractal_data.get('win_rate', 0)
                    win_row.append(f"{win_rate:.2f}%")
                else:
                    win_row.append("N/A")
            rows.append(win_row)
    
    # Display the table
    table_data = pd.DataFrame(rows[1:], columns=rows[0])
    st.table(table_data)
    
    # Add interpretation
    st.write("""
    **Interpretation:**
    - **MAPE (Mean Absolute Percentage Error)**: Lower is better
    - **RMSE (Root Mean Square Error)**: Lower is better
    - **Direction Accuracy**: Higher is better (percentage of correct direction predictions)
    - **Win Rate**: Percentage of samples where the fractal model outperformed the benchmark
    """)

def documentation_page():
    """Display documentation about the fracTime package."""
    st.header("FracTime Documentation")
    
    st.subheader("Overview")
    st.write("""
    FracTime is an advanced time series forecasting tool based on fractal geometry principles
    and chaos theory. Drawing inspiration from Benoit Mandelbrot's work on financial markets, 
    FracTime analyzes the self-similar patterns in financial data across different time scales.
    """)
    
    st.subheader("Key Features")
    st.markdown("""
    - **Fractal Pattern Analysis**: Identifies self-similar patterns across different timeframes
    - **Path Simulation**: Generates thousands of possible future price paths
    - **Volatility Clustering**: Accounts for the tendency of volatility to cluster in financial markets
    - **Regime Detection**: Identifies different market regimes and matches forecasts to similar historical periods
    - **Quantum Enhancements**: Uses quantum-inspired methods for improved pattern recognition
    - **Backtesting System**: Rigorously evaluates forecast accuracy on historical data
    """)
    
    st.subheader("How to Use")
    
    st.write("#### Forecast Dashboard")
    st.markdown("""
    1. **Enter Symbols**: Input stock or index symbols from Yahoo Finance
    2. **Configure Parameters**: Set the number of paths and forecast steps
    3. **Run Analysis**: Process the data and generate forecasts
    4. **Interpret Results**: View forecasts, probability clouds, and support/resistance levels
    """)
    
    st.write("#### Backtest System")
    st.markdown("""
    1. **Select Symbols**: Choose which stocks or indices to backtest
    2. **Set Parameters**: Configure sample count, date ranges, and metrics
    3. **Run Backtest**: Evaluate model performance on historical data
    4. **Analyze Results**: Compare fractal model performance against benchmarks
    """)
    
    st.subheader("Mathematical Foundation")
    st.markdown("""
    FracTime uses several key concepts from fractal geometry and chaos theory:
    
    - **Hurst Exponent**: Measures the long-term memory of a time series
    - **Fractal Dimension**: Quantifies the complexity and roughness of price movements
    - **Self-Similarity**: Identifies repeating patterns across different time scales
    - **Rescaled Range Analysis**: Detects cycles and persistence in the data
    """)
    
    st.subheader("Further Reading")
    st.markdown("""
    - Mandelbrot, B. (1997). Fractals and Scaling in Finance
    - Peters, E. (1994). Fractal Market Analysis
    - Taleb, N. N. (2007). The Black Swan
    - Peitgen, H.O., et al. (2004). Chaos and Fractals: New Frontiers of Science
    """)

def performance_benchmark_page():
    """Page for benchmarking system performance."""
    from fractime.optimization import benchmark_system
    
    st.header("Performance Benchmarking")
    st.write("Run performance tests to identify and compare optimization strategies.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol to test", "AAPL")
        start_date = st.date_input(
            "Start date",
            datetime.now() - timedelta(days=365*5)
        ).strftime('%Y-%m-%d')
    
    with col2:
        n_steps = st.slider("Forecast steps", 10, 100, 30)
        n_paths = st.slider("Number of paths", 100, 10000, 1000)
    
    if st.button("Run Benchmark"):
        with st.spinner("Running performance benchmark..."):
            st.text("Starting benchmark - this may take a minute...")
            
            # Create placeholders for results
            time_info = st.empty()
            profile_info = st.expander("Detailed Profiling Results", expanded=False)
            
            # Capture stdout to display profiling results
            import io
            import sys
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            # Run the benchmark
            stats = benchmark_system(symbol, start_date, n_steps, n_paths)
            
            # Reset stdout and get captured output
            sys.stdout = old_stdout
            profile_output = new_stdout.getvalue()
            
            # Display results
            with time_info:
                st.subheader("Benchmark Results")
                st.text(profile_output.split("\nTop 20")[0])  # Just the timing info
                
            with profile_info:
                st.text(profile_output)

if __name__ == "__main__":
    main() 