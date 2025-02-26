import streamlit as st
import numpy as np
import polars as pl
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from fractime import FractalAnalyzer, FractalSimulator, PathAnalyzer, FractalVisualizer, get_yahoo_data

def analyze_symbol(symbol: str, start_date: str) -> dict:
    """Analyze a single symbol."""
    try:
        data = get_yahoo_data(symbol, start_date)
        prices = data['Close'].to_numpy()
        dates = data['Date'].to_numpy()  # Get dates
        
        analyzer = FractalAnalyzer()
        simulator = FractalSimulator(prices, analyzer)
        
        # Get analysis results
        analysis_results = analyzer.analyze_patterns(prices)
        
        # Get simulation results - now returns tuple of (paths, path_analysis)
        paths, path_analysis = simulator.simulate_paths(n_steps=30, n_paths=1000)
        
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

def main():
    st.title("Fractal Time Series Analysis")
    
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
    n_paths = st.sidebar.slider("Number of Simulation Paths", 100, 10000, 1000)
    n_steps = st.sidebar.slider("Forecast Steps", 10, 100, 30)
    
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
            result = analyze_symbol(symbol, start_date)
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
                # Create visualization with new structure
                visualizer = FractalVisualizer()
                fig = visualizer.plot_analysis_and_forecast(
                    result['prices'],
                    result['simulation_results'],
                    result['analysis_results'],
                    result['dates']  # Pass dates to visualization
                )
                st.plotly_chart(fig, use_container_width=True)
                
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

if __name__ == "__main__":
    main() 