# Standard library imports
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import warnings
import logging
import os
import sys

# Completely silence Streamlit warnings - must be set before importing streamlit
os.environ['STREAMLIT_LOG_LEVEL'] = 'critical'
logging.root.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Redirect stdout to suppress specific warnings
class WarningFilter:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        
    def write(self, message):
        # Filter out the specific warning messages
        if "missing ScriptRunContext" not in message and "run it with the following command" not in message:
            self.original_stdout.write(message)
            
    def flush(self):
        self.original_stdout.flush()

sys.stdout = WarningFilter(sys.stdout)

# Third-party library imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st
import yfinance as yf

# Set page configuration
st.set_page_config(
    page_title="FracTime - Home",
    page_icon="🏠",
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
    MultidimensionalFractalAnalyzer,
    QuantumPriceLevelGenerator,
    demo_quantum_fractal_analysis
)

# Clear the cache to ensure we get fresh data
st.cache_data.clear()
st.cache_resource.clear()

# Home page content
st.title("FracTime: Fractal Time Series Analysis")

# Hero section with illustration
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ### Advanced Market Analysis with Fractal Geometry

    FracTime applies Mandelbrot's fractal mathematics to financial markets, combining:
    
    * **Fractal Pattern Recognition** - Identifying self-similar market structures
    * **Quantum Price Levels** - Support/resistance from quantum mechanics principles
    * **Trading Time Warping** - Time flows at different rates based on volatility
    * **Cross-Dimensional Analysis** - Correlating price and volume fractal behaviors

    Navigate to the **Analysis page** to generate comprehensive market insights, or visit the 
    **Explanations page** for in-depth information on the methods used.
    """)
    
    st.info("""
    **Getting Started:**
    1. Go to the Analysis page (sidebar)
    2. Select a ticker symbol (e.g., ^GSPC for S&P 500)
    3. Set simulation parameters
    4. Generate your analysis
    """)

with col2:
    # Display a fractal image
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Mandel_zoom_00_mandelbrot_set.jpg/640px-Mandel_zoom_00_mandelbrot_set.jpg", 
             caption="Mandelbrot Set - The foundational fractal")
    
    # Navigation buttons
    st.markdown("### Quick Navigation")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📊 Analysis", use_container_width=True):
            st.switch_page("pages/01_Analysis.py")
    with col_b:
        if st.button("📚 Explanations", use_container_width=True):
            st.switch_page("pages/02_Explanations.py")

# About section
st.markdown("---")
st.markdown("""
## About FracTime

FracTime is built on the pioneering work of Benoit Mandelbrot, who discovered that financial markets 
follow fractal patterns rather than the random walks proposed by traditional financial theory.

### Key Innovations:

* **Scale-Invariant Pattern Recognition** - Patterns repeat at different timeframes
* **Non-Normal Distribution Modeling** - Accounts for fat tails and extreme events
* **Wavelet-Based Feature Extraction** - Captures both time and frequency domain
* **Quantum-Inspired Support/Resistance** - Applies quantum mechanics to price barriers

For more information on the scientific foundations, visit the Explanations page.
""")

def analyze_symbol(
    symbol: str, 
    start_date: str, 
    n_paths: int = 1000, 
    n_steps: int = 30,
    use_trading_time: bool = True,
    warping_alpha: float = 0.5,
    enable_time_forecast: bool = True,
    use_cross_dim: bool = False,
    use_quantum_levels: bool = False,
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
    # Sidebar - Input Parameters
    st.sidebar.header("Analysis Parameters")
    
    # Data Source Selection
    input_method = st.sidebar.radio(
        "Data Source",
        ["Yahoo Finance", "CSV Upload"]
    )
    
    # Date Range
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365)
    ).strftime('%Y-%m-%d')
    
    # Symbol or File Input
    if input_method == "Yahoo Finance":
        symbols_input = st.sidebar.text_area(
            "Enter symbols (one per line)",
            "^GSPC"
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
    
    # Simulation Parameters
    st.sidebar.header("Simulation Settings")
    
    n_paths = st.sidebar.slider(
        "Number of Simulation Paths", 
        min_value=100, 
        max_value=10000, 
        value=1000, 
        step=100
    )
    
    n_steps = st.sidebar.slider(
        "Forecast Steps", 
        min_value=10, 
        max_value=100, 
        value=30, 
        step=5
    )
    
    # Analysis Features (all enabled by default)
    st.sidebar.header("Analysis Features")
    use_trading_time = st.sidebar.checkbox("Trading Time Warping", value=True)
    use_cross_dim = st.sidebar.checkbox("Cross-Dimensional Analysis", value=True)
    use_quantum_levels = st.sidebar.checkbox("Quantum Price Levels", value=True)
    
    # Run Analysis Button
    run_analysis = st.sidebar.button("Generate Complete Analysis", use_container_width=True)
    
    # Main content area
    if run_analysis:
        if not symbols:
            st.warning("Please enter at least one symbol or upload a CSV file")
        else:
            # Create comprehensive report
            generate_report(
                symbols, 
                start_date, 
                n_paths,
                n_steps,
                use_trading_time=use_trading_time,
                use_cross_dim=use_cross_dim,
                use_quantum_levels=use_quantum_levels
            )
    else:
        # Show instructions when not yet run
        st.info("Set your analysis parameters in the sidebar and click 'Generate Complete Analysis' to create a comprehensive report.")
        
        # Sample visualization explanation
        st.subheader("What You'll Get in the Analysis")
        
        st.markdown("""
        ### 1. Fractal Pattern Analysis
        - Hurst exponent and fractal dimension calculation
        - Self-similar patterns identification across timeframes
        - Volatility clusters detection
        
        ### 2. Price Forecasting
        - Multiple forecast paths with probability density
        - Most likely price trajectory identification
        - Trading time warping for improved accuracy
        
        ### 3. Quantum Price Levels
        - Support and resistance levels derived from quantum mechanics
        - Price level strength and significance estimations
        - Wave function visualization of price stability zones
        
        ### 4. Cross-Dimensional Analysis
        - Price-volume fractal coherence assessment
        - Market regime identification
        - Multi-dimensional attractor visualization
        """)

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
    
    # Generate the comprehensive report for each symbol
    for result in results:
        if result['status'] == 'success':
            symbol = result['symbol']
            
            # 1. Symbol Header and Basic Info
            st.header(f"Analysis Report: {symbol}")
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Hurst Exponent", f"{result['analysis_results']['hurst']:.3f}")
                h = result['analysis_results']['hurst']
                if h > 0.55:
                    st.caption("Trending market (persistent)")
                elif h < 0.45:
                    st.caption("Mean-reverting market")
                else:
                    st.caption("Random walk (efficient)")
                    
            with col2:
                st.metric("Fractal Dimension", f"{result['analysis_results']['fractal_dim']:.3f}")
                st.caption("Complexity/roughness indicator")
                
            with col3:
                st.metric("Historical Volatility", f"{result.get('quantum_levels', {}).get('volatility', 0):.2%}")
                st.caption("Annualized volatility")
                
            with col4:
                st.metric("Identified Patterns", f"{len(result['analysis_results']['self_similar_patterns'])}")
                st.caption("Self-similar structures")
            
            # 2. Price Forecast with Projected Paths
            st.subheader("Price Forecast and Simulation Paths")
            
            # Display price forecast visualization
            visualizer = FractalVisualizer()
            paths, path_analysis = result['simulation_results']
            
            # Main forecast figure
            forecast_fig = visualizer.plot_high_density_forecast(
                result['prices'],
                result['simulation_results'],
                result['analysis_results'],
                result['dates']
            )
            
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Explain the forecast
            with st.expander("**Forecast Explanation**"):
                st.markdown("""
                This forecast visualization shows:
                
                - **Probability Cloud**: The colored area shows the probability distribution of future price paths
                - **Darker Colors**: Higher probability regions
                - **Top Paths**: Several high-probability paths are highlighted
                - **Pattern Matches**: Historical patterns matching current conditions
                
                The forecast is generated from {n_paths:,} Monte Carlo simulations using the fractal properties 
                of the historical data. This is not a simple random walk - patterns from history are selected based 
                on fractal self-similarity.
                """.format(n_paths=n_paths))
            
            # Divider
            st.markdown("---")
            
            # 3. Quantum Price Levels (if enabled)
            if use_quantum_levels and 'quantum_levels' in result:
                st.subheader("Quantum Price Level Analysis")
                
                # Create quantum price level visualization
                try:
                    qpl_generator = QuantumPriceLevelGenerator()
                    qpl_generator.price_levels = result['quantum_levels']
                    
                    # Also set potential_function and wave_function if they're missing
                    if qpl_generator.potential_function is None or qpl_generator.wave_function is None:
                        # Generate a minimal potential function for visualization
                        price_range = np.max(result['prices']) - np.min(result['prices'])
                        p_min = np.min(result['prices']) - 0.1 * price_range
                        p_max = np.max(result['prices']) + 0.1 * price_range
                        price_grid = np.linspace(p_min, p_max, 100)
                        
                        # Use a simple smoothed histogram for the potential
                        from scipy import stats
                        kde = stats.gaussian_kde(result['prices'], bw_method=0.3)
                        density = kde(price_grid)
                        potential = 1.0 - density / np.max(density)
                        
                        # Store the potential function
                        qpl_generator.potential_function = (price_grid, potential)
                        
                        # Create simplified wave functions if needed
                        if qpl_generator.wave_function is None and qpl_generator.price_levels and 'levels' in qpl_generator.price_levels:
                            # Create wave functions based on levels
                            levels = qpl_generator.price_levels['levels']
                            n_levels = len(levels)
                            eigenvectors = np.zeros((len(price_grid), n_levels))
                            
                            for i, level in enumerate(levels):
                                center_idx = np.argmin(np.abs(price_grid - level['price']))
                                width_idx = max(5, int(level['width'] / (price_range / 100) / 2))
                                
                                # Create Gaussian peak centered at price level
                                for j in range(len(price_grid)):
                                    eigenvectors[j, i] = np.exp(-0.5 * ((j - center_idx) / width_idx)**2)
                                
                                # Normalize
                                if np.max(np.abs(eigenvectors[:, i])) > 0:
                                    eigenvectors[:, i] = eigenvectors[:, i] / np.max(np.abs(eigenvectors[:, i]))
                                    
                            qpl_generator.wave_function = eigenvectors
                    
                    # Get path weights if available
                    quantum_weights = None
                    if 'quantum_weights' in path_analysis and path_analysis['quantum_weights'] is not None:
                        quantum_weights = np.array(path_analysis['quantum_weights'])
                        
                    # Generate visualization
                    qpl_fig = qpl_generator.visualize_price_levels(
                        result['prices'],
                        paths,
                        result['dates'],
                        quantum_weights,
                        path_analysis.get('most_likely_path')
                    )
                    
                    st.plotly_chart(qpl_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Couldn't create quantum price level visualization: {e}")
                    
                    # Create a basic visualization instead
                    if 'quantum_levels' in result and 'levels' in result['quantum_levels']:
                        levels = result['quantum_levels']['levels']
                        
                        # Create simple visualization using plotly
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        
                        # Add price history
                        fig.add_trace(go.Scatter(
                            x=[str(d) for d in result['dates']],
                            y=result['prices'],
                            name="Price",
                            line=dict(color='blue')
                        ))
                        
                        # Add horizontal lines for each price level
                        for i, level in enumerate(levels):
                            price = level['price']
                            strength = level.get('strength', 0.5)
                            width = level.get('width', 0.0)
                            
                            # Line width based on strength
                            line_width = 1 + 3 * strength
                            
                            # Line color with opacity based on strength
                            line_color = f'rgba(128,0,128,{0.3 + 0.7 * strength})'
                            
                            # Add horizontal line
                            fig.add_trace(go.Scatter(
                                x=[str(result['dates'][0]), str(result['dates'][-1])],
                                y=[price, price],
                                name=f"QPL {i+1}: {price:.2f}",
                                line=dict(color=line_color, width=line_width, dash='dash')
                            ))
                            
                            # Add band if width is specified
                            if width > 0:
                                fig.add_trace(go.Scatter(
                                    x=[str(result['dates'][0]), str(result['dates'][-1]), str(result['dates'][-1]), str(result['dates'][0]), str(result['dates'][0])],
                                    y=[price + width/2, price + width/2, price - width/2, price - width/2, price + width/2],
                                    fill="toself",
                                    fillcolor=f'rgba(128,0,128,0.1)',
                                    line=dict(width=0),
                                    name=f"QPL Band {i+1}",
                                    showlegend=False
                                ))
                                
                        fig.update_layout(
                            title=f"Quantum Price Levels - {symbol}",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Show quantum price levels as a table
                if 'quantum_levels' in result and 'levels' in result['quantum_levels']:
                    levels = result['quantum_levels']['levels']
                    level_data = [{
                        'Level': i+1,
                        'Price': f"${level['price']:.2f}",
                        'Width': f"±${level['width']:.2f}",
                        'Strength': f"{level['strength']:.2f}",
                        'Probability': f"{level['probability']:.2f}"
                    } for i, level in enumerate(levels)]
                    
                    # Use Polars for better performance
                    levels_df = pl.DataFrame(level_data)
                    # Convert to pandas for Streamlit compatibility
                    st.table(levels_df.to_pandas())
                
                # Quantum price level explanation
                with st.expander("**Quantum Price Level Explanation**"):
                    st.markdown("""
                    **Quantum Price Level Analysis** applies quantum mechanics to identify significant price barriers:
                    
                    - **Price Levels**: Major support/resistance zones derived from quantum energy states
                    - **Width**: Uncertainty range around each level (wider = less precise)
                    - **Strength**: How significant each level is as support/resistance
                    - **Probability**: Likelihood of price interacting with this level
                    
                    The **Quantum Potential** shows how price tends to be attracted to certain levels. 
                    The **Wave Functions** represent the quantum states of the price, with peaks 
                    indicating where price is most likely to stabilize.
                    
                    This is based on solving the Schrödinger equation with a potential derived from 
                    price distribution.
                    """)
                
                # Divider
                st.markdown("---")
            
            # 4. Trading Time Analysis (if enabled)
            if use_trading_time and 'time_map' in result:
                st.subheader("Trading Time Warping Analysis")
                
                # Create trading time visualization if method exists
                try:
                    trading_time_fig = visualizer.plot_trading_time_analysis(
                        result['prices'],
                        result['time_map'],
                        result['dates']
                    )
                    
                    st.plotly_chart(trading_time_fig, use_container_width=True)
                except AttributeError:
                    st.info("Trading time visualization not available in this version.")
                
                # Display metrics about time warping
                if 'dilation_factors' in result['time_map']:
                    dilation_factors = result['time_map']['dilation_factors']
                    
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_dilation = np.mean(dilation_factors)
                        st.metric("Average Time Dilation", f"{avg_dilation:.2f}x")
                        
                    with col2:
                        max_dilation = np.max(dilation_factors)
                        st.metric("Maximum Time Dilation", f"{max_dilation:.2f}x")
                        
                    with col3:
                        min_dilation = np.min(dilation_factors)
                        st.metric("Minimum Time Dilation", f"{min_dilation:.2f}x")
                
                # Trading time explanation
                with st.expander("**Trading Time Warping Explanation**"):
                    st.markdown("""
                    **Trading Time Warping** recognizes that market time flows at different rates depending on volatility:
                    
                    - **Faster Time**: Red regions where time runs faster during high volatility
                    - **Slower Time**: Blue regions where time runs slower during low volatility
                    - **Time Mapping**: Shows how calendar time is transformed into "trading time"
                    
                    This concept is inspired by Mandelbrot's observation that financial markets have a fractal
                    dimension in both price and time. Higher volatility periods experience more "market events"
                    per calendar day, effectively speeding up trading time.
                    
                    By warping time based on volatility, forecasts can better account for periods of market stress
                    and calm.
                    """)
                
                # Divider
                st.markdown("---")
            
            # 5. Cross-Dimensional Analysis (if enabled)
            if use_cross_dim and 'cross_dim_results' in result:
                st.subheader("Cross-Dimensional Fractal Analysis")
                
                # Only show if we have volume data
                if result.get('volumes') is not None:
                    # Skip visualization or handle errors
                    try:
                        # Display a simpler alternative instead
                        st.write("**Cross-Dimensional Price-Volume Analysis:**")
                        
                        # Create a simple scatterplot
                        # Using the imports from the top of the file
                        
                        # If we have both price and volume, show their relationship
                        if len(result['prices']) == len(result['volumes']):
                            # Create dataframe for plotting using Polars
                            df = pl.DataFrame({
                                'Price': result['prices'],
                                'Volume': result['volumes'],
                                'Date': [str(d) for d in result['dates']]
                            })
                            # Convert to pandas for Plotly compatibility
                            pdf = df.to_pandas()
                            
                            # Create scatter plot of price vs volume
                            fig = px.scatter(pdf, x='Price', y='Volume', 
                                            hover_data=['Date'],
                                            title=f"Price-Volume Relationship for {symbol}",
                                            labels={'Price': 'Price', 'Volume': 'Volume'})
                            
                            # Add trendline
                            fig.update_layout(width=800, height=500)
                            
                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Price and volume data have different lengths, cannot create cross-dimensional visualization.")
                    except Exception as e:
                        # Show error in a user-friendly way
                        st.info(f"Could not create cross-dimensional visualization: {e}")
                    
                    # Display metrics
                    if result.get('cross_dim_results'):
                        # Create columns for metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            regime_info = result.get('cross_dim_results', {}).get('regime', {})
                            regime_type = regime_info.get('regime', 0)
                            regime_names = ["Trending", "Mean-Reverting", "Random Walk"]
                            regime_name = regime_names[regime_type] if regime_type < len(regime_names) else f"Regime {regime_type}"
                            st.metric("Market Regime", regime_name)
                            
                        with col2:
                            coherence = result.get('cross_dim_results', {}).get('fractal_coherence', {}).get('overall', 0)
                            st.metric("Price-Volume Coherence", f"{coherence:.2f}")
                            
                        with col3:
                            if 'cross_correlation' in result.get('cross_dim_results', {}):
                                corr_matrix = result['cross_dim_results']['cross_correlation']
                                if len(corr_matrix) >= 2 and len(corr_matrix[0]) >= 2:
                                    price_vol_corr = corr_matrix[0][1]
                                    st.metric("Price-Volume Correlation", f"{price_vol_corr:.2f}")
                
                    # Cross-dimensional explanation
                    with st.expander("**Cross-Dimensional Analysis Explanation**"):
                        st.markdown("""
                        **Cross-Dimensional Analysis** examines how fractal patterns relate across different market variables:
                        
                        - **Market Regime**: Identifies if the market is trending, mean-reverting, or random
                        - **Fractal Coherence**: Measures how closely price and volume fractal patterns align
                        - **Phase Space**: Shows the relationship between price, volume, and their changes
                        
                        When multiple market dimensions (like price and volume) exhibit coherent fractal patterns,
                        it suggests a more stable market structure. Divergence between dimensions often precedes
                        major market shifts.
                        
                        This analysis helps identify which paths are most likely based on how they maintain the
                        cross-dimensional relationships observed in the historical data.
                        """)
                else:
                    st.info("Cross-dimensional analysis requires volume data, which is not available for this symbol.")
                
                # Divider
                st.markdown("---")
            
            # 6. Self-Similar Patterns
            st.subheader("Identified Self-Similar Patterns")
            
            # Display patterns if available
            if 'self_similar_patterns' in result['analysis_results'] and result['analysis_results']['self_similar_patterns']:
                patterns = result['analysis_results']['self_similar_patterns']
                
                # Display number of patterns found
                st.write(f"Found {len(patterns)} self-similar patterns in the historical data.")
                
                # Pattern explanation
                with st.expander("**Pattern Matching Explanation**"):
                    st.markdown("""
                    **Self-Similar Pattern Analysis** identifies recurring fractal structures in the price history:
                    
                    - **Pattern Matches**: Highlighted segments show where historical patterns match current conditions
                    - **Scaling**: Patterns are identified across different timeframes (fractal self-similarity)
                    - **Forecasting**: These patterns inform the simulation by providing likely continuations
                    
                    The pattern matching algorithm finds self-similar structures regardless of their absolute scale,
                    allowing the model to learn from both short-term and long-term historical behaviors.
                    
                    This is a core feature of fractal market analysis - patterns repeat across different timescales.
                    """)
            else:
                st.info("No significant self-similar patterns were identified for this symbol.")
            
            # Divider
            st.markdown("---")
            
            # 7. Download Options
            st.subheader("Download Analysis Data")
            
            # Prepare data for download - we'll use pandas for this
            # as it's more forgiving with different length columns
            csv_data = {
                'Date': [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in result['dates']],
                'Price': result['prices'].tolist()
            }
            
            # Use pandas for flexible handling of different length columns
            download_df = pd.DataFrame(csv_data)
            
            # Column for future dates (needed for paths)
            n_dates = len(result['dates'])
            last_date = result['dates'][-1]
            
            # Create future dates
            if hasattr(last_date, 'strftime'):
                # Create date sequence if possible
                import pandas as pd_dates  # Use pandas for date handling
                future_dates = pd_dates.date_range(
                    start=last_date, periods=n_steps+1
                )[1:]  # Skip first date (last historical)
                
                # Convert to strings
                future_dates = [d.strftime("%Y-%m-%d") for d in future_dates]
            else:
                # Just use numbered forecast days
                future_dates = [f"Forecast {i+1}" for i in range(n_steps)]
                
            # Add projected paths - each in a separate dataframe
            path_dfs = []
            
            for i in range(min(5, paths.shape[0])):
                # Create path dataframe
                path_data = {
                    'Date': future_dates,
                    f'Path_{i+1}': paths[i].tolist()
                }
                path_dfs.append(pd.DataFrame(path_data))
                
            # Combine into path dataframe
            if path_dfs:
                path_df = path_dfs[0]
                for df in path_dfs[1:]:
                    path_df = path_df.merge(df, on='Date', how='outer')
                
                # Export paths separately
                path_csv = path_df.to_csv(index=False)
                
                # Add download button for paths
                st.download_button(
                    "Download Forecast Paths as CSV",
                    path_csv,
                    f"fractime_paths_{symbol}.csv",
                    "text/csv"
                )
            
            # Create download button
            st.download_button(
                "Download Historical Data as CSV",
                download_df.to_csv(index=False),
                f"fractime_history_{symbol}.csv",
                "text/csv"
            )
        else:
            # Show error message for failed analyses
            st.error(f"Analysis failed for {result['symbol']}: {result.get('error', 'Unknown error')}")
            
    # Final summary
    st.subheader("Analysis Summary")
    st.success(f"Completed analysis for {len([r for r in results if r['status'] == 'success'])} of {len(symbols)} symbols.")

# No main function needed for the home page
