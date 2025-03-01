# Explanations page for FracTime - detailed information about methods

import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="FracTime Explanations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title and introduction
st.title("Fractal Market Analysis - Detailed Explanations")
st.markdown("""
This page provides in-depth explanations of the theoretical foundations and practical
applications of the fractal analysis methods used in FracTime.
""")

# Sidebar with table of contents
st.sidebar.title("Contents")
section = st.sidebar.radio(
    "Navigate to:",
    ["Fractal Market Theory", "Hurst Exponent & Fractal Dimension", "Quantum Price Levels", 
     "Trading Time Warping", "Cross-Dimensional Analysis", "Chart Interpretation Guide", 
     "Scientific References"]
)

# Main content area
if section == "Fractal Market Theory":
    st.header("Fractal Market Theory")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### The Mandelbrot View of Markets

        Benoit Mandelbrot (1924-2010), the father of fractal geometry, fundamentally challenged 
        traditional financial models with his groundbreaking work, "The (Mis)behavior of Markets."
        
        #### Key Insights:
        
        1. **Markets are not random walks** - They exhibit self-similarity and long-memory effects.
        
        2. **Price movements follow power laws** - Not the Gaussian bell curve assumed by modern portfolio theory.
        
        3. **Risk is often underestimated** - Extreme events ("black swans") occur more frequently than predicted by normal distributions.
        
        4. **Markets have fractal structure** - Similar patterns appear at different time scales.
        
        The FracTime analysis implements these principles through several interconnected methods that identify,
        measure, and forecast based on the fractal properties of market data.
        """)
        
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Benoit_Mandelbrot%2C_TED_2010.jpg/330px-Benoit_Mandelbrot%2C_TED_2010.jpg", 
             caption="Benoit Mandelbrot - Mathematician & Economist")
        
        st.markdown("""
        *"Market prices are not the result of a dice game... Behind the bewildering, 
        seemingly random, up-and-down of the market charts, many of the same patterns repeat themselves."*
        
        — Benoit Mandelbrot
        """)
    
    st.subheader("Self-Similar Patterns")
    
    st.markdown("""
    Self-similarity is a defining characteristic of fractals. In markets, this manifests as patterns that 
    repeat at different scales - hourly charts may exhibit similar structures to weekly or monthly charts.
    
    FracTime identifies these self-similar patterns using:
    
    - **Wavelet decomposition** - Breaking down price movements across different frequencies
    - **Scale-invariant feature detection** - Finding patterns regardless of their absolute size
    - **Clustering of pattern matches** - Grouping similar historical episodes
    
    When the system identifies a pattern match between current conditions and historical data, it can 
    generate forecasts based on how those similar patterns evolved in the past.
    """)

elif section == "Hurst Exponent & Fractal Dimension":
    st.header("Hurst Exponent & Fractal Dimension")
    
    st.markdown("""
    ### Hurst Exponent
    
    The Hurst exponent (H) measures the long-term memory of a time series. It quantifies the tendency 
    of a time series to regress to the mean or cluster in a direction.
    
    #### Interpretation:
    
    - **H < 0.5**: Mean-reverting series (anti-persistent)
    - **H = 0.5**: Random walk (Brownian motion)
    - **H > 0.5**: Trending series (persistent)
    
    In financial markets:
    
    - Mean-reverting markets tend to oscillate and return to an average value
    - Trending markets tend to continue in the same direction
    - Random markets have no directional bias
    
    The Hurst exponent helps determine the appropriate forecasting strategy for a given market.
    """)
    
    st.subheader("Fractal Dimension")
    
    st.markdown("""
    Fractal dimension (D) measures the complexity or "roughness" of a time series. 
    For financial time series, it is related to the Hurst exponent by the formula D = 2 - H.
    
    #### Interpretation:
    
    - **Lower D (closer to 1)**: Smoother, more trending price movements
    - **Higher D (closer to 2)**: Choppier, more volatile price movements
    
    The fractal dimension helps determine:
    
    - Appropriate stop loss distances
    - Expected noise levels in the price signal
    - Volatility-adjusted position sizing
    
    By measuring both H and D, FracTime can classify market regimes and adjust forecasting methods accordingly.
    """)
    
    st.subheader("Calculation Methods")
    
    with st.expander("Technical Details of Calculation"):
        st.markdown("""
        FracTime calculates the Hurst exponent using the rescaled range (R/S) analysis method:
        
        1. Convert price series to returns series
        2. Divide the time series into subseries of various lengths
        3. For each subseries, calculate:
           - The range R (max - min of cumulative deviation from mean)
           - The standard deviation S
        4. Plot log(R/S) against log(subseries length)
        5. The slope of this line is the Hurst exponent
        
        The fractal dimension is then calculated as D = 2 - H.
        
        A robust implementation uses multiple lag values and employs statistical techniques to ensure 
        the stability of the estimation.
        """)

elif section == "Quantum Price Levels":
    st.header("Quantum Price Levels")
    
    st.markdown("""
    ### Quantum Mechanics Meets Finance
    
    Quantum Price Levels (QPLs) apply concepts from quantum mechanics to identify significant price levels 
    in financial markets. This approach models price dynamics as a quantum system where:
    
    - Price levels behave like energy states in a quantum system
    - Price tends to be attracted to certain discrete levels
    - The probability of price occupying different levels can be calculated
    
    The key insight is that financial markets, like quantum systems, exhibit quantized behavior rather than
    purely continuous changes.
    """)
    
    st.subheader("The Quantum Price Model")
    
    st.markdown("""
    FracTime's quantum price level model works through the following process:
    
    1. **Create a Quantum Potential Function**
       - Convert price density into a quantum potential landscape
       - Areas where price spends more time become potential "wells"
       - Rarely visited price zones become potential "barriers"
       
    2. **Solve the Schrödinger Equation**
       - The fundamental equation of quantum mechanics
       - Applied to the price potential function
       - Yields energy states (eigenvalues) and wave functions (eigenvectors)
       
    3. **Extract Quantum Price Levels**
       - Each energy state corresponds to a quantum price level
       - Wave function peaks indicate where price is most likely to stabilize
       - Wave function width represents uncertainty in the price level
       
    4. **Calculate Support/Resistance Strength**
       - Probability amplitudes determine level strength
       - Narrower wave functions indicate more precise levels
       
    This approach provides mathematically rigorous support and resistance levels based on
    the collective behavior of market participants.
    """)
    
    st.subheader("Interpreting Quantum Price Levels")
    
    st.markdown("""
    In the quantum price level visualization:
    
    - **Horizontal Purple Lines**: The quantum price levels
    - **Line Thickness**: Indicates strength of the level
    - **Purple Bands**: Uncertainty range around each level
    - **Potential Function**: Shows the energy landscape of price
    - **Wave Functions**: Show probability distributions for each level
    - **Strength Metrics**: Numeric values for level significance
    
    These levels often align with traditional support/resistance but have several advantages:
    
    - Mathematically derived rather than subjectively drawn
    - Include quantified uncertainty bands
    - Provide probability measures for price interaction
    - Account for the quantum-like behavior of markets
    """)

elif section == "Trading Time Warping":
    st.header("Trading Time Warping")
    
    st.markdown("""
    ### Time is Not Uniform in Markets
    
    Traditional financial analysis treats time as flowing at a constant rate. However, market activity varies dramatically:
    
    - High volatility periods contain more "market time" per calendar day
    - Low volatility periods contain less "market time" per calendar day
    
    Trading Time Warping implements Mandelbrot's insight that market time flows at variable rates depending on volatility and trading activity.
    """)
    
    st.subheader("How Trading Time Warping Works")
    
    st.markdown("""
    FracTime implements trading time warping through the following process:
    
    1. **Measuring Time Dilation**
       - Calculate local volatility over multiple timeframes
       - Map volatility to time dilation factors
       - Higher volatility = faster time flow
       
    2. **Creating the Time Map**
       - Transform calendar time into "trading time"
       - Stretches periods of high activity
       - Compresses periods of low activity
       
    3. **Forecasting in Trading Time**
       - Generate paths in trading time space
       - Apply inverse transformation to calendar time
       - Account for expected future volatility changes
    
    This approach allows the model to generate more accurate forecasts by properly accounting for the non-uniform flow of market time.
    """)
    
    st.subheader("Interpreting the Time Warping Visualization")
    
    st.markdown("""
    In the Trading Time Warping visualization:
    
    - **Red Areas**: Time flows faster (high volatility periods)
    - **Blue Areas**: Time flows slower (low volatility periods)
    - **Green Areas**: Normal time flow (average volatility)
    - **Dilation Factors**: Quantify how much faster/slower time flows
    
    Applications of trading time warping:
    
    - More accurate pattern matching across different volatility regimes
    - Improved forecast timing
    - Better volatility clustering models
    - More realistic path simulations
    """)

elif section == "Cross-Dimensional Analysis":
    st.header("Cross-Dimensional Fractal Analysis")
    
    st.markdown("""
    ### Beyond Price: Multi-Dimensional Fractals
    
    Markets are not one-dimensional - they have multiple interrelated dimensions such as:
    
    - Price
    - Volume
    - Volatility
    - Open interest
    - Orderflow metrics
    
    Cross-Dimensional Analysis examines how fractal patterns relate across these different dimensions,
    providing a more complete picture of market structure.
    """)
    
    st.subheader("Dimensional Relationships")
    
    st.markdown("""
    FracTime's cross-dimensional analysis examines:
    
    1. **Fractal Coherence**
       - How closely aligned fractal patterns are across dimensions
       - High coherence = stable market structure
       - Low coherence = unstable or transitioning market
       
    2. **Regime Identification**
       - Classifies market states based on multi-dimensional properties
       - Identifies trending, mean-reverting, or random regimes
       - Detects regime changes earlier than single-dimension analysis
       
    3. **Correlation Dynamics**
       - Measures relationship strength between dimensions
       - Identifies when correlations are breaking down
       - Detects divergences between dimensions
    """)
    
    st.subheader("Interpreting Cross-Dimensional Analysis")
    
    st.markdown("""
    In the cross-dimensional visualization:
    
    - **Scatter Plot**: Shows relationship between price and volume
    - **Regime Metrics**: Identifies current market regime
    - **Coherence Value**: Measures alignment of fractal patterns
    - **Correlation**: Quantifies linear relationship strength
    
    Applications:
    
    - Earlier detection of regime changes
    - More robust path filtering
    - Divergence identification
    - Improved market state classification
    """)

elif section == "Chart Interpretation Guide":
    st.header("Chart Interpretation Guide")
    
    st.markdown("""
    This comprehensive guide explains how to interpret each chart, visualization, and metric 
    in the Analysis page. Use this guide to get the most insights from your FracTime analysis.
    """)
    
    # Summary metrics interpretation
    st.subheader("Summary Metrics")
    
    st.markdown("""
    At the top of each analysis, you'll find summary metrics that provide key insights at a glance:
    
    **Hurst Exponent (H):**
    - **Value Range:** 0 to 1
    - **Interpretation:**
        - **H < 0.45:** Mean-reverting market. Price tends to revert to a mean value. Look for overbought/oversold opportunities.
        - **H = 0.45-0.55:** Random walk. Price moves randomly without persistent trends. Consider range-trading strategies.
        - **H > 0.55:** Trending market. Price tends to continue in the same direction. Look for trend-following opportunities.
    - **Action Items:**
        - In trending markets (H > 0.55), trend-following strategies work better
        - In mean-reverting markets (H < 0.45), range-trading and reversal strategies work better
        - In random markets (H ≈ 0.5), look for other factors or consider non-directional strategies
    
    **Fractal Dimension (D):**
    - **Value Range:** 1 to 2
    - **Interpretation:**
        - **Lower D (≈ 1):** Smoother price movement with clearer trends
        - **Higher D (≈ 2):** Choppier, more volatile price movement
    - **Action Items:**
        - Higher D requires wider stops and more filtering to avoid whipsaws
        - Lower D allows for tighter stops and more aggressive entries
    
    **Historical Volatility:**
    - **Interpretation:** Annualized standard deviation of returns
    - **Action Items:**
        - Use for position sizing - higher volatility requires smaller positions
        - Adjust stop distances based on volatility
    
    **Identified Patterns:**
    - Number of self-similar fractal patterns found in historical data
    - More patterns generally means more reliable forecasting
    """)
    
    # Price forecast chart interpretation
    st.subheader("Price Forecast & Simulation Paths")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        The price forecast chart shows predicted price movement based on fractal analysis:
        
        **Blue Line:**
        - Historical price data used for analysis
        
        **Colored Cloud Area:**
        - Probability density of future price paths
        - Darker colors indicate higher probability areas
        - Lighter colors indicate lower probability areas
        
        **Bright Lines Within Cloud:**
        - High-probability individual price paths
        - When multiple bright paths converge, it indicates a strong probability zone
        
        **Red Line (if present):**
        - Most likely path based on quantum price level analysis
        - Represents the highest probability trajectory
        
        **Vertical Patterns:**
        - Vertical concentrations (narrow bands) indicate potential support/resistance
        - Vertical spreads (wide bands) indicate high uncertainty periods
        """)
        
    with col2:
        st.markdown("""
        **How to Use This Chart:**
        
        1. Identify high-probability zones (darker areas)
        2. Note where multiple paths converge
        3. Look for vertical concentrations of probability
        4. Compare to quantum price levels to confirm support/resistance
        5. Use for target setting and stop placement
        6. Consider position sizing based on forecast confidence
        """)
    
    st.markdown("""
    **Additional Notes:**
    - The darker the area, the higher the probability of price visiting that region
    - When the cloud narrows, it indicates lower uncertainty and higher confidence
    - When the cloud widens, it indicates higher uncertainty and lower confidence
    - Pay attention to where bright paths and dark regions align with quantum price levels
    """)
    
    # Quantum Price Level chart interpretation
    st.subheader("Quantum Price Level Chart")
    
    st.markdown("""
    The Quantum Price Level (QPL) visualization has four main components:
    
    **1. Price with Quantum Levels (Top Left):**
    - **Blue Line:** Historical price data
    - **Dashed Purple Lines:** Quantum price levels
    - **Line Thickness:** Indicates strength of the level (thicker = stronger)
    - **Purple Bands:** Uncertainty range around each level
    - **Future Paths:** If displayed, show potential price trajectories
    - **Red Line (if present):** Most likely price path
    
    **Interpretation:**
    - Quantum price levels act as support/resistance
    - Thicker lines indicate stronger support/resistance
    - Wider bands indicate less precise levels
    - When price approaches a level, expect potential bounces or slower transitions through the level
    
    **2. Quantum Potential Function (Top Right):**
    - **Purple Line:** Shows the quantum potential landscape
    - **Red Circle:** Current price position in the potential
    - **Green Stars:** Quantum price level positions
    
    **Interpretation:**
    - Valleys (low points) in the potential are "attractive" price zones
    - Hills (high points) are "repulsive" price zones
    - Price tends to spend more time in valleys and move quickly through hills
    - Current price position shows whether price is in a stable (valley) or unstable (hill/slope) zone
    
    **3. Price Level Wave Functions (Bottom Left):**
    - **Different Colored Lines:** Wave functions for different quantum states
    - **Wave Peaks:** Most likely price positions for each quantum state
    
    **Interpretation:**
    - Taller wave function peaks indicate higher probability
    - Narrower wave functions indicate more precise price levels
    - Where wave functions overlap, price can "tunnel" between levels
    
    **4. Price Level Strength (Bottom Right):**
    - **Bar Chart:** Shows the relative strength of each quantum price level
    
    **Interpretation:**
    - Taller bars indicate stronger levels
    - Compare strengths to prioritize which levels are most significant
    - Stronger levels are more likely to cause reversals or consolidations
    """)
    
    # Quantum Price Level Table Interpretation
    st.subheader("Quantum Price Level Table")
    
    st.markdown("""
    The Quantum Price Level table provides numerical data for each level:
    
    **Level:** Identification number for the quantum price level
    
    **Price:** The exact price of the quantum level
    
    **Width:** Uncertainty range around the level
    - Wider width = less precise level
    - Narrower width = more precise level
    
    **Strength:** Indicates how powerful the level is as support/resistance
    - **Value Range:** 0 to 1
    - Higher strength = more likely to cause reversals or consolidation
    - Lower strength = more likely to be broken
    
    **Probability:** The likelihood of price interacting with this level
    - **Value Range:** 0 to 1
    - Higher probability = more likely to be visited
    - Lower probability = less likely to be visited
    
    **Using the Table:**
    - Sort levels by strength to identify the most significant support/resistance
    - Compare current price to level prices to identify nearby support/resistance
    - Use width as a buffer zone around the level
    - Combine strength and probability to prioritize which levels to focus on
    """)
    
    # Trading Time Warping Chart Interpretation
    st.subheader("Trading Time Warping Chart")
    
    st.markdown("""
    The Trading Time Warping chart shows how market time flows at different rates:
    
    **Price Line:** Historical price data
    
    **Color Scale:**
    - **Red Areas:** Time flows faster (high volatility periods)
    - **Blue Areas:** Time flows slower (low volatility periods)
    - **Green Areas:** Normal time flow (average volatility)
    
    **Dilation Metrics:**
    - **Average Time Dilation:** Overall time flow rate compared to calendar time
    - **Maximum Time Dilation:** Fastest time flow rate in the dataset
    - **Minimum Time Dilation:** Slowest time flow rate in the dataset
    
    **Interpretation:**
    - Red areas are high-activity periods where more "market time" passes per calendar day
    - Blue areas are low-activity periods where less "market time" passes per calendar day
    - Transitions from blue to red often mark breakouts or increased volatility
    - Transitions from red to blue often mark consolidation or decreased volatility
    
    **Trading Applications:**
    - Expect more rapid price movements in red areas
    - Expect slower price movements in blue areas
    - Adjust holding periods based on expected time flow
    - Use time dilation to adjust forecast timing
    - Look for regime changes at color transitions
    """)
    
    # Cross-Dimensional Analysis Chart Interpretation
    st.subheader("Cross-Dimensional Analysis Chart")
    
    st.markdown("""
    The Cross-Dimensional Analysis examines relationships between price and volume:
    
    **Price-Volume Scatter Plot:**
    - Each point represents a trading day
    - X-axis: Price
    - Y-axis: Volume
    - Hover data: Date information
    
    **Regime Metrics:**
    - **Market Regime:** Identifies the current market state
      - Trending: Price tends to continue in the same direction
      - Mean-Reverting: Price tends to revert to a mean value
      - Random Walk: Price moves without directional bias
    
    - **Price-Volume Coherence:** Measures alignment of fractal patterns
      - **Value Range:** 0 to 1
      - Higher values indicate stronger fractal relationship
      - Lower values indicate weaker fractal relationship
    
    - **Price-Volume Correlation:** Linear relationship between price and volume
      - **Value Range:** -1 to 1
      - Positive: Volume tends to increase with price increases
      - Negative: Volume tends to increase with price decreases
      - Near zero: No consistent relationship
    
    **Pattern Interpretation:**
    - Positive slope trend: Rising prices with rising volume (typically bullish)
    - Negative slope trend: Falling prices with rising volume (typically bearish)
    - Horizontal bands: Price consolidation at different volume levels
    - Vertical bands: Volume consolidation at different price levels
    - Clusters: Stability zones where price and volume relationship is consistent
    - Outliers: Unusual price-volume relationships, often significant events
    
    **Trading Applications:**
    - Use regime identification to select appropriate strategies
    - Look for divergences between price and volume fractal patterns
    - Higher coherence suggests more stable market conditions
    - Lower coherence suggests changing market conditions
    - Compare current position to historical clusters
    """)
    
    # Self-Similar Patterns Interpretation
    st.subheader("Self-Similar Patterns")
    
    st.markdown("""
    This section identifies recurring fractal patterns in the price history:
    
    **Number of Patterns:** Quantity of self-similar patterns identified
    
    **Pattern Significance:**
    - More patterns generally leads to more reliable forecasting
    - Patterns are matched across different time scales
    - Current market conditions are compared to historical patterns
    - Forecast paths are influenced by how similar patterns evolved in the past
    
    **Trading Applications:**
    - Higher pattern count suggests more reliable forecasting
    - Consider how similar historical patterns evolved
    - Look for common outcomes across multiple pattern matches
    - Use pattern recognition to anticipate potential market behavior
    - More patterns with similar outcomes increase forecast confidence
    """)
    
    # Download options
    st.subheader("Download Options")
    
    st.markdown("""
    The Analysis page provides two download options:
    
    **Historical Data CSV:**
    - Contains the date and price data used in the analysis
    - Useful for further analysis in other software
    
    **Forecast Paths CSV:**
    - Contains future dates and price projections
    - Includes up to 5 different forecast paths
    - Useful for quantitative target setting and scenario analysis
    
    **How to Use Downloaded Data:**
    - Import into spreadsheets or other analysis tools
    - Create custom visualizations
    - Perform additional calculations
    - Integrate with other trading systems
    - Develop custom risk models
    """)
    
    # Summary and Next Steps
    st.subheader("Putting It All Together")
    
    st.markdown("""
    **Step-by-Step Analysis Process:**
    
    1. **Check Summary Metrics**
       - Determine market regime (trending, mean-reverting, or random)
       - Note volatility and complexity (fractal dimension)
       
    2. **Review Price Forecast**
       - Identify high-probability areas (darker regions)
       - Note path convergence points
       
    3. **Examine Quantum Price Levels**
       - Identify significant support/resistance levels
       - Note their strength and width
       
    4. **Consider Trading Time**
       - Note periods of faster/slower time flow
       - Adjust timing expectations accordingly
       
    5. **Analyze Cross-Dimensional Relationships**
       - Check price-volume relationship
       - Note coherence and correlation metrics
       
    6. **Synthesize Insights**
       - Look for confirmation across different analyses
       - Prioritize insights with multiple confirmations
       - Develop a coherent market view integrating all elements
    """)
    
    # Example Analysis walkthrough
    with st.expander("Example Analysis Walkthrough"):
        st.markdown("""
        **Example: S&P 500 Analysis**
        
        **1. Summary Metrics**
        - Hurst Exponent: 0.58 → Trending market
        - Fractal Dimension: 1.42 → Moderate complexity
        - Volatility: 15% → Moderate volatility
        - Patterns: 12 → Good number of historical patterns
        
        **2. Price Forecast**
        - Dark cloud concentration at 4500-4600 range
        - Multiple paths converging around 4550
        - Wider dispersion after 4600, indicating uncertainty
        
        **3. Quantum Price Levels**
        - Strong level at 4480 (0.82 strength)
        - Moderate level at 4620 (0.65 strength)
        - Current price between these levels
        
        **4. Trading Time**
        - Recent red zone indicating accelerated time
        - Moving toward green (normal) time flow
        - Average dilation of 1.2x
        
        **5. Cross-Dimensional**
        - Trending regime
        - High coherence (0.78)
        - Positive price-volume correlation (0.42)
        
        **6. Integrated Insights**
        - Market in uptrend (Hurst > 0.5) with strong resistance at 4620
        - Current price likely to continue toward 4550 (path convergence)
        - Time flow normalizing, suggesting steady rather than rapid movement
        - Good price-volume relationship supports trend continuation
        - Key decision points: support at 4480, resistance at 4620
        """)

elif section == "Scientific References":
    st.header("Scientific References")
    
    st.markdown("""
    ### Foundational Works
    
    **Fractal Market Analysis:**
    
    - Mandelbrot, B. (1997). *Fractals and Scaling in Finance*. Springer.
    - Peters, E. (1994). *Fractal Market Analysis: Applying Chaos Theory to Investment and Economics*. Wiley.
    - Mandelbrot, B. & Hudson, R.L. (2004). *The (Mis)behavior of Markets: A Fractal View of Financial Turbulence*. Basic Books.
    
    **Quantum Finance:**
    
    - Baaquie, B.E. (2007). *Quantum Finance: Path Integrals and Hamiltonians for Options and Interest Rates*. Cambridge University Press.
    - Khrennikov, A. (2010). *Ubiquitous Quantum Structure: From Psychology to Finance*. Springer.
    - Haven, E. & Khrennikov, A. (2013). *Quantum Social Science*. Cambridge University Press.
    
    **Trading Time & Volatility:**
    
    - Müller, U.A., Dacorogna, M.M., et al. (1990). *Statistical study of foreign exchange rates, empirical evidence of a price change scaling law, and intraday analysis*. Journal of Banking & Finance, 14(6), 1189-1208.
    - Dacorogna, M.M., Gençay, R., et al. (2001). *An Introduction to High-Frequency Finance*. Academic Press.
    - Mandelbrot, B. (1997). *Multifractals and 1/ƒ Noise: Wild Self-Affinity in Physics*. Springer.
    """)
    
    st.subheader("Advanced Topics")
    
    with st.expander("Further Reading on Specific Methods"):
        st.markdown("""
        **Hurst Exponent & Fractal Dimension:**
        
        - Hurst, H.E. (1951). *Long-term storage capacity of reservoirs*. Transactions of the American Society of Civil Engineers, 116, 770-808.
        - Di Matteo, T. (2007). *Multi-scaling in finance*. Quantitative Finance, 7(1), 21-36.
        - Mandelbrot, B. (1967). *How long is the coast of Britain? Statistical self-similarity and fractional dimension*. Science, 156(3775), 636-638.
        
        **Quantum Price Level Generation:**
        
        - Sornette, D. (1998). *Discrete scale invariance and complex dimensions*. Physics Reports, 297(5), 239-270.
        - Baaquie, B.E. (2004). *Quantum Finance: Path Integrals and Hamiltonians for Options*. Cambridge University Press.
        - Haven, E. (2002). *A discussion on embedding the Black-Scholes option pricing model in a quantum physics setting*. Physica A, 304(3-4), 507-524.
        
        **Trading Time Warping:**
        
        - Clark, P.K. (1973). *A subordinated stochastic process model with finite variance for speculative prices*. Econometrica, 41(1), 135-155.
        - Ané, T., & Geman, H. (2000). *Order flow, transaction clock, and normality of asset returns*. The Journal of Finance, 55(5), 2259-2284.
        - Mandelbrot, B., & Taylor, H.M. (1967). *On the distribution of stock price differences*. Operations Research, 15(6), 1057-1062.
        
        **Cross-Dimensional Analysis:**
        
        - Calvet, L.E., & Fisher, A.J. (2001). *Forecasting multifractal volatility*. Journal of Econometrics, 105(1), 27-58.
        - Schmitt, F., Schertzer, D., & Lovejoy, S. (1999). *Multifractal analysis of foreign exchange data*. Applied Stochastic Models and Data Analysis, 15(1), 29-53.
        - Kantelhardt, J.W., et al. (2002). *Multifractal detrended fluctuation analysis of nonstationary time series*. Physica A, 316(1-4), 87-114.
        """)

    st.subheader("Software Implementation")
    
    st.markdown("""
    FracTime implements these advanced concepts using modern numerical methods:
    
    - **Numerical Integration**: Solving Schrödinger equation for quantum levels
    - **Wavelet Analysis**: Pattern detection across multiple scales
    - **Monte Carlo Simulation**: Path generation with fractal properties
    - **Eigenvalue Decomposition**: Finding quantum states in price potential
    - **Kernel Density Estimation**: Creating smooth potential functions
    - **Time Series Transformations**: Converting between calendar and trading time
    
    For technical details on implementation, refer to the codebase documentation.
    """)

# Footer with navigation and resources
st.markdown("---")
st.markdown("""
**Navigation:** [Home](/Home) | [Analysis](/Analysis) | [GitHub](https://github.com/anthropics/claude-code)
""")

if st.button("← Back to Analysis", use_container_width=True):
    st.switch_page("pages/01_Analysis.py")