# FracTime

FracTime is an advanced time series forecasting tool that leverages fractal geometry and chaos theory principles for market analysis and prediction.

## Features

- **Pattern Recognition**
  - Hurst exponent calculation (R/S analysis)
  - Fractal dimension estimation
  - Self-similarity detection
  - Multi-scale price movement analysis

- **Advanced Simulation**
  - Fractional Brownian motion via spectral synthesis
  - Pattern-weighted path generation
  - Non-parametric bootstrapping with recency bias
  - Regime detection using HMMs

- **Path Analysis**
  - K-means clustering of trajectories
  - Probability density estimation
  - Representative path selection
  - Empirical confidence intervals

## Quick Start

FracTime uses `uv` for dependency management:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"

# Run the Streamlit application
streamlit run Home.py
```

The application will be available at http://localhost:8501 in your browser.

## App Structure

The FracTime app consists of three main pages:

1. **Home**: Introduction and overview of capabilities
2. **Analysis**: Interactive analysis tools and visualizations
3. **Explanations**: Detailed explanations of the theory and how to interpret results

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run a specific test
pytest tests/test_core.py::test_sample_data
```

## Development

```bash
# Formatting
black fractime/ tests/

# Linting
ruff check fractime/ tests/

# Type checking
mypy fractime/
```

## Requirements

- Python >=3.11
- Modern web browser for interactive visualizations
- Internet connection for real-time market data

## License

MIT

## Theoretical Framework

### The Fractal Market Hypothesis

FracTime is built on the Fractal Market Hypothesis which challenges traditional market theories by recognizing that:

1. Markets have memory (long-range dependence)
2. Price movements exhibit self-similarity across time scales
3. Returns follow power-law distributions rather than Gaussian distributions
4. Volatility clusters in a scale-invariant manner

### Trading Time vs. Clock Time

Following Mandelbrot's insights, FracTime implements the concept that markets operate on their own internal time scale:

- Market time "speeds up" during high volatility/volume periods
- The warping of time creates a non-linear relationship between clock time and trading time
- When prices are resampled to uniform trading time, their statistical properties become more tractable

### Multi-Scale Analysis

Our methodology analyzes financial time series through multiple components:

- **Fractal Pattern Recognition**: Identifies self-similar patterns using Hurst exponent calculation and normalized cross-correlation
- **Scaling Analysis**: Quantifies how volatility scales with time (σ(t) ~ t^H) and measures self-similarity across scales
- **Regime Identification**: Uses Hidden Markov Models to identify distinct fractal regimes and predict transitions between them
- **Distribution Modeling**: Forecasts entire probability distributions conditional on the current fractal regime

### Applications

This approach enables novel analytical capabilities:

- Scale-dependent analysis across different time horizons
- Regime-specific forecasting based on fractal patterns
- Trading time optimization for strategy development
- Improved fat-tail risk modeling for extreme events

By integrating these concepts, FracTime offers a more nuanced understanding of market dynamics than traditional forecasting methods, particularly during periods of high volatility and regime transitions.

## Advanced Theoretical Concepts

### Multi-Dimensional Fractal Analysis

FracTime extends beyond one-dimensional time series analysis by incorporating:

- **Multi-dimensional fractal equations** that simulate market behavior across multiple factors simultaneously
- Cross-correlation analysis between different fractal dimensions
- Factor-based decomposition of market movements using fractal geometry
- Higher-dimensional attractors that capture complex market dynamics

### Quantum Finance Applications

Drawing from quantum mechanics principles applied to finance:

#### Quantum Price Levels (QPLs)

The framework implements Quantum Price Levels (QPLs) from Chapter 10 of Quantum Finance theory:

- **Quantum Finance Schrödinger Equation (QFSE)** is used to derive price levels that act as support and resistance
- QPLs represent probabilistic price barriers with quantum properties
- Market prices tend to stabilize around these quantum levels during specific regimes

#### Computational Methods

The implementation utilizes advanced numerical methods:

- **Finite difference methods (FDM)** to solve the QFSE for practical application
- **Path integral techniques** that connect quantum mechanics to path simulation
- Discretization approaches that balance computational efficiency with model accuracy

#### Practical Applications

These quantum finance concepts enhance practical market analysis by:

- Providing more robust support/resistance levels based on quantum probabilities
- Identifying price regions with higher probability of reversal or continuation
- Quantifying the energy required for price to break through quantum barriers
- Predicting regime changes using quantum tunneling analogies

By integrating these advanced theoretical concepts, FracTime provides a unified framework that connects classical fractal analysis with quantum finance principles for a more comprehensive understanding of market behavior.