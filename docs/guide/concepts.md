# Core Concepts

Understanding the fractal approach to time series forecasting.

## Fractal Geometry in Time Series

Fractal geometry, pioneered by Benoit Mandelbrot, describes patterns that exhibit self-similarity across scales. Financial time series often display fractal characteristics:

- Price movements look similar whether viewed hourly, daily, or monthly
- Volatility clusters persist across time scales
- Extreme events occur more frequently than Gaussian models predict

## The Hurst Exponent

The Hurst exponent (H) is the fundamental measure of long-term memory in a time series.

| H Value | Interpretation | Market Behavior |
|---------|----------------|-----------------|
| H > 0.5 | Persistent (trending) | Trends tend to continue |
| H = 0.5 | Random walk | No predictable pattern |
| H < 0.5 | Anti-persistent (mean-reverting) | Movements tend to reverse |

```python
import fractime as ft

analyzer = ft.FractalAnalyzer()
hurst = analyzer.compute_hurst(prices)
```

## Fractal Dimension

The fractal dimension measures the complexity or "roughness" of the time series:

- Lower dimension: Smoother, more trending series
- Higher dimension: Rougher, more volatile series

```python
dim = analyzer.compute_fractal_dimension(prices)
```

## How FracTime Forecasting Works

### 1. Fractal Analysis

FracTime first analyzes the series to estimate:
- Hurst exponent
- Fractal dimension
- Volatility structure

### 2. Fractional Brownian Motion

Forecasts are generated using Fractional Brownian Motion (fBm), which extends standard Brownian motion to incorporate long-term memory:

- Standard Brownian motion assumes H = 0.5 (random walk)
- fBm uses the estimated Hurst exponent to preserve memory characteristics

### 3. Path Simulation

Multiple future scenarios are simulated, each respecting the fractal structure of the historical data.

### 4. Probability Weighting

Each simulated path receives a probability weight based on:
- Hurst consistency with historical data
- Volatility pattern matching
- Multi-scale trend alignment

### 5. Uncertainty Quantification

The weighted paths provide:
- Point forecasts (probability-weighted mean)
- Confidence intervals
- Full probability distributions

## Regime Detection

Markets shift between different regimes (trending vs. mean-reverting). FracTime can detect these shifts:

```python
analysis = analyzer.analyze(prices)
print(f"Current Hurst: {analysis['hurst']:.3f}")
print(f"Regime: {analysis['regime']}")
```

## Cross-Dimensional Analysis

For multiple related time series:

```python
from fractime import CrossDimensionalAnalyzer

cda = CrossDimensionalAnalyzer()
cda.add_dimension('Stock A', prices_a)
cda.add_dimension('Stock B', prices_b)

correlation = cda.compute_cross_correlation()
hurst_values = cda.compute_hurst_exponents()
```
