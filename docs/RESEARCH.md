# Fractal Time Series Analysis: Mathematical Foundations

**Research Documentation for FracTime Framework**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Fractal Geometry in Financial Markets](#fractal-geometry-in-financial-markets)
3. [The Hurst Exponent](#the-hurst-exponent)
4. [Fractal Dimension](#fractal-dimension)
5. [Trading Time vs. Clock Time](#trading-time-vs-clock-time)
6. [Cross-Dimensional Fractal Analysis](#cross-dimensional-fractal-analysis)
7. [Pattern Recognition in Self-Similar Systems](#pattern-recognition-in-self-similar-systems)
8. [Regime-Matched Path Simulation](#regime-matched-path-simulation)
9. [Statistical Testing Framework](#statistical-testing-framework)
10. [References](#references)

---

## Introduction

Traditional financial models assume that market returns follow a Gaussian (normal) distribution with independent increments. However, empirical evidence consistently shows that financial markets exhibit:

1. **Fat tails**: Extreme events occur more frequently than predicted by normal distributions
2. **Volatility clustering**: High volatility periods cluster together
3. **Long-range dependence**: Returns show persistent correlations over long time horizons
4. **Self-similarity**: Market patterns repeat across different time scales

**Fractal geometry** provides a mathematical framework for modeling these empirically observed phenomena. Unlike traditional models based on Brownian motion, fractal models can capture the complex, scale-invariant structures inherent in financial time series.

---

## Fractal Geometry in Financial Markets

### Self-Similarity and Scale Invariance

A fractal object exhibits **self-similarity** if it appears similar at different scales of magnification. Mathematically, a time series X(t) is self-similar with Hurst parameter H if:

```
X(ct) =_d c^H X(t)
```

where =_d denotes equality in distribution, and c is a scaling constant.

For financial time series, this implies that the statistical properties of price movements over different time horizons (daily, weekly, monthly) maintain similar characteristics when appropriately rescaled.

### Mandelbrot's Multifractal Model

Mandelbrot proposed that price changes follow a **multifractal** process rather than simple Brownian motion. In a multifractal model, volatility itself varies according to a fractal process, creating:

- **Heterogeneous scaling**: Different parts of the time series scale differently
- **Intermittency**: Bursts of high activity followed by calm periods
- **Fat-tailed distributions**: Power-law tails rather than exponential decay

The log-price process can be modeled as:

```
log P(t) = log P(0) + ∫₀ᵗ σ(s) dB_H(s)
```

where B_H(s) is fractional Brownian motion with Hurst exponent H, and σ(s) is a multifractal volatility process.

---

## The Hurst Exponent

### Definition and Interpretation

The **Hurst exponent** (H) quantifies the long-term memory and persistence of a time series. It ranges from 0 to 1:

- **H = 0.5**: Random walk (no memory, Brownian motion)
- **0.5 < H < 1**: Persistent series (trending behavior, positive autocorrelation)
- **0 < H < 0.5**: Anti-persistent series (mean-reverting, negative autocorrelation)

### R/S Analysis Method

The rescaled range (R/S) analysis method estimates H by examining how the range of cumulative deviations scales with time:

**Step 1**: For a time series of returns {r_i}, divide into subseries of length τ

**Step 2**: For each subseries, compute the mean:
```
m_τ = (1/τ) Σᵢ₌₁ᵗ rᵢ
```

**Step 3**: Create mean-adjusted cumulative deviations:
```
Y(t,τ) = Σᵢ₌₁ᵗ (rᵢ - m_τ)
```

**Step 4**: Calculate the range:
```
R(τ) = max Y(t,τ) - min Y(t,τ)
```

**Step 5**: Calculate the standard deviation:
```
S(τ) = √[(1/τ) Σᵢ₌₁ᵗ (rᵢ - m_τ)²]
```

**Step 6**: The rescaled range:
```
(R/S)(τ) = R(τ)/S(τ)
```

**Step 7**: The Hurst exponent is the slope of:
```
log(R/S) = H log(τ) + constant
```

### Fractional Brownian Motion

A process with Hurst exponent H can be modeled as **fractional Brownian motion** (fBm):

```
B_H(t) ~ N(0, t^(2H))
```

The covariance structure is:
```
Cov[B_H(s), B_H(t)] = (1/2)[|s|^(2H) + |t|^(2H) - |t-s|^(2H)]
```

This captures long-range dependence when H ≠ 0.5.

### Trading Implications

- **H > 0.7**: Strong trending behavior → momentum strategies may be effective
- **H < 0.3**: Strong mean reversion → contrarian strategies may work
- **0.4 < H < 0.6**: Near-random walk → difficult to predict, market efficiency

---

## Fractal Dimension

### Box-Counting Dimension

The **box-counting dimension** D_B measures the complexity and roughness of a time series curve. For a curve embedded in 2D space:

**Algorithm**:
1. Cover the curve with boxes of size ε
2. Count the number N(ε) of boxes needed
3. The dimension is:
```
D_B = lim_(ε→0) log N(ε) / log(1/ε)
```

**Practical Estimation**:
```
D_B ≈ slope of log N(ε) vs log(1/ε) plot
```

### Relationship to Hurst Exponent

For self-affine fractals (like financial time series), the relationship is:

```
D_B = 2 - H
```

This means:
- **Higher H** → Lower dimension → Smoother, more persistent
- **Lower H** → Higher dimension → Rougher, more erratic

### Volatility Estimation

Fractal dimension can estimate volatility without assuming normal distributions:

```
σ_fractal ∝ (D_B - 1)
```

This provides a robust volatility measure that adapts to fat tails and clustering.

---

## Trading Time vs. Clock Time

### Mandelbrot's Subordinated Process

Mandelbrot proposed that financial markets operate on **trading time** rather than clock time. Trading time flows:
- **Faster** during high volatility (more "information" per unit time)
- **Slower** during quiet periods (less market activity)

Mathematically, price returns in trading time follow:

```
dP/P = σ dB(τ(t))
```

where τ(t) is a stochastic time change process, and B is Brownian motion in trading time.

### Time Dilation Model

The time dilation factor λ(t) transforms clock time t to trading time τ:

```
dτ/dt = λ(t) = f(σ(t), V(t))
```

where:
- σ(t) is local volatility
- V(t) is trading volume
- f is a combining function (e.g., geometric mean)

**Power-law transformation**:
```
λ(t) = [σ(t)/σ̄]^α · [V(t)/V̄]^β
```

where α and β control the sensitivity to volatility and volume.

### Implications for Forecasting

1. **Equal trading time intervals** have more similar statistical properties than equal clock time intervals
2. **Volatility forecasts** should account for time dilation
3. **Regime detection** is more accurate in trading time
4. **Pattern matching** works better when patterns are aligned in trading time

---

## Cross-Dimensional Fractal Analysis

### Multidimensional Fractals

Financial systems are inherently multidimensional, with interdependencies between:
- **Price and volume**
- **Multiple securities**
- **Different market indicators**

**Cross-dimensional fractal coherence** measures how fractal properties align across dimensions:

```
C = Σᵢⱼ w_ij · corr(Dᵢ, Dⱼ)
```

where Dᵢ and Dⱼ are fractal dimensions of different variables.

### Joint Hurst Estimation

For bivariate series (X, Y), the joint Hurst exponent H_XY characterizes co-movement:

```
H_XY = [H_X + H_Y + H_(X+Y)]/2
```

**Interpretation**:
- **H_XY > (H_X + H_Y)/2**: Positive co-persistence
- **H_XY < (H_X + H_Y)/2**: Negative co-persistence (divergence)

### Regime Classification

Cross-dimensional analysis enables regime classification based on fractal properties:

**Regime Features**:
```
Feature vector: [H_price, D_price, H_volume, D_volume, H_XY, C]
```

**Clustering**: K-means or Gaussian mixture models on feature space identify:
1. **Trending regimes**: High H, low D across dimensions
2. **Mean-reverting regimes**: Low H, high D
3. **Transition regimes**: Mixed properties, low coherence

---

## Pattern Recognition in Self-Similar Systems

### Fractal Pattern Matching

Traditional pattern recognition fails in self-similar systems because patterns appear at multiple scales. **Fractal pattern matching** accounts for scale invariance:

**Normalized Distance Metric**:
```
d(P₁, P₂) = min_s ||P₁ - s·P₂||_H
```

where s is a scaling factor and ||·||_H is the Hurst-weighted norm:

```
||X||_H = √[Σᵢ (xᵢ - x̄)² · i^(-2H)]
```

This metric:
1. **Rescales patterns** to find best match
2. **Weights by persistence**: More weight on recent values if H > 0.5
3. **Accounts for heteroscedasticity**: Adapts to varying volatility

### Wavelet-Based Pattern Detection

**Continuous wavelet transform** decomposes signals into scale and position:

```
W(a,b) = ∫ f(t) ψ*((t-b)/a) dt
```

where:
- a is the scale parameter
- b is the translation parameter
- ψ is the mother wavelet

**Pattern identification**:
1. Transform both historical and forecast patterns
2. Compare wavelet coefficients across scales
3. Match patterns with similar multi-scale structure

### Self-Similar Pattern Synthesis

**Recursive subdivision**: Generate new patterns by combining historical sub-patterns:

```
P_new(t) = Σₖ wₖ · s_k · P_hist(φₖ(t))
```

where:
- wₖ are weights based on pattern similarity
- sₖ are scaling factors
- φₖ are time warping functions

---

## Regime-Matched Path Simulation

### Volatility Regime Detection

**Multi-scale volatility clustering**: Analyze volatility at different timeframes (daily, weekly, monthly):

```
σ_τ(t) = √[252/τ · Σᵢ₌₀^(τ-1) r²(t-i)]
```

**Regime features**:
- Mean volatility: σ̄_τ
- Volatility of volatility: σ(σ_τ)
- Hurst exponent: H_τ
- Fractal dimension: D_τ

**Similarity metric** between current and historical regimes:

```
S = Σ_τ w_τ [|σ̄_τ^now - σ̄_τ^hist|/σ̄_τ^now + |H_τ^now - H_τ^hist|]
```

### Path Generation Algorithm

**Step 1: Regime Identification**
- Compute current regime features R_current
- Find N most similar historical regimes: {R_1, ..., R_N}

**Step 2: Return Sampling**
- For each historical regime Rᵢ, sample n_steps returns
- Apply trading time transformation if enabled
- Preserve empirical distribution (no parametric assumptions)

**Step 3: Path Assembly**
- Construct cumulative returns:
```
Path(t) = P₀ · exp(Σᵢ₌₁ᵗ rᵢ)
```

**Step 4: Probability Weighting**
- Assign probability to each path based on regime similarity:
```
p_i = exp(-λ · S_i) / Σⱼ exp(-λ · S_j)
```

where λ controls concentration on most similar regimes.

### Volatility Preservation

To ensure forecasts maintain realistic volatility:

**Step 1**: Measure historical volatility
```
σ_hist = std(log-returns)
```

**Step 2**: Measure forecast volatility
```
σ_forecast = mean over paths [std(log-returns per path)]
```

**Step 3**: If σ_forecast < threshold · σ_hist, add scaled noise:
```
Path'(t) = Path(t) · exp(ε(t))
where ε(t) ~ Empirical(historical returns) · scaling_factor
```

---

## Statistical Testing Framework

### Diebold-Mariano Test

Tests whether two forecasts have significantly different accuracy:

**Null hypothesis**: E[L(e₁) - L(e₂)] = 0

where L is a loss function (e.g., squared error) and e₁, e₂ are forecast errors.

**Test statistic**:
```
DM = d̄ / √[Var(d)/T]
```

where d̄ is mean loss differential, T is sample size.

**Distribution**: DM ~ N(0,1) under H₀ for large T

**Interpretation**:
- |DM| > 1.96 → Reject H₀ at 5% level → Forecasts differ significantly

### Model Confidence Set (MCS)

Identifies the set of models that are not significantly outperformed:

**Algorithm**:
1. Start with all M models
2. Compute pairwise loss differentials
3. Eliminate model with worst performance
4. Test if remaining models are equivalent
5. Repeat until no significant differences

**Equivalence test statistic**:
```
T_max = max_i,j |t_ij|
```

where t_ij is the t-statistic for loss differential between models i and j.

**Result**: Superior Set of Models (SSM) with confidence level α

### Continuous Ranked Probability Score (CRPS)

Evaluates probabilistic forecasts using the entire predictive distribution:

```
CRPS(F, x) = ∫_{-∞}^{∞} [F(y) - 1{y ≥ x}]² dy
```

where F is the forecast CDF and x is the realized value.

**Interpretation**:
- **Lower CRPS** → Better calibrated probabilistic forecast
- **CRPS = 0** → Perfect forecast
- Generalizes MAE for probabilistic forecasts

### Coverage Tests

Assess whether prediction intervals have correct coverage:

**Unconditional coverage**:
```
H₀: P(y_t ∈ CI_t) = 1 - α
```

Test using binomial test on hit rate.

**Conditional coverage** (Christoffersen test):
- Tests both correct coverage and independence of violations
- Uses likelihood ratio test on Markov chain of hits/misses

---

## References

### Foundational Papers

1. **Mandelbrot, B. B.** (1963). "The Variation of Certain Speculative Prices." *Journal of Business*, 36(4), 394-419.
   - Introduced fractal concepts to finance
   - Showed fat tails and infinite variance in cotton prices

2. **Hurst, H. E.** (1951). "Long-Term Storage Capacity of Reservoirs." *Transactions of the American Society of Civil Engineers*, 116, 770-808.
   - Original R/S analysis method
   - Long-range dependence in Nile River data

3. **Peters, E. E.** (1994). *Fractal Market Analysis: Applying Chaos Theory to Investment and Economics.* Wiley.
   - Comprehensive treatment of fractals in finance
   - Market microstructure and fractal time

4. **Mandelbrot, B. B., & Van Ness, J. W.** (1968). "Fractional Brownian Motions, Fractional Noises and Applications." *SIAM Review*, 10(4), 422-437.
   - Mathematical foundations of fBm
   - Connection to Hurst exponent

### Fractal Time Series Methods

5. **Granger, C. W., & Joyeux, R.** (1980). "An Introduction to Long-Memory Time Series Models and Fractional Differencing." *Journal of Time Series Analysis*, 1(1), 15-29.
   - ARFIMA models for long memory
   - Fractional integration

6. **Peng, C. K., et al.** (1994). "Mosaic Organization of DNA Nucleotides." *Physical Review E*, 49(2), 1685.
   - Detrended fluctuation analysis (DFA)
   - Alternative to R/S analysis

7. **Kantelhardt, J. W., et al.** (2002). "Multifractal Detrended Fluctuation Analysis of Nonstationary Time Series." *Physica A*, 316(1-4), 87-114.
   - MFDFA method
   - Local Hurst exponents

### Trading Time and Subordinated Processes

8. **Clark, P. K.** (1973). "A Subordinated Stochastic Process Model with Finite Variance for Speculative Prices." *Econometrica*, 41(1), 135-155.
   - Subordinated process framework
   - Volume as directing process

9. **Ané, T., & Geman, H.** (2000). "Order Flow, Transaction Clock, and Normality of Asset Returns." *Journal of Finance*, 55(5), 2259-2284.
   - Transaction time vs. calendar time
   - Conditional normality in transaction time

### Statistical Testing

10. **Diebold, F. X., & Mariano, R. S.** (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253-263.
    - DM test for forecast comparison
    - Asymptotic theory

11. **Hansen, P. R., Lunde, A., & Nason, J. M.** (2011). "The Model Confidence Set." *Econometrica*, 79(2), 453-497.
    - MCS procedure
    - Multiple testing corrections

12. **Gneiting, T., & Raftery, A. E.** (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association*, 102(477), 359-378.
    - Proper scoring rules
    - CRPS and other metrics

### Recent Advances

13. **Cont, R.** (2007). "Volatility Clustering in Financial Markets: Empirical Facts and Agent-Based Models." In *Long Memory in Economics* (pp. 289-309). Springer.
    - Stylized facts of volatility
    - Agent-based explanations

14. **Tsay, R. S.** (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley.
    - Modern time series methods
    - GARCH and stochastic volatility

15. **Taylor, S. J.** (2008). *Modelling Financial Time Series* (2nd ed.). World Scientific.
    - Comprehensive treatment
    - High-frequency data analysis

---

## Appendix: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| H | Hurst exponent |
| D_B | Box-counting dimension |
| B_H(t) | Fractional Brownian motion |
| τ | Time lag or trading time |
| σ(t) | Volatility at time t |
| λ(t) | Time dilation factor |
| P(t) | Price at time t |
| r_t | Log return at time t |
| =_d | Equality in distribution |
| ~ | Distributed as |
| ∫ | Integral |
| Σ | Summation |
| || · || | Norm |
| E[·] | Expected value |
| Var(·) | Variance |
| Cov(·, ·) | Covariance |
| corr(·, ·) | Correlation |

---

*This research document provides the theoretical foundations for the FracTime forecasting framework. For implementation details, see the codebase documentation.*
