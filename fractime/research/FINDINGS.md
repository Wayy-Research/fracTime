# Empirical Findings: FracTime v0.5.0 Research Extensions

## Executive Summary

This document presents empirical findings from rigorous walk-forward testing of FracTime's forecasting models across 12 assets (US equities, sector ETFs, cryptocurrencies, and international markets) over the 2020-2024 period. Our key finding is that **fractal-based features provide significant value for risk management and regime detection, but not for short-term price prediction**.

---

## 1. Price Prediction Accuracy

### 1.1 Methodology

We conducted 1-step ahead walk-forward validation with:
- **Training window**: 252 trading days (1 year)
- **Test samples**: 100 sequential predictions per asset
- **Assets tested**: SPY, QQQ, DIA, XLF, XLE, XLK, XLV, BTCUSDT, ETHUSDT, SOLUSDT, EFA, EEM
- **Models**: ARIMA, ETS, LSTM, FractalLSTM

Directional accuracy was computed only for meaningful price moves (>0.1%) to avoid noise.

### 1.2 Results

| Model | Avg Directional Accuracy | Inverse Accuracy | Avg Sharpe |
|-------|--------------------------|------------------|------------|
| ETS | 52.3% | 47.7% | 0.11 |
| FractalLSTM | 49.5% | 50.5% | -0.38 |
| LSTM | 45.3% | 54.7% | -1.59 |
| ARIMA | 40.0% | 39.4% | 0.00 |

### 1.3 Key Observations

1. **All models hover near 50%**: Consistent with the Efficient Market Hypothesis for liquid assets at daily frequency.

2. **ETS performs best**: Simple exponential smoothing (52.3%) marginally outperforms neural networks, suggesting that complex models overfit to noise.

3. **ARIMA's systematic errors**: ARIMA's 40% accuracy is notable—it's consistently wrong, but inverting predictions yields only 39.4% accuracy, indicating errors are not systematically exploitable.

4. **Sharpe ratios are more revealing**: Even ETS's 52.3% accuracy translates to only a 0.11 Sharpe ratio before transaction costs—not economically significant.

5. **Neural networks underperform**: Both LSTM (-1.59 Sharpe) and FractalLSTM (-0.38 Sharpe) destroy value in directional trading, likely due to overfitting.

### 1.4 Statistical Significance

T-tests against the null hypothesis of 50% accuracy:

| Model | Mean Accuracy | p-value | Significance |
|-------|---------------|---------|--------------|
| ETS | 52.3% | 0.18 | Not significant |
| FractalLSTM | 49.5% | 0.82 | Not significant |
| LSTM | 45.3% | 0.09 | Marginally significant (wrong direction) |
| ARIMA | 40.0% | 0.03 | Significant (wrong direction) |

**Conclusion**: No model achieves statistically significant predictive accuracy above random chance for 1-day price direction.

---

## 2. The Case for Risk Management

### 2.1 Regime-Based Position Sizing

Instead of predicting price direction, we tested using the HMM regime detector for position sizing:

**Strategy**:
- Full position (100%) in detected Bull regime
- Half position (50%) in detected Bear regime

**Results on SPY (2020-2024)**:

| Metric | Buy & Hold | Regime-Based | Improvement |
|--------|------------|--------------|-------------|
| Total Return | 96.8% | 124.5% | +28.6% |
| Sharpe Ratio | 0.64 | 1.00 | +56% |
| Max Drawdown | -41.1% | -23.0% | -44% reduction |
| Calmar Ratio | 0.47 | 1.08 | +130% |

### 2.2 Interpretation

The regime-based strategy achieves:
- **Higher risk-adjusted returns** (Sharpe 1.00 vs 0.64)
- **Dramatically reduced drawdowns** (-23% vs -41%)
- **Superior capital preservation** during bear markets

This demonstrates that fractal/HMM features are valuable for **risk management**, not price prediction.

---

## 3. Feature-Specific Findings

### 3.1 HMM Regime Detection

**What it does well**:
- Identifies regime transitions with 85-95% confidence
- Provides interpretable Bull/Bear/Sideways classifications
- Expected regime durations match market behavior (Bull ~130 days, Bear ~17 days)

**Recommended use**:
- Position sizing (reduce in Bear)
- Stop-loss adjustment (tighter in high-volatility regime)
- Strategy selection (trend-following in Bull, mean-reversion in Sideways)

### 3.2 FractalLSTM Hybrid

**What it does well**:
- Feature importance reveals volatility (39%) and fractal_dim (29%) dominate
- Reduces MAPE by 43% compared to vanilla LSTM (5.87% vs 10.38%)
- Better calibrated uncertainty estimates via MC dropout

**What it doesn't do**:
- Improve directional accuracy (49.5% vs LSTM's 45.3%)
- Generate positive Sharpe from directional trading

**Recommended use**:
- Volatility forecasting (not price direction)
- Uncertainty quantification for confidence-based sizing

### 3.3 Transaction Cost Analysis (TCA)

**Key findings**:
- Average SPY round-trip cost: ~100 bps (commission + spread + impact)
- Corwin-Schultz spread estimates: 25-55 bps for liquid ETFs
- Market impact: 30-40 bps for 1% participation rate

**Critical insight**:
A strategy needs >100 bps per trade expected return to be profitable. With 52% directional accuracy and average move ~1%, expected return per trade is only ~4 bps—**unprofitable after costs**.

### 3.4 Multifractal DFA (MF-DFA)

**Empirical observations**:

| Asset | Spectrum Width | Interpretation |
|-------|----------------|----------------|
| SPY | 0.28 | Moderate multifractality |
| BTC | 0.63 | Strong multifractality |
| ETH | 0.71 | Very strong multifractality |

**Key finding**: Crypto assets exhibit 2-3x stronger multifractal behavior than equity ETFs, consistent with:
- Fatter tails
- More extreme volatility clustering
- Less efficient price discovery

**Recommended use**:
- Detect structural breaks (sudden width changes)
- Identify regime changes before HMM detects them
- Asset-specific model calibration

---

## 4. Ensemble and Combination Strategies

### 4.1 Ensemble Results

| Strategy | Avg Directional Accuracy |
|----------|--------------------------|
| Simple Average | 48.2% |
| Median | 47.8% |
| Majority Vote | 49.1% |
| Inverse ARIMA + FracLSTM | 48.5% |
| Adaptive Weighted | 51.2% |

**Finding**: Ensemble methods do not significantly improve prediction accuracy. The "wisdom of crowds" fails when all models are near random.

### 4.2 What Actually Helps

The only approach showing consistent value is **regime-conditional behavior**:
- Use HMM to determine market state
- Apply different strategies per regime
- Focus on risk management, not prediction

---

## 5. Recommendations

### 5.1 For Practitioners

1. **Don't predict prices at daily frequency** — EMH holds for liquid assets
2. **Use fractal features for risk management**:
   - Position sizing via regime detection
   - Volatility forecasting for hedging
   - Drawdown prediction for capital preservation
3. **Always account for transaction costs** — Most "alpha" disappears after costs
4. **Consider longer horizons** — Fractal features may have more predictive value at weekly/monthly frequencies

### 5.2 For Researchers

1. **Prediction benchmarks should include random walk** — Many papers omit this critical baseline
2. **Report Sharpe ratios, not just accuracy** — Direction matters, but magnitude matters more
3. **Test on multiple asset classes** — Crypto behavior differs fundamentally from equities
4. **Include transaction cost analysis** — Academic alpha often vanishes in practice

### 5.3 For FracTime Development

1. **Reposition the library** — From "forecasting" to "risk analytics"
2. **Add regime-based strategy framework** — The primary value proposition
3. **Emphasize volatility prediction** — More predictable than price
4. **Build drawdown prediction models** — High practical value

---

## 6. Conclusion

Our empirical analysis reveals that FracTime's fractal and regime-based features provide **significant value for risk management** but **no statistically significant edge for short-term price prediction**.

The HMM regime detector, when used for position sizing rather than directional betting, improved Sharpe ratio by 56% and reduced maximum drawdown by 44% on SPY during 2020-2024.

We recommend practitioners use these tools for:
- Regime-aware position sizing
- Volatility forecasting
- Transaction cost budgeting
- Structural break detection

Rather than attempting to predict unpredictable price movements.

---

## Appendix: Reproducibility

All findings can be reproduced using:

```python
from fractime.regime import RegimeDetector
from fractime.baselines import FractalLSTMForecaster
from fractime.research.tca import TCACalculator
from fractime._numba import compute_mfdfa, compute_multifractal_spectrum

# See tests/test_paper_features.py for complete test suite
# See scripts/validation_test.py for walk-forward methodology
```

**Data**: Yahoo Finance via wrdata library (2020-01-01 to 2024-12-31)
**Hardware**: Standard CPU (no GPU required)
**Software**: Python 3.10+, PyTorch 2.0+, hmmlearn 0.3+

---

*FracTime v0.5.0 — Wayy Research, 2025*
