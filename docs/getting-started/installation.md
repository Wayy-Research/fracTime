# Installation

## Quick Install

```bash
pip install fractime
```

This installs FracTime with all required dependencies.

---

## From Source

For the latest development version:

```bash
git clone https://github.com/wayy-research/fracTime.git
cd fracTime
pip install -e .
```

For development with testing tools:

```bash
pip install -e ".[dev]"
```

---

## Dependencies

FracTime automatically installs these dependencies:

| Package | Purpose |
|---------|---------|
| **numpy** | Numerical computing |
| **polars** | Fast DataFrames |
| **numba** | JIT compilation for speed |
| **wrchart** | Interactive visualizations |
| **scikit-learn** | Clustering and preprocessing |

### Optional Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| **pymc** | Bayesian forecasting | `pip install pymc` |

---

## Verify Installation

```python
import fractime as ft

# Check version
print(f"FracTime version: {ft.__version__}")

# Quick test
import numpy as np
prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.02)
analyzer = ft.Analyzer(prices)
print(f"Hurst exponent: {analyzer.hurst}")
```

Expected output:

```
FracTime version: 0.3.0
Hurst exponent: hurst=0.5234
```

---

## Platform Support

FracTime supports:

- **Python**: 3.10, 3.11, 3.12
- **OS**: Linux, macOS, Windows
- **Architecture**: x86_64, ARM64

---

## Troubleshooting

### Numba Compilation

On first use, Numba compiles optimized functions. This causes a one-time delay of a few seconds. Subsequent runs are fast.

### Memory Issues

For very large datasets (>100,000 points), consider:

```python
# Reduce bootstrap samples
analyzer = ft.Analyzer(prices, n_samples=500)

# Use smaller rolling windows
analyzer = ft.Analyzer(prices, window=30)
```

### Import Errors

If you see import errors, ensure you have the latest version:

```bash
pip install --upgrade fractime
```
