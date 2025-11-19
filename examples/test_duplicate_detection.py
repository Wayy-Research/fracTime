#!/usr/bin/env python3
"""
Test duplicate path detection and diagnosis.
"""

import sys
sys.path.insert(0, '/home/rcgalbo/wayy-research/wayy-fin/wf/fracTime')

import numpy as np
import polars as pl
import fractime as ft
from datetime import datetime, timedelta

print("=" * 70)
print("DUPLICATE PATH DETECTION TEST")
print("=" * 70)

# Generate sample data
np.random.seed(42)
prices = 100 + np.random.randn(200).cumsum()
end_date = datetime(2025, 11, 19)
start_date = end_date - timedelta(days=199)
dates = pl.datetime_range(start=start_date, end=end_date, interval='1d', eager=True).to_numpy()

print(f"\n1. FITTING FORECASTER")
forecaster = ft.FractalForecaster()
forecaster.fit(prices, dates=dates)
print(f"   ✓ Fitted with {len(prices)} data points")

print(f"\n2. GENERATING FORECAST WITH WARNING CHECK")
print(f"   Watch for duplicate path warnings...")

result = forecaster.predict(end_date='2025-11-27', n_paths=500)

print(f"\n3. ANALYZING RESULTS")

paths = result['paths']
probs = result['probabilities']

# Check for duplicates manually
unique_paths, unique_indices, counts = np.unique(
    paths, axis=0, return_index=True, return_counts=True
)

print(f"\n   Total paths:     {len(paths)}")
print(f"   Unique paths:    {len(unique_paths)}")
print(f"   Duplicate paths: {np.sum(counts > 1)}")

if np.sum(counts > 1) > 0:
    print(f"\n   ⚠️  Duplicates found!")
    print(f"\n   Most duplicated paths:")
    sorted_idx = np.argsort(counts)[::-1]
    for i in range(min(5, np.sum(counts > 1))):
        idx = sorted_idx[i]
        if counts[idx] > 1:
            print(f"      Path appears {counts[idx]} times, final value: ${unique_paths[idx, -1]:.2f}")
else:
    print(f"\n   ✓ No duplicates found!")

# Check probability distribution
unique_probs = np.unique(probs)
print(f"\n   Unique probabilities: {len(unique_probs)}")
print(f"   Min probability:      {np.min(probs):.6f}")
print(f"   Max probability:      {np.max(probs):.6f}")
print(f"   Mean probability:     {np.mean(probs):.6f}")
print(f"   Expected (uniform):   {1.0/len(probs):.6f}")

# Show top 10 paths
print(f"\n4. TOP 10 MOST LIKELY PATHS")
print(f"   (Looking for identical values...)")

top_indices = np.argsort(probs)[-10:][::-1]

print(f"\n   {'Rank':<6} {'Probability':<15} {'Final Value':<15}")
print(f"   {'-' * 40}")

final_values = []
for rank, idx in enumerate(top_indices, 1):
    prob = probs[idx]
    final_val = paths[idx, -1]
    final_values.append(final_val)
    print(f"   #{rank:<5} {prob:.6f}      ${final_val:>8.2f}")

# Check if all top paths have the same final value
if len(set(np.round(final_values, 2))) == 1:
    print(f"\n   ⚠️  CRITICAL: All top 10 paths end at the same value!")
    print(f"   This indicates duplicate paths.")
else:
    print(f"\n   ✓ Top paths have different final values")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)

print(f"""
If you see duplicate paths, possible solutions:

1. Increase n_paths to get more variety:
   result = forecaster.predict(end_date='2025-11-27', n_paths=2000)

2. Use longer historical data for fitting:
   forecaster.fit(prices, dates=dates)

3. Check if your data has enough variation

4. The duplicate detection will now automatically warn you when
   this happens in future predictions.
""")
