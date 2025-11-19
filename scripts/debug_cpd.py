"""Debug GP-CPD to understand why severity is always 0."""
import pandas as pd
import torch

from xtrend.cpd import CPDConfig, GPCPDSegmenter
from xtrend.cpd.gp_fitter import GPFitter
from xtrend.data.sources import BloombergParquetSource

# Load ES data for 2019-2020
source = BloombergParquetSource(root_path="data/bloomberg/processed")
prices_df = source.load_prices(["ES"], start="2019-01-01", end="2020-12-31")
price_series = prices_df["ES"].ffill().bfill()

print(f"Loaded {len(price_series)} days of ES prices")
print(f"Date range: {price_series.index[0]} to {price_series.index[-1]}")
print()

# Test GP-CPD on a single window
lookback = 63
window_start = 200  # Start somewhere in the middle
window_end = window_start + lookback - 1

window = price_series.iloc[window_start:window_end + 1]
print(f"Testing window: days {window_start}-{window_end} ({len(window)} days)")
print(f"Price range: {window.min():.2f} to {window.max():.2f}")
print()

# Convert to tensors
x = torch.arange(len(window)).float().unsqueeze(-1)
y = torch.tensor(window.values).float()

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"y stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")
print()

# Fit stationary GP
print("="*60)
print("FITTING STATIONARY GP")
print("="*60)
fitter = GPFitter(max_iter=200, lr=0.1)
stat_model, log_mll_M = fitter.fit_stationary_gp(x, y)
print(f"Stationary GP log marginal likelihood: {log_mll_M:.4f}")
print()

# Fit change-point GP
print("="*60)
print("FITTING CHANGE-POINT GP")
print("="*60)
cp_model, log_mll_C, t_cp = fitter.fit_changepoint_gp(x, y, stat_model)
print(f"Change-point GP log marginal likelihood: {log_mll_C:.4f}")
print(f"Detected change-point location: {t_cp:.0f}")
print()

# Compute severity
delta = log_mll_C - log_mll_M
severity = torch.sigmoid(torch.tensor(delta)).item()

print("="*60)
print("SEVERITY COMPUTATION")
print("="*60)
print(f"L_M (stationary):     {log_mll_M:.4f}")
print(f"L_C (change-point):   {log_mll_C:.4f}")
print(f"Delta (L_C - L_M):    {delta:.4f}")
print(f"Severity (sigmoid(Δ)): {severity:.6f}")
print()

if severity >= 0.9:
    print("✓ Change-point DETECTED (severity >= 0.9)")
else:
    print("✗ No change-point (severity < 0.9) - FALLBACK")

print()
print("="*60)
print("INTERPRETATION")
print("="*60)
if delta < -10:
    print("Delta is very negative → Stationary model MUCH better")
    print("This suggests:")
    print("  - Change-point GP fit failed")
    print("  - Or: data truly has no regime change in this window")
elif delta < 0:
    print("Delta is negative → Stationary model slightly better")
    print("No strong evidence for change-point")
elif delta < 2.2:
    print("Delta is positive but < 2.2 → Weak evidence for change-point")
elif delta >= 2.2:
    print("Delta >= 2.2 → Strong evidence for change-point (severity >= 0.9)")
