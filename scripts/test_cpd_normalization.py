"""Test CPD normalization fix with detailed diagnostics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from xtrend.cpd.gp_fitter import GPFitter


def load_es_data():
    """Load ES futures data."""
    data_path = project_root / "data/bloomberg/processed/ES.parquet"
    df = pd.read_parquet(data_path)
    df = df.sort_index()
    return df


def test_multiple_windows():
    """Test CPD on multiple windows to see severity distribution."""
    print("Loading ES data...")
    df = load_es_data()

    # Filter to 2019-2020
    df = df.loc["2019-01-01":"2020-12-31"]
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total days: {len(df)}\n")

    prices = df["price"].values
    lookback = 63

    fitter = GPFitter(max_iter=200, lr=0.1)

    severities = []
    deltas = []

    # Test every 10th window to get a sample
    for start_idx in range(0, len(prices) - lookback, 10):
        end_idx = start_idx + lookback
        window_prices = prices[start_idx:end_idx]

        # Convert to torch tensors
        x = torch.arange(lookback, dtype=torch.float32).reshape(-1, 1)
        y = torch.tensor(window_prices, dtype=torch.float32)

        # Fit both models
        try:
            _, log_mll_stat = fitter.fit_stationary_gp(x, y)
            _, log_mll_cp, t_cp = fitter.fit_changepoint_gp(x, y, None)

            severity = fitter.compute_severity(log_mll_stat, log_mll_cp)
            delta = log_mll_cp - log_mll_stat

            severities.append(severity)
            deltas.append(delta)

            if len(severities) % 5 == 0:
                print(f"Window {len(severities):3d}: L_M={log_mll_stat:7.2f}, L_C={log_mll_cp:7.2f}, "
                      f"Δ={delta:6.2f}, severity={severity:.4f}")
        except Exception as e:
            print(f"Failed at window {start_idx}: {e}")
            continue

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total windows tested: {len(severities)}")
    print(f"\nSeverity distribution:")
    print(f"  Min:    {np.min(severities):.6f}")
    print(f"  25th:   {np.percentile(severities, 25):.6f}")
    print(f"  Median: {np.median(severities):.6f}")
    print(f"  75th:   {np.percentile(severities, 75):.6f}")
    print(f"  Max:    {np.max(severities):.6f}")
    print(f"\nDelta (L_C - L_M) distribution:")
    print(f"  Min:    {np.min(deltas):.4f}")
    print(f"  25th:   {np.percentile(deltas, 25):.4f}")
    print(f"  Median: {np.median(deltas):.4f}")
    print(f"  75th:   {np.percentile(deltas, 75):.4f}")
    print(f"  Max:    {np.max(deltas):.4f}")
    print(f"\nDetection rate (severity >= 0.9): {np.mean(np.array(severities) >= 0.9) * 100:.1f}%")
    print(f"Detection rate (severity >= 0.8): {np.mean(np.array(severities) >= 0.8) * 100:.1f}%")
    print(f"Detection rate (severity >= 0.7): {np.mean(np.array(severities) >= 0.7) * 100:.1f}%")


if __name__ == "__main__":
    test_multiple_windows()
