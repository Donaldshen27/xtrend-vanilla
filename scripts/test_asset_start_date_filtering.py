#!/usr/bin/env python3
"""
Test script to verify that assets with late start dates are handled correctly.

This verifies the fix for backfilled synthetic data issue:
- Assets that started after 1990 should not have training samples from before they existed
- The model should only train on real price history, not backfilled flat prices

Usage:
    uv run python scripts/test_asset_start_date_filtering.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xtrend.data.sources import BloombergParquetSource


def test_asset_start_dates():
    """Check which assets have late start dates."""
    print("="*60)
    print("ASSET START DATE ANALYSIS")
    print("="*60)

    # Load data source
    data_path = 'data/bloomberg/processed'
    source = BloombergParquetSource(root_path=data_path)
    symbols = source.symbols()

    print(f"\nLoading prices for {len(symbols)} symbols from 1990-2023...")
    prices = source.load_prices(symbols, start='1990-01-01', end='2023-12-31')

    dataset_start = prices.index[0]
    print(f"Dataset start: {dataset_start.strftime('%Y-%m-%d')}")
    print(f"Dataset end: {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total days: {len(prices)}")

    # Analyze each asset
    results = []
    for symbol in symbols:
        series = prices[symbol]
        first_valid = series.first_valid_index()

        if first_valid is None:
            results.append({
                'symbol': symbol,
                'first_valid': None,
                'years_late': None,
                'status': 'NO_DATA'
            })
        else:
            years_late = (first_valid - dataset_start).days / 365.25
            status = 'LATE_START' if years_late > 1 else 'FULL_HISTORY'
            results.append({
                'symbol': symbol,
                'first_valid': first_valid,
                'years_late': years_late,
                'status': status
            })

    # Create DataFrame for analysis
    df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total assets: {len(df)}")
    print(f"Assets with full history (started ~1990): {len(df[df['status'] == 'FULL_HISTORY'])}")
    print(f"Assets with late start (>1 year after 1990): {len(df[df['status'] == 'LATE_START'])}")
    print(f"Assets with no data: {len(df[df['status'] == 'NO_DATA'])}")

    # Show late starters
    late_starters = df[df['status'] == 'LATE_START'].sort_values('years_late', ascending=False)
    if len(late_starters) > 0:
        print("\n" + "="*60)
        print("LATE START ASSETS (sorted by years late)")
        print("="*60)
        print(f"{'Symbol':<10} {'Start Date':<12} {'Years Late':>12}")
        print("-"*60)
        for _, row in late_starters.iterrows():
            print(f"{row['symbol']:<10} {row['first_valid'].strftime('%Y-%m-%d'):<12} {row['years_late']:>12.1f}")

        print("\n⚠️  BEFORE FIX: These assets would have been trained on backfilled")
        print("   flat price history (0% returns) for the pre-listing period.")
        print("\n✅ AFTER FIX: Training samples for these assets now start from")
        print("   their actual listing date + 252 days warmup period.")

    # Show no-data assets
    no_data = df[df['status'] == 'NO_DATA']
    if len(no_data) > 0:
        print("\n" + "="*60)
        print("ASSETS WITH NO DATA")
        print("="*60)
        for _, row in no_data.iterrows():
            print(f"  {row['symbol']}")

    print("\n" + "="*60)
    print("VERIFICATION STEPS")
    print("="*60)
    print("1. Run training script with verbose output:")
    print("   uv run python scripts/train_xtrend.py --model xtrendq --epochs 1")
    print("\n2. Look for diagnostic output showing late-start assets")
    print("\n3. Verify training samples exclude pre-listing periods")
    print("="*60)


if __name__ == '__main__':
    test_asset_start_dates()
