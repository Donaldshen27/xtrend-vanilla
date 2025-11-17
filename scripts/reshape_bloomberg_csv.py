#!/usr/bin/env python3
"""
Reshape wide-format Bloomberg CSV into individual ticker CSV files.

Input: bloomberg_historical_data_50_original.csv
  - Format: ticker, date, price, empty, ticker, date, price, empty, ...
  - 50 tickers × 4 columns = 200 columns wide

Output: individual_csvs/
  - One CSV per ticker: {ticker_name}.csv
  - Format: date,price (ISO 8601 dates)

Usage:
    python scripts/reshape_bloomberg_csv.py [input_csv] [output_dir]

    If no arguments provided, uses defaults:
      input_csv: data/bloomberg/future_data/bloomberg_historical_data_50_original.csv
      output_dir: data/bloomberg/future_data/original_individual_csvs
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def identify_ticker_columns(first_row: pd.Series) -> List[Tuple[int, str]]:
    """
    Scan first row to find ticker name columns.

    Returns:
        List of (column_index, ticker_name) tuples
    """
    tickers = []
    ticker_markers = ["Comdty", "Index", "Curncy"]

    for col_idx, value in enumerate(first_row):
        if pd.notna(value) and isinstance(value, str):
            value = value.strip()
            if value and any(marker in value for marker in ticker_markers):
                tickers.append((col_idx, value))

    return tickers


def sanitize_filename(ticker_name: str) -> str:
    """Convert ticker name to safe filename: 'CC1 Comdty' -> 'CC1_Comdty'."""
    return ticker_name.replace(' ', '_').replace('/', '_')


def extract_ticker_data(df: pd.DataFrame, ticker_col: int, ticker_name: str) -> pd.DataFrame:
    """
    Extract date and price columns for a single ticker.

    Args:
        df: Full DataFrame (raw CSV data)
        ticker_col: Column index where ticker name appears
        ticker_name: Name of the ticker (for logging)

    Returns:
        DataFrame with 'date' and 'price' columns
    """
    date_col = ticker_col + 1
    price_col = ticker_col + 2

    # Skip first row (contains ticker name, not data)
    dates = df.iloc[1:, date_col]
    prices = df.iloc[1:, price_col]

    # Create DataFrame
    ticker_df = pd.DataFrame({
        'date': dates,
        'price': prices
    })

    # Remove rows with missing data
    ticker_df = ticker_df.dropna()

    # Convert date strings to datetime
    ticker_df['date'] = pd.to_datetime(ticker_df['date'])

    # Sort by date
    ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

    return ticker_df


def parse_wide_bloomberg_csv(input_path: Path, output_dir: Path) -> Dict[str, int]:
    """
    Parse wide-format Bloomberg CSV into individual ticker CSVs.

    Args:
        input_path: Path to bloomberg_historical_data_50_original.csv
        output_dir: Directory to save individual CSV files

    Returns:
        Dict mapping ticker_name -> row_count
    """
    print(f"Reading CSV: {input_path}")
    print(f"File size: {input_path.stat().st_size / (1024**2):.1f} MB")
    print()

    # Read entire CSV (no header inference - treat as raw data)
    df = pd.read_csv(input_path, header=None, dtype=str, encoding='utf-8-sig')

    print(f"CSV dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print()

    # Extract ticker names from first row
    first_row = df.iloc[0]
    tickers = identify_ticker_columns(first_row)

    print(f"Found {len(tickers)} tickers:")
    for idx, (col_idx, ticker_name) in enumerate(tickers, 1):
        print(f"  {idx:2d}. {ticker_name:20s} (column {col_idx})")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each ticker
    results = {}
    print("Processing tickers...")
    print("-" * 60)

    for ticker_col, ticker_name in tickers:
        try:
            # Extract data
            ticker_df = extract_ticker_data(df, ticker_col, ticker_name)

            # Validate data
            if len(ticker_df) == 0:
                print(f"  WARNING: {ticker_name} has no data - skipping")
                continue

            # Save to CSV
            safe_filename = sanitize_filename(ticker_name)
            output_path = output_dir / f"{safe_filename}.csv"
            ticker_df.to_csv(output_path, index=False, date_format='%Y-%m-%d')

            # Record results
            date_range = f"{ticker_df['date'].min().date()} to {ticker_df['date'].max().date()}"
            results[ticker_name] = len(ticker_df)

            print(f"  ✓ {ticker_name:20s} → {len(ticker_df):5d} rows ({date_range})")

        except Exception as e:
            print(f"  ERROR: {ticker_name} - {e}")
            continue

    return results


def main():
    # Paths
    project_root = Path(__file__).parent.parent

    # Handle command-line arguments
    if len(sys.argv) >= 3:
        input_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        input_path = Path(sys.argv[1])
        # Auto-generate output dir based on input filename
        output_dir = input_path.parent / f"{input_path.stem}_individual_csvs"
    else:
        # Defaults
        input_path = project_root / "data" / "bloomberg" / "future_data" / "bloomberg_historical_data_50_original.csv"
        output_dir = project_root / "data" / "bloomberg" / "future_data" / "original_individual_csvs"

    # Verify input exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("Bloomberg CSV Reshape: Wide → Individual Files")
    print("=" * 60)
    print()

    # Parse and reshape
    results = parse_wide_bloomberg_csv(input_path, output_dir)

    # Summary
    print()
    print("=" * 60)
    print("Reshape complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Files created: {len(results)}")
    print(f"Total rows: {sum(results.values()):,}")
    print()

    # List output files
    output_files = sorted(output_dir.glob("*.csv"))
    print(f"Generated {len(output_files)} CSV files:")
    for csv_file in output_files[:10]:
        print(f"  - {csv_file.name}")
    if len(output_files) > 10:
        print(f"  ... and {len(output_files) - 10} more")
    print()


if __name__ == "__main__":
    main()
