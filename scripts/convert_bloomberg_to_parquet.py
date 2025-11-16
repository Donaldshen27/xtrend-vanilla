#!/usr/bin/env python3
"""
Convert Bloomberg CSV exports to Parquet format.

Usage:
    python scripts/convert_bloomberg_to_parquet.py

This script:
1. Reads all CSVs from data/bloomberg/raw/
2. Converts each to Parquet format
3. Saves to data/bloomberg/[SYMBOL].parquet

Input format: Bloomberg CSV exports (date, price columns)
Output format: Parquet with standardized schema (date index, price column)
"""

import pandas as pd
from pathlib import Path
import sys

def convert_csv_to_parquet(csv_path: Path, output_dir: Path) -> None:
    """
    Convert a single Bloomberg CSV to Parquet.

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save Parquet file
    """
    symbol = csv_path.stem
    print(f"Processing {symbol}...")

    try:
        # Read CSV - Bloomberg typically has date in first column, price in second
        df = pd.read_csv(csv_path, parse_dates=[0])

        # Standardize column names
        if len(df.columns) >= 2:
            df.columns = ['date', 'price']
        else:
            print(f"  WARNING: {symbol} - unexpected columns: {list(df.columns)}")
            return

        # Set date as index
        df = df.set_index('date')

        # Sort by date
        df = df.sort_index()

        # Remove any NaN prices
        df = df.dropna()

        # Save to Parquet
        output_path = output_dir / f"{symbol}.parquet"
        df.to_parquet(output_path, engine='pyarrow', compression='snappy')

        print(f"  ✓ Converted {symbol}: {len(df)} rows ({df.index.min()} to {df.index.max()})")

    except Exception as e:
        print(f"  ERROR processing {symbol}: {e}")


def convert_excel_to_parquet(excel_path: Path, output_dir: Path) -> None:
    """
    Convert Bloomberg Excel export (multiple symbols) to individual Parquet files.

    Assumes BDH formula created a wide table with:
    - First column: dates
    - Subsequent columns: prices for each symbol

    Args:
        excel_path: Path to Excel file
        output_dir: Directory to save Parquet files
    """
    print(f"Processing Excel file: {excel_path}")

    try:
        # Read Excel file
        df = pd.read_excel(excel_path, engine='openpyxl')

        # First column should be dates
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        # Each remaining column is a symbol
        for symbol_col in df.columns:
            symbol = symbol_col.replace(' Comdty', '').replace(' ', '_')

            print(f"Processing {symbol}...")

            # Extract single symbol series
            prices = df[symbol_col].dropna()

            if len(prices) == 0:
                print(f"  WARNING: {symbol} has no data")
                continue

            # Create DataFrame with standard schema
            symbol_df = pd.DataFrame({'price': prices})

            # Save to Parquet
            output_path = output_dir / f"{symbol}.parquet"
            symbol_df.to_parquet(output_path, engine='pyarrow', compression='snappy')

            print(f"  ✓ Converted {symbol}: {len(symbol_df)} rows ({symbol_df.index.min()} to {symbol_df.index.max()})")

    except Exception as e:
        print(f"  ERROR processing Excel file: {e}")
        sys.exit(1)


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "bloomberg" / "raw"
    output_dir = project_root / "data" / "bloomberg"

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bloomberg Data Conversion: CSV/Excel → Parquet")
    print("=" * 60)
    print()

    # Check for Excel file first (single file with all symbols)
    excel_files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))

    if excel_files:
        print(f"Found Excel file(s): {[f.name for f in excel_files]}")
        print("Processing Excel export...")
        print()

        for excel_file in excel_files:
            convert_excel_to_parquet(excel_file, output_dir)

    # Also check for individual CSVs
    csv_files = list(raw_dir.glob("*.csv"))

    if csv_files:
        print(f"\nFound {len(csv_files)} CSV file(s)")
        print("Processing individual CSVs...")
        print()

        for csv_file in csv_files:
            convert_csv_to_parquet(csv_file, output_dir)

    if not excel_files and not csv_files:
        print(f"ERROR: No CSV or Excel files found in {raw_dir}")
        print("\nExpected file locations:")
        print(f"  - Excel: {raw_dir}/*.xlsx")
        print(f"  - CSVs: {raw_dir}/*.csv")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Conversion complete!")
    print(f"Parquet files saved to: {output_dir}")
    print("=" * 60)

    # List output files
    parquet_files = list(output_dir.glob("*.parquet"))
    print(f"\nGenerated {len(parquet_files)} Parquet files:")
    for pf in sorted(parquet_files)[:10]:  # Show first 10
        print(f"  - {pf.name}")
    if len(parquet_files) > 10:
        print(f"  ... and {len(parquet_files) - 10} more")


if __name__ == "__main__":
    main()
