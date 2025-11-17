#!/usr/bin/env python3
"""
Convert Bloomberg CSV/Excel exports to the Pinnacle-aligned Parquet layout.

Usage:
    python scripts/convert_bloomberg_to_parquet.py [--symbol-map PATH ...]

Workflow:
1. Reads symbol metadata from the canonical & expanded symbol map CSVs
2. Scans data/bloomberg/raw for Excel/CSV exports
3. Converts each column to Parquet using Pinnacle IDs for filenames

Input format: Bloomberg BDH exports (Excel worksheet or CSV per symbol)
Output format: data/bloomberg/<PinnacleID>.parquet with standardized schema
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


IGNORED_TICKER_TOKENS = {"<TBD>", "TBD"}


def _clean_field(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_security_name(value: str) -> str:
    """Uppercase + strip non-alphanumerics so CL1 Comdty -> CL1COMDTY."""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def sanitize_symbol_for_filename(value: str) -> str:
    """Fallback filename component when no map entry exists."""
    clean = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    clean = clean.strip("_")
    return clean or "unknown_symbol"


SymbolEntry = Tuple[str, str]  # (pinnacle_id, bloomberg_ticker)


class SymbolMapper:
    """Lightweight helper that maps Bloomberg tickers to Pinnacle IDs."""

    def __init__(self, csv_paths: Iterable[Path]) -> None:
        self.csv_paths = list(csv_paths)
        self.by_norm: Dict[str, List[SymbolEntry]] = {}
        self._load_all()

    def _load_all(self) -> None:
        for csv_path in self.csv_paths:
            self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        if not csv_path.exists():
            print(f"WARNING: Symbol map not found at {csv_path} – filenames will use Bloomberg tickers.")
            return

        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ticker = _clean_field(row.get("bloomberg_ticker"))
                pinnacle = _clean_field(row.get("pinnacle_id"))
                if not ticker or not pinnacle:
                    continue
                if pinnacle.startswith("#") or ticker.startswith("#"):
                    continue
                if any(token in ticker.upper() for token in IGNORED_TICKER_TOKENS):
                    continue
                info: SymbolEntry = (pinnacle, ticker)
                for key in self._variant_keys(ticker, pinnacle):
                    bucket = self.by_norm.setdefault(key, [])
                    if info not in bucket:
                        bucket.append(info)

    def _variant_keys(self, ticker: str, pinnacle: str) -> set:
        """Track multiple normalized aliases per security."""
        variants = {normalize_security_name(ticker)}
        variants.add(normalize_security_name(pinnacle))
        suffixes = ("COMDTY", "INDEX", "CURNCY")
        for suff in suffixes:
            variant = normalize_security_name(ticker)
            if variant.endswith(suff):
                variants.add(variant[: -len(suff)])
        if ticker.endswith("Index"):
            variants.add(normalize_security_name(ticker.replace(" Index", "")))
        return {v for v in variants if v}

    def resolve(self, raw_label: str) -> List[SymbolEntry]:
        """Return all Pinnacle IDs that map to a header/filename."""
        cleaned = normalize_security_name(raw_label)
        if cleaned in self.by_norm:
            return self.by_norm[cleaned]
        # Fallback: substring search to catch headers like 'CL1COMDTYPX_LAST'
        matches: List[SymbolEntry] = []
        for key, info in self.by_norm.items():
            if key and key in cleaned:
                for entry in info:
                    if entry not in matches:
                        matches.append(entry)
        return matches


def determine_output_symbols(raw_label: str, mapper: SymbolMapper) -> List[Tuple[str, Optional[str]]]:
    """Return all output symbol targets for a given raw label."""
    if mapper:
        matches = mapper.resolve(raw_label)
        if matches:
            return [(pinnacle, ticker) for pinnacle, ticker in matches]
    return [(sanitize_symbol_for_filename(raw_label), None)]


def convert_csv_to_parquet(csv_path: Path, output_dir: Path, mapper: SymbolMapper) -> None:
    """
    Convert a single Bloomberg CSV to Parquet.

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save Parquet file
    """
    raw_symbol = csv_path.stem
    targets = determine_output_symbols(raw_symbol, mapper)
    pretty_name = targets[0][1] or raw_symbol
    target_labels = ", ".join(sym for sym, _ in targets)
    print(f"Processing {pretty_name} → {target_labels}...")

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

        # Save to Parquet (duplicate if multiple Pinnacle IDs map to the same ticker)
        for output_symbol, _ in targets:
            output_path = output_dir / f"{output_symbol}.parquet"
            df.to_parquet(output_path, engine='pyarrow', compression='snappy')
            print(f"  ✓ Converted {output_symbol}: {len(df)} rows ({df.index.min()} to {df.index.max()})")

    except Exception as e:
        print(f"  ERROR processing {pretty_name}: {e}")


def convert_excel_to_parquet(excel_path: Path, output_dir: Path, mapper: SymbolMapper) -> None:
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
            targets = determine_output_symbols(symbol_col, mapper)
            pretty_name = targets[0][1] or symbol_col
            target_labels = ", ".join(sym for sym, _ in targets)
            print(f"Processing {pretty_name} → {target_labels}...")

            # Extract single symbol series
            prices = df[symbol_col].dropna()

            if len(prices) == 0:
                print(f"  WARNING: {pretty_name} has no data")
                continue

            # Create DataFrame with standard schema
            symbol_df = pd.DataFrame({'price': prices})

            # Save to Parquet
            for output_symbol, _ in targets:
                output_path = output_dir / f"{output_symbol}.parquet"
                symbol_df.to_parquet(output_path, engine='pyarrow', compression='snappy')
                print(f"  ✓ Converted {output_symbol}: {len(symbol_df)} rows ({symbol_df.index.min()} to {symbol_df.index.max()})")

    except Exception as e:
        print(f"  ERROR processing Excel file: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bloomberg Excel/CSV exports into Parquet files."
    )
    parser.add_argument(
        "--symbol-map",
        action="append",
        type=Path,
        help=(
            "Path to a symbol map CSV (repeatable). "
            "Defaults to data/bloomberg/symbol_map.csv plus the expanded map if present."
        ),
    )
    return parser.parse_args()


def build_symbol_map_paths(project_root: Path, cli_paths: Optional[List[Path]]) -> List[Path]:
    if cli_paths:
        return [p if p.is_absolute() else (Path.cwd() / p) for p in cli_paths]

    bloomberg_dir = project_root / "data" / "bloomberg"
    paths = [bloomberg_dir / "symbol_map.csv"]
    expanded = bloomberg_dir / "symbol_map_expanded.csv"
    if expanded.exists():
        paths.append(expanded)
    return paths


def main():
    args = parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "bloomberg" / "raw"
    output_dir = project_root / "data" / "bloomberg"
    symbol_map_paths = build_symbol_map_paths(project_root, args.symbol_map)
    mapper = SymbolMapper(symbol_map_paths)

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
            convert_excel_to_parquet(excel_file, output_dir, mapper)

    # Also check for individual CSVs
    csv_files = list(raw_dir.glob("*.csv"))

    if csv_files:
        print(f"\nFound {len(csv_files)} CSV file(s)")
        print("Processing individual CSVs...")
        print()

        for csv_file in csv_files:
            convert_csv_to_parquet(csv_file, output_dir, mapper)

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
