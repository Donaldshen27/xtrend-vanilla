#!/usr/bin/env python3
"""
Convert Bloomberg CSV exports to the Pinnacle-aligned Parquet layout.

Usage:
    python scripts/convert_bloomberg_to_parquet.py INPUT_DIR [--output-dir DIR] [--symbol-map PATH ...]

Examples:
    # Convert CSVs from specific folder to default output location
    python scripts/convert_bloomberg_to_parquet.py data/bloomberg/future_data/original_individual_csvs

    # Specify custom output directory
    python scripts/convert_bloomberg_to_parquet.py data/bloomberg/raw --output-dir data/bloomberg/processed

Workflow:
1. Reads symbol metadata from the canonical & expanded symbol map CSVs
2. Scans the input directory for CSV files
3. Converts each CSV to individual Parquet files using Pinnacle IDs for filenames

Input format: Individual Bloomberg CSV files (one per symbol)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bloomberg CSV exports into Parquet files."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Bloomberg CSV files to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save Parquet files (defaults to data/bloomberg)",
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
    input_dir = args.input_dir if args.input_dir.is_absolute() else (Path.cwd() / args.input_dir)
    output_dir = args.output_dir if args.output_dir else (project_root / "data" / "bloomberg")
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    symbol_map_paths = build_symbol_map_paths(project_root, args.symbol_map)
    mapper = SymbolMapper(symbol_map_paths)

    # Validate input directory exists
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bloomberg Data Conversion: CSV → Parquet")
    print("=" * 60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"ERROR: No CSV files found in {input_dir}")
        print("\nExpected file pattern: *.csv")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s)")
    print("Processing individual CSVs...")
    print()

    for csv_file in csv_files:
        convert_csv_to_parquet(csv_file, output_dir, mapper)

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
