"""Data loading utilities for Bloomberg parquet files."""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


DATA_DIR = Path("data/bloomberg")


@st.cache_data
def get_available_symbols() -> List[str]:
    """Scan processed/ directory for parquet files.

    Returns:
        Sorted list of symbol names (without .parquet extension)
    """
    if not DATA_DIR.exists():
        return []
    return sorted([p.stem for p in DATA_DIR.glob("*.parquet")])


@st.cache_data
def load_symbol(symbol: str) -> pd.DataFrame:
    """Load single parquet file with caching.

    Args:
        symbol: Symbol name (without .parquet extension)

    Returns:
        DataFrame with datetime index and 'price' column
    """
    path = DATA_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def load_bloomberg_data(
    symbols: List[str],
    date_range: Tuple[datetime, datetime]
) -> Dict[str, pd.DataFrame]:
    """Load multiple symbols and filter by date range.

    Args:
        symbols: List of symbol names to load
        date_range: (start_date, end_date) tuple

    Returns:
        Dict mapping symbol -> filtered DataFrame
    """
    data = {}
    for symbol in symbols:
        df = load_symbol(symbol)  # Cached per symbol
        # Filter to date range
        mask = (df.index >= date_range[0]) & (df.index <= date_range[1])
        data[symbol] = df.loc[mask]
    return data


def get_date_range(symbols: List[str]) -> Tuple[datetime, datetime]:
    """Get min and max dates across all selected symbols.

    Args:
        symbols: List of symbol names

    Returns:
        (min_date, max_date) tuple
    """
    if not symbols:
        # Default range if no symbols selected
        return (datetime(1990, 1, 1), datetime(2025, 12, 31))

    all_dates = []
    for symbol in symbols:
        df = load_symbol(symbol)
        if not df.empty:
            all_dates.extend([df.index.min(), df.index.max()])

    if not all_dates:
        return (datetime(1990, 1, 1), datetime(2025, 12, 31))

    return (min(all_dates), max(all_dates))
