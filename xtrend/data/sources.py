"""
Data sources & storage adapters (no vendor logic).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
class DataSource(Protocol):
    """
    Protocol for data sources that provide continuous futures panels.

    Methods:
        symbols() -> List[str]
        load_prices(symbols: Sequence[str], start=None, end=None) -> DataFrame[date × symbol]
    """
    def symbols(self) -> "List[str]": ...
    def load_prices(self, symbols: "Sequence[str]", start: Optional[Any] = None, end: Optional[Any] = None) -> "Any": ...

class PinnacleCLCSource:
    """
    Thin adapter for Pinnacle Data CLC (ratio-adjusted continuous futures).

    Notes:
        - Do NOT implement roll logic; expect vendor-provided CLC series.
        - Provide a loader that returns a wide DataFrame indexed by date with columns per symbol.
    """
    def __init__(self, root_path: str):
        """Initialize with root path to vendor files (declaration only)."""
        pass

    def symbols(self) -> "List[str]":
        """Return available Pinnacle CLC symbols (declaration only)."""
        pass

    def load_prices(self, symbols: "Sequence[str]", start: Optional[Any] = None, end: Optional[Any] = None) -> "Any":
        """Load price panel for given symbols and date range (declaration only)."""
        pass

class NasdaqQuandlContinuousSource:
    """
    Adapter for Nasdaq Data Link (Quandl) continuous futures where available.

    Notes:
        - Availability/access terms change; this adapter abstracts retrieval so the rest of the code stays stable.
        - Use vendor SDK/HTTP clients externally; here we only define the interface.
    """
    def __init__(self, api_key_env: str = "NASDAQ_API_KEY"):
        """Initialize with API key environment variable name (declaration only)."""
        pass

    def symbols(self) -> "List[str]":
        """Return supported symbols as maintained by your project (declaration only)."""
        pass

    def load_prices(self, symbols: "Sequence[str]", start: Optional[Any] = None, end: Optional[Any] = None) -> "Any":
        """Fetch price panel (wide DataFrame) for symbols and date range (declaration only)."""
        pass

class DuckDBParquetStore:
    """
    Local lake adapter for Parquet files with DuckDB/Polars-friendly layout.

    Methods:
        write_panel(df) -> None
        read_panel(symbols, start, end) -> DataFrame

    Notes:
        - Use this to cache normalized panels; no custom binary formats.
    """
    def __init__(self, root_path: str):
        """Initialize with a root folder for Parquet datasets (declaration only)."""
        pass

    def write_panel(self, df: "Any") -> None:
        """Persist a price panel to Parquet with a deterministic schema (declaration only)."""
        pass

    def read_panel(self, symbols: "Sequence[str]", start: Optional[Any] = None, end: Optional[Any] = None) -> "Any":
        """Load a price panel subset into memory (declaration only)."""
        pass

class BloombergParquetSource:
    """
    Adapter for Bloomberg continuous futures data stored as local Parquet files.

    Data flow:
        1. Export from Bloomberg Terminal using Excel BDH() formula
        2. Convert to Parquet using scripts/convert_bloomberg_to_parquet.py
        3. Load using this class

    File structure:
        data/bloomberg/[SYMBOL].parquet
        - Each file contains: date index, 'price' column
        - Example: data/bloomberg/CL1.parquet

    Methods:
        symbols() -> List[str]
        load_prices(symbols, start, end) -> DataFrame[date × symbol]

    Notes:
        - Expects one Parquet file per symbol
        - Parquet schema: date (index), price (float)
        - Returns wide DataFrame with dates as index, symbols as columns
    """
    def __init__(self, root_path: str = "data/bloomberg/processed"):
        """
        Initialize with root path to Bloomberg Parquet files.

        Args:
            root_path: Directory containing [SYMBOL].parquet files
        """
        from pathlib import Path
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")

    def symbols(self) -> "List[str]":
        """
        Return available Bloomberg symbols from Parquet files.

        Returns:
            List of symbol names (e.g., ['CL', 'ES', 'GC'])
        """
        import glob
        from pathlib import Path
        parquet_files = glob.glob(str(self.root_path / "*.parquet"))
        symbols = [Path(f).stem for f in parquet_files]
        return sorted(symbols)

    def load_prices(self, symbols: "Sequence[str]", start: Optional[Any] = None, end: Optional[Any] = None) -> "Any":
        """
        Load price panel for given symbols and date range.

        Args:
            symbols: List of symbols to load (e.g., ['CL', 'ES'])
            start: Start date (optional, format: 'YYYY-MM-DD' or datetime)
            end: End date (optional, format: 'YYYY-MM-DD' or datetime)

        Returns:
            Wide DataFrame with:
                - Index: dates
                - Columns: symbols
                - Values: prices

        Raises:
            FileNotFoundError: If symbol Parquet file not found
            ValueError: If invalid date range
        """
        import pandas as pd
        from pathlib import Path

        # Load each symbol and combine into wide format
        price_dfs = {}
        for symbol in symbols:
            parquet_path = self.root_path / f"{symbol}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"Symbol not found: {symbol} at {parquet_path}")

            df = pd.read_parquet(parquet_path)
            # Extract price column, rename to symbol
            price_dfs[symbol] = df['price']

        # Combine into wide DataFrame
        prices = pd.DataFrame(price_dfs)
        prices.index.name = 'date'

        # Apply date filter if specified
        if start is not None:
            prices = prices[prices.index >= pd.Timestamp(start)]
        if end is not None:
            prices = prices[prices.index <= pd.Timestamp(end)]

        if len(prices) == 0:
            raise ValueError(f"No data in date range: {start} to {end}")

        return prices
