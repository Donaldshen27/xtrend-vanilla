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
