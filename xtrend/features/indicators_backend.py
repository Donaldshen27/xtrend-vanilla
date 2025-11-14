"""
Indicator adapters for TA-Lib / 'ta' (no hand-rolled indicators).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def macd(prices: "Any",
         short: int,
         long: int,
         signal: int = 9,
         backend: Literal["talib", "ta"] = "ta") -> "Any":
    """
    Compute MACD using TA-Lib or 'ta' library.

    Args:
        prices: Series or DataFrame of close prices.
        short: Fast EMA span.
        long: Slow EMA span.
        signal: Signal EMA span.
        backend: 'talib' (C-backed) or 'ta' (pure Python).

    Returns:
        A structure with MACD line, signal, and histogram.

    Notes:
        This is a declaration only; actual calls delegate to chosen backend.
    """
    pass
