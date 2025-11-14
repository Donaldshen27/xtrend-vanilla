"""
Tables and figures using vectorbt + empyrical/quantstats (adapters only).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def build_results_table(results: "Any") -> "Any":
    """
    Construct a summary table of performance across windows and seeds.

    Args:
        results: Structured results object.

    Returns:
        Table-like object (e.g., pandas.DataFrame).
    """
    pass

def save_tables(tables: "Any", out_dir: str) -> None:
    """Persist tables to CSV/LaTeX (declaration only)."""
    pass

def generate_figures(returns_by_model: "Mapping[str, Any]", out_dir: str) -> None:
    """Generate equity/drawdown figures (declaration only)."""
    pass
