"""
Vectorized backtesting via vectorbt (adapter only).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def evaluate_with_vectorbt(prices: "Any",
                           signals: "Any",
                           sigma_t: "Any",
                           sigma_target: float,
                           costs_bps: float = 0.0) -> "Any":
    """
    Run a vectorized backtest using vectorbt.

    Args:
        prices: Wide price panel.
        signals: Wide panel of raw or vol-targeted positions.
        sigma_t: Ex-ante volatility panel.
        sigma_target: Target volatility for scaling.
        costs_bps: Round-trip cost in basis points for what-if analysis.

    Returns:
        Portfolio returns series suitable for reporting.

    Notes:
        This skeleton only declares the interface.
    """
    pass
