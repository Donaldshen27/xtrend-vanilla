"""
Baseline forecaster without cross-attention (DMN/LSTM).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
class DMNBaseline:
    """
    Simple neural forecaster mapping features to a bounded position.

    Notes:
        - Train with the same joint losses as X‑Trend (without contexts).
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        """Initialize baseline (declaration only)."""
        pass
    def forward(self, X: "Any") -> "Any":
        """Return positions in [-1,1] (declaration only)."""
        pass
