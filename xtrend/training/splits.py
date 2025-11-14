"""
Time-based splits: prefer sklearn/sktime utilities.

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def expanding_windows(dates: "Sequence[Any]",
                      train_start: "Any",
                      window_years: int = 5) -> "List[Tuple[Any, Any, Any, Any]]":
    """
    Generate expanding-window (train, test) intervals.

    Notes:
        - In real code, prefer sktime's ExpandingWindowSplitter or sklearn's TimeSeriesSplit with custom folds.
    """
    pass
