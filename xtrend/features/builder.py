"""
Feature construction using pandas.ewm + indicator adapters.

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def make_features(prices: "Any",
                  returns: "Any",
                  sigma: "Any",
                  horizons: "Sequence[int]" = (1, 21, 63, 126, 252),
                  macd_pairs: "Sequence[Tuple[int, int]]" = ((8, 24), (16, 28), (32, 96)),
                  indicator_backend: Literal["talib", "ta"] = "ta") -> "FeatureMatrix":
    """
    Build feature matrix for the model.

    Blocks:
        1) Normalized multi-horizon returns.
        2) MACD features via TA-Lib or 'ta' backend.
        3) Optional standardization using EWM vol.

    Args:
        prices: Price panel (wide DataFrame).
        returns: Daily returns panel aligned to prices.
        sigma: Ex-ante volatility panel.
        horizons: Lookback horizons in trading days.
        macd_pairs: Short/long EMA pairs for MACD.
        indicator_backend: Which indicator library to use.

    Returns:
        FeatureMatrix with names and metadata.
    """
    pass
