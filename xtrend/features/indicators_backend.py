"""
Indicator adapters for TA-Lib / 'ta' (no hand-rolled indicators).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
import numpy as np

def _extract_macd_column(macd_result: "Any") -> "Any":
    """
    Extract the MACD line from MACD indicator result DataFrame.

    Tries common column name patterns used by different libraries.

    Args:
        macd_result: DataFrame containing MACD indicator outputs

    Returns:
        Series containing the MACD line values

    Raises:
        ValueError: If MACD column cannot be identified
    """
    import pandas as pd

    # Try known column names in order of preference
    for col_pattern in ['trend_macd', 'MACD', 'macd']:
        matching_cols = [c for c in macd_result.columns
                        if col_pattern in c and 'signal' not in c.lower() and 'diff' not in c.lower()]
        if matching_cols:
            return macd_result[matching_cols[0]]

    # Fallback: assume first column is MACD line
    if len(macd_result.columns) >= 1:
        return macd_result.iloc[:, 0]

    raise ValueError(f"Could not extract MACD column from result with columns: {macd_result.columns.tolist()}")

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
        A DataFrame with MACD line, signal, and histogram.

    Notes:
        This is a wrapper around the ta library's MACD indicator.
    """
    import pandas as pd

    if backend == "ta":
        from ta.trend import MACD

        # ta library expects Series
        if isinstance(prices, pd.DataFrame):
            raise ValueError("MACD expects a Series, not DataFrame. Process one column at a time.")

        macd_indicator = MACD(close=prices, window_slow=long, window_fast=short, window_sign=signal)

        # Return DataFrame with MACD components
        return pd.DataFrame({
            'trend_macd': macd_indicator.macd(),
            'trend_macd_signal': macd_indicator.macd_signal(),
            'trend_macd_diff': macd_indicator.macd_diff(),
        }, index=prices.index)

    elif backend == "talib":
        import talib
        macd_line, signal_line, histogram = talib.MACD(
            prices.values,
            fastperiod=short,
            slowperiod=long,
            signalperiod=signal
        )
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
        }, index=prices.index)

    else:
        raise ValueError(f"Unknown backend: {backend}")

def macd_multi_scale(prices: "Any",
                     timescale_pairs: "List[Tuple[int, int]]" = [(8, 24), (16, 28), (32, 96)],
                     backend: Literal["talib", "ta"] = "ta") -> "Any":
    """
    Compute MACD for multiple timescale pairs across all assets.

    Args:
        prices: Wide DataFrame indexed by date (columns = symbols)
        timescale_pairs: List of (short, long) EMA pairs
        backend: 'talib' or 'ta'

    Returns:
        Wide DataFrame with MACD features for each asset and timescale pair
        Columns: [ASSET_MACD_SHORT_LONG, ...]

    Raises:
        TypeError: If prices is not a DataFrame
        ValueError: If prices is empty or timescale pairs are invalid

    Notes:
        Paper uses timescale pairs: (8,24), (16,28), (32,96)
    """
    import pandas as pd

    # Validate input DataFrame
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(prices).__name__}")

    if prices.empty:
        raise ValueError("Cannot compute MACD on empty DataFrame")

    # Validate timescale pairs
    for short, long in timescale_pairs:
        if short >= long:
            raise ValueError(f"Short period ({short}) must be < long period ({long})")
        if short < 2 or long < 2:
            raise ValueError(f"Periods must be >= 2, got short={short}, long={long}")

    macd_features = {}

    for asset in prices.columns:
        for short, long in timescale_pairs:
            # Compute MACD for this asset and timescale
            macd_result = macd(prices[asset], short=short, long=long, backend=backend)

            # Extract MACD line using robust helper function
            macd_line = _extract_macd_column(macd_result)

            # Store with descriptive column name
            col_name = f'{asset}_MACD_{short}_{long}'
            macd_features[col_name] = macd_line

    return pd.DataFrame(macd_features, index=prices.index)

def macd_normalized(prices: "Any",
                   short: int,
                   long: int,
                   norm_window: int = 252,
                   backend: Literal["talib", "ta"] = "ta") -> "Any":
    """
    Compute normalized MACD: MACD / rolling_std(MACD, norm_window).

    Args:
        prices: Series of close prices
        short: Fast EMA span
        long: Slow EMA span
        norm_window: Window for rolling normalization (default 252 days)
        backend: 'talib' or 'ta'

    Returns:
        Series of normalized MACD values

    Raises:
        TypeError: If prices is not a Series
        ValueError: If prices is empty or parameters are invalid

    Notes:
        Equation 4 from paper: MACD normalized by 252-day rolling std
    """
    import pandas as pd

    # Validate input Series
    if not isinstance(prices, pd.Series):
        raise TypeError(f"Expected Series, got {type(prices).__name__}")

    if prices.empty:
        raise ValueError("Cannot compute MACD on empty Series")

    # Validate parameters
    if short >= long:
        raise ValueError(f"Short period ({short}) must be < long period ({long})")
    if short < 2 or long < 2:
        raise ValueError(f"Periods must be >= 2, got short={short}, long={long}")
    if norm_window < 2:
        raise ValueError(f"Normalization window must be >= 2, got {norm_window}")

    # Calculate raw MACD
    macd_result = macd(prices, short=short, long=long, backend=backend)

    # Extract MACD line using robust helper function
    raw_macd = _extract_macd_column(macd_result)

    # Normalize by rolling standard deviation
    rolling_std = raw_macd.rolling(window=norm_window, min_periods=20).std()

    # Avoid division by zero
    normalized = raw_macd / rolling_std.replace(0, np.nan)

    return normalized
