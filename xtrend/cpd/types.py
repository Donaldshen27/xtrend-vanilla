"""Type definitions for GP-CPD."""
from dataclasses import dataclass
from typing import List, NamedTuple

import pandas as pd


@dataclass
class CPDConfig:
    """Configuration for GP Change-Point Detection.

    Attributes:
        lookback: Window size for CPD detection (trading days)
        threshold: Severity threshold for detecting change-points [0, 1]
        min_length: Minimum regime length (trading days)
        max_length: Maximum regime length (trading days)
    """
    lookback: int = 21
    threshold: float = 0.9
    min_length: int = 5
    max_length: int = 21

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_length <= 0:
            raise ValueError(f"min_length ({self.min_length}) must be positive")
        if self.max_length <= 0:
            raise ValueError(f"max_length ({self.max_length}) must be positive")
        if self.lookback <= 0:
            raise ValueError(f"lookback ({self.lookback}) must be positive")
        if self.lookback < self.min_length:
            raise ValueError(f"lookback ({self.lookback}) must be >= min_length ({self.min_length})")
        if self.min_length >= self.max_length:
            raise ValueError(f"min_length ({self.min_length}) must be < max_length ({self.max_length})")
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold ({self.threshold}) must be in [0, 1]")


class RegimeSegment(NamedTuple):
    """A single regime segment from CPD.

    Attributes:
        start_idx: Start index in price series
        end_idx: End index (inclusive) in price series
        severity: Detection severity [0, 1] (higher = stronger change-point)
        start_date: Start date of regime
        end_date: End date of regime
    """
    start_idx: int
    end_idx: int
    severity: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp


@dataclass
class RegimeSegments:
    """Collection of regime segments with validation methods.

    Attributes:
        segments: List of detected regime segments
        config: Configuration used for detection
    """
    segments: List[RegimeSegment]
    config: CPDConfig

    def __len__(self) -> int:
        """Number of detected regimes."""
        return len(self.segments)
