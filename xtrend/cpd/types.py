"""Type definitions for GP-CPD."""
from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np
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
    lookback: int = 63  # Increased from 21 for more robust GP fits
    threshold: float = 0.7  # Tuned for real-market severity distribution
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

    def validate_statistics(self, prices: pd.Series) -> 'ValidationReport':
        """Enhanced statistical validation with dispersion and quality metrics.

        Args:
            prices: Original price series used for segmentation

        Returns:
            ValidationReport with statistical checks
        """
        from xtrend.cpd.validation import ValidationCheck, ValidationReport

        checks = []
        lengths = [seg.end_idx - seg.start_idx + 1 for seg in self.segments]

        # 1. Length statistics
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        cv_length = std_length / mean_length if mean_length > 0 else 0

        checks.append(ValidationCheck(
            name="Mean regime length",
            expected=(10, 15),
            actual=f"{mean_length:.1f}",
            passed=(10 <= mean_length <= 15)
        ))

        checks.append(ValidationCheck(
            name="Length dispersion (CV)",
            expected=(0.3, 0.7),
            actual=f"{cv_length:.2f}",
            passed=(0.3 <= cv_length <= 0.7)
        ))

        # 2. Min/max constraints
        all_within_bounds = all(
            self.config.min_length <= length <= self.config.max_length
            for length in lengths
        )

        checks.append(ValidationCheck(
            name="All segments within length bounds",
            expected=f"[{self.config.min_length}, {self.config.max_length}]",
            actual="Yes" if all_within_bounds else "No",
            passed=all_within_bounds
        ))

        # 3. No gaps or overlaps
        sorted_segs = sorted(self.segments, key=lambda s: s.start_idx)
        no_gaps = all(
            sorted_segs[i].end_idx + 1 == sorted_segs[i+1].start_idx
            for i in range(len(sorted_segs) - 1)
        )

        checks.append(ValidationCheck(
            name="No gaps or overlaps",
            expected="Contiguous coverage",
            actual="Yes" if no_gaps else "No",
            passed=no_gaps
        ))

        # 4. Full coverage check
        full_coverage = (
            len(sorted_segs) > 0 and
            sorted_segs[0].start_idx == 0 and
            sorted_segs[-1].end_idx == len(prices) - 1
        )

        if sorted_segs:
            actual_range = f"[{sorted_segs[0].start_idx}, {sorted_segs[-1].end_idx}]"
        else:
            actual_range = "No segments"

        checks.append(ValidationCheck(
            name="Full coverage",
            expected=f"[0, {len(prices) - 1}]",
            actual=actual_range,
            passed=full_coverage
        ))

        # 5. Severity calibration
        severities = [seg.severity for seg in self.segments]
        severity_p90 = np.percentile(severities, 90) if severities else 0

        checks.append(ValidationCheck(
            name="Severity 90th percentile",
            expected=(0.85, 0.95),
            actual=f"{severity_p90:.3f}",
            passed=(0.85 <= severity_p90 <= 0.95)
        ))

        return ValidationReport(checks=checks)
