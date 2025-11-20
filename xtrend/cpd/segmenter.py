"""GP-based change-point detection segmenter."""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments


class GPCPDSegmenter:
    """Segment time-series into regimes using GP change-point detection.

    Implements recursive backward segmentation from X-Trend paper Algorithm 1.
    """

    def __init__(self, config: CPDConfig, fitter: Optional[GPFitter] = None):
        """Initialize segmenter.

        Args:
            config: CPD configuration
            fitter: Optional GP fitter (useful to inject faster settings in tests)
        """
        self.config = config
        self.fitter = fitter or GPFitter()

    def fit_segment(self, prices: pd.Series) -> RegimeSegments:
        """Segment entire price series into regimes.

        Uses recursive backward segmentation from X-Trend paper Algorithm 1.

        Args:
            prices: Price time-series with DatetimeIndex

        Returns:
            RegimeSegments containing detected regimes
        """
        segments = []
        t1 = len(prices) - 1  # Current regime end (inclusive)
        t = t1                # Scanning pointer that walks backward

        while t >= 0:
            remaining_len = t1 + 1  # Unsegmented portion length
            window_start = max(0, t - self.config.lookback + 1)
            window = prices.iloc[window_start:t + 1]

            # Handle leftover stub only when the entire remaining data is too short
            if remaining_len < self.config.min_length:
                if segments:
                    last_seg = segments[-1]
                    segments[-1] = RegimeSegment(
                        start_idx=0,
                        end_idx=last_seg.end_idx,
                        severity=last_seg.severity,
                        start_date=prices.index[0],
                        end_date=last_seg.end_date
                    )
                elif t1 >= 0:
                    segments.append(RegimeSegment(
                        start_idx=0,
                        end_idx=t1,
                        severity=0.0,
                        start_date=prices.index[0],
                        end_date=prices.index[t1]
                    ))
                break

            # Detect change-point in window ending at t (Algorithm 1)
            x = torch.arange(len(window)).float().unsqueeze(-1)
            y = torch.tensor(window.values).float()

            stat_model, log_mll_M = self.fitter.fit_stationary_gp(x, y)
            cp_model, log_mll_C, t_cp_relative = self.fitter.fit_changepoint_gp(
                x, y, stat_model
            )
            severity = self.fitter.compute_severity(log_mll_M, log_mll_C)

            committed_cp = False

            if severity >= self.config.threshold:
                t_cp_absolute = window_start + round(t_cp_relative)

                right_length = t1 - t_cp_absolute + 1
                left_length = t_cp_absolute - window_start

                if (right_length >= self.config.min_length and
                    left_length >= self.config.min_length):

                    # Clamp to max_length if CP-generated regime is too long
                    if right_length > self.config.max_length:
                        t_cp_absolute = t1 - self.config.max_length + 1
                        right_length = self.config.max_length

                    segments.append(RegimeSegment(
                        start_idx=t_cp_absolute,
                        end_idx=t1,
                        severity=severity,
                        start_date=prices.index[t_cp_absolute],
                        end_date=prices.index[t1]
                    ))

                    # Reset for next regime (Algorithm 1 lines 10-11)
                    t1 = t_cp_absolute - 1
                    t = t1
                    committed_cp = True

            if committed_cp:
                continue

            # No change-point detected: move back by one step (Algorithm 1 line 13)
            t -= 1

            # Prevent regime from exceeding max_length (Algorithm 1 line 14)
            if t >= 0 and (t1 - t + 1) > self.config.max_length:
                t = t1 - self.config.max_length + 1

            # If we've stepped past the start, commit remaining data
            if t < 0:
                if t1 >= 0:
                    segments.append(RegimeSegment(
                        start_idx=0,
                        end_idx=t1,
                        severity=severity,
                        start_date=prices.index[0],
                        end_date=prices.index[t1]
                    ))
                break

            # Commit regime once it reaches max_length (Algorithm 1 lines 17-20)
            if (t1 - t + 1) == self.config.max_length:
                segments.append(RegimeSegment(
                    start_idx=t,
                    end_idx=t1,
                    severity=severity,
                    start_date=prices.index[t],
                    end_date=prices.index[t1]
                ))
                t1 = t - 1
                t = t1

        # Reverse (built backward)
        segments.reverse()

        return RegimeSegments(segments=segments, config=self.config)
