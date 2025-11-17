"""GP-based change-point detection segmenter."""
from typing import List

import numpy as np
import pandas as pd
import torch

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments


class GPCPDSegmenter:
    """Segment time-series into regimes using GP change-point detection.

    Implements recursive backward segmentation from X-Trend paper Algorithm 1.
    """

    def __init__(self, config: CPDConfig):
        """Initialize segmenter.

        Args:
            config: CPD configuration
        """
        self.config = config
        self.fitter = GPFitter()

    def fit_segment(self, prices: pd.Series) -> RegimeSegments:
        """Segment entire price series into regimes.

        Uses recursive backward segmentation from X-Trend paper Algorithm 1.

        Args:
            prices: Price time-series with DatetimeIndex

        Returns:
            RegimeSegments containing detected regimes
        """
        segments = []
        current_end = len(prices) - 1

        while current_end >= self.config.min_length:
            window_start = max(0, current_end - self.config.lookback + 1)
            window = prices.iloc[window_start:current_end + 1]

            # Handle leftover stub at beginning
            if len(window) < self.config.min_length:
                if current_end + 1 >= self.config.min_length:
                    segments.append(RegimeSegment(
                        start_idx=0,
                        end_idx=current_end,
                        severity=0.0,
                        start_date=prices.index[0],
                        end_date=prices.index[current_end]
                    ))
                break

            # Detect change-point in window
            x = torch.arange(len(window)).float().unsqueeze(-1)
            y = torch.tensor(window.values).float()

            stat_model, log_mll_M = self.fitter.fit_stationary_gp(x, y)
            cp_model, log_mll_C, t_cp_relative = self.fitter.fit_changepoint_gp(
                x, y, stat_model
            )
            severity = self.fitter.compute_severity(log_mll_M, log_mll_C)

            if severity >= self.config.threshold:
                # Change-point detected!
                t_cp_absolute = window_start + round(t_cp_relative)

                # Validate CP creates valid regimes on both sides
                regime_length = current_end - t_cp_absolute + 1
                left_length = t_cp_absolute - window_start

                if (regime_length >= self.config.min_length and
                    left_length >= self.config.min_length):
                    # Valid CP
                    segments.append(RegimeSegment(
                        start_idx=t_cp_absolute,
                        end_idx=current_end,
                        severity=severity,
                        start_date=prices.index[t_cp_absolute],
                        end_date=prices.index[current_end]
                    ))
                    current_end = t_cp_absolute - 1
                else:
                    # CP too close to edge, treat as no CP
                    regime_len = min(self.config.max_length, current_end - window_start + 1)
                    segments.append(RegimeSegment(
                        start_idx=current_end - regime_len + 1,
                        end_idx=current_end,
                        severity=severity,
                        start_date=prices.index[current_end - regime_len + 1],
                        end_date=prices.index[current_end]
                    ))
                    current_end -= regime_len
            else:
                # No change-point detected - jump by max_length
                regime_len = min(self.config.max_length, current_end - window_start + 1)

                if regime_len >= self.config.min_length:
                    segments.append(RegimeSegment(
                        start_idx=current_end - regime_len + 1,
                        end_idx=current_end,
                        severity=severity,
                        start_date=prices.index[current_end - regime_len + 1],
                        end_date=prices.index[current_end]
                    ))
                    current_end -= regime_len
                else:
                    current_end -= 1

        # Reverse (built backward)
        segments.reverse()

        return RegimeSegments(segments=segments, config=self.config)
