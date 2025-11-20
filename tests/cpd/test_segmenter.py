import pandas as pd
import torch

from xtrend.cpd import CPDConfig, GPCPDSegmenter


class DummyFitter:
    """Deterministic fitter to exercise pending CP logic without heavy GP work."""

    def fit_stationary_gp(self, x, y):
        return None, 0.0

    def fit_changepoint_gp(self, x, y, stationary_model):
        # Propose a split that leaves a 1-day right segment (pending)
        return None, 1.0, float(len(x) - 1)

    def compute_severity(self, log_mll_stationary, log_mll_changepoint):
        return 1.0


def test_pending_cp_when_jump_at_end():
    """Massive jump at the end should produce a pending CP without NaN severity."""
    n = 12
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Flat then jump on last day
    values = [2.0] * (n - 1) + [5.0]
    prices = pd.Series(values, index=dates)

    config = CPDConfig(
        lookback=12,
        threshold=0.3,  # permissive to trigger CP
        min_length=5,
        max_length=12,
    )
    segmenter = GPCPDSegmenter(config, fitter=DummyFitter())

    segments = segmenter.fit_segment(prices)

    # Should keep main segments valid and record at least one pending split
    assert len(segments.segments) >= 1
    assert len(segments.pending_segments) >= 1

    # All pending severities must be finite and in range
    for pending in segments.pending_segments:
        assert torch.isfinite(torch.tensor(pending.severity))
        assert 0.0 <= pending.severity <= 1.0
