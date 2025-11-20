"""Tests for GPCPDSegmenter."""
import numpy as np
import pandas as pd
import pytest
from xtrend.cpd import CPDConfig, GPCPDSegmenter, GPFitter


class TestGPCPDSegmenter:
    def test_segmenter_initialization(self):
        """GPCPDSegmenter can be initialized with config."""
        config = CPDConfig(lookback=21, threshold=0.9)
        segmenter = GPCPDSegmenter(config)

        assert segmenter.config == config

    def test_fit_segment_returns_regime_segments(self, sample_prices):
        """fit_segment returns RegimeSegments object."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        fast_fitter = GPFitter(max_iter=15, patience=2, lr=0.15,
                               grid_stride=5, grid_max_candidates=6, quick_iter=5)
        segmenter = GPCPDSegmenter(config, fitter=fast_fitter)

        segments = segmenter.fit_segment(sample_prices)

        # Should return RegimeSegments
        assert hasattr(segments, 'segments')
        assert hasattr(segments, 'config')
        assert segments.config == config


class TestSegmentationProperties:
    def test_no_gaps_or_overlaps(self, sample_prices):
        """Segmentation produces contiguous coverage."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        fast_fitter = GPFitter(max_iter=15, patience=2, lr=0.15,
                               grid_stride=5, grid_max_candidates=6, quick_iter=5)
        segmenter = GPCPDSegmenter(config, fitter=fast_fitter)

        segments = segmenter.fit_segment(sample_prices)

        # Should have at least one segment
        assert len(segments) > 0

        # Check contiguity
        sorted_segs = sorted(segments.segments, key=lambda s: s.start_idx)

        # First segment should start at 0
        assert sorted_segs[0].start_idx == 0

        # Last segment should end at len(prices) - 1
        assert sorted_segs[-1].end_idx == len(sample_prices) - 1

        # No gaps between segments
        for i in range(len(sorted_segs) - 1):
            assert sorted_segs[i].end_idx + 1 == sorted_segs[i+1].start_idx

    def test_all_segments_within_length_bounds(self, sample_prices):
        """All segments satisfy min/max length constraints."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        fast_fitter = GPFitter(max_iter=15, patience=2, lr=0.15,
                               grid_stride=5, grid_max_candidates=6, quick_iter=5)
        segmenter = GPCPDSegmenter(config, fitter=fast_fitter)

        segments = segmenter.fit_segment(sample_prices)

        for seg in segments.segments:
            length = seg.end_idx - seg.start_idx + 1
            assert config.min_length <= length <= config.max_length

    def test_stable_market_reaches_max_length(self):
        """Stable series should accumulate to full max_length (not lookback)."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        prices = pd.Series(np.ones(30), index=dates, name='Close')

        # lookback < max_length to expose the previous bug
        config = CPDConfig(lookback=5, threshold=0.95, min_length=2, max_length=10)
        fast_fitter = GPFitter(max_iter=10, patience=2, lr=0.2,
                               grid_stride=2, grid_max_candidates=4, quick_iter=4)
        segmenter = GPCPDSegmenter(config, fitter=fast_fitter)

        segments = segmenter.fit_segment(prices)
        lengths = [seg.end_idx - seg.start_idx + 1 for seg in segments.segments]

        assert max(lengths) == config.max_length
        assert any(length > config.lookback for length in lengths)


class TestKnownEventDetection:
    def test_detects_obvious_regime_change(self):
        """Detects obvious regime change in synthetic data."""
        # Create synthetic data with clear regime change at day 50
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Regime 1: low volatility
        regime1 = 100 + np.cumsum(np.random.randn(50) * 0.1)
        # Regime 2: high volatility + different mean
        regime2 = 110 + np.cumsum(np.random.randn(50) * 0.5)

        prices = pd.Series(
            np.concatenate([regime1, regime2]),
            index=dates,
            name='Close'
        )

        config = CPDConfig(lookback=21, threshold=0.85, min_length=5, max_length=30)
        fast_fitter = GPFitter(max_iter=15, patience=2, lr=0.1,
                               grid_stride=5, grid_max_candidates=6, quick_iter=5)
        segmenter = GPCPDSegmenter(config, fitter=fast_fitter)

        segments = segmenter.fit_segment(prices)

        # Should detect change-point near day 50 (±10 days tolerance)
        change_points = [seg.start_idx for seg in segments.segments[1:]]  # Skip first

        nearby_cps = [cp for cp in change_points if abs(cp - 50) <= 10]
        assert len(nearby_cps) > 0, "Should detect regime change near day 50"
