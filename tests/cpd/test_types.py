"""Tests for CPD types and configuration."""
import pandas as pd
import pytest
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments


class TestCPDConfig:
    def test_default_values(self):
        """CPDConfig has sensible defaults."""
        config = CPDConfig()

        assert config.lookback == 21
        assert config.threshold == 0.9
        assert config.min_length == 5
        assert config.max_length == 21

    def test_custom_values(self):
        """CPDConfig accepts custom parameters."""
        config = CPDConfig(
            lookback=42,
            threshold=0.85,
            min_length=10,
            max_length=30
        )

        assert config.lookback == 42
        assert config.threshold == 0.85
        assert config.min_length == 10
        assert config.max_length == 30


class TestRegimeSegment:
    def test_creation(self):
        """RegimeSegment can be created with all fields."""
        seg = RegimeSegment(
            start_idx=0,
            end_idx=20,
            severity=0.95,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-31')
        )

        assert seg.start_idx == 0
        assert seg.end_idx == 20
        assert seg.severity == 0.95
        assert seg.start_date == pd.Timestamp('2020-01-01')
        assert seg.end_date == pd.Timestamp('2020-01-31')

    def test_length_property(self):
        """Regime length is end - start + 1."""
        seg = RegimeSegment(
            start_idx=10,
            end_idx=30,
            severity=0.9,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-31')
        )

        assert seg.end_idx - seg.start_idx + 1 == 21


class TestRegimeSegments:
    def test_empty_segments(self):
        """RegimeSegments can be created with empty list."""
        config = CPDConfig()
        segments = RegimeSegments(segments=[], config=config)

        assert len(segments) == 0

    def test_multiple_segments(self):
        """RegimeSegments tracks multiple regimes."""
        config = CPDConfig()
        segs = [
            RegimeSegment(0, 10, 0.9, pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-15')),
            RegimeSegment(11, 20, 0.85, pd.Timestamp('2020-01-16'), pd.Timestamp('2020-01-31')),
        ]
        segments = RegimeSegments(segments=segs, config=config)

        assert len(segments) == 2


class TestCPDConfigValidation:
    def test_negative_min_length_raises(self):
        """min_length must be positive (negative case)."""
        with pytest.raises(ValueError, match="min_length.*must be positive"):
            CPDConfig(min_length=-5, max_length=21, lookback=21)

    def test_negative_max_length_raises(self):
        """max_length must be positive (negative case)."""
        with pytest.raises(ValueError, match="max_length.*must be positive"):
            CPDConfig(min_length=5, max_length=-10, lookback=21)

    def test_negative_lookback_raises(self):
        """lookback must be positive (negative case)."""
        with pytest.raises(ValueError, match="lookback.*must be positive"):
            CPDConfig(lookback=-21, min_length=5, max_length=21)

    def test_zero_min_length_raises(self):
        """min_length must be positive (zero case)."""
        with pytest.raises(ValueError, match="min_length.*must be positive"):
            CPDConfig(min_length=0, max_length=21, lookback=21)

    def test_zero_max_length_raises(self):
        """max_length must be positive (zero case)."""
        with pytest.raises(ValueError, match="max_length.*must be positive"):
            CPDConfig(min_length=5, max_length=0, lookback=21)

    def test_zero_lookback_raises(self):
        """lookback must be positive (zero case)."""
        with pytest.raises(ValueError, match="lookback.*must be positive"):
            CPDConfig(lookback=0, min_length=5, max_length=21)

    def test_lookback_less_than_min_length_raises(self):
        """lookback must be >= min_length."""
        with pytest.raises(ValueError, match="lookback.*must be >= min_length"):
            CPDConfig(lookback=3, min_length=5)

    def test_min_length_gte_max_length_raises(self):
        """min_length must be < max_length."""
        with pytest.raises(ValueError, match="min_length.*must be < max_length"):
            CPDConfig(min_length=21, max_length=21)

    def test_threshold_out_of_range_raises(self):
        """threshold must be in [0, 1]."""
        with pytest.raises(ValueError, match="threshold.*must be in"):
            CPDConfig(threshold=1.5)
