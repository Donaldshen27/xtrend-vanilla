"""End-to-end integration tests for Phase 2 GP-CPD."""
import pandas as pd
import pytest

from xtrend.cpd import CPDConfig, GPCPDSegmenter
from xtrend.data.sources import BloombergParquetSource


class TestPhase2Integration:
    @pytest.mark.skip(reason="Requires Bloomberg data files")
    def test_full_pipeline_on_real_data(self):
        """Full Phase 2 pipeline on real Bloomberg data."""
        # Load real data
        source = BloombergParquetSource()
        prices = source.load_symbol('ES', '2020-01-01', '2020-12-31')['Close']

        # Run CPD
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)
        segments = segmenter.fit_segment(prices)

        # Basic checks
        assert len(segments) > 0
        assert all(s.end_idx - s.start_idx + 1 >= config.min_length for s in segments.segments)

        # Validate statistics
        report = segments.validate_statistics(prices)
        passed_count = sum(1 for c in report.checks if c.passed)

        # At least 60% of checks should pass
        assert passed_count / len(report.checks) >= 0.6

    def test_covid_detection_on_synthetic_data(self):
        """Detect COVID-like crash in synthetic data."""
        # Create synthetic data with COVID-like crash
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

        # Pre-COVID: stable
        pre_covid = 100 + pd.Series(range(60)) * 0.1

        # COVID crash: sharp drop
        covid_crash = pd.Series([100 - i * 2 for i in range(20)])

        # Post-COVID: recovery
        post_covid = 60 + pd.Series(range(len(dates) - 80)) * 0.2

        prices = pd.concat([
            pd.Series(pre_covid.values, index=dates[:60]),
            pd.Series(covid_crash.values, index=dates[60:80]),
            pd.Series(post_covid.values, index=dates[80:])
        ])

        # Run CPD
        config = CPDConfig(lookback=21, threshold=0.85, min_length=5, max_length=30)
        segmenter = GPCPDSegmenter(config)
        segments = segmenter.fit_segment(prices)

        # Should detect change-point near day 60 (±10 days)
        change_points = [seg.start_idx for seg in segments.segments[1:]]
        nearby_cps = [cp for cp in change_points if abs(cp - 60) <= 10]

        assert len(nearby_cps) > 0, "Should detect COVID-like crash"
