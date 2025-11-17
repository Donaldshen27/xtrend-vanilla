"""Tests for validation methods."""
import numpy as np
import pandas as pd
import pytest
from xtrend.cpd import CPDConfig, GPCPDSegmenter


class TestStatisticalValidation:
    def test_validate_statistics_runs_without_error(self, sample_prices):
        """validate_statistics executes without error."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)
        report = segments.validate_statistics(sample_prices)

        # Should return a report with checks
        assert hasattr(report, 'checks')
        assert len(report.checks) > 0

    def test_validation_checks_length_statistics(self, sample_prices):
        """Validation includes mean length and dispersion checks."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)
        report = segments.validate_statistics(sample_prices)

        # Check that length statistics are included
        check_names = [check.name for check in report.checks]
        assert any('length' in name.lower() for name in check_names)


class TestValidationReport:
    def test_validation_report_str_formatting(self, sample_prices):
        """ValidationReport formats nicely as string."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)
        report = segments.validate_statistics(sample_prices)

        report_str = str(report)

        # Should contain header
        assert "Validation Report" in report_str

        # Should show pass/fail counts
        assert "Passed:" in report_str

        # Should show check details
        assert "Expected:" in report_str
        assert "Actual:" in report_str
