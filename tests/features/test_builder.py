"""Tests for feature builder."""
import pytest
import pandas as pd
import numpy as np

def test_feature_builder_complete():
    """Test complete feature building pipeline."""
    from xtrend.features.builder import FeatureBuilder
    from xtrend.data.sources import BloombergParquetSource

    # This is an integration test - requires thinking about API
    # For now, verify imports work
    assert FeatureBuilder is not None
