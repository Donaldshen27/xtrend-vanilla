"""Smoke test to verify pytest works."""

def test_imports():
    """Test that core libraries import successfully."""
    import pandas as pd
    import numpy as np
    import ta
    assert pd.__version__
    assert np.__version__

def test_fixtures(sample_prices):
    """Test that fixtures work."""
    assert len(sample_prices) == 366  # 2020 is a leap year
    assert list(sample_prices.columns) == ['ES', 'CL', 'GC']
