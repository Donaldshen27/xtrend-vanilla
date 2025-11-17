"""Shared test fixtures for X-Trend tests."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(42)
    # Simulate 3 assets with realistic price movements
    prices_data = {
        'ES': 3000 + np.cumsum(np.random.randn(len(dates)) * 10),
        'CL': 50 + np.cumsum(np.random.randn(len(dates)) * 2),
        'GC': 1500 + np.cumsum(np.random.randn(len(dates)) * 20),
    }
    return pd.DataFrame(prices_data, index=dates)

@pytest.fixture
def sample_returns(sample_prices):
    """Create sample returns data."""
    return sample_prices.pct_change()

@pytest.fixture
def temp_bloomberg_data(tmp_path):
    """Create temporary Bloomberg parquet files for testing."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    np.random.seed(42)

    bloomberg_dir = tmp_path / "bloomberg" / "processed"
    bloomberg_dir.mkdir(parents=True)

    # Create sample parquet files for 3 symbols
    for symbol in ['ES', 'CL', 'GC']:
        prices = 100 + np.cumsum(np.random.randn(len(dates)))
        df = pd.DataFrame({'price': prices}, index=dates)
        df.index.name = 'date'
        df.to_parquet(bloomberg_dir / f"{symbol}.parquet")

    return str(bloomberg_dir.parent)
