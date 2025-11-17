"""Tests for data sources."""
import pytest
import pandas as pd
from pathlib import Path
from xtrend.data.sources import BloombergParquetSource

def test_bloomberg_symbols(temp_bloomberg_data):
    """Test that symbols() returns available parquet files."""
    source = BloombergParquetSource(root_path=f"{temp_bloomberg_data}/processed")
    symbols = source.symbols()

    assert isinstance(symbols, list)
    assert set(symbols) == {'ES', 'CL', 'GC'}
    assert len(symbols) == 3

def test_bloomberg_symbols_empty_directory(tmp_path):
    """Test symbols() with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    source = BloombergParquetSource(root_path=str(empty_dir))

    assert source.symbols() == []

def test_bloomberg_load_prices(temp_bloomberg_data):
    """Test loading prices for specific symbols."""
    source = BloombergParquetSource(root_path=f"{temp_bloomberg_data}/processed")

    prices = source.load_prices(['ES', 'CL'])

    # Check structure
    assert isinstance(prices, pd.DataFrame)
    assert list(prices.columns) == ['ES', 'CL']
    assert prices.index.name == 'date'
    assert len(prices) == 366  # 2020 is leap year
    assert not prices.isna().any().any()

def test_bloomberg_load_prices_with_date_filter(temp_bloomberg_data):
    """Test loading prices with date range."""
    source = BloombergParquetSource(root_path=f"{temp_bloomberg_data}/processed")

    prices = source.load_prices(['ES'], start='2020-06-01', end='2020-06-30')

    assert len(prices) == 30
    assert prices.index.min() >= pd.Timestamp('2020-06-01')
    assert prices.index.max() <= pd.Timestamp('2020-06-30')

def test_bloomberg_load_prices_missing_symbol(temp_bloomberg_data):
    """Test error handling for missing symbol."""
    source = BloombergParquetSource(root_path=f"{temp_bloomberg_data}/processed")

    with pytest.raises(FileNotFoundError, match="Symbol not found"):
        source.load_prices(['INVALID'])
