"""Integration tests with real Bloomberg data."""
import pytest
from pathlib import Path
from xtrend.data.sources import BloombergParquetSource

@pytest.mark.skipif(
    not Path("data/bloomberg/processed").exists(),
    reason="Bloomberg data not available"
)
def test_real_bloomberg_data():
    """Test loading real Bloomberg data."""
    source = BloombergParquetSource()

    symbols = source.symbols()
    assert len(symbols) > 0
    print(f"Found {len(symbols)} symbols: {symbols[:5]}...")

    # Load first 5 symbols
    prices = source.load_prices(symbols[:5], start='2020-01-01', end='2020-12-31')
    assert len(prices) > 200  # Should have ~252 trading days

    # Check that at least some symbols have non-NaN data
    non_empty_cols = (~prices.isna().all()).sum()
    assert non_empty_cols > 0, "All columns are NaN"
    print(f"Found {non_empty_cols} symbols with data out of {len(symbols[:5])}")

    print(f"Loaded prices shape: {prices.shape}")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
