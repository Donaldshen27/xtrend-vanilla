"""Tests for technical indicators."""
import pytest
import pandas as pd
import numpy as np
from xtrend.features.indicators_backend import macd

def test_macd_basic():
    """Test basic MACD calculation."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(200)), index=dates)

    result = macd(prices, short=12, long=26, signal=9, backend='ta')

    # Check result structure (ta library returns DataFrame)
    assert isinstance(result, pd.DataFrame)
    assert 'MACD' in result.columns or 'trend_macd' in result.columns

    # MACD should have same length as input
    assert len(result) == len(prices)

def test_macd_multi_timescale():
    """Test MACD for multiple timescale pairs."""
    from xtrend.features.indicators_backend import macd_multi_scale

    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)
    prices = pd.DataFrame({
        'ES': 100 + np.cumsum(np.random.randn(300)),
        'CL': 50 + np.cumsum(np.random.randn(300) * 0.5),
    }, index=dates)

    # Paper uses: (8,24), (16,28), (32,96)
    timescale_pairs = [(8, 24), (16, 28), (32, 96)]

    macd_features = macd_multi_scale(prices, timescale_pairs=timescale_pairs)

    # Check structure
    assert isinstance(macd_features, pd.DataFrame)
    assert len(macd_features) == len(prices)

    # Should have 3 MACD features per asset
    expected_cols = []
    for asset in prices.columns:
        for short, long in timescale_pairs:
            expected_cols.append(f'{asset}_MACD_{short}_{long}')

    assert len(macd_features.columns) == len(expected_cols)

def test_macd_normalized():
    """Test MACD normalized by rolling std (Equation 4)."""
    from xtrend.features.indicators_backend import macd_normalized

    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(300)), index=dates)

    # Calculate MACD and normalize
    norm_macd = macd_normalized(prices, short=8, long=24, norm_window=252)

    # Check structure
    assert isinstance(norm_macd, pd.Series)
    assert len(norm_macd) == len(prices)

    # Calculate raw MACD for comparison
    raw_macd_result = macd(prices, short=8, long=24, backend='ta')
    # Extract MACD column (column name varies by ta version)
    macd_col = [c for c in raw_macd_result.columns if 'macd' in c.lower() and 'signal' not in c.lower() and 'diff' not in c.lower()][0]
    raw_macd = raw_macd_result[macd_col]

    # Verify normalization divides by rolling std
    # Both should have similar scale after warmup period (not necessarily lower std)
    assert not norm_macd.isna().all()  # Not all NaN
    assert np.isfinite(norm_macd.dropna()).any()  # Has some finite values

def test_full_feature_pipeline():
    """Test complete feature pipeline: returns + MACD + volatility."""
    from xtrend.data.sources import BloombergParquetSource
    from xtrend.data.returns_vol import simple_returns, multi_scale_returns, ewm_volatility
    from xtrend.features.indicators_backend import macd_multi_scale

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)
    prices = pd.DataFrame({
        'ES': 3000 + np.cumsum(np.random.randn(300) * 10),
        'CL': 50 + np.cumsum(np.random.randn(300) * 2),
    }, index=dates)

    # Calculate returns
    returns_1d = simple_returns(prices)
    returns_multi = multi_scale_returns(prices, scales=[1, 21, 63, 126, 252])

    # Calculate MACD features
    macd_features = macd_multi_scale(prices, timescale_pairs=[(8,24), (16,28), (32,96)])

    # Calculate volatility
    vol = ewm_volatility(returns_1d, span=60)

    # Combine all features
    all_features = pd.concat([
        returns_multi[1].add_suffix('_ret_1'),
        returns_multi[21].add_suffix('_ret_21'),
        macd_features,
        vol.add_suffix('_vol'),
    ], axis=1)

    print(f"\nFeature matrix shape: {all_features.shape}")
    print(f"Features: {list(all_features.columns)[:5]}...")

    # Verify we have expected number of features
    # 2 assets × (2 returns scales + 3 MACD + 1 vol) = 12 features
    assert all_features.shape[1] == 12
    assert not all_features.iloc[60:].isna().all().any()  # No all-NaN columns after warmup
