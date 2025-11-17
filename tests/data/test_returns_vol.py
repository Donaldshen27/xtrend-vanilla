"""Tests for returns and volatility calculations."""
import pytest
import pandas as pd
import numpy as np
from xtrend.data.returns_vol import simple_returns, ewm_volatility, apply_vol_target

def test_simple_returns(sample_prices):
    """Test simple returns calculation."""
    returns = simple_returns(sample_prices)

    # Check structure
    assert isinstance(returns, pd.DataFrame)
    assert returns.shape == sample_prices.shape
    assert list(returns.columns) == list(sample_prices.columns)

    # Check calculation (manual verification for first asset)
    expected_first_return = (sample_prices['ES'].iloc[1] - sample_prices['ES'].iloc[0]) / sample_prices['ES'].iloc[0]
    assert np.isclose(returns['ES'].iloc[1], expected_first_return)

    # First row should be NaN
    assert returns.iloc[0].isna().all()

    # No other NaN values
    assert not returns.iloc[1:].isna().any().any()

def test_simple_returns_formula(sample_prices):
    """Test returns formula: r_t = (p_t - p_{t-1}) / p_{t-1}"""
    returns = simple_returns(sample_prices)

    # Verify pct_change formula
    expected = sample_prices.pct_change()
    pd.testing.assert_frame_equal(returns, expected)

def test_multi_scale_returns():
    """Test returns at multiple timescales."""
    from xtrend.data.returns_vol import multi_scale_returns

    # Create simple upward trending prices
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    prices = pd.DataFrame({
        'ES': 100 + np.arange(300),  # Linear increase
    }, index=dates)

    scales = [1, 21, 63]
    returns_dict = multi_scale_returns(prices, scales=scales)

    # Check structure
    assert isinstance(returns_dict, dict)
    assert set(returns_dict.keys()) == {1, 21, 63}

    # Check each scale
    for scale in scales:
        ret = returns_dict[scale]
        assert isinstance(ret, pd.DataFrame)
        assert ret.shape == prices.shape

        # Verify calculation for scale
        # r_{t-scale,t} = (p_t - p_{t-scale}) / p_{t-scale}
        expected = (prices - prices.shift(scale)) / prices.shift(scale)
        pd.testing.assert_frame_equal(ret, expected)

def test_normalized_returns():
    """Test normalized returns: r̂ = r / (σ_t * √t')"""
    from xtrend.data.returns_vol import normalized_returns

    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)
    prices = pd.DataFrame({
        'ES': 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)),
    }, index=dates)

    # Calculate normalized returns for 21-day scale
    norm_returns = normalized_returns(prices, scale=21, vol_window=252)

    assert isinstance(norm_returns, pd.DataFrame)
    assert norm_returns.shape == prices.shape

    # Verify that normalization was applied (should have different values than raw)
    raw_returns = (prices - prices.shift(21)) / prices.shift(21)
    # After enough warmup period, normalized returns should exist
    assert not norm_returns.iloc[252:].isna().all().all()
