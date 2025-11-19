"""Tests for X-Trend feature computation."""
import pandas as pd
import numpy as np
import pytest

from xtrend.data.features import compute_xtrend_features


@pytest.fixture
def deterministic_prices():
    """Generate deterministic price series for stable tests."""
    np.random.seed(42)  # Fixed seed for reproducibility
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Create price series with known volatility
    returns = np.random.randn(500) * 0.02  # 2% daily vol
    prices = pd.Series(
        100 * (1 + returns).cumprod(),
        index=dates,
        name='TEST'
    )
    return prices


def test_feature_columns(deterministic_prices):
    """Test that features have exactly the expected columns."""
    features = compute_xtrend_features(deterministic_prices)

    # Exactly these 8 columns should be present (5 returns + 3 MACDs)
    expected_cols = {
        'ret_1d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
        'macd_8_24', 'macd_16_28', 'macd_32_96'
    }

    assert set(features.columns) == expected_cols, \
        f"Expected columns {expected_cols}, got {set(features.columns)}"

    assert len(features.columns) == 8, \
        f"Expected exactly 8 features, got {len(features.columns)}"


def test_return_timescales_present(deterministic_prices):
    """Test that return features use correct timescales [1, 21, 63, 126, 252]."""
    features = compute_xtrend_features(deterministic_prices)

    # Should have columns for each return timescale
    expected_return_cols = ['ret_1d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d']
    for col in expected_return_cols:
        assert col in features.columns, f"Missing expected return column: {col}"


def test_macd_indicators_present(deterministic_prices):
    """Test that MACD features use correct (S,L) pairs [(8,24), (16,28), (32,96)]."""
    features = compute_xtrend_features(deterministic_prices)

    # Should have columns for each MACD pair
    expected_macd_cols = ['macd_8_24', 'macd_16_28', 'macd_32_96']
    for col in expected_macd_cols:
        assert col in features.columns, f"Missing expected MACD column: {col}"


def test_returns_normalized_by_volatility(deterministic_prices):
    """Test that returns are normalized by volatility: r_hat = r / (σ_t * sqrt(t')).

    With fixed seed, normalized returns should have predictable magnitude.
    """
    features = compute_xtrend_features(deterministic_prices)

    # Check that normalized returns have reasonable magnitude
    # (should be ~O(1) due to volatility normalization)
    for col in ['ret_1d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d']:
        valid_values = features[col].dropna()

        assert len(valid_values) > 100, \
            f"{col} should have sufficient non-NaN values"

        # With seed=42, these specific bounds hold
        # Normalized returns should typically be in [-5, 5] range
        mean_abs = valid_values.abs().mean()
        max_abs = valid_values.abs().max()

        assert mean_abs < 2.0, \
            f"{col} mean magnitude {mean_abs:.2f} too high (not normalized)"
        assert max_abs < 10.0, \
            f"{col} max magnitude {max_abs:.2f} too high (not normalized)"


def test_feature_index_alignment(deterministic_prices):
    """Test that features maintain same index as input prices."""
    features = compute_xtrend_features(deterministic_prices)

    assert features.index.equals(deterministic_prices.index), \
        "Feature index should match input price index"


def test_no_inf_values(deterministic_prices):
    """Test that feature computation doesn't produce inf values."""
    features = compute_xtrend_features(deterministic_prices)

    for col in features.columns:
        assert not features[col].isin([np.inf, -np.inf]).any(), \
            f"Column {col} contains inf values"
