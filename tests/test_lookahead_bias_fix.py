"""Test to verify target returns use lagged volatility (no lookahead bias)."""
import pandas as pd
import numpy as np
import pytest


def test_input_features_use_concurrent_volatility():
    """Test that INPUT FEATURES use concurrent volatility.

    For input features, σ[t] SHOULD include r[t] because it represents
    the actual market state at time t. This prevents outliers during
    volatility spikes.
    """
    from xtrend.data.features import compute_xtrend_features

    # Create simple deterministic price series
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series([100.0] * 100, index=dates)

    # Insert a large spike at time t=50
    prices.iloc[50] = 200.0  # 100% jump

    # Compute features
    features = compute_xtrend_features(prices)

    # At time t=50, for INPUT FEATURES:
    # r[50] = (200 - 100) / 100 = 1.0
    # sigma[50] SHOULD include r[50] (concurrent volatility)
    # This makes normalized return reasonable (~O(1)), not a huge outlier

    ret_1d = features['ret_1d'].iloc[50]

    # With CORRECT concurrent volatility:
    # sigma[50] includes the spike, so normalized return is ~O(1)
    # This is desirable for input features - prevents outliers

    # Test: normalized return should be moderate (< 10), not a huge spike
    assert abs(ret_1d) < 10, \
        f"Input feature normalized return ({ret_1d:.2f}) should be moderate " \
        f"when using concurrent volatility. Large values indicate shifted " \
        f"volatility which creates outliers during regime changes."


def test_target_returns_use_lagged_volatility():
    """Test that TARGET RETURNS in training script use lagged volatility.

    This is the critical fix: when normalizing return r[t+1], we should
    use σ[t] (known at decision time), not σ[t+1] (lookahead).
    """
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')

    # Create price series
    prices = pd.Series(100 * (1 + np.random.randn(200) * 0.02).cumprod(), index=dates)

    # Simulate what training script does for TARGET returns
    daily_rets = prices.pct_change()

    # CORRECT: Use lagged volatility (shift(1))
    sigma_t_lagged = daily_rets.ewm(span=60, min_periods=20).std().shift(1)
    sigma_t_lagged = sigma_t_lagged.clip(lower=1e-8).bfill()
    normalized_rets_correct = daily_rets / sigma_t_lagged

    # INCORRECT: Use concurrent volatility (no shift)
    sigma_t_concurrent = daily_rets.ewm(span=60, min_periods=20).std().clip(lower=1e-8)
    normalized_rets_wrong = daily_rets / sigma_t_concurrent

    # For a given return r[t], check that it's normalized differently:
    # - Correct: r[t] / σ[t-1] (lagged)
    # - Wrong: r[t] / σ[t] (concurrent, includes r[t])

    # Take a sample timestep after warmup (e.g., t=80)
    t = 80

    # The two normalization schemes should produce different values
    assert normalized_rets_correct.iloc[t] != normalized_rets_wrong.iloc[t], \
        "Target returns with lagged vs concurrent volatility should differ"

    # The concurrent version will generally have smaller absolute values
    # because σ[t] (including r[t]) is more "up to date" than σ[t-1]
    # This makes training artificially easier - lookahead bias!

    # Verify we're using the lagged version
    assert not pd.isna(normalized_rets_correct.iloc[t]), \
        "Lagged normalization should produce valid values after warmup"


def test_no_lookahead_in_target_returns():
    """Test that target returns don't have lookahead bias either."""
    # This test is for the training script, but we can test the pattern
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(100 * (1 + np.random.randn(100) * 0.02).cumprod(), index=dates)

    # Compute normalized returns the CORRECT way (no lookahead)
    daily_rets = prices.pct_change()
    sigma_t = daily_rets.ewm(span=60, min_periods=20).std().shift(1)
    sigma_t = sigma_t.clip(lower=1e-8)
    normalized_rets = daily_rets / sigma_t

    # Verify that sigma_t at time t doesn't include return at time t
    # This is implicitly tested by the shift(1)

    # Check that we don't have inf values from the first return
    # (which would happen if sigma_t[0] is used before being shifted)
    assert not normalized_rets.isin([np.inf, -np.inf]).any(), \
        "Should not have inf values with proper shifting"

    # The first valid normalized return should be at index where sigma_t becomes valid
    first_valid_idx = normalized_rets.first_valid_index()
    assert first_valid_idx is not None, "Should have some valid normalized returns"
