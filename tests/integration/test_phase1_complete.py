"""Integration test for complete Phase 1 pipeline."""
import pytest
import pandas as pd
from pathlib import Path

@pytest.mark.skipif(
    not Path("data/bloomberg/processed").exists(),
    reason="Bloomberg data not available"
)
def test_phase1_pipeline_complete():
    """Test complete Phase 1 pipeline with real data."""
    from xtrend.data.sources import BloombergParquetSource
    from xtrend.data.returns_vol import simple_returns, multi_scale_returns, normalized_returns, ewm_volatility, apply_vol_target
    from xtrend.features.indicators_backend import macd_multi_scale

    # Load real data
    source = BloombergParquetSource()
    symbols = source.symbols()[:5]  # First 5 symbols

    print(f"\nLoading {len(symbols)} symbols: {symbols}")

    prices = source.load_prices(symbols, start='2020-01-01', end='2023-12-31')
    print(f"Prices shape: {prices.shape}")

    # Calculate returns at multiple scales
    returns_dict = multi_scale_returns(prices, scales=[1, 21, 63, 126, 252])
    print(f"Returns scales: {list(returns_dict.keys())}")

    # Calculate MACD features
    macd_features = macd_multi_scale(prices, timescale_pairs=[(8,24), (16,28), (32,96)])
    print(f"MACD features shape: {macd_features.shape}")

    # Calculate volatility
    daily_returns = simple_returns(prices)
    vol = ewm_volatility(daily_returns, span=60)
    print(f"Volatility shape: {vol.shape}")

    # Apply volatility targeting
    dummy_positions = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    targeted = apply_vol_target(dummy_positions, vol, sigma_target=0.15)
    print(f"Targeted positions shape: {targeted.shape}")

    # Verify feature dimensions match paper expectations
    # Paper uses 8 features per asset: 5 returns + 3 MACD
    expected_features_per_asset = 5 + 3  # 5 return scales + 3 MACD
    assert macd_features.shape[1] == len(symbols) * 3  # 3 MACD per asset

    print("\n✅ Phase 1 pipeline complete!")
    print(f"   - Loaded {len(symbols)} assets")
    print(f"   - {len(prices)} timesteps")
    print(f"   - {expected_features_per_asset} features per asset")
