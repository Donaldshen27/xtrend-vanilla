"""Feature computation for X-Trend model training.

Implements paper-specified features:
- 5 volatility-normalized returns at timescales [1, 21, 63, 126, 252] days
- 3 MACD indicators at (S,L) pairs [(8,24), (16,28), (32,96)]
"""
import numpy as np
import pandas as pd


def compute_xtrend_features(prices: pd.Series) -> pd.DataFrame:
    """Compute 8 features for a price series (matching X-Trend paper).

    Features:
    - 5 volatility-normalized returns at scales [1, 21, 63, 126, 252] days
    - 3 MACD indicators at (S,L) pairs [(8,24), (16,28), (32,96)]

    Returns are normalized by EWMA volatility as per paper:
        r_hat[t-t',t] = r[t-t',t] / (σ_t * sqrt(t'))

    Args:
        prices: Price series (pd.Series with DatetimeIndex)

    Returns:
        DataFrame with 8 feature columns
    """
    df = pd.DataFrame(index=prices.index)

    # Compute EWMA volatility (span=60 as recommended by x-trend-architecture skill)
    daily_returns = prices.pct_change()
    sigma_t = daily_returns.ewm(span=60, min_periods=20).std()

    # Clip to prevent division by zero
    sigma_t = sigma_t.clip(lower=1e-8)

    # Paper specification: normalized returns at timescales [1, 21, 63, 126, 252]
    # Formula: r_hat[t-t',t] = r[t-t',t] / (σ_t * sqrt(t'))
    for scale in [1, 21, 63, 126, 252]:
        # Compute raw return at this scale
        raw_ret = prices.pct_change(scale)

        # Normalize by volatility and sqrt(scale)
        normalized_ret = raw_ret / (sigma_t * np.sqrt(scale))

        df[f'ret_{scale}d'] = normalized_ret

    # Paper specification: MACD at (S,L) pairs [(8,24), (16,28), (32,96)]
    df['macd_8_24'] = (prices.ewm(span=8).mean() - prices.ewm(span=24).mean()) / prices
    df['macd_16_28'] = (prices.ewm(span=16).mean() - prices.ewm(span=28).mean()) / prices
    df['macd_32_96'] = (prices.ewm(span=32).mean() - prices.ewm(span=96).mean()) / prices

    return df.fillna(0.0)
