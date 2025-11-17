"""Analysis functions for Phase 1: Returns, MACD, and Volatility."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Import X-Trend Phase 1 functions
from xtrend.data.returns_vol import simple_returns, normalized_returns, ewm_volatility, apply_vol_target
from xtrend.features.indicators_backend import macd_normalized


def compute_normalized_returns_analysis(
    data: Dict[str, pd.DataFrame],
    scales: List[int] = [1, 21, 63, 126, 252],
    vol_window: int = 252
) -> Dict[int, pd.DataFrame]:
    """
    Compute normalized returns at multiple timescales for all symbols.

    Args:
        data: Dict mapping symbol -> DataFrame with 'price' column
        scales: List of timescales in days
        vol_window: Rolling window for volatility calculation

    Returns:
        Dict mapping scale -> DataFrame of normalized returns (symbols as columns)
    """
    # Combine all prices into wide DataFrame
    prices_wide = pd.DataFrame({
        symbol: df['price'] for symbol, df in data.items()
    })

    # Calculate normalized returns at each scale
    results = {}
    for scale in scales:
        norm_rets = normalized_returns(prices_wide, scale=scale, vol_window=vol_window)
        results[scale] = norm_rets

    return results


def compute_macd_analysis(
    data: Dict[str, pd.DataFrame],
    symbol: str,
    timescale_pairs: List[Tuple[int, int]] = [(8, 24), (16, 28), (32, 96)],
    norm_window: int = 252
) -> Dict[str, pd.Series]:
    """
    Compute MACD indicators at multiple timescales for a single symbol.

    Args:
        data: Dict mapping symbol -> DataFrame with 'price' column
        symbol: Symbol to analyze
        timescale_pairs: List of (short, long) EMA pairs
        norm_window: Window for normalization

    Returns:
        Dict mapping 'MACD_{short}_{long}' -> Series of MACD values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")

    price_series = data[symbol]['price']

    results = {}
    for short, long in timescale_pairs:
        macd_vals = macd_normalized(price_series, short=short, long=long, norm_window=norm_window)
        results[f'MACD_{short}_{long}'] = macd_vals

    return results


def compute_volatility_targeting_analysis(
    data: Dict[str, pd.DataFrame],
    symbol: str,
    sigma_target: float = 0.15,
    span: int = 60
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute volatility targeting components for a single symbol.

    Args:
        data: Dict mapping symbol -> DataFrame with 'price' column
        symbol: Symbol to analyze
        sigma_target: Target annual volatility (default 15%)
        span: EWM span for volatility calculation (default 60 days)

    Returns:
        Tuple of (realized_vol_annual, leverage_factor, price_series)
            - realized_vol_annual: Annualized rolling volatility
            - leverage_factor: σ_target / σ_t (capped at 10x)
            - price_series: Original price series
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")

    price_series = data[symbol]['price']

    # Calculate daily returns
    returns = simple_returns(price_series)

    # Calculate ex-ante volatility (60-day EWMA of daily returns std)
    sigma_t_daily = ewm_volatility(returns, span=span)

    # Annualize volatility (multiply by sqrt(252))
    sigma_t_annual = sigma_t_daily * np.sqrt(252)

    # Calculate leverage factor with clipping
    sigma_t_clipped = sigma_t_annual.clip(lower=1e-8)
    leverage = sigma_target / sigma_t_clipped
    leverage = leverage.clip(upper=10.0)

    return sigma_t_annual, leverage, price_series


def compute_returns_distribution_stats(
    normalized_returns_dict: Dict[int, pd.DataFrame],
    scale: int
) -> pd.DataFrame:
    """
    Compute distribution statistics for normalized returns at a given scale.

    Args:
        normalized_returns_dict: Dict from compute_normalized_returns_analysis()
        scale: Timescale to analyze

    Returns:
        DataFrame with distribution stats (mean, std, skew, kurtosis, count)
    """
    if scale not in normalized_returns_dict:
        raise ValueError(f"Scale {scale} not found in results")

    norm_rets = normalized_returns_dict[scale]

    # Flatten all returns across symbols
    all_returns = norm_rets.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]

    stats = {
        'scale': scale,
        'count': len(all_returns),
        'mean': all_returns.mean(),
        'std': all_returns.std(),
        'skew': pd.Series(all_returns).skew(),
        'kurtosis': pd.Series(all_returns).kurtosis(),
        'min': all_returns.min(),
        'max': all_returns.max(),
        'p05': np.percentile(all_returns, 5),
        'p95': np.percentile(all_returns, 95)
    }

    return pd.DataFrame([stats])


def compute_correlation_matrix(data: Dict[str, pd.DataFrame], period: str = 'daily') -> pd.DataFrame:
    """
    Compute correlation matrix of returns across symbols.

    Args:
        data: Dict mapping symbol -> DataFrame with 'price' column
        period: Not used yet (placeholder for future functionality)

    Returns:
        DataFrame correlation matrix
    """
    # Combine all prices into wide DataFrame
    prices_wide = pd.DataFrame({
        symbol: df['price'] for symbol, df in data.items()
    })

    # Calculate simple returns
    returns_wide = simple_returns(prices_wide)

    # Compute correlation matrix
    corr_matrix = returns_wide.corr()

    return corr_matrix
