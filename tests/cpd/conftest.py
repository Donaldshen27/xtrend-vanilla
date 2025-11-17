"""Pytest fixtures for CPD tests."""
import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_prices():
    """Sample price series for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates,
        name='Close'
    )
    return prices


@pytest.fixture
def synthetic_changepoint_data():
    """Generate synthetic data with known change-point at t=15.

    Returns:
        tuple: (x, y, true_cp) where x is time indices, y is observations,
               true_cp is the true change-point location
    """
    torch.manual_seed(42)
    x = torch.arange(30).float().unsqueeze(-1)

    # Two different regimes
    y1 = torch.sin(x[:15] / 5.0).squeeze() + 0.1 * torch.randn(15)
    y2 = -torch.sin(x[15:] / 5.0).squeeze() + 2.0 + 0.1 * torch.randn(15)
    y = torch.cat([y1, y2])

    return x, y, 15
