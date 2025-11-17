"""Plotly chart builders for Bloomberg data visualization."""

import plotly.graph_objects as go
import pandas as pd
from typing import Dict


def create_price_chart(
    data: Dict[str, pd.DataFrame],
    normalize: bool = False
) -> go.Figure:
    """Create interactive Plotly line chart for price data.

    Args:
        data: Dict mapping symbol -> DataFrame with price column
        normalize: If True, normalize all series to start at 100

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for symbol, df in data.items():
        if df.empty:
            continue

        prices = df['price'].copy()

        if normalize and len(prices) > 0:
            # Normalize to 100 at start
            first_price = prices.iloc[0]
            if first_price != 0:
                prices = (prices / first_price) * 100

        fig.add_trace(go.Scatter(
            x=df.index,
            y=prices,
            mode='lines',
            name=symbol,
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title="Bloomberg Futures Prices",
        xaxis_title="Date",
        yaxis_title="Normalized Price (base=100)" if normalize else "Price",
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def display_summary_stats(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate summary statistics table.

    Args:
        data: Dict mapping symbol -> DataFrame with price column

    Returns:
        DataFrame with summary statistics per symbol
    """
    stats = []

    for symbol, df in data.items():
        if df.empty:
            continue

        prices = df['price']
        stats.append({
            'Symbol': symbol,
            'Count': len(prices),
            'Mean': f"{prices.mean():.2f}",
            'Std': f"{prices.std():.2f}",
            'Min': f"{prices.min():.2f}",
            'Max': f"{prices.max():.2f}",
            'Start Date': df.index.min().strftime('%Y-%m-%d'),
            'End Date': df.index.max().strftime('%Y-%m-%d')
        })

    return pd.DataFrame(stats)
