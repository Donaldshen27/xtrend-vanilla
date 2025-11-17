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


# ============================================================================
# Phase 1: Returns Analysis Charts
# ============================================================================

def create_returns_distribution_chart(
    normalized_returns_dict: Dict[int, pd.DataFrame],
    scales: list = [1, 21, 63, 126, 252]
) -> go.Figure:
    """Create returns distribution histograms for multiple timescales.

    Args:
        normalized_returns_dict: Dict mapping scale -> DataFrame of normalized returns
        scales: List of timescales to display

    Returns:
        Plotly Figure with subplots for each timescale
    """
    from plotly.subplots import make_subplots
    import numpy as np
    from scipy import stats as scipy_stats

    n_scales = len(scales)
    rows = (n_scales + 1) // 2
    cols = 2

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{scale}-Day Normalized Returns" for scale in scales],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    for idx, scale in enumerate(scales):
        if scale not in normalized_returns_dict:
            continue

        row = (idx // 2) + 1
        col = (idx % 2) + 1

        # Flatten all returns
        all_returns = normalized_returns_dict[scale].values.flatten()
        all_returns = all_returns[~np.isnan(all_returns)]

        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=all_returns,
                nbinsx=100,
                name=f"{scale}d",
                histnorm='probability density',
                marker_color='steelblue',
                opacity=0.7,
                showlegend=False
            ),
            row=row, col=col
        )

        # Overlay normal distribution
        mu, std = all_returns.mean(), all_returns.std()
        x_range = np.linspace(all_returns.min(), all_returns.max(), 100)
        y_normal = scipy_stats.norm.pdf(x_range, mu, std)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name=f'N({mu:.2f}, {std:.2f})',
                line=dict(color='red', width=2),
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_xaxes(title_text="Normalized Return r̂", row=rows, col=1)
    fig.update_xaxes(title_text="Normalized Return r̂", row=rows, col=2)
    fig.update_yaxes(title_text="Density")

    fig.update_layout(
        title_text="Phase 1: Returns Distribution (Normalized by Volatility)",
        height=300 * rows,
        showlegend=False
    )

    return fig


def create_macd_overlay_chart(
    price_series: pd.Series,
    macd_dict: Dict[str, pd.Series],
    symbol: str
) -> go.Figure:
    """Create MACD overlay chart with price and multiple MACD indicators.

    Args:
        price_series: Price series for the symbol
        macd_dict: Dict mapping 'MACD_{short}_{long}' -> Series
        symbol: Symbol name for title

    Returns:
        Plotly Figure with price + MACD subplots
    """
    from plotly.subplots import make_subplots

    n_macd = len(macd_dict)
    fig = make_subplots(
        rows=n_macd + 1, cols=1,
        subplot_titles=[f"{symbol} Price"] + list(macd_dict.keys()),
        vertical_spacing=0.08,
        row_heights=[0.3] + [0.7 / n_macd] * n_macd
    )

    # Plot 1: Price series
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode='lines',
            name='Price',
            line=dict(color='black', width=1.5),
            showlegend=True
        ),
        row=1, col=1
    )

    # Plots 2+: MACD indicators
    colors = ['steelblue', 'orange', 'green', 'purple']
    for idx, (macd_name, macd_series) in enumerate(macd_dict.items()):
        row = idx + 2
        color = colors[idx % len(colors)]

        # MACD line
        fig.add_trace(
            go.Scatter(
                x=macd_series.index,
                y=macd_series.values,
                mode='lines',
                name=macd_name,
                line=dict(color=color, width=1.5),
                showlegend=True
            ),
            row=row, col=1
        )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=row, col=1)

        # Shade positive/negative regions
        fig.add_trace(
            go.Scatter(
                x=macd_series.index,
                y=macd_series.where(macd_series > 0, 0),
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(0, 255, 0, 0.1)',
                showlegend=False
            ),
            row=row, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=macd_series.index,
                y=macd_series.where(macd_series <= 0, 0),
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=False
            ),
            row=row, col=1
        )

    fig.update_xaxes(title_text="Date", row=n_macd + 1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    for i in range(n_macd):
        fig.update_yaxes(title_text="Normalized MACD", row=i + 2, col=1)

    fig.update_layout(
        title_text=f"Phase 1: MACD Indicators Overlay ({symbol})",
        height=800,
        hovermode='x unified'
    )

    return fig


def create_volatility_targeting_chart(
    price_series: pd.Series,
    realized_vol: pd.Series,
    leverage: pd.Series,
    symbol: str,
    sigma_target: float = 0.15
) -> go.Figure:
    """Create volatility targeting chart showing realized vol, target, and leverage.

    Args:
        price_series: Price series for the symbol
        realized_vol: Realized annualized volatility
        leverage: Leverage factor (σ_target / σ_t)
        symbol: Symbol name
        sigma_target: Target volatility

    Returns:
        Plotly Figure with 3 subplots
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Realized vs Target Volatility",
            "Volatility Targeting Leverage",
            f"{symbol} Price"
        ],
        vertical_spacing=0.10,
        row_heights=[0.35, 0.30, 0.35]
    )

    # Plot 1: Volatility comparison
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=realized_vol.values * 100,
            mode='lines',
            name='Realized Vol (σ_t)',
            line=dict(color='steelblue', width=1.5)
        ),
        row=1, col=1
    )

    fig.add_hline(
        y=sigma_target * 100,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Target ({sigma_target*100:.0f}%)",
        row=1, col=1
    )

    # Plot 2: Leverage factor
    fig.add_trace(
        go.Scatter(
            x=leverage.index,
            y=leverage.values,
            mode='lines',
            name='Leverage Factor',
            line=dict(color='darkorange', width=1.5)
        ),
        row=2, col=1
    )

    fig.add_hline(y=1.0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    fig.add_hline(
        y=10.0,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text="Max Cap (10x)",
        row=2, col=1
    )

    # Plot 3: Price
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode='lines',
            name='Price',
            line=dict(color='black', width=1.5)
        ),
        row=3, col=1
    )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Annualized Vol (%)", row=1, col=1)
    fig.update_yaxes(title_text="Leverage Factor", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=3, col=1)

    fig.update_layout(
        title_text=f"Phase 1: Volatility Targeting Effect ({symbol})",
        height=900,
        hovermode='x unified',
        showlegend=True
    )

    return fig
