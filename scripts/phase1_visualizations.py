#!/usr/bin/env python3
"""
Phase 1 Visualization Script for X-Trend Project

Generates the remaining completion plots:
1. Returns distribution histogram (normalized returns)
2. MACD indicators overlaid on price
3. Volatility time series with targeting effect

Usage:
    uv run python scripts/phase1_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import X-Trend modules
from xtrend.data.sources import BloombergParquetSource
from xtrend.data.returns_vol import simple_returns, normalized_returns, ewm_volatility, apply_vol_target
from xtrend.features.indicators_backend import macd_normalized


def setup_plotting_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def plot_returns_distribution(prices, symbols=['ES', 'GC', 'CL'], scales=[1, 21, 63, 126, 252]):
    """
    Plot 1: Returns distribution histogram showing normalized returns.

    Shows distribution of normalized returns at multiple timescales.
    """
    print("\n" + "="*60)
    print("PLOT 1: Returns Distribution Histogram")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Calculate normalized returns for each scale
    for idx, scale in enumerate(scales):
        ax = axes[idx]

        # Calculate normalized returns for selected symbols
        norm_returns = normalized_returns(prices[symbols], scale=scale, vol_window=252)

        # Flatten all returns across symbols
        all_returns = norm_returns.values.flatten()
        all_returns = all_returns[~np.isnan(all_returns)]

        # Plot histogram
        ax.hist(all_returns, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')

        # Overlay normal distribution for reference
        mu, std = all_returns.mean(), all_returns.std()
        x = np.linspace(all_returns.min(), all_returns.max(), 100)
        ax.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2),
                'r-', linewidth=2, label=f'N({mu:.3f}, {std:.3f})')

        # Statistics
        ax.axvline(mu, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mu:.3f}')
        ax.axvline(mu + 2*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±2σ: {2*std:.3f}')
        ax.axvline(mu - 2*std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_title(f'{scale}-Day Normalized Returns', fontweight='bold')
        ax.set_xlabel('Normalized Return r̂')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f'N={len(all_returns):,}\nSkew={pd.Series(all_returns).skew():.3f}\nKurt={pd.Series(all_returns).kurtosis():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)

    # Hide unused subplot
    if len(scales) < len(axes):
        axes[-1].axis('off')

    plt.suptitle('Phase 1: Returns Distribution (Normalized by Volatility)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save figure
    output_path = Path('outputs/plots/phase1_returns_distribution.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    plt.show()
    plt.close()


def plot_macd_overlay(prices, symbol='ES', start_date='2020-01-01', end_date='2020-12-31'):
    """
    Plot 2: MACD indicators overlaid on price for sample period.

    Shows price series with three MACD indicators at different timescales.
    """
    print("\n" + "="*60)
    print("PLOT 2: MACD Indicators Overlaid on Price")
    print("="*60)

    # Filter date range
    mask = (prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))
    price_series = prices.loc[mask, symbol]

    # Calculate MACD at multiple timescales (as per paper)
    timescale_pairs = [(8, 24), (16, 28), (32, 96)]
    macd_results = {}

    for short, long in timescale_pairs:
        macd_results[f'MACD_{short}_{long}'] = macd_normalized(
            price_series,
            short=short,
            long=long,
            norm_window=252
        )

    # Create figure with 4 subplots (price + 3 MACD indicators)
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Price series
    ax1 = axes[0]
    ax1.plot(price_series.index, price_series.values, linewidth=1.5, color='black', label='Price')
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.set_title(f'{symbol} Price Series', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2-4: MACD indicators
    colors = ['steelblue', 'orange', 'green']
    for idx, ((short, long), color) in enumerate(zip(timescale_pairs, colors)):
        ax = axes[idx + 1]
        macd_key = f'MACD_{short}_{long}'
        macd_data = macd_results[macd_key]

        # Plot MACD line
        ax.plot(macd_data.index, macd_data.values, linewidth=1.5, color=color,
                label=f'MACD({short},{long})')

        # Add zero line
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        # Shade positive/negative regions
        ax.fill_between(macd_data.index, 0, macd_data.values,
                        where=(macd_data.values > 0), color='green', alpha=0.2, label='Bullish')
        ax.fill_between(macd_data.index, 0, macd_data.values,
                        where=(macd_data.values <= 0), color='red', alpha=0.2, label='Bearish')

        ax.set_ylabel('Normalized MACD', fontweight='bold')
        ax.set_title(f'MACD Indicator ({short}-day, {long}-day)', fontweight='bold', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontweight='bold')

    plt.suptitle(f'Phase 1: MACD Indicators Overlay ({symbol}, {start_date} to {end_date})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    output_path = Path(f'outputs/plots/phase1_macd_overlay_{symbol}.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    plt.show()
    plt.close()


def plot_volatility_targeting(prices, symbol='ES', sigma_target=0.15,
                              start_date='2018-01-01', end_date='2023-12-31'):
    """
    Plot 3: Volatility time series showing targeting effect.

    Shows realized volatility, target volatility, and leverage factor over time.
    """
    print("\n" + "="*60)
    print("PLOT 3: Volatility Time Series with Targeting Effect")
    print("="*60)

    # Filter date range
    mask = (prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))
    price_series = prices.loc[mask, symbol]

    # Calculate returns
    returns = simple_returns(price_series)

    # Calculate ex-ante volatility (60-day EWMA)
    sigma_t = ewm_volatility(returns, span=60)

    # Annualize volatility for comparison with target
    sigma_t_annual = sigma_t * np.sqrt(252)

    # Calculate leverage factor
    leverage = sigma_target / sigma_t_annual.clip(lower=1e-8)
    leverage = leverage.clip(upper=10.0)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Volatility comparison
    ax1 = axes[0]
    ax1.plot(sigma_t_annual.index, sigma_t_annual.values * 100, linewidth=1.5,
             color='steelblue', label='Realized Volatility (σ_t)')
    ax1.axhline(sigma_target * 100, color='red', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Target Volatility ({sigma_target*100:.0f}%)')

    # Shade high/low volatility regions
    ax1.fill_between(sigma_t_annual.index, sigma_target * 100, sigma_t_annual.values * 100,
                     where=(sigma_t_annual.values > sigma_target), color='orange', alpha=0.2,
                     label='High Vol (reduce exposure)')
    ax1.fill_between(sigma_t_annual.index, sigma_target * 100, sigma_t_annual.values * 100,
                     where=(sigma_t_annual.values <= sigma_target), color='green', alpha=0.2,
                     label='Low Vol (increase exposure)')

    ax1.set_ylabel('Annualized Volatility (%)', fontweight='bold')
    ax1.set_title('Realized vs Target Volatility', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add COVID-19 marker
    covid_date = pd.Timestamp('2020-03-01')
    if start_date <= '2020-03-01' <= end_date:
        ax1.axvline(covid_date, color='purple', linestyle=':', linewidth=2, alpha=0.7, label='COVID-19')

    # Plot 2: Leverage factor
    ax2 = axes[1]
    ax2.plot(leverage.index, leverage.values, linewidth=1.5, color='darkorange',
             label='Leverage Factor (σ_tgt / σ_t)')
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='1x Leverage')
    ax2.axhline(10.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max Leverage Cap (10x)')

    ax2.set_ylabel('Leverage Factor', fontweight='bold')
    ax2.set_title('Volatility Targeting Leverage', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 12)

    if start_date <= '2020-03-01' <= end_date:
        ax2.axvline(covid_date, color='purple', linestyle=':', linewidth=2, alpha=0.7)

    # Plot 3: Price for reference
    ax3 = axes[2]
    ax3.plot(price_series.index, price_series.values, linewidth=1.5, color='black', label='Price')
    ax3.set_ylabel('Price ($)', fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_title(f'{symbol} Price Series', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    if start_date <= '2020-03-01' <= end_date:
        ax3.axvline(covid_date, color='purple', linestyle=':', linewidth=2, alpha=0.7)

    plt.suptitle(f'Phase 1: Volatility Targeting Effect ({symbol}, {start_date} to {end_date})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    output_path = Path(f'outputs/plots/phase1_volatility_targeting_{symbol}.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    plt.show()
    plt.close()


def main():
    """Main execution function."""
    print("="*60)
    print("Phase 1 Visualization Script")
    print("X-Trend: Data Pipeline & Feature Engineering")
    print("="*60)

    # Setup plotting style
    setup_plotting_style()

    # Load Bloomberg data
    print("\nLoading Bloomberg futures data...")
    source = BloombergParquetSource(root_path="data/bloomberg/processed")

    # Load a representative set of symbols (diverse asset classes)
    symbols = ['ES', 'XAU', 'BR', 'ZC', 'ZN', 'DX']  # Equity, Gold, Oil, Grain, Bond, FX
    prices = source.load_prices(symbols, start='1995-01-01', end='2023-12-31')
    print(f"✅ Loaded {len(symbols)} symbols: {symbols}")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Total observations: {len(prices):,}")

    # Generate plots
    print("\n" + "="*60)
    print("Generating Phase 1 Completion Plots")
    print("="*60)

    # Plot 1: Returns distribution
    plot_returns_distribution(prices, symbols=symbols)

    # Plot 2: MACD overlay (focus on COVID-19 period for interest)
    plot_macd_overlay(prices, symbol='ES', start_date='2020-01-01', end_date='2020-12-31')

    # Plot 3: Volatility targeting (show full 2018-2023 period including COVID)
    plot_volatility_targeting(prices, symbol='ES', sigma_target=0.15,
                             start_date='2018-01-01', end_date='2023-12-31')

    print("\n" + "="*60)
    print("✅ All Phase 1 visualizations complete!")
    print("="*60)
    print("\nPlots saved to: outputs/plots/")
    print("  1. phase1_returns_distribution.png")
    print("  2. phase1_macd_overlay_ES.png")
    print("  3. phase1_volatility_targeting_ES.png")
    print("\n✨ Phase 1 completion criteria met! ✨")


if __name__ == "__main__":
    main()
