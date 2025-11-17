"""Phase 1 Returns Analysis Tab for Streamlit App."""

import streamlit as st
from typing import Dict
import pandas as pd

from bloomberg_viz.analysis import (
    compute_normalized_returns_analysis,
    compute_macd_analysis,
    compute_volatility_targeting_analysis,
    compute_returns_distribution_stats
)
from bloomberg_viz.charts import (
    create_returns_distribution_chart,
    create_macd_overlay_chart,
    create_volatility_targeting_chart
)


def render_returns_tab(available_symbols: list, data: Dict[str, pd.DataFrame]):
    """Render Phase 1: Returns Analysis tab.

    Args:
        available_symbols: List of available symbols
        data: Dict mapping symbol -> DataFrame with price data
    """
    st.markdown("## Phase 1: Returns, MACD, and Volatility Analysis")

    # Subsection selector
    analysis_type = st.radio(
        "Select Analysis:",
        options=["Returns Distribution", "MACD Indicators", "Volatility Targeting"],
        horizontal=True
    )

    if analysis_type == "Returns Distribution":
        render_returns_distribution(data)

    elif analysis_type == "MACD Indicators":
        render_macd_analysis(available_symbols, data)

    elif analysis_type == "Volatility Targeting":
        render_volatility_targeting(available_symbols, data)


def render_returns_distribution(data: Dict[str, pd.DataFrame]):
    """Render returns distribution analysis."""
    st.markdown("### Normalized Returns Distribution")
    st.markdown("""
    Shows the distribution of normalized returns at multiple timescales.
    Returns are normalized by realized volatility: **r̂ = r / (σ_t ×  √t')**
    """)

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        vol_window = st.slider(
            "Volatility Window (days)",
            min_value=21,
            max_value=504,
            value=252,
            step=21,
            help="Rolling window for volatility calculation"
        )

    with col2:
        st.markdown("**Timescales to analyze:**")
        scales = st.multiselect(
            "Select timescales (days)",
            options=[1, 21, 63, 126, 252],
            default=[1, 21, 63, 126, 252],
            help="Returns calculated over these horizons"
        )

    if not scales:
        st.warning("⚠️ Please select at least one timescale")
        return

    if len(data) == 0:
        st.warning("⚠️ No data loaded")
        return

    # Compute analysis
    with st.spinner("Computing normalized returns..."):
        try:
            norm_returns_dict = compute_normalized_returns_analysis(
                data,
                scales=scales,
                vol_window=vol_window
            )

            # Create chart
            fig = create_returns_distribution_chart(norm_returns_dict, scales=scales)
            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            st.markdown("### Distribution Statistics")
            stats_dfs = []
            for scale in scales:
                stats_df = compute_returns_distribution_stats(norm_returns_dict, scale)
                stats_dfs.append(stats_df)

            combined_stats = pd.concat(stats_dfs, ignore_index=True)
            st.dataframe(
                combined_stats.style.format({
                    'mean': '{:.4f}',
                    'std': '{:.4f}',
                    'skew': '{:.4f}',
                    'kurtosis': '{:.4f}',
                    'min': '{:.4f}',
                    'max': '{:.4f}',
                    'p05': '{:.4f}',
                    'p95': '{:.4f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            # Interpretation
            with st.expander("💡 How to interpret"):
                st.markdown("""
                **Distribution Shape:**
                - Normal distribution (red line) shows ideal Gaussian shape
                - **Skew** > 0: Right tail longer (more extreme positive returns)
                - **Skew** < 0: Left tail longer (more extreme negative returns)
                - **Kurtosis** > 0: Fat tails (more extreme events than normal)

                **Normalization Effect:**
                - Normalized returns account for varying volatility regimes
                - Standardized across different timescales for comparison
                - Useful for machine learning features (similar scale)
                """)

        except Exception as e:
            st.error(f"❌ Error computing returns: {e}")
            st.exception(e)


def render_macd_analysis(available_symbols: list, data: Dict[str, pd.DataFrame]):
    """Render MACD indicators analysis."""
    st.markdown("### MACD Indicators Overlay")
    st.markdown("""
    **Moving Average Convergence Divergence (MACD)** compares fast and slow exponential moving averages.
    Normalized by 252-day rolling standard deviation for stationarity.
    """)

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol",
            options=available_symbols,
            index=available_symbols.index('ES') if 'ES' in available_symbols else 0,
            help="Symbol to analyze"
        )

    with col2:
        norm_window = st.slider(
            "Normalization Window (days)",
            min_value=21,
            max_value=504,
            value=252,
            step=21,
            help="Rolling window for MACD normalization"
        )

    # Timescale pairs
    st.markdown("**MACD Timescale Pairs (Short, Long):**")
    use_default = st.checkbox(
        "Use default pairs: (8,24), (16,28), (32,96)",
        value=True,
        help="Paper's recommended timescale pairs"
    )

    if use_default:
        timescale_pairs = [(8, 24), (16, 28), (32, 96)]
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pair1_short = st.number_input("Pair 1 - Short", value=8, min_value=2)
            pair1_long = st.number_input("Pair 1 - Long", value=24, min_value=pair1_short+1)
        with col2:
            pair2_short = st.number_input("Pair 2 - Short", value=16, min_value=2)
            pair2_long = st.number_input("Pair 2 - Long", value=28, min_value=pair2_short+1)
        with col3:
            pair3_short = st.number_input("Pair 3 - Short", value=32, min_value=2)
            pair3_long = st.number_input("Pair 3 - Long", value=96, min_value=pair3_short+1)

        timescale_pairs = [(pair1_short, pair1_long), (pair2_short, pair2_long), (pair3_short, pair3_long)]

    if selected_symbol not in data:
        st.warning(f"⚠️ Data not loaded for {selected_symbol}")
        return

    # Compute analysis
    with st.spinner(f"Computing MACD for {selected_symbol}..."):
        try:
            macd_dict = compute_macd_analysis(
                data,
                symbol=selected_symbol,
                timescale_pairs=timescale_pairs,
                norm_window=norm_window
            )

            price_series = data[selected_symbol]['price']

            # Create chart
            fig = create_macd_overlay_chart(price_series, macd_dict, selected_symbol)
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            with st.expander("💡 How to interpret"):
                st.markdown("""
                **MACD Signals:**
                - **Positive MACD**: Fast EMA > Slow EMA → Bullish (uptrend)
                - **Negative MACD**: Fast EMA < Slow EMA → Bearish (downtrend)
                - **Zero crossings**: Trend reversals

                **Multi-Scale Analysis:**
                - (8,24): Short-term momentum (days to weeks)
                - (16,28): Medium-term momentum (weeks)
                - (32,96): Long-term momentum (months)

                **Normalization:**
                - Dividing by rolling std makes values comparable across time
                - Accounts for changing volatility regimes
                """)

        except Exception as e:
            st.error(f"❌ Error computing MACD: {e}")
            st.exception(e)


def render_volatility_targeting(available_symbols: list, data: Dict[str, pd.DataFrame]):
    """Render volatility targeting analysis."""
    st.markdown("### Volatility Targeting")
    st.markdown("""
    **Volatility targeting** scales positions to maintain constant risk exposure.
    Leverage factor: **L = σ_target / σ_t** (capped at 10x)
    """)

    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol",
            options=available_symbols,
            index=available_symbols.index('ES') if 'ES' in available_symbols else 0,
            help="Symbol to analyze"
        )

    with col2:
        sigma_target = st.slider(
            "Target Volatility (%)",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Annual target volatility (default 15%)"
        ) / 100

    with col3:
        span = st.slider(
            "EWM Span (days)",
            min_value=20,
            max_value=252,
            value=60,
            step=10,
            help="Exponentially weighted moving average span"
        )

    if selected_symbol not in data:
        st.warning(f"⚠️ Data not loaded for {selected_symbol}")
        return

    # Compute analysis
    with st.spinner(f"Computing volatility targeting for {selected_symbol}..."):
        try:
            realized_vol, leverage, price_series = compute_volatility_targeting_analysis(
                data,
                symbol=selected_symbol,
                sigma_target=sigma_target,
                span=span
            )

            # Create chart
            fig = create_volatility_targeting_chart(
                price_series,
                realized_vol,
                leverage,
                selected_symbol,
                sigma_target
            )
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            st.markdown("### Volatility Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Volatility", f"{sigma_target*100:.1f}%")
            with col2:
                mean_vol = (realized_vol * 100).mean()
                st.metric("Mean Realized Vol", f"{mean_vol:.1f}%")
            with col3:
                mean_leverage = leverage.mean()
                st.metric("Mean Leverage", f"{mean_leverage:.2f}x")

            # Interpretation
            with st.expander("💡 How to interpret"):
                st.markdown("""
                **Volatility Targeting Benefits:**
                - **Constant risk**: Portfolio volatility stays near target
                - **Automatic rebalancing**: Reduce exposure in high vol, increase in low vol
                - **Risk-adjusted returns**: Focus on Sharpe ratio, not absolute returns

                **Leverage Factor:**
                - **L > 1**: Realized vol < target → Increase position size
                - **L < 1**: Realized vol > target → Decrease position size
                - **L = 1**: At target volatility

                **10x Cap:**
                - Prevents infinite positions during extremely low volatility
                - Typical for futures markets (1x-10x range)
                """)

        except Exception as e:
            st.error(f"❌ Error computing volatility targeting: {e}")
            st.exception(e)
