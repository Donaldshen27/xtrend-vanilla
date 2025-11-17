"""Streamlit tab for Phase 2: Regime Detection & Validation."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from xtrend.cpd import CPDConfig, GPCPDSegmenter


@st.cache_data
def run_cpd_cached(asset: str, start_date, end_date,
                   lookback: int, threshold: float,
                   min_length: int, max_length: int):
    """Run GP-CPD with proper caching.

    Cache key = (asset, dates, hyperparams)

    Args:
        asset: Asset symbol
        start_date: Start date
        end_date: End date
        lookback: Lookback window
        threshold: Severity threshold
        min_length: Min regime length
        max_length: Max regime length

    Returns:
        tuple: (segments, prices, config)
    """
    from xtrend.data import BloombergParquetSource

    data_source = BloombergParquetSource()
    prices = data_source.load_symbol(asset, start_date, end_date)['Close']

    config = CPDConfig(
        lookback=lookback,
        threshold=threshold,
        min_length=min_length,
        max_length=max_length
    )

    segmenter = GPCPDSegmenter(config)
    segments = segmenter.fit_segment(prices)

    return segments, prices, config


def render_regimes_tab(data_source):
    """Render the Regimes tab with GP-CPD analysis.

    Args:
        data_source: BloombergParquetSource instance
    """
    st.header("Phase 2: Regime Detection & Validation")

    # Sidebar: Configuration
    with st.sidebar:
        st.subheader("CPD Configuration")
        lookback = st.slider("Lookback window", 10, 63, 21)
        threshold = st.slider("Severity threshold", 0.5, 0.99, 0.9, 0.01)
        min_length = st.slider("Min regime length", 3, 10, 5)
        max_length = st.slider("Max regime length", 21, 126, 21)

    # Asset selection
    selected_asset = st.selectbox(
        "Select asset for regime analysis",
        data_source.available_symbols()
    )

    date_range = st.date_input(
        "Date range",
        value=(pd.Timestamp('2019-01-01'), pd.Timestamp('2023-12-31'))
    )

    # Show last run info if available
    if 'last_run_params' in st.session_state:
        last_run = st.session_state['last_run_params']
        with st.expander("ℹ️ Last Run Info", expanded=False):
            st.write(f"**Asset:** {last_run['asset']}")
            st.write(f"**Date Range:** {last_run['start']} to {last_run['end']}")

            current_params = {
                'asset': selected_asset,
                'start': date_range[0],
                'end': date_range[1],
                'lookback': lookback,
                'threshold': threshold
            }

            if current_params != last_run:
                st.warning("⚠️ Settings differ from last run")

    # Run CPD button
    run_button = st.button("🔍 Detect Regimes", type="primary")

    # Execute if button clicked
    if run_button:
        with st.status("Running GP-CPD...", state="running") as status:
            try:
                status.update(label="Loading price data...", state="running")

                segments, prices, config = run_cpd_cached(
                    selected_asset, date_range[0], date_range[1],
                    lookback, threshold, min_length, max_length
                )

                status.update(label="Segmentation complete!", state="complete")

                st.session_state['current_results'] = {
                    'segments': segments,
                    'prices': prices,
                    'asset': selected_asset
                }

                st.session_state['last_run_params'] = {
                    'asset': selected_asset,
                    'start': date_range[0],
                    'end': date_range[1],
                    'lookback': lookback,
                    'threshold': threshold
                }

                st.toast(f"✅ Detected {len(segments.segments)} regimes")

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {str(e)}")
                return

    # Display results
    if 'current_results' in st.session_state:
        results = st.session_state['current_results']
        segments = results['segments']
        prices = results['prices']

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Regimes", len(segments.segments))
        with col2:
            avg_len = np.mean([s.end_idx - s.start_idx + 1 for s in segments.segments])
            st.metric("Avg Length", f"{avg_len:.1f} days")
        with col3:
            high_sev = sum(s.severity >= 0.9 for s in segments.segments) / len(segments.segments)
            st.metric("High Severity %", f"{high_sev:.1%}")

        # Three sub-tabs
        tab1, tab2 = st.tabs([
            "📊 Visualization",
            "✅ Validation"
        ])

        with tab1:
            render_regime_chart(prices, segments, results['asset'])
        with tab2:
            render_validation_results(segments, prices)
    else:
        st.info("👆 Click 'Detect Regimes' to begin")


def render_regime_chart(prices, segments, asset):
    """Interactive Plotly chart with regime coloring.

    Args:
        prices: Price series
        segments: RegimeSegments object
        asset: Asset symbol
    """
    # Downsample if needed
    if len(prices) > 2000:
        prices_plot = prices.resample('W').last()
        st.caption(f"ℹ️ Downsampled to {len(prices_plot)} points")
    else:
        prices_plot = prices

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=prices_plot.index, y=prices_plot.values,
        mode='lines', name='Price',
        line=dict(color='black', width=1.5), opacity=0.7
    ))

    # Colored regime segments
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3

    for i, seg in enumerate(segments.segments):
        color = colors[i % len(colors)]

        fig.add_vrect(
            x0=seg.start_date, x1=seg.end_date,
            fillcolor=color,
            opacity=0.15 + 0.25 * seg.severity,
            layer="below", line_width=0,
            annotation_text=f"R{i+1}",
            annotation_position="top left"
        )

        if i > 0:
            fig.add_vline(
                x=seg.start_date,
                line_dash="dash", line_color="red",
                line_width=1.5, opacity=seg.severity
            )

    fig.update_layout(
        title=f"{asset} - {len(segments.segments)} Regimes",
        xaxis_title="Date", yaxis_title="Price",
        hovermode='x unified', height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Regime details table
    regime_df = pd.DataFrame([
        {
            'ID': i + 1,
            'Start': seg.start_date.strftime('%Y-%m-%d'),
            'End': seg.end_date.strftime('%Y-%m-%d'),
            'Days': seg.end_idx - seg.start_idx + 1,
            'Severity': f"{seg.severity:.3f}",
            'Status': '🔴 High' if seg.severity >= 0.9 else '🟡 Med' if seg.severity >= 0.7 else '🟢 Low'
        }
        for i, seg in enumerate(segments.segments)
    ])

    st.dataframe(regime_df, use_container_width=True, hide_index=True)


def render_validation_results(segments, prices):
    """Render statistical validation results.

    Args:
        segments: RegimeSegments object
        prices: Price series
    """
    st.subheader("Statistical Validation")

    report = segments.validate_statistics(prices)

    # Pass/fail summary
    passed_count = sum(1 for c in report.checks if c.passed)
    total_count = len(report.checks)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Checks Passed", f"{passed_count}/{total_count}")
    with col2:
        pass_rate = passed_count / total_count if total_count > 0 else 0
        st.metric("Pass Rate", f"{pass_rate:.1%}")

    # Detailed checks
    st.write("### Detailed Checks")

    for check in report.checks:
        status_icon = "✅" if check.passed else "❌"
        with st.expander(f"{status_icon} {check.name}", expanded=not check.passed):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Expected:** {check.expected}")
            with col2:
                st.write(f"**Actual:** {check.actual}")
