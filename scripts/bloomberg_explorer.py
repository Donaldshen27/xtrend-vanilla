"""Bloomberg Parquet Explorer - Streamlit App

Interactive visualization tool for 72 Bloomberg futures price series.

Launch command:
    uv run streamlit run scripts/bloomberg_explorer.py
"""

import streamlit as st
from pathlib import Path

from bloomberg_viz.data_loader import (
    get_available_symbols,
    load_bloomberg_data,
    get_date_range,
    DATA_DIR
)
from bloomberg_viz.charts import create_price_chart, display_summary_stats
from bloomberg_viz.returns_tab import render_returns_tab
from bloomberg_viz.regimes_tab import render_regimes_tab


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Bloomberg Futures Explorer",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Bloomberg Futures Explorer")

    # Startup validation
    if not DATA_DIR.exists():
        st.error(f"❌ Data directory not found: {DATA_DIR}")
        st.info("Run: `uv run python scripts/convert_bloomberg_to_parquet.py`")
        st.stop()

    available_symbols = get_available_symbols()
    if len(available_symbols) == 0:
        st.error("❌ No parquet files found")
        st.info("Run: `uv run python scripts/convert_bloomberg_to_parquet.py`")
        st.stop()

    st.success(f"✅ Found {len(available_symbols)} symbols")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Prices",
        "📊 Returns",
        "🎯 Regimes",
        "🔍 Quality",
        "📉 Correlations"
    ])

    with tab1:
        render_price_explorer(available_symbols)

    with tab2:
        # Load default data for Returns tab (same symbols as Price tab would be ideal)
        default_symbols_returns = ['ES', 'XAU', 'BR', 'ZC', 'ZN', 'DX']
        available_for_returns = [s for s in default_symbols_returns if s in available_symbols]
        if not available_for_returns:
            available_for_returns = available_symbols[:6]  # fallback to first 6

        try:
            # Load data for returns analysis
            data_returns = {}
            for symbol in available_for_returns:
                from xtrend.data.sources import BloombergParquetSource
                source = BloombergParquetSource(root_path="data/bloomberg/processed")
                prices_df = source.load_prices([symbol], start='1995-01-01', end='2023-12-31')
                data_returns[symbol] = prices_df.rename(columns={symbol: 'price'})

            render_returns_tab(available_symbols, data_returns)
        except Exception as e:
            st.error(f"❌ Error loading returns data: {e}")
            st.exception(e)

    with tab3:
        # Phase 2: Regimes tab
        try:
            from xtrend.data.sources import BloombergParquetSource
            source = BloombergParquetSource(root_path="data/bloomberg/processed")
            render_regimes_tab(source)
        except Exception as e:
            st.error(f"❌ Error loading regimes tab: {e}")
            st.exception(e)

    with tab4:
        st.info("🔜 Coming soon: Data Quality Checks")
        st.markdown("""
        **Planned features:**
        - Missing date detection
        - Price anomaly identification (>5% jumps)
        - Quality report per symbol
        """)

    with tab5:
        st.info("🔜 Coming soon: Correlation Analysis")
        st.markdown("""
        **Planned features:**
        - Correlation matrix heatmap
        - Rolling correlation charts
        - Cross-asset correlation analysis
        """)


def render_price_explorer(available_symbols):
    """Render the Price Explorer tab (Tab 1)."""
    # Sidebar controls
    st.sidebar.header("Data Selection")

    # Symbol selection
    default_symbols = ['ES', 'CL', 'GC'] if all(s in available_symbols for s in ['ES', 'CL', 'GC']) else available_symbols[:3]
    selected_symbols = st.sidebar.multiselect(
        "Select symbols to plot",
        options=available_symbols,
        default=default_symbols,
        help="Select 1-10 symbols to overlay on the chart"
    )

    if not selected_symbols:
        st.warning("⚠️ Please select at least one symbol to display")
        return

    # Date range
    min_date, max_date = get_date_range(selected_symbols)
    date_range = st.sidebar.slider(
        "Date range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        help="Filter all symbols to this date range"
    )

    # Convert date to datetime for filtering
    from datetime import datetime
    date_range = (
        datetime.combine(date_range[0], datetime.min.time()),
        datetime.combine(date_range[1], datetime.max.time())
    )

    # Normalization
    normalize = st.sidebar.checkbox(
        "Normalize to 100",
        value=False,
        help="Show relative performance (all series start at 100)"
    )

    # Load data
    try:
        data = load_bloomberg_data(selected_symbols, date_range)

        # Check if any data loaded
        if not data or all(df.empty for df in data.values()):
            st.warning("⚠️ No data found for selected symbols in this date range")
            return

        # Warn about symbols with sparse data
        for symbol, df in data.items():
            if len(df) < 100:
                st.warning(f"⚠️ {symbol}: Only {len(df)} data points in range")

        # Main chart
        st.subheader("Price Chart")
        fig = create_price_chart(data, normalize=normalize)
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")
        summary_df = display_summary_stats(data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.info("Run conversion script: `uv run python scripts/convert_bloomberg_to_parquet.py`")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
