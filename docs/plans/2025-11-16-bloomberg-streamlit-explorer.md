# Bloomberg Streamlit Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Streamlit web app for visualizing 72 Bloomberg futures price series with interactive charts.

**Architecture:** Multi-tab Streamlit app with modular supporting modules. Tab 1 (Price Explorer) is MVP, remaining tabs are stubs. Uses Streamlit caching for efficient data loading.

**Tech Stack:** Streamlit, Plotly, pandas, pyarrow (all parquet files already exist in `data/bloomberg/processed/`)

---

## Task 1: Set Up Directory Structure

**Files:**
- Create: `scripts/bloomberg_viz/__init__.py`
- Create: `scripts/bloomberg_viz/data_loader.py`
- Create: `scripts/bloomberg_viz/charts.py`
- Create: `scripts/bloomberg_viz/analysis.py`
- Create: `scripts/bloomberg_viz/quality.py`

**Step 1: Create directory and __init__.py**

```bash
mkdir -p scripts/bloomberg_viz
touch scripts/bloomberg_viz/__init__.py
```

Expected: Directory created with empty __init__.py

**Step 2: Create data_loader.py stub**

File: `scripts/bloomberg_viz/data_loader.py`

```python
"""Data loading utilities for Bloomberg parquet files."""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


DATA_DIR = Path("data/bloomberg/processed")


@st.cache_data
def get_available_symbols() -> List[str]:
    """Scan processed/ directory for parquet files.

    Returns:
        Sorted list of symbol names (without .parquet extension)
    """
    if not DATA_DIR.exists():
        return []
    return sorted([p.stem for p in DATA_DIR.glob("*.parquet")])


@st.cache_data
def load_symbol(symbol: str) -> pd.DataFrame:
    """Load single parquet file with caching.

    Args:
        symbol: Symbol name (without .parquet extension)

    Returns:
        DataFrame with datetime index and 'price' column
    """
    path = DATA_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def load_bloomberg_data(
    symbols: List[str],
    date_range: Tuple[datetime, datetime]
) -> Dict[str, pd.DataFrame]:
    """Load multiple symbols and filter by date range.

    Args:
        symbols: List of symbol names to load
        date_range: (start_date, end_date) tuple

    Returns:
        Dict mapping symbol -> filtered DataFrame
    """
    data = {}
    for symbol in symbols:
        df = load_symbol(symbol)  # Cached per symbol
        # Filter to date range
        mask = (df.index >= date_range[0]) & (df.index <= date_range[1])
        data[symbol] = df.loc[mask]
    return data


def get_date_range(symbols: List[str]) -> Tuple[datetime, datetime]:
    """Get min and max dates across all selected symbols.

    Args:
        symbols: List of symbol names

    Returns:
        (min_date, max_date) tuple
    """
    if not symbols:
        # Default range if no symbols selected
        return (datetime(1990, 1, 1), datetime(2025, 12, 31))

    all_dates = []
    for symbol in symbols:
        df = load_symbol(symbol)
        if not df.empty:
            all_dates.extend([df.index.min(), df.index.max()])

    if not all_dates:
        return (datetime(1990, 1, 1), datetime(2025, 12, 31))

    return (min(all_dates), max(all_dates))
```

**Step 3: Create charts.py stub**

File: `scripts/bloomberg_viz/charts.py`

```python
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
```

**Step 4: Create analysis.py stub (future features)**

File: `scripts/bloomberg_viz/analysis.py`

```python
"""Analysis functions for returns, volatility, correlations (future features)."""

# Stub file - implement when Tab 2 is needed

def calculate_returns(data: dict, period: str = 'daily') -> dict:
    """Calculate returns for each symbol."""
    raise NotImplementedError("Tab 2: Returns Analysis - coming soon")


def calculate_volatility(data: dict, window: int = 20) -> dict:
    """Rolling volatility calculation."""
    raise NotImplementedError("Tab 2: Returns Analysis - coming soon")


def calculate_correlation_matrix(data: dict, period: str = 'daily'):
    """Correlation matrix of returns."""
    raise NotImplementedError("Tab 4: Correlations - coming soon")
```

**Step 5: Create quality.py stub (future features)**

File: `scripts/bloomberg_viz/quality.py`

```python
"""Data quality check functions (future features)."""

# Stub file - implement when Tab 3 is needed

def check_missing_dates(df, symbol: str):
    """Identify gaps in trading dates."""
    raise NotImplementedError("Tab 3: Data Quality - coming soon")


def detect_price_anomalies(df, threshold: float = 5.0):
    """Find >5% daily jumps."""
    raise NotImplementedError("Tab 3: Data Quality - coming soon")


def generate_quality_report(data: dict):
    """Summary: date range, missing %, anomaly count."""
    raise NotImplementedError("Tab 3: Data Quality - coming soon")
```

**Step 6: Commit directory structure**

```bash
git add scripts/bloomberg_viz/
git commit -m "feat: add bloomberg_viz module structure

- data_loader.py: caching utilities for parquet files
- charts.py: Plotly chart builders
- analysis.py: stub for future returns/volatility features
- quality.py: stub for future data quality checks

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Build Main Streamlit App (Tab 1: Price Explorer)

**Files:**
- Create: `scripts/bloomberg_explorer.py`

**Step 1: Create bloomberg_explorer.py**

File: `scripts/bloomberg_explorer.py`

```python
"""Bloomberg Futures Price Explorer - Streamlit App

Launch with: uv run streamlit run scripts/bloomberg_explorer.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bloomberg_viz.data_loader import (
    get_available_symbols,
    load_bloomberg_data,
    get_date_range,
    DATA_DIR
)
from bloomberg_viz.charts import create_price_chart, display_summary_stats


# Page config
st.set_page_config(
    page_title="Bloomberg Futures Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Title
st.title("📈 Bloomberg Futures Price Explorer")


# Startup validation
if not DATA_DIR.exists():
    st.error(f"❌ Data directory not found: {DATA_DIR}")
    st.info("Run: `uv run python scripts/convert_bloomberg_to_parquet.py`")
    st.stop()

parquets = list(DATA_DIR.glob("*.parquet"))
if len(parquets) == 0:
    st.error("❌ No parquet files found")
    st.stop()

st.success(f"✅ Found {len(parquets)} symbols")


# Sidebar controls
st.sidebar.header("📊 Data Selection")

available_symbols = get_available_symbols()

if not available_symbols:
    st.error("No symbols available")
    st.stop()

# Symbol selection with defaults
default_symbols = []
for sym in ['ES', 'CL', 'GC']:
    if sym in available_symbols:
        default_symbols.append(sym)

selected_symbols = st.sidebar.multiselect(
    "Select symbols to plot",
    options=available_symbols,
    default=default_symbols if default_symbols else available_symbols[:3]
)

if not selected_symbols:
    st.warning("⚠️ Please select at least one symbol")
    st.stop()

# Get date range for selected symbols
min_date, max_date = get_date_range(selected_symbols)

# Date range slider
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.date(),
    max_value=max_date.date(),
    value=(min_date.date(), max_date.date())
)

# Convert back to datetime
from datetime import datetime
date_range = (
    datetime.combine(date_range[0], datetime.min.time()),
    datetime.combine(date_range[1], datetime.max.time())
)

# Normalization toggle
normalize = st.sidebar.checkbox("Normalize to 100", value=False)


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Prices",
    "📊 Returns",
    "🔍 Data Quality",
    "🔗 Correlations"
])


# Tab 1: Price Explorer (MVP)
with tab1:
    try:
        # Load data
        data = load_bloomberg_data(selected_symbols, date_range)

        # Check if any data loaded
        if not data or all(df.empty for df in data.values()):
            st.warning("No data found for selected symbols in this date range")
            st.stop()

        # Warn about symbols with sparse data
        for symbol, df in data.items():
            if len(df) < 100:
                st.warning(f"⚠️ {symbol}: Only {len(df)} data points in range")

        # Create and display chart
        fig = create_price_chart(data, normalize=normalize)
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("📋 Summary Statistics")
        stats_df = display_summary_stats(data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        st.info("Check that parquet files exist in data/bloomberg/processed/")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.exception(e)


# Tab 2: Returns Analysis (stub)
with tab2:
    st.info("📊 Returns Analysis - Coming soon")
    st.markdown("""
    **Planned features:**
    - Daily/weekly/monthly returns
    - Cumulative returns
    - Rolling volatility
    - Drawdown analysis
    """)


# Tab 3: Data Quality (stub)
with tab3:
    st.info("🔍 Data Quality - Coming soon")
    st.markdown("""
    **Planned features:**
    - Missing date detection
    - Price anomaly detection (>5% jumps)
    - Data completeness report
    """)


# Tab 4: Correlations (stub)
with tab4:
    st.info("🔗 Correlations - Coming soon")
    st.markdown("""
    **Planned features:**
    - Correlation matrix heatmap
    - Rolling correlations over time
    - Pairwise correlation analysis
    """)
```

**Step 2: Commit main app**

```bash
git add scripts/bloomberg_explorer.py
git commit -m "feat: add Bloomberg Streamlit explorer main app

Tab 1 (Price Explorer) fully functional:
- Multi-symbol selection with defaults
- Date range slider
- Normalize toggle
- Interactive Plotly charts
- Summary statistics table
- Error handling and validation

Tabs 2-4 are stubs for future features.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add streamlit and plotly to dependencies**

Edit `pyproject.toml`, add to `dependencies` list:

```toml
[project]
dependencies = [
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "openpyxl>=3.1.0",
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
]
```

**Step 2: Sync dependencies**

```bash
uv sync
```

Expected output: streamlit and plotly installed successfully

**Step 3: Commit dependency changes**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add streamlit and plotly for visualization

Required for Bloomberg futures price explorer app.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Manual Testing

**No git commits - just verification**

**Step 1: Launch Streamlit app**

```bash
uv run streamlit run scripts/bloomberg_explorer.py
```

Expected: Browser opens to http://localhost:8501

**Step 2: Test basic functionality**

Manual test checklist:

1. ✅ App loads without errors
2. ✅ Shows "Found X symbols" success message
3. ✅ Sidebar has symbol multiselect
4. ✅ Default symbols (ES, CL, GC) are pre-selected
5. ✅ Date range slider appears and is interactive
6. ✅ Chart displays with selected symbols
7. ✅ Normalize checkbox toggles between absolute/relative prices
8. ✅ Summary statistics table appears below chart
9. ✅ Can select different symbols and chart updates
10. ✅ Can adjust date range and chart updates
11. ✅ Tabs 2-4 show "Coming soon" messages

**Step 3: Test error conditions**

1. Deselect all symbols → Should show warning "Please select at least one symbol"
2. Select symbol with short history → Should show warning about sparse data
3. Hover over chart → Should show date and price tooltips

**Step 4: Performance check**

1. Select 10 symbols at once
2. Verify chart renders smoothly
3. Toggle normalize on/off
4. Verify updates are instant (due to caching)

---

## Task 5: Update Documentation

**Files:**
- Create: `scripts/README.md`

**Step 1: Create scripts README**

File: `scripts/README.md`

```markdown
# Scripts Directory

## Bloomberg Data Tools

### bloomberg_explorer.py

Interactive Streamlit app for visualizing Bloomberg futures price data.

**Launch:**
```bash
uv run streamlit run scripts/bloomberg_explorer.py
```

**Features:**
- **Tab 1: Price Explorer** (MVP)
  - Multi-symbol selection
  - Date range filtering
  - Normalize prices to 100
  - Interactive Plotly charts
  - Summary statistics
- **Tabs 2-4:** Stubs for future features (returns, quality, correlations)

**Requirements:**
- Parquet files in `data/bloomberg/processed/`
- Run `uv sync` to install streamlit and plotly

**Architecture:**
```
bloomberg_explorer.py          # Main Streamlit app
bloomberg_viz/
  data_loader.py               # Cached parquet loading
  charts.py                    # Plotly chart builders
  analysis.py                  # Future: returns, volatility
  quality.py                   # Future: data quality checks
```

### Other Scripts

- `convert_bloomberg_to_parquet.py` - Convert CSV/Excel to Parquet
- `reshape_bloomberg_csv.py` - Reshape wide-format CSVs
- `generate_bloomberg_workbook.py` - Generate Excel workbook for Bloomberg Terminal

See `data/bloomberg/README.md` for full workflow.
```

**Step 2: Commit documentation**

```bash
git add scripts/README.md
git commit -m "docs: add scripts/README.md with explorer usage

Documents Bloomberg Streamlit explorer launch and features.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

Before marking complete:

- [ ] All 5 tasks completed and committed (6 commits total)
- [ ] App launches without errors
- [ ] Can select/deselect symbols
- [ ] Chart updates correctly
- [ ] Normalize toggle works
- [ ] Summary statistics display correctly
- [ ] Error handling works (no symbols, no data, etc.)
- [ ] Tabs 2-4 show "Coming soon" stubs
- [ ] Documentation complete

---

## Future Enhancements (Not in This Plan)

**Tab 2: Returns Analysis**
- Implement `calculate_returns()`, `calculate_volatility()`
- Add return chart and volatility chart
- Add cumulative returns comparison

**Tab 3: Data Quality**
- Implement `check_missing_dates()`, `detect_price_anomalies()`
- Add data quality dashboard
- Show gaps, anomalies, completeness stats

**Tab 4: Correlations**
- Implement `calculate_correlation_matrix()`, `rolling_correlation()`
- Add correlation heatmap
- Add rolling correlation time series

These can be added incrementally as needed.
