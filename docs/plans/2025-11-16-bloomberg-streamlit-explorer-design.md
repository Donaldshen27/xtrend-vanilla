# Bloomberg Parquet Explorer - Streamlit App Design

**Date:** 2025-11-16
**Purpose:** Interactive visualization tool for 72 Bloomberg futures price series
**Use Case:** Quick exploration & data quality checks, with extensibility for analysis features

---

## Overview

A Streamlit web application for visualizing Bloomberg continuous futures data stored in Parquet format. Starts with basic price exploration, designed to grow into a comprehensive analysis dashboard.

**Launch command:**
```bash
uv run streamlit run scripts/bloomberg_explorer.py
```

---

## Architecture

### File Structure

```
scripts/
  bloomberg_explorer.py          # Main Streamlit app (~100 lines)
  bloomberg_viz/                 # Supporting modules
    __init__.py
    data_loader.py               # Parquet loading + caching
    charts.py                    # Plotly chart builders
    analysis.py                  # Returns, volatility, correlation (stubs initially)
    quality.py                   # Data quality checks (stubs initially)
```

### Multi-Tab Layout

```
Tab 1: Price Explorer    ✅ Build first (MVP)
Tab 2: Returns Analysis  🔜 Add later
Tab 3: Data Quality      🔜 Add later
Tab 4: Correlations      🔜 Add later
```

Each tab implemented as separate function:
```python
tab1, tab2, tab3, tab4 = st.tabs(["Prices", "Returns", "Quality", "Correlations"])

with tab1:
    render_price_explorer()  # MVP implementation

with tab2:
    st.info("Coming soon")   # Placeholder
```

**Rationale:**
- Modular structure keeps main app clean
- Easy to add features without breaking existing functionality
- Supporting modules can be reused in Jupyter notebooks

---

## Tab 1: Price Explorer (MVP)

### Sidebar Controls

```python
st.sidebar.header("Data Selection")

# Symbol selection
available_symbols = get_available_symbols()  # ['ES', 'CL', 'GC', ...]
selected_symbols = st.sidebar.multiselect(
    "Select symbols to plot",
    options=available_symbols,
    default=['ES', 'CL', 'GC']  # Sensible defaults
)

# Date range
min_date, max_date = get_date_range(selected_symbols)
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Normalization
normalize = st.sidebar.checkbox("Normalize to 100", value=False)
```

### Main Chart Area

```python
# Load data
data = load_bloomberg_data(selected_symbols, date_range)

# Interactive Plotly chart
fig = create_price_chart(data, normalize=normalize)
st.plotly_chart(fig, use_container_width=True)

# Summary statistics table
st.subheader("Summary Statistics")
display_summary_stats(data)  # mean, std, min, max, date range per symbol
```

**Features:**
- Multi-select for 1-10 symbols (overlay on same chart)
- Date slider filters all symbols to common period
- Normalize toggle: show relative performance (all start at 100)
- Summary table: quick stats per symbol

---

## Data Loading & Caching

### Strategy: Load-on-Demand with Caching

```python
# data_loader.py

import streamlit as st
import pandas as pd
from pathlib import Path

@st.cache_data
def get_available_symbols():
    """Scan processed/ directory for parquet files"""
    parquet_dir = Path("data/bloomberg/processed")
    return sorted([p.stem for p in parquet_dir.glob("*.parquet")])

@st.cache_data
def load_symbol(symbol: str) -> pd.DataFrame:
    """Load single parquet file with caching"""
    path = Path(f"data/bloomberg/processed/{symbol}.parquet")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)  # Ensure datetime index
    return df

def load_bloomberg_data(symbols: list, date_range: tuple) -> dict:
    """Load multiple symbols and filter by date range"""
    data = {}
    for symbol in symbols:
        df = load_symbol(symbol)  # Cached per symbol
        # Filter to date range
        mask = (df.index >= date_range[0]) & (df.index <= date_range[1])
        data[symbol] = df.loc[mask]
    return data
```

**Why this works:**
- `@st.cache_data` on `load_symbol()`: each parquet loaded once, cached in memory
- Switching symbols in UI is instant for already-loaded data
- Only loads selected symbols (not all 72 upfront)
- Date filtering happens after caching (fast)

**Memory footprint:**
- ~8MB per parquet × 10 symbols = ~80MB cached (acceptable)
- Can add cache eviction if memory becomes an issue

---

## Future Tabs (Extensibility Design)

### Tab 2: Returns Analysis

Stub functions in `analysis.py`:

```python
def calculate_returns(data: dict, period: str = 'daily') -> dict:
    """Calculate returns for each symbol

    Args:
        data: {symbol: DataFrame} from load_bloomberg_data()
        period: 'daily', 'weekly', 'monthly'
    Returns:
        {symbol: DataFrame} with returns column
    """
    pass  # Implement when needed

def calculate_volatility(data: dict, window: int = 20) -> dict:
    """Rolling volatility calculation"""
    pass

def compare_performance(data: dict, normalize: bool = True) -> pd.DataFrame:
    """Cumulative returns comparison table"""
    pass
```

### Tab 3: Data Quality

Stub functions in `quality.py`:

```python
def check_missing_dates(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Identify gaps in trading dates"""
    pass

def detect_price_anomalies(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Find >5% daily jumps (potential data errors)"""
    pass

def generate_quality_report(data: dict) -> pd.DataFrame:
    """Summary: date range, missing %, anomaly count per symbol"""
    pass
```

### Tab 4: Correlations

Additional functions in `analysis.py`:

```python
def calculate_correlation_matrix(data: dict, period: str = 'daily') -> pd.DataFrame:
    """Correlation matrix of returns across selected symbols"""
    pass

def rolling_correlation(data: dict, window: int = 60) -> dict:
    """Time-varying correlation between pairs"""
    pass
```

**Implementation approach:**
- Start with Tab 1 fully working
- Add stub tabs with `st.info("Coming soon")`
- Implement features incrementally as needed
- Each tab is independent - no breaking changes

---

## Error Handling

### Startup Validation

```python
# At top of bloomberg_explorer.py
DATA_DIR = Path("data/bloomberg/processed")

if not DATA_DIR.exists():
    st.error(f"❌ Data directory not found: {DATA_DIR}")
    st.info("Run: `uv run python scripts/convert_bloomberg_to_parquet.py`")
    st.stop()

parquets = list(DATA_DIR.glob("*.parquet"))
if len(parquets) == 0:
    st.error("❌ No parquet files found")
    st.stop()

st.success(f"✅ Found {len(parquets)} symbols")
```

### Runtime Error Handling

```python
try:
    data = load_bloomberg_data(selected_symbols, date_range)

    # Check if any data loaded
    if not data or all(df.empty for df in data.values()):
        st.warning("No data found for selected symbols in this date range")
        st.stop()

    # Warn about symbols with sparse data
    for symbol, df in data.items():
        if len(df) < 100:
            st.warning(f"⚠️ {symbol}: Only {len(df)} data points in range")

    # Plot
    fig = create_price_chart(data, normalize=normalize)
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"Data directory not found: {e}")
    st.info("Run conversion script first: `uv run python scripts/convert_bloomberg_to_parquet.py`")

except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.exception(e)  # Show full traceback in development
```

**Key safeguards:**
- Validate data directory exists before loading
- Graceful handling of missing/empty data
- Helpful error messages with actionable steps
- Warn about sparse data (e.g., crypto symbols with short history)

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
]
```

Install with:
```bash
uv sync
```

---

## Testing Plan

### Manual Testing Checklist

1. **Startup:**
   - [ ] App launches without errors
   - [ ] Shows correct count of available symbols
   - [ ] Sidebar loads with defaults

2. **Symbol Selection:**
   - [ ] Can select/deselect symbols
   - [ ] Chart updates immediately
   - [ ] Handles 1, 5, 10 symbols gracefully

3. **Date Range:**
   - [ ] Slider covers full data range
   - [ ] Filtering works correctly
   - [ ] Handles edge cases (all data filtered out)

4. **Normalization:**
   - [ ] Toggle works
   - [ ] All series start at 100 when enabled
   - [ ] Preserves relative performance

5. **Error Conditions:**
   - [ ] Graceful handling of missing data directory
   - [ ] Warning for sparse data symbols
   - [ ] Clear error messages

### Data Quality Checks

```python
# Quick validation script
import pandas as pd
from pathlib import Path

parquets = Path("data/bloomberg/processed").glob("*.parquet")
for p in parquets:
    df = pd.read_parquet(p)
    print(f"{p.stem}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
```

---

## Next Steps

### Phase 1: MVP (Tab 1 only)
1. Create directory structure (`scripts/bloomberg_viz/`)
2. Implement `data_loader.py` with caching
3. Implement `charts.py` (basic Plotly line chart)
4. Build `bloomberg_explorer.py` with Tab 1
5. Add `streamlit` + `plotly` to dependencies
6. Test with 3-5 symbols

### Phase 2: Polish
7. Add summary statistics table
8. Improve chart styling (colors, legends, tooltips)
9. Test with all 72 symbols

### Phase 3: Extended Features (as needed)
10. Implement Tab 2 (Returns Analysis)
11. Implement Tab 3 (Data Quality)
12. Implement Tab 4 (Correlations)

---

## Open Questions

- **Chart library:** Plotly confirmed (interactive, works well with Streamlit)
- **Deployment:** Local only for now (run via `uv run streamlit run`)
- **Max symbols:** UI tested up to 10 overlays (more gets cluttered)
- **Performance:** Caching should handle 72 symbols × 35 years of data (~8MB each)

---

## References

- Streamlit docs: https://docs.streamlit.io
- Plotly docs: https://plotly.com/python/
- Bloomberg data: `data/bloomberg/README.md`
- Parquet schema: date (index), price (float64)
