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
