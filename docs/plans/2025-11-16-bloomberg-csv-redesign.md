# Bloomberg CSV Redesign: Wide to Individual Files

**Date:** 2025-11-16
**Status:** Approved
**Context:** Redesigning `data/bloomberg/future_data/bloomberg_historical_data_50_original.csv` for easier parquet conversion

> **Environment note:** Run `uv sync` once in the repo root, then execute every script below via `uv run python …` (for example, `uv run python scripts/reshape_bloomberg_csv.py`).

## Problem Statement

The current Bloomberg export CSV has a problematic "wide format":
- Structure: `ticker_name, date, price, empty, ticker_name, date, price, empty, ...`
- Each ticker occupies 4 columns with data flowing vertically
- Ticker names only appear in row 1, with empty cells below
- Different tickers have different start dates (e.g., CC1 starts 1990, DA1 starts 1996)
- 50 tickers total, creating a 200-column wide file (50 × 4)

This format makes it difficult to:
- Parse and manipulate data programmatically
- Convert to parquet using existing conversion script
- Handle tickers with different date ranges

## Solution: Individual CSV Files Per Ticker

### Target Output Structure

Transform into 50 individual CSV files (one per ticker):

```
data/bloomberg/future_data/individual_csvs/
├── CC1_Comdty.csv
├── DA1_Comdty.csv
├── LB1_Comdty.csv
├── SB1_Comdty.csv
├── ...
└── MP1_Curncy.csv
```

**Each CSV structure:**
```csv
date,price
1990-01-02,929
1990-01-03,912
1990-01-04,940
...
```

**Key features:**
- Header row: `date,price`
- Date format: `YYYY-MM-DD` (ISO 8601 standard)
- No empty rows or columns
- Each ticker's data starts from its first available date
- Compatible with existing `convert_bloomberg_to_parquet.py` script

### Parsing Strategy

**Current structure pattern (repeating every 4 columns):**
```
Column N:   Ticker name (only in row 1, empty below)
Column N+1: Date (all rows have dates)
Column N+2: Price (all rows have prices)
Column N+3: Empty separator
```

**Parsing algorithm:**

1. **Extract ticker names from row 1:**
   - Scan first row for non-empty values containing "Comdty", "Index", or "Curncy"
   - Record column index for each ticker

2. **For each ticker, extract data columns:**
   - Date column = ticker_col + 1
   - Price column = ticker_col + 2

3. **Build per-ticker DataFrames:**
   ```python
   df = pd.DataFrame({
       'date': all_rows[date_col],
       'price': all_rows[price_col]
   })
   df = df.dropna()  # Remove empty rows
   df['date'] = pd.to_datetime(df['date'])  # Parse M/D/YYYY format
   ```

4. **Convert date format:**
   - Input: `M/D/YYYY` (e.g., "1/2/1990")
   - Output: `YYYY-MM-DD` (ISO 8601)

### Implementation

**Script:** `scripts/reshape_bloomberg_csv.py`

**Core function:**
```python
def parse_wide_bloomberg_csv(input_path, output_dir):
    """
    Parse wide-format Bloomberg CSV into individual ticker CSVs.

    Args:
        input_path: Path to bloomberg_historical_data_50_original.csv
        output_dir: Directory to save individual CSV files

    Returns:
        dict of {ticker_name: row_count}
    """
```

**Processing steps:**
1. Read CSV with `header=None` (raw data, no pandas header inference)
2. Parse first row to identify ticker columns
3. For each ticker:
   - Extract date and price columns
   - Skip row 1 (contains ticker name)
   - Create DataFrame with date/price
   - Remove NaN values
   - Convert dates to datetime
   - Sort by date
   - Save to individual CSV with sanitized filename

**Filename sanitization:**
- Replace spaces with underscores: `"CC1 Comdty"` → `"CC1_Comdty.csv"`

### Integration with Existing Pipeline

**Compatibility with `convert_bloomberg_to_parquet.py`:**

The existing conversion script expects individual CSVs with structure:
```python
df = pd.read_csv(csv_path, parse_dates=[0])
df.columns = ['date', 'price']
```

Our output format matches this exactly, so no changes needed to conversion script.

**Workflow after redesign:**
```bash
# Step 1: Reshape wide CSV to individual files
uv run python scripts/reshape_bloomberg_csv.py

# Step 2: Convert individual CSVs to parquet (existing script)
uv run python scripts/convert_bloomberg_to_parquet.py
```

### Benefits

1. **Handles different start dates** - Each ticker has its own date range
2. **Clean separation** - One file per ticker, easy to inspect/debug
3. **Works with existing tools** - Compatible with current conversion script
4. **Standard format** - Follows "tidy data" principles (one observation per row)
5. **Easy to extend** - Adding new tickers just means adding new CSV files
6. **Parquet-friendly** - Simple date/price schema converts cleanly

### Testing Checklist

After running the reshape script:
- [ ] Verify 50 CSV files created
- [ ] Check sample tickers have correct row counts
- [ ] Confirm dates are sorted chronologically
- [ ] Validate date format is `YYYY-MM-DD`
- [ ] Ensure no empty rows or NaN prices
- [ ] Test conversion to parquet using existing script
- [ ] Compare parquet output against original data (spot check)

### Future Expansion

When expanding to 72 assets (per XTREND_ASSET_EXPANSION_NOTES.md):
- Same reshape script works for new Bloomberg exports
- Just increase ticker count (50 → 72)
- Handle crypto assets with limited history (start ~2017)
