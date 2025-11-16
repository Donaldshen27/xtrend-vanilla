# Bloomberg Data Export Guide

Complete workflow for exporting Bloomberg continuous futures data and using it in X-Trend.

## Overview

This directory contains Bloomberg continuous futures data exported from Bloomberg Terminal and converted to Parquet format.

**Data Flow:**
```
Bloomberg Terminal (Excel) → CSV/Excel → Parquet → BloombergParquetSource
```

## Quick Start

### 1. At Bloomberg Terminal (University Library)

Follow the detailed instructions in: [`BLOOMBERG_EXPORT_INSTRUCTIONS.md`](./BLOOMBERG_EXPORT_INSTRUCTIONS.md)

**Summary:**
- Open Excel at Bloomberg Terminal
- Use `=BDH()` formula to download all 50 symbols at once
- Save Excel file and copy to USB drive

**Time required:** ~30 minutes (including formula loading time)

### 2. At Home - Convert to Parquet

Place your Bloomberg export in `data/bloomberg/raw/`:
- Excel file: `data/bloomberg/raw/bloomberg_export.xlsx`
- OR CSV files: `data/bloomberg/raw/*.csv`

Run the conversion script:

```bash
python scripts/convert_bloomberg_to_parquet.py
```

**Output:** Creates `.parquet` files in `data/bloomberg/`

Example:
```
data/bloomberg/CL1.parquet
data/bloomberg/BN1.parquet
data/bloomberg/ES1.parquet
...
```

### 3. Use in X-Trend

```python
from xtrend.data.sources import BloombergParquetSource

# Initialize source
source = BloombergParquetSource(root_path="data/bloomberg")

# Get available symbols
symbols = source.symbols()  # ['CL1', 'BN1', 'ES1', ...]

# Load prices for specific symbols
prices = source.load_prices(
    symbols=['CL1', 'BN1', 'ES1'],
    start='2000-01-01',
    end='2023-12-31'
)
# Returns: DataFrame with dates as index, symbols as columns
```

## Directory Structure

```
data/bloomberg/
├── README.md                              # This file
├── BLOOMBERG_EXPORT_INSTRUCTIONS.md       # Detailed Bloomberg Terminal instructions
├── raw/                                   # Place Bloomberg exports here
│   ├── bloomberg_export.xlsx              # Excel export from Terminal
│   └── *.csv                              # OR individual CSV files
├── CL1.parquet                            # Converted Parquet files
├── BN1.parquet
├── ES1.parquet
└── ...                                    # All 50 symbols
```

## Parquet File Schema

Each `.parquet` file contains:

| Column | Type | Description |
|--------|------|-------------|
| date (index) | datetime64 | Trading date |
| price | float64 | Continuous futures price |

**Example:**
```
           price
date
1990-01-02  45.23
1990-01-03  45.67
1990-01-04  45.12
...
```

## Bloomberg Symbol Convention

**Continuous contracts:** `[ROOT]1 Comdty`

Examples:
- `CL1 Comdty` - Crude Oil front month
- `BN1 Comdty` - Brent Crude front month
- `ES1 Comdty` - E-mini S&P 500 front month
- `ZN1 Comdty` - 10-Year T-Note front month

The `1` indicates front-month continuous contract (Bloomberg auto-rolls).

## Data Source Configuration

Update `conf/data/futures.yaml`:

```yaml
source: bloomberg_parquet
symbols: ${oc.env:XTREND_SYMBOLS,[]}
start: 1990-01-01
end: 2023-12-31
```

## Troubleshooting

### Problem: Conversion script fails with "No files found"

**Solution:** Check that files are in `data/bloomberg/raw/`:
```bash
ls -la data/bloomberg/raw/
```

### Problem: Missing data for some symbols

**Possible causes:**
1. Bloomberg doesn't have data for that symbol back to 1990
2. Symbol name incorrect (check Bloomberg Terminal)
3. Export failed for that specific symbol

**Solution:** Re-export those specific symbols with adjusted date range

### Problem: `#N/A` errors in Excel

**Causes:**
- Invalid symbol format
- Symbol not available in Bloomberg
- Wrong price field (try `PX_SETTLE` instead of `PX_LAST`)

### Problem: Excel formula takes very long to load

**Normal behavior:** 50 symbols over 33 years = ~165,000 data points
- Expected load time: 5-15 minutes
- Don't close Excel while loading

## Data Quality Checks

After conversion, verify your data:

```python
import pandas as pd
from pathlib import Path

# Check all Parquet files
data_dir = Path("data/bloomberg")
for pq_file in data_dir.glob("*.parquet"):
    df = pd.read_parquet(pq_file)
    print(f"{pq_file.stem}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
```

Expected output:
```
CL1: 8500 rows, 1990-01-02 to 2023-12-29
BN1: 7200 rows, 1992-06-15 to 2023-12-29
...
```

## Next Steps

1. ✅ Export data from Bloomberg Terminal
2. ✅ Convert to Parquet format
3. ⏭️ Implement `BloombergParquetSource` class (skeleton exists in `xtrend/data/sources.py`)
4. ⏭️ Test with initial development symbols (BN, ZN, ES)
5. ⏭️ Integrate with X-Trend pipeline

## Files Reference

| File | Purpose |
|------|---------|
| `BLOOMBERG_EXPORT_INSTRUCTIONS.md` | Step-by-step Bloomberg Terminal guide |
| `scripts/convert_bloomberg_to_parquet.py` | CSV/Excel → Parquet converter |
| `xtrend/data/sources.py` | Data source classes (including `BloombergParquetSource`) |
| `conf/data/futures.yaml` | Data source configuration |

## Questions?

Common questions:

**Q: Can I use Bloomberg API instead?**
A: Not on university library computers (can't install Python packages). Excel is simpler.

**Q: How long does the entire process take?**
A: ~1 hour total:
- 30 min at Bloomberg Terminal
- 5 min conversion at home
- 25 min verification and setup

**Q: Do I need all 50 symbols upfront?**
A: No! Start with 3-5 symbols (BN, ZN, ES) for development, add more later.

**Q: What if Bloomberg doesn't have 1990-2023 for all symbols?**
A: Adjust date range per symbol. Some futures started later than 1990.
