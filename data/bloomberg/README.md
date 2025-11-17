# Bloomberg Data Export Guide

Complete workflow for exporting Bloomberg continuous futures data and using it in X-Trend.

## Overview

This directory contains Bloomberg continuous futures data exported from Bloomberg Terminal and converted to Parquet format.

**Data Flow:**
```
Bloomberg Terminal (Excel) → CSV/Excel → Parquet → BloombergParquetSource
```

## Quick Start

### 0. Before Going to Bloomberg Terminal (Recommended)

#### Install Python dependencies (first time only)

The helper scripts require `openpyxl`, `pandas`, and `pyarrow`. Create a virtual environment (recommended) and install the bundled requirements:

```bash
python3 -m venv .venv-bloomberg
source .venv-bloomberg/bin/activate
pip install -r requirements/bloomberg.txt
```

Run the commands from the project root so the `requirements/` folder is in the working directory.

> Skip the `source` step on Windows; run `.venv-bloomberg\Scripts\activate` instead.

Generate the Excel workbook with pre-configured BDH formula:

```bash
# For 50-asset set
python scripts/generate_bloomberg_workbook.py

# For 72-asset set (recommended; currently 69 fully-specified tickers)
python scripts/generate_bloomberg_workbook.py --expanded
```

Copy the generated file from `data/bloomberg/raw/` to USB drive.

### 1. At Bloomberg Terminal (University Library)

Follow the detailed instructions in: [`BLOOMBERG_EXPORT_INSTRUCTIONS.md`](./BLOOMBERG_EXPORT_INSTRUCTIONS.md)

**Summary:**
- Open the generated Excel file (or create manually with `=BDH()` formula)
- Press Enter in cell F1 to download all symbols at once
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

> The converter auto-loads both `symbol_map.csv` and `symbol_map_expanded.csv` when present. Use `--symbol-map path/to/custom.csv` (repeatable) to override.

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

## Canonical Symbol Map (Matches Table 5 & 6)

To stay faithful to the X-Trend paper portfolio (Tables 5 & 6 in *Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies*), we maintain a canonical mapping between Pinnacle IDs and Bloomberg tickers in [`data/bloomberg/symbol_map.csv`](./symbol_map.csv). Copy/paste this list into Excel so column **A** holds the Pinnacle IDs (useful for traceability) and column **B** holds the Bloomberg tickers that the `BDH()` formula should reference:

| Pinnacle | Bloomberg | Asset Class | Description |
|----------|-----------|-------------|-------------|
| CC | CC1 Comdty | CM | Cocoa |
| DA | DA1 Comdty | CM | Class III Milk |
| LB | LB1 Comdty | CM | Random Length Lumber |
| SB | SB1 Comdty | CM | Sugar #11 |
| ZA | PA1 Comdty | CM | Palladium |
| ZC | ZC1 Comdty | CM | Corn |
| ZF | FC1 Comdty | CM | Feeder Cattle |
| ZI | SI1 Comdty | CM | Silver |
| ZO | ZO1 Comdty | CM | Oats |
| ZR | ZR1 Comdty | CM | Rough Rice (electronic) |
| ZU | CL1 Comdty | CM | WTI Crude Oil |
| ZW | ZW1 Comdty | CM | Chicago Wheat |
| ZZ | HE1 Comdty | CM | Lean Hogs |
| GI | GI1 Comdty | CM | Goldman Sachs Commodity Index |
| JO | OJ1 Comdty | CM | Orange Juice |
| KC | KC1 Comdty | CM | Coffee |
| KW | KW1 Comdty | CM | Kansas City Wheat |
| NR | RR1 Comdty | CM | Rough Rice (legacy pit) |
| ZG | GC1 Comdty | CM | Gold |
| ZH | HO1 Comdty | CM | Heating Oil |
| ZK | HG1 Comdty | CM | Copper |
| ZL | ZL1 Comdty | CM | Soybean Oil |
| ZN | NG1 Comdty | CM | Natural Gas |
| ZP | PL1 Comdty | CM | Platinum |
| ZT | LC1 Comdty | CM | Live Cattle |
| EN | NQ1 Index | EQ | Nasdaq 100 E-Mini |
| ES | ES1 Index | EQ | S&P 500 E-Mini |
| MD | MID1 Index | EQ | S&P 400 E-Mini |
| SC | SP1 Index | EQ | S&P 500 composite |
| SP | SP1 Index | EQ | S&P 500 day session |
| XX | VG1 Index | EQ | Dow Jones STOXX 50 |
| YM | YM1 Index | EQ | Mini Dow Jones |
| CA | CAC1 Index | EQ | CAC 40 Index |
| ER | RTY1 Index | EQ | Russell 2000 E-Mini |
| LX | Z 1 Index | EQ | FTSE 100 Index |
| NK | NK1 Index | EQ | Nikkei 225 Index |
| XU | VG1 Index | EQ | Dow Jones EURO STOXX 50 |
| DT | RX1 Comdty | FI | Euro Bund |
| FB | FV1 Comdty | FI | U.S. 5Y Treasury Note |
| TY | TY1 Comdty | FI | U.S. 10Y Treasury Note |
| UB | OE1 Comdty | FI | Euro Bobl |
| US | US1 Comdty | FI | U.S. 30Y Treasury Bond |
| AN | AD1 Curncy | FX | Australian Dollar |
| DX | DX1 Curncy | FX | U.S. Dollar Index |
| FN | EC1 Curncy | FX | Euro FX |
| JN | JY1 Curncy | FX | Japanese Yen |
| SN | SF1 Curncy | FX | Swiss Franc |
| BN | BP1 Curncy | FX | British Pound |
| CN | CD1 Curncy | FX | Canadian Dollar |
| MP | MP1 Curncy | FX | Mexican Peso |

**Notes**
- `NR` only exists in the Pinnacle dataset as a pit-composite Rough Rice series; map it to the same Bloomberg contract (`RR1 Comdty`) used by `ZR` if you cannot source a separate pit session.
- `SC` and `SP` both refer to S&P 500 futures (composite vs. day session). Bloomberg publishes a single continuous contract (`SP1 Index`), so both IDs share that feed.
- `XX` and `XU` are historical labels for the Dow Jones (Euro) STOXX 50. Bloomberg consolidates them under `VG1 Index`.
- Update the CSV if you need to add or swap assets; the converter script consumes it directly.

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

## 2025-11-16 Export Snapshot (Reference USB Pull)

The raw files delivered from the November 16, 2025 Bloomberg session are stored at
`data/bloomberg/future_data-20251117T014559Z-1-001.zip`. The archive contains:

| File | Notes |
|------|-------|
| `future_data/bloomberg_historical_data_50_original.csv` | Original 50-symbol context set pulled from the Pinnacle table. |
| `future_data/bloomberg_extend_tickers.csv` | 24 additional instruments (LME forwards, crypto, microstructure stress proxies). |

Extract with:

```bash
unzip data/bloomberg/future_data-20251117T014559Z-1-001.zip -d data/bloomberg
```

### CSV layout (four-column blocks)

Bloomberg wrote each asset into **four adjacent columns**: `[Ticker Label, Date, Price, blank]`. The first row holds the ticker string, and all following rows in that block hold the date/price pairs. To reshape into a standard wide table (dates down, tickers across), drop the empty columns and stack the `[Date, Price]` pairs per ticker. Example:

```python
import pandas as pd

raw = pd.read_csv("future_data/bloomberg_extend_tickers.csv", header=None)
series = []
for offset in range(0, raw.shape[1], 4):
    ticker = raw.iat[0, offset]
    if not isinstance(ticker, str) or not ticker.strip():
        continue
    block = raw.iloc[1:, [offset + 1, offset + 2]].dropna()
    block.columns = ["date", ticker.strip()]
    block["date"] = pd.to_datetime(block["date"])
    series.append(block.set_index("date"))

extended_df = pd.concat(series, axis=1).sort_index()
```

Use the same pattern for the `bloomberg_historical_data_50_original.csv` file.

### Extended ticker coverage (Nov 16, 2025 pull)

| Ticker | Description | Date Range | Rows |
|--------|-------------|------------|------|
| LMAHDS03 Comdty | LME Aluminum 3M forward | 1990-01-02 → 2025-11-14 | 9,065 |
| LMZSDS03 Comdty | LME Zinc 3M forward | 1990-01-02 → 2025-11-14 | 9,067 |
| LMCADS03 Comdty | LME Copper 3M forward | 1990-01-02 → 2025-11-14 | 9,065 |
| LMNIDS03 Comdty | LME Nickel 3M forward | 1990-01-02 → 2025-11-14 | 9,056 |
| LMPBDS03 Comdty | LME Lead 3M forward | 1990-01-02 → 2025-11-14 | 9,062 |
| LMSNDS03 Comdty | LME Tin 3M forward | 1990-01-02 → 2025-11-14 | 9,063 |
| GX1 Index | Eurex DAX future | 1990-11-23 → 2025-11-14 | 8,867 |
| TP1 Index | TOPIX large future | 1990-05-16 → 2025-11-14 | 8,716 |
| HI1 Index | Hang Seng future | 1992-04-01 → 2025-11-14 | 8,356 |
| ZTSA Index | FTSE/JSE Top 40 | 2013-12-09 → 2025-11-14 | 1,258 |
| BTCA Curncy | Bitcoin aggregate index | 2025-05-30 → 2025-11-14 | 118 |
| DCRA Curncy | Decred aggregate index | 2025-05-30 → 2025-11-14 | 118 |
| MERA Curncy | Monero aggregate index | 2025-05-30 → 2025-11-14 | 118 |
| BMR Curncy | Bitcoin Cash aggregate index | 2023-02-28 → 2025-11-14 | 683 |
| XAUUSD Curncy | Gold spot vs USD | 1990-01-02 → 2025-11-14 | 9,304 |
| XAGUSD Curncy | Silver spot vs USD | 1990-01-02 → 2025-11-14 | 9,303 |
| XPTUSD Curncy | Platinum spot vs USD | 1990-01-01 → 2025-11-14 | 9,346 |
| XPDUSD Curncy | Palladium spot vs USD | 1993-10-29 → 2025-11-14 | 8,256 |
| XB1 Comdty | NYMEX RBOB gasoline | 2005-10-03 → 2025-11-14 | 5,068 |
| VIX Index | CBOE Volatility Index | 1990-01-02 → 2025-11-14 | 9,059 |
| FF1 Comdty | CME Fed Funds (front) | 1990-01-02 → 2025-11-14 | 9,035 |
| FF4  Comdty | CME Fed Funds (4th contract) | 1990-01-02 → 2025-11-14 | 9,036 |
| TU1 Comdty | CBOT 2Y Treasury Note | 1990-06-25 → 2025-11-14 | 8,913 |
| LF98OAS Index | ICE/BofA US HY OAS | 1994-01-31 → 2025-11-14 | 6,413 |

These 24 additions extend the context set beyond the 50 futures in Table 5 of the paper. The workbook generator currently ships 69 concrete tickers (50 originals + 19 of the planned additions) until Bloomberg publishes SOL/XRP codes; use the reference CSV above if you need the full list immediately.

### Notable quirks in the 50-symbol CSV

- Several agricultural contracts roll over to new tickers mid-history; e.g. `ZC1 Comdty` only spans 2006-11-16 → 2009-12-30 (197 rows) in this pull.
- Thin or discontinued contracts (`ZR1 Comdty`, `HE1 Comdty`, `MID1 Index`) only show hundreds of rows, so expect #N/A gaps when you rebuild the workbook.
- Some columns are duplicated because Pinnacle listed separate session IDs (e.g. both `SC` and `SP` map to `SP1 Index`; both appear in the CSV).
- The `YM1  Comdty` header retains multiple spaces from Bloomberg; trim whitespace before feeding into scripts.

Keep these quirks in mind when validating conversions so you do not mistake genuine contract limits for download errors.

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
| `scripts/generate_bloomberg_workbook.py` | Generate Excel workbook with BDH formula |
| `scripts/convert_bloomberg_to_parquet.py` | CSV/Excel → Parquet converter |
| `BLOOMBERG_EXPORT_INSTRUCTIONS.md` | Step-by-step Bloomberg Terminal guide |
| `symbol_map.csv` | 50-asset symbol mapping (Pinnacle → Bloomberg) |
| `symbol_map_expanded.csv` | 72-asset symbol mapping (includes new additions) |
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
