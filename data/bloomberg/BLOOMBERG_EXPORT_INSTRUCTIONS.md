# Bloomberg Excel Export Instructions

**UPDATED:** Now includes 72 assets (50 original + 22 new for enhanced X-Trend context diversity)

## Quick Start

**🚀 AUTOMATED WORKBOOK GENERATION (RECOMMENDED):**
```bash
# Generate 50-asset workbook
python scripts/generate_bloomberg_workbook.py

# Generate 72-asset workbook (expanded set)
python scripts/generate_bloomberg_workbook.py --expanded

# Custom date range
python scripts/generate_bloomberg_workbook.py --expanded --start 1995-01-01 --end 2024-12-31
```

This creates an Excel file with the BDH formula pre-configured. Just copy to USB, open at Bloomberg Terminal, and press Enter in cell F1.

> Dependencies: `openpyxl`, `pandas`, and `pyarrow`. Install once with `python3 -m venv .venv-bloomberg && source .venv-bloomberg/bin/activate && pip install -r requirements/bloomberg.txt`.

> The expanded workbook currently includes **69 fully-configured tickers** (50 original + 19 new). Add SOL/XRP once Bloomberg codes are confirmed to reach the planned 72.

**Manual Setup:**
**For original 50 assets:** See [Section A](#section-a-original-50-assets)
**For expanded 72 assets:** See [Section B](#section-b-expanded-72-asset-set)

---

## Section B: Expanded 72-Asset Set

### Why 72 Assets?

The expanded set adds **22 new instruments** to enhance X-Trend's few-shot learning via:
- **Venue diversity**: LME, Eurex, CME crypto
- **Currency diversity**: EUR, JPY, HKD, ZAR
- **Microstructure diversity**: 3M forwards, spot prices, crypto 24/7 trading
- **Expected performance**: +25-35% Sharpe improvement vs 50-asset baseline

See [`XTREND_ASSET_EXPANSION_NOTES.md`](./XTREND_ASSET_EXPANSION_NOTES.md) for detailed rationale.

### Automated Method (Recommended)

Run the workbook generator script before going to the terminal:

```bash
python scripts/generate_bloomberg_workbook.py --expanded
```

This creates `data/bloomberg/raw/bloomberg_export_72assets.xlsx` with all symbols and the BDH formula pre-configured (currently 69 populated tickers; add SOL/XRP later when Bloomberg codes are published).

### Manual Method

### Step 1: At the Bloomberg Terminal

1. Open Excel (Bloomberg Add-in should be pre-loaded)
2. Create a new workbook
3. Set up your symbol list

### Step 2: Symbol List Setup (Expanded Set)

1. Open [`symbol_map_expanded.csv`](./symbol_map_expanded.csv)
2. In Excel set:
   - `A1 = Pinnacle ID`
   - `B1 = Asset Class`
   - `C1 = Description`
   - `D1 = Bloomberg Ticker`
   - `E1 = Notes (optional, matches CSV column 5)`
3. Paste all rows from the expanded symbol map (currently 69 tickers; add the remaining assets when Bloomberg releases the SOL/XRP codes)

### Step 3: Bulk Download Formula (Expanded Set)

In cell F1, enter this formula:

```excel
=BDH($D$2:$D$70,"PX_LAST","19900101","20251116","Dir=V")
```

**Formula breakdown:**
- `$D$2:$D$70` = Bloomberg ticker strings for the expanded asset list
- `"PX_LAST"` = Last price field
- `"19900101"` = Start date (Jan 1, 1990)
- `"20251116"` = End date (Nov 16, 2025)
- `"Dir=V"` = Vertical direction (dates down, symbols across)

**⚠️ Important Notes:**
- The expanded CSV includes asset class + description columns, so the tickers live in column **D** by default. Adjust the BDH range if you reorder columns in Excel.
- Press Enter and wait **10-20 minutes** for Bloomberg to load data (≈70 assets × 30+ years)
- Some assets (crypto, recent int'l indices) may not have data back to 1990 - this is expected
- Bloomberg will show #N/A for dates before asset inception (handle in Python preprocessing)

#### Reference export (Nov 16, 2025)

See `data/bloomberg/future_data-20251117T014559Z-1-001.zip` for the exact files we downloaded at the library:

| File | Coverage |
|------|----------|
| `future_data/bloomberg_extend_tickers.csv` | 24 assets: the 6 LME 3M forwards, 4 global equity indices (GX1, TP1, HI1, ZTSA), 4 crypto aggregates (BTCA, DCRA, MERA, BMR), 4 spot metals (XAU/XAG/XPT/XPD), RBOB gasoline (XB1), plus risk proxies (VIX Index, FF1 Comdty, FF4  Comdty, TU1 Comdty, LF98OAS Index). |
| `future_data/bloomberg_historical_data_50_original.csv` | Canonical 50 futures and FX contracts. Some (e.g., `ZC1 Comdty`, `ZR1 Comdty`, `MID1 Index`) have short histories because Bloomberg only stores portions of the legacy contracts. |

Each CSV is formatted as repeating four-column blocks `[Ticker label, Date, Price, blank column]`. When importing into pandas or Excel, copy each `[Date, Price]` pair into its own tab/column or adapt the sample reshaping snippet in `data/bloomberg/README.md`.

---

## Section A: Original 50 Assets

### Automated Method (Recommended)

Run the workbook generator script before going to the terminal:

```bash
python scripts/generate_bloomberg_workbook.py
```

This creates `data/bloomberg/raw/bloomberg_export.xlsx` with all symbols and the BDH formula pre-configured.

### Manual Method

### Step 1: At the Bloomberg Terminal

1. Open Excel (Bloomberg Add-in should be pre-loaded)
2. Create a new workbook
3. Set up your symbol list

### Step 2: Symbol List Setup (Original)

1. Open [`symbol_map.csv`](./symbol_map.csv) (or the copy in the README).
2. In Excel set:
   - `A1 = Pinnacle ID`
   - `B1 = Asset Class`
   - `C1 = Description`
   - `D1 = Bloomberg Ticker`
3. Paste the 50 rows from the symbol map CSV

### Step 3: Bulk Download Formula (Original)

In cell F1, enter this formula:

```excel
=BDH($D$2:$D$51,"PX_LAST","19900101","20251116","Dir=V")
```

**Formula breakdown:**
- `$D$2:$D$51` = Bloomberg ticker strings from column D
- `"PX_LAST"` = Last price field
- `"19900101"` = Start date (Jan 1, 1990)
- `"20251116"` = End date (Nov 16, 2025)
- `"Dir=V"` = Vertical direction (dates down, symbols across)

Press Enter and wait 5-10 minutes for Bloomberg to load all the data.

---

## Step 4: Export Options (Both 50 & 72 Asset Sets)

### Option A: Export entire workbook (EASIEST)
1. File → Save As
2. Save as:
   - `bloomberg_export.xlsx` (for 50 assets)
   - `bloomberg_export_72assets.xlsx` (for 72 assets)
3. Copy to USB drive

### Option B: Export individual CSVs (if needed)
For each symbol column:
1. Select the date + price columns for one symbol
2. Copy to new sheet
3. File → Save As → CSV
4. Name it: `[SYMBOL].csv` (e.g., `CL1.csv`)

---

## Step 5: Verify Your Data

### For 50-Asset Export:
- [ ] All 50 symbols loaded (check for #N/A errors)
- [ ] Date range is correct (1990-2023)
- [ ] Data looks reasonable (no all zeros)

### For 72-Asset Export:
- [ ] Original 50 symbols loaded completely
- [ ] Check new additions (22 assets) - expect some #N/A for early dates:
  - **LME metals (6)**: Should have data back to 1990s
  - **Global indices (4)**: DAX, TOPIX, Hang Seng back to 1990s; JSE may be later
  - **Crypto (4)**: Expect #N/A before 2017-2018 (asset didn't exist)
  - **Spot metals (4)**: Should have full history
  - **RBOB (1)**: Should have data from ~2006 (MTBE replacement)
- [ ] Date range is 1990-2023 where data exists
- [ ] No all-zeros columns (indicates data error)

## Step 6: Transfer to Your Computer

Copy the Excel file (or CSVs) to USB drive.

## Alternative Fields (if PX_LAST doesn't work)

Try these Bloomberg fields if PX_LAST gives errors:
- `PX_SETTLE` - Settlement price
- `PX_CLOSE` - Close price
- `PX_MID` - Mid price

---

## 50 Symbols to Export

Copy these into Excel Column A (A2:A51):

| Pinnacle ID | Bloomberg ticker |
|-------------|------------------|
| CC | CC1 Comdty |
| DA | DA1 Comdty |
| LB | LB1 Comdty |
| SB | SB1 Comdty |
| ZA | PA1 Comdty |
| ZC | ZC1 Comdty |
| ZF | FC1 Comdty |
| ZI | SI1 Comdty |
| ZO | ZO1 Comdty |
| ZR | ZR1 Comdty |
| ZU | CL1 Comdty |
| ZW | ZW1 Comdty |
| ZZ | HE1 Comdty |
| GI | GI1 Comdty |
| JO | OJ1 Comdty |
| KC | KC1 Comdty |
| KW | KW1 Comdty |
| NR | RR1 Comdty |
| ZG | GC1 Comdty |
| ZH | HO1 Comdty |
| ZK | HG1 Comdty |
| ZL | ZL1 Comdty |
| ZN | NG1 Comdty |
| ZP | PL1 Comdty |
| ZT | LC1 Comdty |
| EN | NQ1 Index |
| ES | ES1 Index |
| MD | MID1 Index |
| SC | SP1 Index |
| SP | SP1 Index |
| XX | VG1 Index |
| YM | YM1 Index |
| CA | CAC1 Index |
| ER | RTY1 Index |
| LX | Z 1 Index |
| NK | NK1 Index |
| XU | VG1 Index |
| DT | RX1 Comdty |
| FB | FV1 Comdty |
| TY | TY1 Comdty |
| UB | OE1 Comdty |
| US | US1 Comdty |
| AN | AD1 Curncy |
| DX | DX1 Curncy |
| FN | EC1 Curncy |
| JN | JY1 Curncy |
| SN | SF1 Curncy |
| BN | BP1 Curncy |
| CN | CD1 Curncy |
| MP | MP1 Curncy |

---

## 22 New Assets in Expanded Set

Copy these into Excel **after** the 50 original assets (rows A52:E73):

| Pinnacle ID | Asset Class | Description | Bloomberg Ticker | Notes |
|-------------|-------------|-------------|------------------|-------|
| **LME Base Metals (6)** | | | | |
| AL | CM | LME Aluminum 3M | LMAHDS03 Comdty | LME 3-month aluminum - microstructure: forward curve vs spot |
| ZN_LME | CM | LME Zinc 3M | LMZSDS03 Comdty | LME 3-month zinc - complements existing COMEX metals |
| CU | CM | LME Copper 3M | LMCADS03 Comdty | LME 3M copper - comp to ZK (COMEX). Different venue dynamics |
| NI | CM | LME Nickel 3M | LMNIDS03 Comdty | LME 3M nickel - EV battery metal exposure |
| PB | CM | LME Lead 3M | LMPBDS03 Comdty | LME 3M lead - industrial metal context |
| TN | CM | LME Tin 3M | LMSNDS03 Comdty | LME 3M tin - electronics/solder demand |
| **Global Equity Indices (4)** | | | | |
| GX | EQ | DAX Index (Germany) | GX1 Index | Eurex DAX - German equity benchmark. Euro-denominated |
| TP | EQ | TOPIX Index (Japan) | TP1 Index | JPX TOPIX - broader Japan market vs NK (Nikkei 225) |
| HI | EQ | Hang Seng Index (Hong Kong) | HI1 Index | HKEX Hang Seng - Hong Kong/China exposure |
| TS | EQ | FTSE/JSE Top 40 (South Africa) | ZTSA Index | JSE South Africa Top 40 - emerging market equity |
| **Cryptocurrency Futures (4)** | | | | |
| BC | CR | Bitcoin (aggregate) | BTCA Curncy | Bitcoin aggregate index - crypto flagship |
| DC | CR | Decred | DCRA Curncy | Decred aggregate - alt L1 blockchain |
| XM | CR | Monero | MERA Curncy | Monero aggregate - privacy coin |
| BR | CR | Bitcoin Cash | BMR Curncy | Bitcoin Cash - Bitcoin fork |
| **Spot Precious Metals (4)** | | | | |
| XAU | SP | Gold Spot vs USD | XAUUSD Curncy | Spot gold - basis vs ZG futures. London fix reference |
| XAG | SP | Silver Spot vs USD | XAGUSD Curncy | Spot silver - basis vs ZI futures |
| XPT | SP | Platinum Spot vs USD | XPTUSD Curncy | Spot platinum - basis vs ZP futures |
| XPD | SP | Palladium Spot vs USD | XPDUSD Curncy | Spot palladium - basis vs ZA futures |
| **Refined Energy (1)** | | | | |
| RB | CM | RBOB Gasoline | XB1 Comdty | NYMEX RBOB - gasoline crack spread vs crude oil (ZU) |

**Note:** LME Zinc uses Pinnacle ID `ZN_LME` so the legacy `ZN` natural gas contract remains queryable in Bloomberg and in downstream scripts.

**Pending Confirmation (add when available):**
- SOL (Solana futures) - Check CME vendor file for Bloomberg code
- XR (XRP futures) - Check CME vendor file for Bloomberg code

---

## Troubleshooting

**Formula returns #N/A:**
- Check symbol format (should be "[ROOT]1 Comdty")
- Try different price field (PX_SETTLE instead of PX_LAST)
- Check date format (YYYYMMDD)

**Formula loads slowly:**
- This is normal for 50 symbols over 30+ years
- Wait 10-15 minutes if needed
- Don't close Excel while loading

**Some symbols missing:**
- Bloomberg may not have data for all contracts back to 1990
- Note which ones failed and adjust date range for those

---

## Next Step

After exporting, use the conversion script at:
`scripts/convert_bloomberg_to_parquet.py`
