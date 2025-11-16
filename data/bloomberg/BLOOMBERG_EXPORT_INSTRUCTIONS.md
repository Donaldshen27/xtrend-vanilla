# Bloomberg Excel Export Instructions

## Step 1: At the Bloomberg Terminal

1. Open Excel (Bloomberg Add-in should be pre-loaded)
2. Create a new workbook
3. Set up your symbol list

## Step 2: Symbol List Setup

1. Open [`symbol_map.csv`](./symbol_map.csv) (or the copy in the README).
2. In Excel set:
   - `A1 = Pinnacle ID`
   - `B1 = Bloomberg Ticker`
3. Paste the 50 rows from the table below so every Pinnacle ID aligns with its Bloomberg ticker. The `BDH()` formula will reference **column B** while column **A** preserves the IDs needed by the paper.

## Step 3: Bulk Download Formula

In cell C1, enter this formula:

```excel
=BDH($B$2:$B$51,"PX_LAST","19900101","20231231","Dir=V")
```

**Formula breakdown:**
- `B2:B51` = Bloomberg ticker strings listed below
- `"PX_LAST"` = Last price field
- `"19900101"` = Start date (Jan 1, 1990)
- `"20231231"` = End date (Dec 31, 2023)
- `"Dir=V"` = Vertical direction (dates down, symbols across)

Press Enter and wait 5-10 minutes for Bloomberg to load all the data.

## Step 4: Export Options

### Option A: Export entire workbook (EASIEST)
1. File → Save As
2. Save as: `bloomberg_export.xlsx`
3. Copy to USB drive

### Option B: Export individual CSVs (if needed)
For each symbol column:
1. Select the date + price columns for one symbol
2. Copy to new sheet
3. File → Save As → CSV
4. Name it: `[SYMBOL].csv` (e.g., `CL1.csv`)

## Step 5: Verify Your Data

Before leaving the terminal, check:
- [ ] All 50 symbols loaded (check for #N/A errors)
- [ ] Date range is correct (1990-2023)
- [ ] Data looks reasonable (no all zeros)

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
