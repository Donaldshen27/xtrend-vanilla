# Bloomberg Excel Export Instructions

## Step 1: At the Bloomberg Terminal

1. Open Excel (Bloomberg Add-in should be pre-loaded)
2. Create a new workbook
3. Set up your symbol list

## Step 2: Symbol List Setup

In Column A, starting from A2, enter all your symbols:

```
A2: CL1 Comdty
A3: BN1 Comdty
A4: ZN1 Comdty
A5: ES1 Comdty
... (continue for all 50 symbols)
A51: [Your 50th symbol]
```

## Step 3: Bulk Download Formula

In cell C1, enter this formula:

```excel
=BDH(A2:A51,"PX_LAST","19900101","20231231","Dir=V")
```

**Formula breakdown:**
- `A2:A51` = Range of all your symbols
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

```
CL1 Comdty
BN1 Comdty
ZN1 Comdty
ES1 Comdty
```

(Add your complete list of 50 futures symbols here)

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
`scripts/convert_bloomberg_csv_to_parquet.py`
