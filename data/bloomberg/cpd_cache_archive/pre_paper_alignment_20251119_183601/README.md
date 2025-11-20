# Archived CPD Cache Files

**Archive Date:** 2025-11-19 18:36:01
**Reason:** Parameter alignment with X-Trend paper specification

## Why Archived

These cache files were generated with parameters that don't match the paper's best X-Trend-Q configuration (2.70 Sharpe ratio).

### Old Configurations Found

1. **Configuration A:** `lb21_th0.90_min5_max21`
   - Lookback: 21 days ✓
   - Threshold: 0.90 ✗ (should be 0.95)
   - Max length: 21 days ✗ (should be 63)
   - Date ranges: Various (2018-2021, 2018-2023)

2. **Configuration B:** `lb63_th0.66_min5_max21`
   - Lookback: 63 days ✗ (should be 21)
   - Threshold: 0.66 ✗ (should be 0.95)
   - Max length: 21 days ✗ (should be 63)
   - Date ranges: 2018-2023

### Paper-Optimal Configuration

The new configuration matches Table 1 (best X-Trend-Q result):
- **Lookback:** 21 days
- **Threshold:** 0.95
- **Max length:** 63 days
- **Date range:** 1990-2023

## Files Archived

- **Total files:** 130
- **Types:** .pkl (cache files), .json (config logs), .csv (generation results)

## Restoration

If you need to restore these files (not recommended):

```bash
# Copy back to active cache directory
cp data/bloomberg/cpd_cache_archive/pre_paper_alignment_20251119_183601/*.pkl \
   data/bloomberg/cpd_cache/

# Or just reference for comparison
ls data/bloomberg/cpd_cache_archive/pre_paper_alignment_20251119_183601/
```

## Next Steps

1. Generate new cache files with paper-optimal parameters:
   ```bash
   uv run python scripts/batch_generate_cpd_cache.py
   ```

2. Train with matching configuration:
   ```bash
   uv run python scripts/train_xtrend.py --model xtrendq
   ```

The new cache files will use: `lb21_th0.95_min5_max63` with 1990-2023 date range.
