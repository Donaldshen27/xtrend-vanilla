# GP-CPD Normalization Fix

## Problem Summary

GP-CPD segmentation is producing **100% fallback** (zero change-point detection) with `lookback=63`. Investigation revealed the root cause is **unnormalized financial data** causing GP optimization to fail.

## Root Cause Analysis

### Symptoms
- ALL segments have `severity = 0.000000`
- 100% fallback to fixed 21-day tiles
- Detection rate: 0.0% across all symbols

### Investigation Results

Testing on ES (2019-2020, window of 63 days):

```
Price range: [2988.25, 3316.50]  # Large absolute values
Stationary GP log marginal likelihood: -3,298.36
Change-point GP log marginal likelihood: -34,571.42  # 10× worse!
Delta (L_C - L_M): -31,273.07
Severity: sigmoid(-31,273) ≈ 0.000000
```

### Grid Search Results

All 26 candidates succeeded (no exceptions), but produced catastrophic likelihoods:

```
Best candidate (t_cp=32):
  Segment 1 (32 days): log_mll = -16,396
  Segment 2 (31 days): log_mll = -18,175
  Combined: -34,571

vs. Single stationary GP (63 days): -3,298
```

### Root Cause

**Unnormalized financial data causes numerical instability in GP optimization:**

1. **Scale issue:** Prices in range [2988, 3316] are large absolute values
2. **GP kernel sensitivity:** Matérn kernel expects standardized inputs
3. **Optimization failure:** Adam optimizer struggles with badly scaled data
4. **Small segments amplify problem:** 10-30 day segments with high variance fail catastrophically

## Solution

### Normalize data before GP fitting

Modify `xtrend/cpd/gp_fitter.py` to:

1. **Standardize inputs:** Convert y to z-scores before fitting
2. **Fit on standardized data:** GP operates on (y - mean) / std
3. **Restore scale for predictions:** Not needed for CPD (only comparing likelihoods)

### Implementation

```python
def fit_stationary_gp(self, x: Tensor, y: Tensor) -> Tuple[ExactGPModel, float]:
    # Standardize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0  # Avoid division by zero
    y_norm = (y - y_mean) / y_std

    # Fit GP on normalized data
    # ... existing code using y_norm instead of y ...
```

Apply same normalization to `fit_changepoint_gp` for both segments.

### Expected Impact

With normalization:
- Log marginal likelihoods should be in range [-100, 0] instead of [-50,000, -3,000]
- GP optimization will converge properly
- Change-point detection rate: 0% → 30-50%
- Severity values will be meaningful (currently all 0.0)

## Files to Modify

1. **xtrend/cpd/gp_fitter.py** (lines 46-97, 99-196):
   - Add normalization to `fit_stationary_gp`
   - Add normalization to `fit_changepoint_gp` (both segments)

2. **Test validation:** Run on ES 2019-2020
   - Expected: 30-50% detection rate
   - Expected: Mean severity 0.85-0.95
   - Expected: Diverse regime lengths (not all 21 days)

## References

- GP best practices require standardized inputs for numerical stability
- GPyTorch documentation recommends z-score normalization for time-series
- X-Trend paper (Wood et al. 2024) likely normalized returns, not raw prices

## Next Steps

1. Implement normalization in `gp_fitter.py`
2. Test on ES 2019-2020 (should complete in ~2 minutes with proper convergence)
3. If successful, regenerate all caches with `batch_generate_cpd_cache.py`
4. Validate detection rates across all symbols

---

**Status:** Root cause identified
**Priority:** HIGH - Blocks CPD from working at all
**Effort:** 1 hour (implementation + testing)
