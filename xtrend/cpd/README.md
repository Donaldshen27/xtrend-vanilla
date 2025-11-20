# GP-CPD Regime Segmentation

Utilities for generating and consuming GP-based change-point (CPD) regimes.

## What lives here
- `segmenter.py`: backward (oracular) CPD used by the paper (Algorithm 1).
- `gp_fitter.py`: GP model fitting and severity scoring.
- `types.py`: small dataclasses for configs and segments.
- `validation.py`: basic stats/consistency checks.

## Generating caches
Use `scripts/batch_generate_cpd_cache.py` (runs backward CPD per symbol).

### Single-span (legacy)
```bash
uv run python scripts/batch_generate_cpd_cache.py \
  --data-path data/bloomberg/processed \
  --cache-dir data/bloomberg/cpd_cache \
  --start 1990-01-01 --end 2023-12-29
```

### Rolling/expanding cutoffs (validation-safe)
```bash
uv run python scripts/batch_generate_cpd_cache.py \
  --data-path data/bloomberg/processed \
  --cache-dir data/bloomberg/cpd_cache \
  --start 2023-12-30 \            # window start for this split
  --rolling-start 2023-12-30 \     # first cutoff
  --rolling-end   2025-11-15 \     # last cutoff
  --rolling-step-days 30           # monthly cutoffs
```
This emits one cache set per end-date; each set only sees data up to its cutoff, eliminating intra-window lookahead.

## Consumption rules (train vs. validation/test)
- Training: `XTrendDataset` passes `allow_future_regimes=True` so targets may attend to regimes that end after the target (paper-approved hindsight) **within the training split only**.
- Validation/Test/Backtest: `allow_future_regimes=False` to enforce causality; use caches whose end date is ≤ the evaluation cutoff (rolling/expanding generation above).

## Safety checks / pitfalls
- Do not reuse caches across splits; filenames encode start/end/n_days, and fallback-to-mismatched-span is disabled.
- A cache that extends past the target dates will leak future info when used with `allow_future_regimes=False`; generate rolling caches instead.
- If you change CPD hyperparams (lookback/threshold/min/max), regenerate caches; filenames encode these params.

## Quick sanity test
```bash
uv run python - <<'PY'
import scripts.batch_generate_cpd_cache as m
print('batch_generate_cpd_cache loaded:', callable(m.batch_generate_cpd_cache))
PY
```

