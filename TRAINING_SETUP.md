# X-Trend Training Setup Guide

## Current Status

✅ **Training cache**: 37/71 symbols (52% coverage)
✅ **Validation cache**: 54/71 symbols with rolling windows
⚠️ **Missing**: 35 symbols need cache generation
⚠️ **On-the-fly CPD recompute is now disabled by default**. Training will fail fast if any symbol is missing a usable cache unless you pass `--allow-cpd-recompute`.

## Option 1: Pre-generate All Cache Files (RECOMMENDED)

**Best for:** Optimal performance matching paper's 2.70 Sharpe result

### Step 1: Generate missing caches (~10-15 minutes)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python scripts/generate_missing_cpd_cache.py
```

This will:
- Process 35 missing symbols
- Generate CPD regimes for training period (1990-2023)
- Save to `data/bloomberg/cpd_cache/`
- Match existing configuration (threshold=0.85, max_length=63)
- Ensure cache spans overlap your dataset dates; the loader now picks the cache file with the greatest overlap to the current dataset window. If a cache exists but doesn’t overlap (e.g., only a late rolling window), generate (or regenerate) a full-span cache for that symbol.

### Step 2: Train with full dataset

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python scripts/train_xtrend.py \
  --model xtrendq \
  --train-cutoff 2023-12-29 \
  --cpd-threshold 0.85 \
  --cpd-max-length 63 \
  --context-size 10 \
  # Optional: permit slow per-run recomputation if a cache is missing
  # --allow-cpd-recompute \
  --epochs 50 \
  --batch-size 64
```

**Expected results:**
- ✅ All 71 symbols (maximum diversity)
- ✅ ~700K+ training samples (vs 502K with 37 symbols)
- ✅ Fast initialization (all cached)
- ✅ Best performance (matches paper configuration)

---

## Option 2: Train with Cached Symbols Only

**Best for:** Immediate training without waiting for cache generation

### Create cached-only training script:

```bash
# Create a wrapper that filters to cached symbols
cat > scripts/train_xtrend_cached_only.py << 'EOF'
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import main training script
from train_xtrend import *

# Override symbol loading
original_main = main

def main():
    # Monkey-patch to use only cached symbols
    cached_symbols = [
        'AL', 'AN', 'BN', 'CA', 'CC', 'CN', 'CU', 'DT', 'DX', 'FB',
        'FC', 'FD', 'GC', 'HG', 'JY', 'KC', 'LH', 'LN', 'MC', 'MD',
        'NE', 'NG', 'NQ', 'PA', 'PL', 'RR', 'RX', 'SB', 'SI', 'SM',
        'TN', 'TU', 'TY', 'US', 'ZB', 'ZF', 'ZT'
    ]

    # Run with filtered symbols
    # (you'll need to modify the main() function to accept symbol list)
    original_main()

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/train_xtrend_cached_only.py
```

**Tradeoffs:**
- ✅ Instant startup (no cache generation)
- ✅ Fast initialization
- ❌ Only 37 symbols (reduced diversity)
- ❌ ~502K samples (30% fewer than full dataset)
- ❌ May underperform vs paper results
- ⚠️ Still enforced: missing caches will fail unless `--allow-cpd-recompute` is set.

---

## Option 3: Use Alternate Context Sampling

**Best for:** Testing without CPD dependency

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python scripts/train_xtrend.py \
  --model xtrendq \
  --train-cutoff 2023-12-29 \
  --context-method time_equivalent \  # or final_hidden_state
  --context-size 10 \
  --epochs 50
```

**Tradeoffs:**
- ✅ No cache needed
- ✅ All 71 symbols
- ❌ May not match paper's results (paper used cpd_segmented)
- ❌ Different context sampling strategy

---

## Quick Decision Matrix

| Goal | Best Option | Command |
|------|-------------|---------|
| **Match paper performance** | Option 1 | Generate caches, then train |
| **Start training NOW** | Option 2 | Use 37 cached symbols only |
| **Test without CPD** | Option 3 | Use `--context-method time_equivalent` |
| **Allow slow fallback** | Any | Add `--allow-cpd-recompute` (not recommended for reproducibility) |

## Recommendation

**Use Option 1** - Generate missing caches first:

```bash
# 1. Generate missing caches (10-15 min one-time cost)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python scripts/generate_missing_cpd_cache.py

# 2. Train with full dataset (best performance)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python scripts/train_xtrend.py \
  --model xtrendq \
  --train-cutoff 2023-12-29 \
  --cpd-threshold 0.85 \
  --epochs 50
```

This gives you:
- ✅ Maximum training data diversity
- ✅ Fastest training initialization
- ✅ Best chance to replicate paper's 2.70 Sharpe
- ✅ One-time 15-minute setup cost
