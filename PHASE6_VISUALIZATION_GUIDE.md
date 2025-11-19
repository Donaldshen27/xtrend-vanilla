# Phase 6 Visualization Guide

Phase 6 (Decoder & Loss Functions) is **complete** with all tests passing! ✅

This guide shows you how to visualize the results.

## Quick Start

### Option 1: Command-Line Script (Recommended)

Run the standalone Python script to generate all visualizations:

```bash
uv run python scripts/phase6_visualizations.py
```

This will create 5 publication-quality plots in `outputs/plots/`:
1. **phase6_position_predictions.png** - Trading positions from all three models
2. **phase6_quantile_predictions.png** - Quantile predictions (fan chart)
3. **phase6_gaussian_predictions.png** - Gaussian distributions with uncertainty
4. **phase6_distribution_snapshots.png** - Distribution PDFs at specific times
5. **phase6_loss_comparison.png** - Loss function comparison

### Option 2: Interactive Jupyter Notebook

For interactive exploration:

```bash
# Start Jupyter (if not installed: uv pip install jupyter)
uv run jupyter notebook notebooks/phase6_visualizations.ipynb
```

The notebook allows you to:
- Modify synthetic data parameters
- Experiment with different model configurations
- Inspect tensor shapes and intermediate outputs
- Create custom visualizations

## What Gets Visualized

### 1. Position Predictions
Compare trading positions (-1 to 1) from all three model variants:
- **XTrend**: Direct position prediction
- **XTrendG**: Gaussian distribution → PTP
- **XTrendQ**: Quantile distribution → PTP (best performance)

### 2. Quantile Predictions (XTrendQ)
Fan chart showing:
- 13 quantile levels (1%, 5%, 10%, ..., 95%, 99%)
- Median prediction
- Confidence intervals
- Actual returns overlaid

### 3. Gaussian Distributions (XTrendG)
Uncertainty quantification:
- Predicted mean (μ)
- Standard deviation (σ) over time
- ±1σ and ±2σ confidence bands
- Actual returns for comparison

### 4. Distribution Snapshots
PDFs at specific time points showing:
- Gaussian distribution curves
- Quantile markers (5%, 50%, 95%)
- Actual return values

### 5. Loss Functions
Performance comparison:
- Sharpe loss (XTrend)
- Joint Gaussian loss (XTrendG)
- Joint Quantile loss (XTrendQ)

## Implementation Details

### Models Tested

All three variants are fully implemented:

```python
# XTrend: Direct position prediction
model = XTrend(config, entity_embedding)
positions = model(target_features, cross_attn_output, entity_ids)

# XTrendG: Gaussian prediction + PTP
model_g = XTrendG(config, entity_embedding)
outputs_g = model_g(target_features, cross_attn_output, entity_ids)
# Returns: {'mean': ..., 'std': ..., 'positions': ...}

# XTrendQ: Quantile prediction + PTP (best)
model_q = XTrendQ(config, entity_embedding, num_quantiles=13)
outputs_q = model_q(target_features, cross_attn_output, entity_ids)
# Returns: {'quantiles': ..., 'positions': ...}
```

### Loss Functions Implemented

```python
# Sharpe ratio loss (core innovation)
loss = sharpe_loss(positions, returns, warmup_steps=63)

# Gaussian joint loss
loss = joint_gaussian_loss(mean, std, positions, returns, alpha=1.0)

# Quantile joint loss
loss = joint_quantile_loss(quantiles, levels, positions, returns, alpha=5.0)
```

## Phase 6 Completion Criteria ✅

All requirements met:

- [x] LSTM Decoder with cross-attention fusion (Equations 19a-19d)
- [x] Three prediction heads: Position, Gaussian, Quantile
- [x] PTP modules: PTP_G, PTP_Q
- [x] Loss functions: Sharpe, Gaussian NLL, Quantile, Joint
- [x] Three model variants: XTrend, XTrendG, XTrendQ
- [x] Entity-conditioned mode (50 entities)
- [x] Zero-shot mode (no entity embeddings)
- [x] End-to-end gradient flow verified
- [x] All 45+ tests passing
- [x] Integration tests passing

## Data Note

**Important:** The visualizations use **synthetic data** since no trained models exist yet.

To see **real performance**:
1. Train the models on your Bloomberg futures data
2. Save trained checkpoints (`.pt` files)
3. Modify the visualization scripts to load trained weights
4. Run on test data to see actual predictions

Example workflow for future training:

```python
# After training (not yet implemented)
model = XTrendQ.load('checkpoints/xtrend_q_best.pt')
test_data = load_test_data('2022-01-01', '2023-12-31')
predictions = model(test_data)

# Visualize real predictions
plot_trading_performance(predictions, test_data.returns)
plot_sharpe_ratio_over_time(predictions, test_data.returns)
plot_cumulative_returns(predictions, test_data.returns)
```

## Next Steps

1. **✅ You are here:** Phase 6 complete, visualizations working
2. **Next:** Implement training loop (Phase 7?)
3. **Then:** Train on real Bloomberg data
4. **Finally:** Backtest and evaluate trading performance

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Sync dependencies
uv sync

# Verify installation
uv run python -c "from xtrend.models import XTrendQ; print('✅ OK')"
```

### Visualization Issues

If plots don't display:

```bash
# Install matplotlib backend
uv pip install matplotlib

# For Jupyter: restart kernel after installing
```

### Test the Models Work

Quick sanity check:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_xtrend_models.py -v
```

Expected: All tests PASS ✅

## File Locations

- **Notebook:** `notebooks/phase6_visualizations.ipynb`
- **Script:** `scripts/phase6_visualizations.py`
- **Outputs:** `outputs/plots/phase6_*.png`
- **Tests:** `tests/models/test_*.py`
- **Implementation:** `xtrend/models/`

---

**Questions?** Check the [Phase 6 Plan](docs/plans/2025-11-18-phase6-decoder-and-losses.md) for full implementation details.
