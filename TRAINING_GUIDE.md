# X-Trend Training Guide

Complete guide for training X-Trend models on your Bloomberg futures data.

## Quick Start

```bash
# Train XTrendQ (best model from paper - Sharpe 2.70)
uv run python scripts/train_xtrend.py --model xtrendq --epochs 50

# Monitor training (in another terminal)
tail -f checkpoints/xtrendq_history.json
```

## Training Commands

### Train XTrendQ (Recommended)
```bash
uv run python scripts/train_xtrend.py \
    --model xtrendq \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --hidden-dim 128
```

### Train XTrendG
```bash
uv run python scripts/train_xtrend.py \
    --model xtrendg \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4
```

### Train XTrend (Baseline)
```bash
uv run python scripts/train_xtrend.py \
    --model xtrend \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4
```

### Resume Training
```bash
uv run python scripts/train_xtrend.py \
    --model xtrendq \
    --resume checkpoints/xtrendq_latest.pt
```

## Data Preparation

The training script automatically:
1. **Loads** Bloomberg parquet files from `data/bloomberg/processed/`
2. **Computes** 8 features per asset:
   - Returns at 1, 5, 21 day scales
   - MACD indicators (8/24, 16/48)
   - Rolling volatility (20-day)
   - Normalized price
   - Volume proxy
3. **Normalizes** returns using 252-day volatility window
4. **Creates** rolling sequences of 126 time steps (~6 months)
5. **Splits** data: 2000-2020 (train), 2021-2023 (validation)

## Training Details

### What Happens During Training

**Forward Pass:**
```
Target Features → Encoder → Cross-Attention ← Context
                     ↓
                  Decoder → Prediction Head → Positions
                     ↓
                 Loss Function (Sharpe/Gaussian/Quantile)
```

**Loss Functions:**
- **XTrend**: Sharpe ratio loss (maximize risk-adjusted returns)
- **XTrendG**: Joint Gaussian loss (α × MLE + Sharpe)
- **XTrendQ**: Joint Quantile loss (α × QRE + Sharpe)

### Hyperparameters (Default)

```python
Model Architecture:
  - Input dim: 8 (features)
  - Hidden dim: 128
  - Attention heads: 4
  - Dropout: 0.1
  - Entities: 50+ (number of futures contracts)

Training:
  - Batch size: 32
  - Learning rate: 1e-4 (AdamW)
  - Weight decay: 1e-5
  - Gradient clipping: 1.0
  - Scheduler: CosineAnnealing
  - Epochs: 50
  - Warmup: 63 steps (ignored in loss)

Sequences:
  - Target length: 126 (~6 months)
  - Context size: 20 (random assets)
  - Min history: 252 days (for features)
```

## Checkpoints

Training saves three checkpoints:

1. **`{model}_best.pt`** - Best validation loss
2. **`{model}_latest.pt`** - Most recent epoch
3. **`{model}_epoch_{N}.pt`** - Every 10 epochs

Each checkpoint contains:
```python
{
    'epoch': int,
    'encoder': state_dict,
    'cross_attn': state_dict,
    'model': state_dict,
    'optimizer': state_dict,
    'train_loss': float,
    'val_loss': float,
    'config': ModelConfig,
    'model_type': str,
    'best_val_loss': float
}
```

## Monitoring Training

### Training History (JSON)

```bash
# View training progress
cat checkpoints/xtrendq_history.json
```

```json
{
  "train_loss": [3.2, 2.8, 2.5, ...],
  "val_loss": [3.4, 3.0, 2.7, ...],
  "lr": [0.0001, 0.00009, ...]
}
```

### Expected Training Time

**On CPU:**
- ~10-15 min/epoch with 32 batch size
- ~8-12 hours for 50 epochs

**On GPU (recommended):**
- ~1-2 min/epoch
- ~1-2 hours for 50 epochs

Use GPU if available:
```bash
# Automatic GPU detection
uv run python scripts/train_xtrend.py --model xtrendq --device cuda
```

## Evaluation & Backtesting

After training, evaluate the model:

```python
# Load best checkpoint
checkpoint = torch.load('checkpoints/xtrendq_best.pt')

# Create model
config = ModelConfig(**checkpoint['config'])
model = XTrendQ(config, ...)
model.load_state_dict(checkpoint['model'])

# Evaluate on test data (2024+)
test_data = load_test_data('2024-01-01', '2024-12-31')
predictions = model(test_data)

# Compute metrics
sharpe_ratio = compute_sharpe(predictions, returns)
cumulative_returns = compute_cumulative_returns(predictions, returns)
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
uv run python scripts/train_xtrend.py --model xtrendq --batch-size 16
```

Or reduce hidden dimension:
```bash
uv run python scripts/train_xtrend.py --model xtrendq --hidden-dim 64
```

### Training Too Slow

Enable GPU:
```bash
uv run python scripts/train_xtrend.py --model xtrendq --device cuda
```

Reduce workers if I/O bound:
```bash
# Edit script: num_workers=2 instead of 4
```

### Loss Not Decreasing

1. Check data quality:
```bash
# Verify parquet files load correctly
uv run python -c "from xtrend.data.sources import BloombergParquetSource; s = BloombergParquetSource('data/bloomberg/processed'); print(s.symbols())"
```

2. Reduce learning rate:
```bash
uv run python scripts/train_xtrend.py --model xtrendq --lr 5e-5
```

3. Check for NaN gradients:
```python
# Training script includes gradient clipping
# If issues persist, add more aggressive clipping
```

### Data Loading Errors

Verify Bloomberg data:
```bash
ls data/bloomberg/processed/*.parquet | wc -l  # Should be 50+
```

Check date coverage:
```python
import pandas as pd
df = pd.read_parquet('data/bloomberg/processed/ES.parquet')
print(f"Date range: {df.index[0]} to {df.index[-1]}")
```

## Performance Targets

Based on the paper (Table 1, Page 12):

| Model | Expected Sharpe | Annual Return | Max Drawdown |
|-------|----------------|---------------|--------------|
| XTrend | ~2.3 | ~15% | ~7% |
| XTrendG | ~2.5 | ~17% | ~7% |
| **XTrendQ** | **~2.7** | **~18%** | **~7%** |

**Note:** These are research paper results. Your results will vary based on:
- Data quality and coverage
- Training hyperparameters
- Market conditions in your data period
- Transaction costs (not included in paper)

## Next Steps After Training

1. **Visualize Predictions** - Use trained models in visualization notebooks
2. **Backtest** - Implement walk-forward backtesting
3. **Live Paper Trading** - Test on recent data before live deployment
4. **Risk Management** - Add position sizing, stop losses
5. **Multi-Asset** - Extend to cross-sectional portfolio

## Advanced: Hyperparameter Tuning

Use Optuna or Ray Tune for automated search:

```python
# Example search space
search_space = {
    'hidden_dim': [64, 128, 256],
    'num_heads': [2, 4, 8],
    'dropout': [0.05, 0.1, 0.2],
    'lr': [5e-5, 1e-4, 5e-4],
    'alpha': [0.5, 1.0, 2.0, 5.0]  # For joint losses
}
```

## Citation

If you use this implementation, cite the original paper:

```
@article{xtrend2024,
  title={Deep Momentum Networks for Time-Series Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

---

**Ready to train?**

```bash
# Start training now!
uv run python scripts/train_xtrend.py --model xtrendq --epochs 50
```
