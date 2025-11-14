# X-Trend Training Guide

## Table of Contents

1. [Loss Functions](#loss-functions)
2. [Training Loop](#training-loop)
3. [Hyperparameters](#hyperparameters)
4. [Training Best Practices](#training-best-practices)

---

## Loss Functions

### Sharpe Loss

```python
def sharpe_loss(positions, returns, volatility, target_vol=0.15):
    """
    Negative Sharpe ratio.

    IMPORTANT: When training with episodic learning where each episode feeds
    a single (asset, time) target, you must accumulate multiple samples in a
    batch before calling this function. Alternatively, use unbiased=False to
    avoid NaN when N=1 (unbiased estimator divides by N-1, which is 0 when N=1).

    Recommended: Accumulate 32+ samples per batch for stable gradient estimates.
    """
    scaled_returns = (target_vol / volatility) * returns * positions

    # Use unbiased=False to avoid NaN with small batches (N=1)
    # This uses N as divisor instead of N-1
    return -torch.sqrt(torch.tensor(252.0)) * scaled_returns.mean() / scaled_returns.std(unbiased=False)
```

### Gaussian MLE Loss

```python
def gaussian_mle_loss(forecast, returns):
    """Negative log-likelihood under Gaussian."""
    mu, log_sigma = forecast[..., 0], forecast[..., 1]
    sigma = torch.exp(log_sigma)

    nll = 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5 * ((returns - mu) / sigma)**2
    return nll.mean()
```

### Quantile Loss

```python
def quantile_loss(forecast, returns, quantiles=None):
    """
    Quantile regression loss.

    Args:
        forecast: Predicted quantiles, shape (..., num_quantiles)
        returns: Actual returns
        quantiles: List of quantile levels to predict. Default is 13 quantiles
                  from 0.05 to 0.95 (as used in X-Trend paper)

    Returns:
        loss: Average quantile loss across all quantiles
    """
    if quantiles is None:
        # Default: 13 quantiles from 0.05 to 0.95 in steps of 0.075
        # This matches the X-Trend paper implementation
        quantiles = [0.05, 0.125, 0.2, 0.275, 0.35, 0.425, 0.5,
                    0.575, 0.65, 0.725, 0.8, 0.875, 0.95]
        # Or equivalently: np.linspace(0.05, 0.95, 13).tolist()

    losses = []
    for i, q in enumerate(quantiles):
        errors = returns - forecast[..., i]
        losses.append(torch.max(q * errors, (q - 1) * errors))

    return torch.stack(losses).mean()
```

### Joint Loss

```python
def joint_loss(positions, forecast, returns, volatility,
              alpha=1.0, forecast_type='gaussian'):
    """
    Joint training objective.

    L_joint = α * L_forecast + L_Sharpe

    Args:
        alpha: Balance parameter (1.0 for Gaussian, 5.0 for Quantile)
    """
    L_sharpe = sharpe_loss(positions, returns, volatility)

    if forecast_type == 'gaussian':
        L_forecast = gaussian_mle_loss(forecast, returns)
    elif forecast_type == 'quantile':
        L_forecast = quantile_loss(forecast, returns)

    return alpha * L_forecast + L_sharpe
```

## Training Loop

### Episodic Training

```python
def train_xtrend(model, train_assets, val_assets, epochs=100,
                context_size=20, seq_len=126, batch_size=32):
    """
    Train X-Trend with episodic learning.

    Each episode:
    1. Sample target (asset, time)
    2. Sample context set C (causally)
    3. Forward pass
    4. Compute loss
    5. Backward pass

    IMPORTANT: Accumulate multiple episodes (batch_size=32+) before computing
    Sharpe loss to avoid NaN from std() with N=1. This ensures stable gradients.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch_idx in range(1000 // batch_size):  # Batches per epoch
            # Accumulate multiple episodes per batch
            batch_positions = []
            batch_forecasts = []
            batch_returns = []
            batch_volatilities = []

            for _ in range(batch_size):
                # Sample target
                target_asset = random.choice(train_assets)
                target_time = random.randint(seq_len, len(target_asset.data))

                target_features = target_asset.features[target_time-seq_len:target_time]
                target_returns = target_asset.returns[target_time]

                # Sample context set (before target_time!)
                context = sample_cpd_context(
                    train_assets,
                    target_time=target_time,
                    size=context_size
                )

                # Forward pass
                position, forecast, _ = model(
                    target_features.unsqueeze(0),
                    target_asset.id,
                    [ctx['features'] for ctx in context],
                    [ctx['asset_id'] for ctx in context]
                )

                # Accumulate batch
                batch_positions.append(position[:, -1, 0])
                batch_forecasts.append(forecast[:, -1, :])
                batch_returns.append(target_returns)
                batch_volatilities.append(target_asset.volatility[target_time])

            # Compute loss on accumulated batch (avoids NaN from std with N=1)
            loss = joint_loss(
                torch.cat(batch_positions),
                torch.stack(batch_forecasts),
                torch.tensor(batch_returns),
                torch.tensor(batch_volatilities),
                alpha=1.0,
                forecast_type='gaussian'
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # Validate
        val_sharpe = validate(model, val_assets)
        print(f"Epoch {epoch}, Val Sharpe: {val_sharpe:.3f}")
```

### Validation Function

```python
def validate(model, val_assets, context_size=20):
    """
    Validate model on validation set.

    Returns annualized Sharpe ratio.
    """
    model.eval()
    all_returns = []

    with torch.no_grad():
        for asset in val_assets:
            for t in range(126, len(asset.data)):
                # Sample context
                context = sample_cpd_context(
                    val_assets,
                    target_time=t,
                    size=context_size
                )

                # Predict
                target_features = asset.features[t-126:t]
                position, _, _ = model(
                    target_features.unsqueeze(0),
                    asset.id,
                    [ctx['features'] for ctx in context],
                    [ctx['asset_id'] for ctx in context]
                )

                # Calculate return
                ret = position[:, -1, 0] * asset.returns[t]
                all_returns.append(ret.item())

    # Calculate Sharpe
    returns_tensor = torch.tensor(all_returns)
    sharpe = torch.sqrt(torch.tensor(252.0)) * returns_tensor.mean() / returns_tensor.std()

    model.train()
    return sharpe.item()
```

## Hyperparameters

### Model Architecture

```python
model_config = {
    'input_dim': 8,           # 5 normalized returns + 3 MACD features
    'hidden_dim': 64,         # Hidden state dimension
    'num_assets': 50,         # Number of assets in universe
    'num_heads': 4,           # Multi-head attention heads
    'forecast_type': 'gaussian'  # 'gaussian' or 'quantile'
}
```

### Training Configuration

```python
training_config = {
    'epochs': 100,
    'batch_size': 32,         # Episodes per batch (CRITICAL: use 32+ to avoid NaN in Sharpe loss)
    'episodes_per_epoch': 1000,
    'context_size': 20,       # Number of context sequences
    'seq_len': 126,           # Target sequence length (6 months)
    'learning_rate': 1e-3,
    'alpha': 1.0,             # Joint loss weight (5.0 for quantile)
    'gradient_clip': 10.0,    # Max gradient norm
    'dropout': 0.3,           # Dropout probability
    'target_vol': 0.15        # Annual target volatility
}
```

### Optimizer Settings

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',           # Maximize Sharpe
    factor=0.5,
    patience=10,
    verbose=True
)
```

## Training Best Practices

### DO:

✅ **Use gradient clipping** (max norm = 10.0) for stable training
✅ **Implement early stopping** based on validation Sharpe
✅ **Monitor attention weights** to ensure meaningful patterns
✅ **Use learning rate scheduling** to fine-tune convergence
✅ **Add dropout** (0.3-0.5) for regularization
✅ **Validate on different time periods** to prevent overfitting
✅ **Save checkpoints** every epoch for recovery

### DON'T:

❌ **Don't skip warm-up** - first 63 predictions are unstable
❌ **Don't use entity embeddings in zero-shot** - model hasn't seen asset
❌ **Don't mix training data** - use episodes, not mini-batches
❌ **Don't ignore attention interpretation** - helps debug
❌ **Don't train too long** - watch for overfitting on validation Sharpe

### Common Training Issues

**Issue: Loss diverges or becomes NaN**
- Solution: Lower learning rate, add gradient clipping, check for data normalization

**Issue: Model predicts constant positions**
- Solution: Check loss function balance (α parameter), ensure sufficient training data

**Issue: Poor zero-shot performance**
- Solution: Increase context diversity, remove entity embeddings from decoder

**Issue: Attention weights are uniform**
- Solution: Increase model capacity, ensure context sequences are informative

---

**Last Updated**: March 2024
**Reference Type**: Training Guide
