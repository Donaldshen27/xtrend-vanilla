# Few-Shot Learning Implementation Patterns

## Table of Contents

1. [Context Set Construction](#context-set-construction)
2. [Meta-Learning Architecture](#meta-learning-architecture)
3. [Implementation Patterns](#implementation-patterns)
4. [Evaluation Protocols](#evaluation-protocols)

---

## Context Set Construction

### Random Context

```python
def construct_context_set(target_time, train_assets, size=20,
                         method='random'):
    """
    Build context set for few-shot prediction.

    Args:
        target_time: Time t of target prediction
        train_assets: Available assets for context
        size: Number of sequences in context |C|
        method: 'random', 'time_equivalent', 'cpd_segmented'

    Returns:
        context_set: List of (asset, start_time, end_time) tuples
    """
    context_set = []

    for _ in range(size):
        # Sample random asset (can be different from target)
        asset = random.choice(train_assets)

        if method == 'random':
            # Random sequence of fixed length before target_time
            length = random.randint(10, 30)
            end_time = random_time_before(target_time)
            start_time = end_time - length

        elif method == 'time_equivalent':
            # Same length as target, time-aligned
            start_time = target_start_time
            end_time = target_time

        elif method == 'cpd_segmented':
            # Use change-point detection to get regime segments
            segment = sample_cpd_segment(asset, before=target_time)
            start_time, end_time = segment

        context_set.append((asset, start_time, end_time))

    return context_set
```

**Context Set Properties:**
- Size `|C|`: Typically 10-30 sequences
- Causality: All context sequences must occur before target time
- Diversity: Include different assets and market conditions
- Quality: Using change-point detection improves performance 11.3%

### Change-Point Segmented Context

```python
def cpd_segmented_context(assets, target_time, size=20):
    """
    Use Gaussian Process change-point detection to create context.

    Improvement: +11.3% Sharpe vs random sequences (X-Trend paper)
    """
    context = []

    for _ in range(size):
        asset = random.choice(assets)
        prices = load_prices(asset, end=target_time)

        # Detect regime boundaries
        changepoints = gaussian_process_cpd(
            prices,
            threshold=0.9,  # LC/(LM + LC) ≥ 0.9
            lookback=21
        )

        # Sample one regime segment
        segment = random.choice(changepoints)
        start, end = segment

        # Limit segment length
        if end - start > 21:  # max 1 month
            start = end - 21

        if end - start >= 5:  # min 5 days
            context.append({
                'asset': asset,
                'features': load_features(asset, start, end),
                'returns': load_returns(asset, start, end)
            })

    return context
```

### Time-Equivalent Context

```python
def time_equivalent_context(target_asset, target_length, target_time,
                           other_assets, size=20):
    """
    Create context with same time structure as target.

    Each context sequence:
    - Same length as target (lc = lt)
    - Time-aligned (can attend to corresponding time steps)
    """
    context = []

    for _ in range(size):
        asset = random.choice(other_assets)

        # Same time window as target
        start = target_time - target_length
        end = target_time

        context.append({
            'asset': asset,
            'features': load_features(asset, start, end),
            'returns': load_returns(asset, start, end),
            'length': target_length
        })

    return context
```

## Meta-Learning Architecture

### Input Structure

```python
# Target sequence (what we predict for)
target = {
    'asset': 'SPY',
    'features': x[t-126:t],  # 126 days of MACD, returns, etc.
    'time': t
}

# Context set (what we condition on)
context = [
    {'asset': 'GLD', 'features': ξ[c][t1-20:t1], 'returns': y[c][t1]},
    {'asset': 'TLT', 'features': ξ[c][t2-20:t2], 'returns': y[c][t2]},
    ...  # C total sequences
]

# Key difference: Context includes labels (returns), target doesn't
```

### Model Flow

```python
class FewShotTradingModel(nn.Module):
    def forward(self, target, context):
        # 1. Encode context sequences
        context_encodings = []
        for ctx in context:
            h_c = self.encoder(ctx['features'])
            context_encodings.append(h_c)

        # 2. Encode target sequence
        h_target = self.encoder(target['features'])

        # 3. Cross-attention: target attends to context
        context_summary = self.cross_attention(
            query=h_target,
            keys=context_encodings,
            values=context_encodings
        )

        # 4. Combine target encoding + context summary
        combined = torch.cat([h_target, context_summary], dim=-1)

        # 5. Predict position
        position = self.decoder(combined)

        return position
```

### Transfer Learning Strategies

**Pattern Transfer:**
```python
def find_similar_patterns(target_sequence, context_set):
    """
    Identify which context sequences are most similar to target.

    This is what cross-attention does automatically!
    """
    similarities = []

    for context_seq in context_set:
        # Attention computes similarity
        sim = cosine_similarity(
            target_sequence.encoding,
            context_seq.encoding
        )
        similarities.append(sim)

    # Attention weights = softmax(similarities)
    attention_weights = softmax(similarities)

    # Weighted combination of context
    transferred_knowledge = sum(
        w * ctx.encoding
        for w, ctx in zip(attention_weights, context_set)
    )

    return transferred_knowledge
```

**Example Transfer Scenarios:**

1. **Same Asset, Different Regime** (Few-shot)
   - Target: SPY in 2020 (COVID crash)
   - Context: SPY in 2008 (financial crisis), SPY in 2018 (correction)
   - Transfer: Crisis response patterns

2. **Different Assets, Similar Dynamics** (Zero-shot)
   - Target: New cryptocurrency (BTC)
   - Context: Gold, Silver, Crude Oil (commodities)
   - Transfer: Trending behavior, volatility patterns

3. **Cross-Asset Momentum Spillover**
   - Target: European equities (CAC40)
   - Context: US equities (SPY), Asian equities (Nikkei)
   - Transfer: Leading indicators, correlation structures

### Training Objectives

**Joint Loss Function:**
```python
def joint_loss(model, target, context, returns, alpha=1.0):
    """
    Combine forecasting accuracy with trading performance.

    L_joint = α * L_MLE + L_Sharpe

    This ensures model learns both to forecast AND to trade profitably.
    """
    # Get model outputs
    forecast_dist = model.forecast(target, context)  # (μ, σ)
    position = model.position(target, context)       # z ∈ [-1, 1]

    # Maximum likelihood loss (forecasting)
    L_MLE = -log_likelihood(returns, forecast_dist)

    # Sharpe ratio loss (trading)
    strategy_returns = position * returns
    L_Sharpe = -sharpe_ratio(strategy_returns)

    return alpha * L_MLE + L_Sharpe
```

**Why Joint Training?**
- Pure forecasting (MLE) doesn't optimize for trading
- Pure Sharpe can overfit to training period
- Joint training balances both objectives
- α balances importance (typically α=1 for Gaussian, α=5 for quantile)

## Implementation Patterns

### Pattern 1: Basic Few-Shot Setup

```python
class BasicFewShotTrader:
    def __init__(self, context_size=20):
        self.context_size = context_size
        self.encoder = LSTMEncoder(input_dim=X, hidden_dim=64)
        self.attention = CrossAttention(dim=64)
        self.decoder = PositionDecoder(input_dim=128, output_dim=1)

    def predict(self, target_features, context_set):
        # Encode all context sequences
        context_encodings = [
            self.encoder(ctx['features'])
            for ctx in context_set
        ]

        # Encode target
        target_encoding = self.encoder(target_features)

        # Cross-attention
        attended = self.attention(
            query=target_encoding,
            context=context_encodings
        )

        # Decode to position
        combined = torch.cat([target_encoding, attended], dim=-1)
        position = torch.tanh(self.decoder(combined))

        return position
```

## Evaluation Protocols

### Expanding Window Backtest

```python
def few_shot_backtest(assets, start_year=1990, end_year=2023):
    """
    Evaluate few-shot model with expanding training window.

    Prevents look-ahead bias and simulates realistic deployment.
    """
    results = []

    # Initial training: 1990-1995
    # Test: 1995-2000
    # Expand training: 1990-2000
    # Test: 2000-2005
    # ... and so on

    train_start = start_year
    test_periods = [
        (1995, 2000),
        (2000, 2005),
        (2005, 2010),
        (2010, 2013),
        (2013, 2018),
        (2018, 2023)
    ]

    for test_start, test_end in test_periods:
        # Train on all data up to test period
        train_data = load_data(train_start, test_start)
        model = train_model(train_data)

        # Test on next 5-year period
        test_data = load_data(test_start, test_end)

        for t in test_data.timestamps:
            # Sample context set (from training period only!)
            context = sample_context_before(t, train_data)

            # Make prediction for target
            position = model.predict(
                target=test_data[t],
                context=context
            )

            # Record returns
            ret = position * test_data.returns[t+1]
            results.append(ret)

        # Expand training window
        train_start = train_start

    return results
```

### Zero-Shot Evaluation

```python
def zero_shot_backtest(train_assets, test_assets):
    """
    Test on completely unseen assets.

    train_assets: 30 assets (e.g., traditional futures)
    test_assets: 20 different assets (e.g., cryptocurrencies)
    """
    # Train only on train_assets
    model = train_model(train_assets, method='episodic')

    # Test on test_assets (model has never seen these!)
    results = []
    for asset in test_assets:
        for t in test_period:
            # Context from train_assets only
            context = sample_context(train_assets, before_time=t)

            # Target from test_assets
            target = load_target(asset, time=t)

            position = model.predict(target, context)
            ret = position * asset.returns[t+1]
            results.append(ret)

    return results
```

### Performance Metrics

```python
def calculate_few_shot_metrics(returns, baseline_returns):
    """
    Calculate metrics specific to few-shot learning evaluation.
    """
    return {
        'sharpe': np.sqrt(252) * returns.mean() / returns.std(),
        'baseline_sharpe': np.sqrt(252) * baseline_returns.mean() / baseline_returns.std(),
        'improvement': (returns.mean() - baseline_returns.mean()) / baseline_returns.mean(),
        'max_drawdown': calculate_max_drawdown(returns),
        'recovery_time': calculate_recovery_time(returns)
    }
```

## Common Pitfalls

### Pitfall 1: Data Leakage

```python
# WRONG - context from future!
context = sample_sequences(all_time_periods)

# CORRECT - context only from past
context = sample_sequences(before=target_time)
```

### Pitfall 2: Overfitting to Context Construction

```python
# WRONG - optimization on test set
best_cpd_threshold = optimize_on_test_set()

# CORRECT - validate on held-out data
best_cpd_threshold = cross_validate_on_train_set()
```

### Pitfall 3: Ignoring Asset Heterogeneity

```python
# WRONG - assume all assets behave identically
encoding = lstm(features)  # Same encoding for all

# CORRECT - use entity embeddings
encoding = lstm(features) + asset_embedding[asset_id]
```

---

**Last Updated**: March 2024
**Reference Type**: Implementation Details
