---
name: few-shot-learning-finance
description: Few-shot learning for financial time-series and trading strategies. Covers episodic learning, context sets, support and query sequences, zero-shot vs few-shot learning, meta-learning for finance, transfer learning across assets and regimes, quick adaptation to market changes. Use when implementing models that learn from minimal data or need to adapt to new market regimes rapidly.
---

# Few-Shot Learning for Finance

## Purpose

Guide for implementing few-shot learning techniques in financial trading strategies, enabling models to quickly adapt to new market regimes or trade previously unseen assets with minimal data.

## When to Use

Activate this skill when:
- Implementing models that adapt to regime changes quickly
- Trading new or low-liquidity assets with limited history
- Building strategies that transfer knowledge across assets
- Dealing with non-stationary markets or structural breaks
- Implementing meta-learning for trading strategies
- Creating context-based prediction systems

## Core Concepts

### 1. Few-Shot vs Zero-Shot Learning

**Few-Shot Learning:**
- Model has seen the target asset during training
- Can use historical data from same asset (in context set)
- Training set and test set overlap: `I_train ∩ I_test = I`
- Example: Adapting to new regime of S&P 500 after COVID-19

**Zero-Shot Learning:**
- Model has NEVER seen the target asset during training
- Must transfer knowledge from different assets entirely
- Training set and test set disjoint: `I_train ∩ I_test = ∅`
- Example: Trading a new cryptocurrency using patterns learned from equities

```python
# Few-shot setting
train_assets = ['SPY', 'GLD', 'TLT']  # 30 assets
test_assets = ['SPY', 'GLD', 'TLT']   # Same 30 assets, different time period

# Zero-shot setting
train_assets = ['SPY', 'GLD', 'TLT']  # 30 assets for training
test_assets = ['BTC', 'ETH', 'SOL']   # 20 different assets for testing
```

### 2. Episodic Learning

Train models the same way they'll be used at test time:

**Traditional Training:**
```python
# Standard mini-batch training - all assets mixed together
for epoch in epochs:
    for batch in shuffle(all_data):
        loss = model(batch)
        optimizer.step()
```

**Episodic Training:**
```python
# Episode-based training - mimics test-time usage
for episode in episodes:
    # Sample target sequence (what we want to predict)
    target_asset, target_time = sample_target()

    # Sample context set (what we condition on)
    context_set = sample_contexts(
        assets=train_assets,
        exclude=(target_asset, target_time),  # Ensure causality
        size=C  # Number of context sequences
    )

    # Make prediction using context
    prediction = model(target=target, context=context_set)

    loss = criterion(prediction, true_value)
    optimizer.step()
```

**Key Principles:**
- Each episode = one prediction task
- Context set must be causal (occurred before target)
- Model learns to transfer patterns from context to target
- Trains on k-shot tasks to perform well on k-shot evaluation

### 3. Context Set Construction

Context set `C` contains sequences from other assets/regimes:

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

### 4. Meta-Learning Architecture

**Input Structure:**
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

**Model Flow:**
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

### 5. Transfer Learning Strategies

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

### 6. Training Objectives

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

### 7. Evaluation Protocols

**Expanding Window Backtest:**
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

**Zero-Shot Evaluation:**
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

### Pattern 2: Change-Point Segmented Context

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

### Pattern 3: Time-Equivalent Context

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

## Performance Insights from X-Trend Paper

**Few-Shot Results (2018-2023):**
- Baseline (no context): Sharpe = 2.27
- X-Trend (with context): Sharpe = 2.70 (+18.9%)
- X-Trend (CPD context): Sharpe = 2.70 (+18.9%)
- vs TSMOM: Sharpe = 0.23 (10× improvement)

**Zero-Shot Results (2018-2023):**
- Baseline: Sharpe = -0.11 (loss-making!)
- X-Trend-G (Gaussian): Sharpe = 0.47
- TSMOM: Sharpe = -0.26
- 5× Sharpe improvement vs baseline

**COVID-19 Recovery:**
- Baseline: 254 days to recover from drawdown
- X-Trend: 162 days (2× faster recovery)

## Best Practices

### DO:

✅ **Use episodic training** - train how you test
✅ **Ensure causality** - context must precede target
✅ **Sample diverse contexts** - different assets, regimes, conditions
✅ **Use change-point detection** - improves Sharpe by 11%+
✅ **Test zero-shot performance** - validates true transfer learning
✅ **Joint optimization** - balance forecasting and trading objectives

### DON'T:

❌ **Don't leak future information** into context set
❌ **Don't use same (asset, time) in context and target**
❌ **Don't assume transferability** without testing
❌ **Don't skip few-shot evaluation** even for zero-shot models
❌ **Don't ignore context set size** - typically 10-30 is optimal

## Common Pitfalls

**Pitfall 1: Data Leakage**
```python
# WRONG - context from future!
context = sample_sequences(all_time_periods)

# CORRECT - context only from past
context = sample_sequences(before=target_time)
```

**Pitfall 2: Overfitting to Context Construction**
```python
# WRONG - optimization on test set
best_cpd_threshold = optimize_on_test_set()

# CORRECT - validate on held-out data
best_cpd_threshold = cross_validate_on_train_set()
```

**Pitfall 3: Ignoring Asset Heterogeneity**
```python
# WRONG - assume all assets behave identically
encoding = lstm(features)  # Same encoding for all

# CORRECT - use entity embeddings
encoding = lstm(features) + asset_embedding[asset_id]
```

## Key Takeaways

1. **Few-shot ≠ Small Model** - Models can be large, but they adapt with minimal examples
2. **Context Quality Matters** - CPD segmentation beats random sampling
3. **Zero-shot Tests Transfer** - If it works on unseen assets, transfer is real
4. **Episodic Training Required** - Don't mix all data; train in episodes
5. **Joint Objectives Help** - Forecasting + trading better than either alone

## Related Files

- `financial-time-series` - Momentum factors, returns, portfolio construction
- `change-point-detection` - GP-CPD for regime segmentation
- `attention-trading-models` - Cross-attention mechanisms

## References

- Matching Networks for One Shot Learning (Vinyals et al. 2016)
- Model-Agnostic Meta-Learning (Finn et al. 2017)
- Neural Processes (Garnelo et al. 2018)
- X-Trend: Few-Shot Learning Patterns (Wood et al. 2024)

---

**Last Updated**: Based on X-Trend paper (March 2024)
**Skill Type**: Domain Knowledge
**Line Count**: ~490 (under 500-line rule ✅)
