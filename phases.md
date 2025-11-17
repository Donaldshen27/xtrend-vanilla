# X-Trend Implementation Phases

> **Purpose**: Step-by-step implementation plan for replicating the X-Trend paper with clear visual completion criteria for each phase.

> **Environment note:** All Python commands in this plan assume the repo has been prepared with `uv sync`. Prefix run commands with `uv run …` (e.g., `uv run python scripts/convert_bloomberg_to_parquet.py`) to keep execution inside the uv-managed virtual environment.

## Table of Contents
- [Phase 1: Data Pipeline & Feature Engineering](#phase-1-data-pipeline--feature-engineering)
- [Phase 2: Change-Point Detection](#phase-2-change-point-detection)
- [Phase 3: Base Neural Architecture](#phase-3-base-neural-architecture)
- [Phase 4: Context Set Construction](#phase-4-context-set-construction)
- [Phase 5: Cross-Attention Mechanism](#phase-5-cross-attention-mechanism)
- [Phase 6: Decoder & Loss Functions](#phase-6-decoder--loss-functions)
- [Phase 7: Episodic Learning Framework](#phase-7-episodic-learning-framework)
- [Phase 8: Training Infrastructure](#phase-8-training-infrastructure)
- [Phase 9: Backtesting & Evaluation](#phase-9-backtesting--evaluation)
- [Phase 10: Performance Optimization](#phase-10-performance-optimization)

---

## Phase 1: Data Pipeline & Feature Engineering

### Objectives
- Load and preprocess futures price data from Pinnacle Data Corp
- Calculate returns at multiple timescales
- Implement MACD indicators
- Apply volatility targeting/scaling

### Tasks
1. **Data Loading** ✅
   - [x] Load 72 continuous futures contracts (1990-2025) ✅
   - [x] Convert Bloomberg Terminal exports to Parquet format ✅
   - [x] Implement data quality checks and validation ✅
   - [x] Implement BloombergParquetSource for parquet loading ✅
   - [x] Handle missing data and date alignment (via pandas) ✅

2. **Returns Calculation** (Equation 1, 5) ✅
   - [x] Implement `r_t = (p_t - p_{t-1}) / p_{t-1}` ✅
   - [x] Calculate returns at timescales: 1, 21, 63, 126, 252 days ✅
   - [x] Normalize returns: `r̂_{t-t',t} = r_{t-t',t} / (σ_t * √t')` ✅

3. **MACD Features** (Equation 4) ✅
   - [x] Implement EWMA with configurable timescales (via 'ta' library) ✅
   - [x] Calculate MACD for (S,L) pairs: (8,24), (16,28), (32,96) ✅
   - [x] Normalize by 252-day rolling standard deviation ✅

4. **Volatility Targeting** (Equation 2) ✅
   - [x] Calculate 60-day exponentially weighted volatility σ_t ✅
   - [x] Implement leverage factor: σ_tgt / σ_t ✅
   - [x] Set target volatility σ_tgt (e.g., 15%) ✅

5. **Data Visualization** ✅
   - [x] Build Streamlit web app for interactive visualization
   - [x] Multi-symbol selection and comparison
   - [x] Date range filtering with slider controls
   - [x] Price normalization for relative performance analysis
   - [x] Interactive Plotly charts with hover tooltips
   - [x] Summary statistics table (mean, std, min, max, dates)
   - [x] Modular architecture for future analysis features

### Visual Completion Criteria

✅ **You should see:**
```python
# Test output example:
print(f"Features shape: {features.shape}")
# Output: Features shape: (T, 50, 8)  # T timesteps, 50 assets, 8 features

print(f"Features: {feature_names}")
# Output: ['r_1', 'r_21', 'r_63', 'r_126', 'r_252',
#          'MACD_8_24', 'MACD_16_28', 'MACD_32_96']

# Volatility stats
print(f"Mean ex-ante vol: {volatilities.mean():.4f}")
# Output: Mean ex-ante vol: 0.1500  # Should be close to target

# Sanity checks pass
assert not features.isna().any().any(), "No NaN values"
assert (volatilities > 0).all(), "All volatilities positive"
```

**Visualization Tools:**
1. ✅ **Bloomberg Streamlit Explorer** - Interactive web app
   - Launch: `uv run streamlit run scripts/bloomberg_explorer.py`
   - Features: Multi-symbol selection, date filtering, normalization, interactive charts
   - Location: `scripts/bloomberg_explorer.py` and `scripts/bloomberg_viz/`

**Plots to generate:**
1. ✅ Price series visualization (via Streamlit explorer)
2. [ ] Returns distribution histogram showing normalized returns
3. [ ] MACD indicators overlaid on price for sample period
4. [ ] Volatility time series showing targeting effect

### ✅ Phase 1 Complete (2025-11-17)

**Implementation:**
- All core functions implemented and tested (21/21 tests passing)
- BloombergParquetSource: Load price data from parquet files
- Returns: simple_returns(), multi_scale_returns(), normalized_returns()
- Volatility: ewm_volatility(), apply_vol_target()
- MACD: macd_multi_scale(), macd_normalized()
- 80% code coverage

**Code Quality:**
- Reviewed by Claude Code + OpenAI Codex
- Fixed all critical numerical stability issues:
  - Added epsilon clipping (1e-8) for division-by-zero
  - Added 10x leverage cap
  - Fixed min_periods logic (min→max)
  - Resolved pandas FutureWarning
- Comprehensive test suite with integration tests

**Files:**
- Implementation: `xtrend/data/{sources.py, returns_vol.py}`, `xtrend/features/indicators_backend.py`
- Tests: `tests/data/`, `tests/features/`, `tests/integration/`
- Documentation: `docs/plans/2025-11-17-phase1-data-pipeline-features.md`

**Known Limitations (for future phases):**
- NaN propagation in edge cases (documented)
- Raw returns division-by-zero for limit-down events (rare)
- Performance optimizations deferred to Phase 10

**Ready for Phase 2:** Change-Point Detection

---

## Phase 2: Change-Point Detection

### Objectives
- Implement Gaussian Process change-point detection (Algorithm 1)
- Segment time series into regimes
- Validate segmentation quality

### Tasks
1. **GP Implementation**
   - [ ] Implement Matérn 3/2 kernel for GP
   - [ ] Implement Change-point kernel (two GPs with soft transition)
   - [ ] Calculate log marginal likelihoods (L_M, L_C)

2. **CPD Algorithm** (Algorithm 1, Appendix A)
   - [ ] Implement lookback window (l_lbw = 21)
   - [ ] Calculate severity: ν = L_C / (L_M + L_C)
   - [ ] Set thresholds: ν = 0.9 (21-day), ν = 0.95 (63-day)
   - [ ] Respect min length (l_min = 5) and max length (l_max = 21 or 63)

3. **Regime Validation**
   - [ ] Verify causality (no future information leakage)
   - [ ] Check regime length distribution
   - [ ] Validate statistical properties of regimes

### Visual Completion Criteria

✅ **You should see:**
```python
# CPD segmentation results
print(f"Total regimes detected: {len(regimes)}")
# Output: Total regimes detected should be in the range [100-200]
#         (depends on asset universe, data quality, and CPD thresholds)

print(f"Average regime length: {np.mean([r[1]-r[0] for r in regimes]):.1f} days")
# Output: Average regime length should be 10-15 days
#         (bounded by min_len=5 and max_len=21 parameters)

print(f"Regime length range: [{min_len}, {max_len}]")
# Output: Regime length range should match your min/max parameters (typically [5, 21])
```

**Plots to generate (like Figure 3 in paper):**
1. Price series with color-coded regime segments
2. Change-point severity over time
3. Regime length histogram
4. Example showing clear regime detection (e.g., British Pound 2016-2022)

**Expected behavior:**
- Regime changes align with visible market shifts
- Uptrends, downtrends, and mean-reversion periods clearly separated
- COVID-19 period (Feb-Mar 2020) shows multiple regime changes

---

## Phase 3: Base Neural Architecture

### Objectives
- Implement Variable Selection Network (VSN)
- Build LSTM encoder with skip connections
- Create entity embeddings for futures contracts
- Implement layer normalization and FFN blocks

### Tasks
1. **Variable Selection Network** (Equation 13)
   - [ ] Implement feature-wise FFN: `FFN_j(x_{t,j})`
   - [ ] Implement softmax attention weights: `w_t = Softmax(FFN(x_t))`
   - [ ] Combine: `VSN(x_t) = Σ w_{t,j} * FFN_j(x_{t,j})`

2. **Entity Embeddings**
   - [ ] Create embedding layer for 50 contract types
   - [ ] Set embedding dimension d_h (e.g., 64 or 128)
   - [ ] Implement conditional FFN with embeddings (Equation 12)

3. **Encoder Architecture** (Equation 14)
   - [ ] LSTM cell implementation
   - [ ] Layer normalization after LSTM
   - [ ] Skip connections: `a_t = LayerNorm(h_t + x'_t)`
   - [ ] Final FFN with skip: `Ξ = LayerNorm(FFN(a_t) + a_t)`

4. **Baseline DMN Model** (Equation 7)
   - [ ] Implement position output: `z_t = tanh(Linear(g(x_t)))`
   - [ ] Verify output range: z_t ∈ (-1, 1)

### Visual Completion Criteria

✅ **You should see:**
```python
# Architecture summary
print(model)
# Output shows:
# - VSN with X feature-wise FFNs
# - Entity embeddings (50, d_h)
# - LSTM (input_size=d_h, hidden_size=d_h)
# - Skip connections and LayerNorm layers

# Forward pass test
batch_size, seq_len, n_features = 32, 126, 8
x = torch.randn(batch_size, seq_len, n_features)
entity_ids = torch.randint(0, 50, (batch_size,))

output = model(x, entity_ids)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([32, 126, 64])

print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
# Output: Output range: [-0.987, 0.991]  # Within tanh range
```

**Tests to pass:**
1. Gradient flow test (no vanishing/exploding gradients)
2. Entity embedding test (similar assets have similar embeddings)
3. VSN attention weights sum to 1
4. LSTM hidden state maintains reasonable magnitude

---

## Phase 4: Context Set Construction

### Objectives
- Implement three context construction methods (Figure 2)
- Ensure causality (context always before target)
- Create random sampling procedure

### Tasks
1. **Final Hidden State Method**
   - [ ] Sample C random sequences of fixed length l_c
   - [ ] Process through encoder to get final hidden states
   - [ ] Use as keys and values for cross-attention

2. **Time-Equivalent Method**
   - [ ] Sample sequences with same length as target: l_c = l_t
   - [ ] Align timesteps: k-th target attends to k-th context
   - [ ] Create time-aligned hidden state representations

3. **CPD Segmented Method** (Primary method)
   - [ ] Use CPD to segment assets into regimes
   - [ ] Randomly sample C regime sequences
   - [ ] Limit to max length (21 or 63 days)
   - [ ] Use final hidden state of each regime

4. **Causality Enforcement**
   - [ ] During training: any time allowed for context
   - [ ] During testing: enforce t_c < t for all context sequences
   - [ ] Implement causal masking in attention

### Visual Completion Criteria

✅ **You should see:**
```python
# Context set construction
target_date = "2020-03-15"  # COVID crash
context_set = build_context_set(
    target_asset="ES",  # S&P 500
    target_date=target_date,
    C=20,  # 20 context sequences
    method="cpd_segmented",
    max_length=21
)

print(f"Context set size: {len(context_set)}")
# Output: Context set size: 20

print(f"Context dates range: {context_set.dates.min()} to {context_set.dates.max()}")
# Output: Context dates range: 1995-01-03 to 2020-03-14

# Verify causality
assert all(d < target_date for d in context_set.dates), "Causality violated!"

# Context composition
print(context_set.asset_distribution())
# Output:
# Commodities: 8
# Equities: 7
# Fixed Income: 3
# FX: 2
```

**Visualization (like Figure 2):**
1. Timeline showing target sequence and context sequences
2. Heatmap of attention patterns for each method
3. Context set diversity metrics (asset classes, time periods)

---

## Phase 5: Cross-Attention Mechanism

### Objectives
- Implement self-attention over context set
- Implement cross-attention between target and context
- Use multi-head attention (4 heads)
- Make attention weights interpretable

### Tasks
1. **Self-Attention** (Equation 17)
   - [ ] Apply self-attention to context values
   - [ ] `V'_t = FFN ∘ Att(V_t, V_t, V_t)`
   - [ ] Use 4 parallel attention heads

2. **Cross-Attention** (Equations 15-18)
   - [ ] Create query from target: `q_t = Ξ_query(x_t, s)`
   - [ ] Create keys from context: `K_t = {Ξ_key(x^c, s^c)}_C`
   - [ ] Create values from context: `V_t = {Ξ_value(ξ^c, s^c)}_C`
   - [ ] Apply attention: `y_t = LayerNorm ∘ FFN ∘ Att(q_t, K_t, V'_t)`

3. **Attention Weights** (Equation 10)
   - [ ] Compute similarity: `α(q,k) = exp(1/√d_att * ⟨W_q q, W_k k⟩)`
   - [ ] Normalize: `p_q(k) = α(q,k) / Σ α(q,k')`
   - [ ] Store for interpretability

4. **Multi-Head Implementation**
   - [ ] 4 parallel attention heads
   - [ ] Concatenate head outputs
   - [ ] Final linear projection

### Visual Completion Criteria

✅ **You should see:**
```python
# Cross-attention test
target_seq = torch.randn(batch_size, l_t, d_h)
context_keys = torch.randn(batch_size, C, d_h)
context_values = torch.randn(batch_size, C, d_h)

output, attention_weights = cross_attention(
    query=target_seq,
    keys=context_keys,
    values=context_values,
    return_weights=True
)

print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([32, 126, 64])

print(f"Attention weights shape: {attention_weights.shape}")
# Output: Attention weights shape: torch.Size([32, 4, 126, 20])
#         [batch, heads, target_steps, context_sequences]

# Attention weights should sum to 1
assert torch.allclose(attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1)))

# Check attention pattern sparsity
top_3_attention = attention_weights.topk(3, dim=-1)[0].sum(dim=-1).mean()
print(f"Top-3 attention weight: {top_3_attention:.2%}")
# Output: Top-3 attention weight: 65%  # Model focuses on few contexts
```

**Visualization (like Figure 9):**
1. Attention weight heatmap (target steps × context sequences)
2. Top-3 attended contexts for specific target points
3. Attention pattern evolution over time
4. Cross-asset attention patterns (commodities vs equities, etc.)

---

## Phase 6: Decoder & Loss Functions

### Objectives
- Implement decoder that fuses encoder output with target
- Create Sharpe ratio loss function
- Implement Maximum Likelihood (Gaussian & QRE) losses
- Create PTP (Predictive To Position) module

### Tasks
1. **Decoder Architecture** (Equation 19)
   - [ ] Fuse encoder output with target: `Concat(VSN(x_t), y_t)`
   - [ ] LSTM decoder with skip connections
   - [ ] Output hidden states for prediction heads

2. **Sharpe Ratio Loss** (Equation 8)
   - [ ] Implement: `-√252 * mean(returns) / std(returns)`
   - [ ] Use warm-up period (l_s = 63) to avoid initial instability
   - [ ] Batch construction with multiple sequences

3. **Gaussian MLE Loss** (Equation 20)
   - [ ] Predict mean μ and volatility σ
   - [ ] Minimize negative log-likelihood
   - [ ] Output predictive distribution N(μ, σ)

4. **Quantile Regression Loss** (Equation 22)
   - [ ] Predict quantiles: [0.01, 0.05, 0.1, ..., 0.95, 0.99]
   - [ ] Implement pinball loss for each quantile
   - [ ] Focus on tail quantiles (important for Sharpe)

5. **Joint Loss Functions** (Equations 21, 23)
   - [ ] X-Trend-G: `α * L_MLE + L_Sharpe^PTP`
   - [ ] X-Trend-Q: `α * L_QRE + L_Sharpe^PTP`
   - [ ] Tune α hyperparameter (1 for Gaussian, 5 for QRE)

6. **PTP Module**
   - [ ] FFN: `PTP(μ, σ) = tanh(FFN([μ, σ]))`
   - [ ] Separate PTP for Gaussian and QRE versions
   - [ ] Output position z ∈ (-1, 1)

### Visual Completion Criteria

✅ **You should see:**
```python
# Loss function test
predictions_mean = model.predict_mean(x, context)
predictions_vol = model.predict_vol(x, context)
positions_ptp = model.ptp_head(predictions_mean, predictions_vol)

print(f"Predicted mean range: [{predictions_mean.min():.4f}, {predictions_mean.max():.4f}]")
# Output: Predicted mean range: [-0.0123, 0.0156]  # Small, realistic

print(f"Predicted vol range: [{predictions_vol.min():.4f}, {predictions_vol.max():.4f}]")
# Output: Predicted vol range: [0.7854, 1.2341]  # Around target vol of 1.0

print(f"Position range: [{positions_ptp.min():.4f}, {positions_ptp.max():.4f}]")
# Output: Position range: [-0.9987, 0.9991]  # Using full position range

# Loss values
sharpe_loss = compute_sharpe_loss(positions_ptp, returns)
mle_loss = compute_mle_loss(predictions_mean, predictions_vol, actual_returns)
joint_loss = 1.0 * mle_loss + sharpe_loss

print(f"Sharpe loss: {sharpe_loss:.4f}")
print(f"MLE loss: {mle_loss:.4f}")
print(f"Joint loss: {joint_loss:.4f}")
# Output:
# Sharpe loss: -2.1234  # Negative is good (Sharpe ~2.12)
# MLE loss: 0.9234
# Joint loss: -1.2000
```

**Plots (like Figures 7, 8):**
1. Predictive mean vs trading signal relationship
2. Predictive volatility vs trading signal
3. 95% confidence interval visualization
4. For QRE: Full distribution with median, IQR, and 95% CI
5. PTP function visualization: (μ, σ) → position

---

## Phase 7: Episodic Learning Framework

### Objectives
- Implement few-shot learning protocol
- Implement zero-shot learning protocol
- Create proper train/val/test splits with expanding window

### Tasks
1. **Episode Construction**
   - [ ] Sample target sequence from test set
   - [ ] Sample context set from training set
   - [ ] Ensure causality: all context before target

2. **Few-Shot Setup**
   - [ ] Training set I_tr and test set I_ts can overlap: I_tr ∩ I_ts = I
   - [ ] Context can include same asset as target (from past)
   - [ ] Use entity embeddings for target asset

3. **Zero-Shot Setup**
   - [ ] Separate assets: I_tr ∩ I_ts = ∅
   - [ ] 30 assets for training, 20 for testing (Tables 5, 6)
   - [ ] Exclude entity embeddings for target (unseen asset)
   - [ ] Keep embeddings in context set

4. **Expanding Window Backtest**
   - [ ] Train on 1990-1995, test on 1996-2000
   - [ ] Expand to 1990-2000, test on 2001-2005
   - [ ] Continue in 5-year increments to 2023
   - [ ] Retrain model for each window

### Visual Completion Criteria

✅ **You should see:**
```python
# Few-shot episode
episode = create_episode(
    target_asset="ES",
    target_date="2020-03-15",
    mode="few-shot"
)
print(f"Target in training set: {episode.target_asset in train_assets}")
# Output: Target in training set: True

# Zero-shot episode
episode = create_episode(
    target_asset="CA",  # CAC40, in zero-shot test set
    target_date="2020-03-15",
    mode="zero-shot"
)
print(f"Target in training set: {episode.target_asset in train_assets}")
# Output: Target in training set: False

print(f"Context assets: {episode.context_assets}")
# Output: Context assets: ['ZC', 'ES', 'BN', ...]  # Only training assets

# Expanding window schedule
for window in backtest_windows:
    print(f"Train: {window.train_start} to {window.train_end}")
    print(f"Test: {window.test_start} to {window.test_end}")
# Output:
# Train: 1990-01-01 to 1995-12-31
# Test: 1996-01-01 to 2000-12-31
# Train: 1990-01-01 to 2000-12-31
# Test: 2001-01-01 to 2005-12-31
# ...
```

**Validation:**
1. No data leakage: context always before or from different assets than target
2. Zero-shot never sees target asset during training
3. Test periods never overlap with training

---

## Phase 8: Training Infrastructure

### Objectives
- Implement Adam optimizer with hyperparameter search
- Add dropout and gradient clipping
- Implement early stopping
- Create 10-seed ensemble

### Tasks
1. **Hyperparameter Search** (Table 3)
   - [ ] Random grid search (10 iterations)
   - [ ] Dropout: {0.3, 0.4, 0.5}
   - [ ] Hidden size d_h: {64, 128}
   - [ ] Batch size: {64, 128}
   - [ ] Max gradient norm: {0.01, 1.0, 100.0}

2. **Fixed Parameters** (Table 4)
   - [ ] Learning rate: 1e-3
   - [ ] Warm-up steps l_s: 63
   - [ ] Total steps l_t: 126
   - [ ] Early stopping patience: 10 epochs
   - [ ] Max iterations: 100 epochs

3. **Training Loop**
   - [ ] Adam optimizer
   - [ ] Gradient clipping
   - [ ] Validation on last 10% of training data
   - [ ] Save best model based on validation Sharpe
   - [ ] Log training metrics

4. **Ensemble**
   - [ ] Train 10 models with different seeds
   - [ ] Average predictions across ensemble
   - [ ] Use ensemble for final backtesting

### Visual Completion Criteria

✅ **You should see:**
```python
# Training progress (example format - actual values vary with seeds/data/hyperparameters)
Epoch 1/100: train_loss=-X.XXX, val_sharpe=Y.YY, lr=0.001000
Epoch 2/100: train_loss=-X.XXX, val_sharpe=Y.YY, lr=0.001000
...
Early stopping triggered: no improvement for 10 epochs
Best val_sharpe: [saved at some epoch before patience limit]

# Hyperparameter search results
Best hyperparameters found (example):
- dropout: typically 0.3-0.5
- hidden_size: typically 64 or 128
- batch_size: typically 64 or 128
- max_grad_norm: typically 0.01-1.0
- val_sharpe: varies significantly with data and seed

# Ensemble performance (qualitative expectations)
Ensemble of 10 models:
- Should show lower variance than individual models
- Mean performance typically better than median individual
- Standard deviation indicates model stability
- Individual models may vary by 10-20% due to random initialization
```

**Note:** Exact loss and Sharpe values are highly sensitive to:
- Random seed initialization
- Data preprocessing choices (e.g., how missing data is handled)
- Hyperparameter configuration
- Training/validation split boundaries

**Qualitative success criteria:**
- Training loss should decrease over epochs (negative values getting more negative is improvement for Sharpe loss)
- Validation Sharpe should generally increase then plateau (early stopping prevents overfitting)
- Ensemble should reduce variance compared to individual runs
- No gradient explosions or NaN values during training

**Plots:**
1. Training and validation loss curves
2. Validation Sharpe ratio over epochs
3. Gradient norm over training
4. Learning rate schedule (if using)
5. Hyperparameter importance (which matters most)

---

## Phase 9: Backtesting & Evaluation

### Objectives
- Implement portfolio construction with volatility targeting
- Calculate performance metrics (Sharpe, drawdown, returns)
- Compare against baselines (TSMOM, MACD, Long, DMN)
- Generate paper-quality plots

### Tasks
1. **Portfolio Construction** (Equation 2)
   - [ ] Calculate portfolio returns: `R_port = (1/N) Σ R_i`
   - [ ] Apply volatility targeting per asset
   - [ ] Assume zero transaction costs (C=0) like paper
   - [ ] Equal risk contribution across assets

2. **Performance Metrics**
   - [ ] Sharpe ratio: `√252 * mean(returns) / std(returns)`
   - [ ] Maximum drawdown
   - [ ] Cumulative returns
   - [ ] Recovery time from COVID-19 drawdown

3. **Baseline Comparisons**
   - [ ] Long: Always z_t = +1
   - [ ] TSMOM: 252-day momentum (Equation 3)
   - [ ] MACD: Multi-timescale MACD signals (Equation 4)
   - [ ] Baseline DMN: No context, just LSTM

4. **Ablation Studies** (Tables 1, 2)
   - [ ] X-Trend variants: Sharpe, J-Gauss, J-QRE
   - [ ] X-Trend-G variants with different contexts
   - [ ] X-Trend-Q variants with different contexts
   - [ ] Context methods: Final/Time/CPD
   - [ ] Context sizes: |C| = 10, 20, 30
   - [ ] Context lengths: l_c = 21, 63, 126

### Visual Completion Criteria

✅ **You should see (Few-Shot, 2018-2023):**
```python
# Performance expectations (actual values depend on data, seeds, and hyperparameters)
# Paper reference ranges from Table 1:

Strategy Performance Tiers:
- Long/Simple Baselines: Sharpe ~0.2-0.7 (basic trend following)
- TSMOM/MACD: Sharpe ~0.2-1.0 (momentum strategies)
- Baseline DMN: Sharpe ~1.9-2.9 (neural network without context)
- X-Trend-Q (best): Sharpe ~2.1-3.1 (with cross-attention and context)

Expected Improvements:
- X-Trend should outperform Baseline DMN by 10-20%
- X-Trend should show 2-3x better performance than traditional momentum (TSMOM/MACD)

# COVID-19 recovery (qualitative expectation)
# X-Trend should recover from COVID drawdown significantly faster than baseline
# Paper shows ~1.5-2x faster recovery (162 vs 254 days in one run)
```

✅ **You should see (Zero-Shot, 2018-2023):**
```python
# Performance expectations for zero-shot (unseen assets)
# Paper reference ranges from Table 2:

Zero-Shot Performance:
- Baseline DMN: Often negative or near-zero Sharpe (struggles with unseen assets)
- X-Trend-G: Positive Sharpe ~0.4-0.5 (successful generalization)

Key Success Metric:
- X-Trend should turn loss-making baseline into profitable strategy
- Demonstrates genuine few-shot learning capability across assets
```

**Plots (replicate Figures 5, 6):**
1. Cumulative returns plot (log scale)
2. Drawdown plot highlighting COVID-19 period
3. Rolling Sharpe ratio (12-month window)
4. Strategy performance by asset class
5. Regime-specific performance

---

## Phase 10: Performance Optimization

### Objectives
- Optimize inference speed
- Reduce memory usage
- Parallelize backtesting
- Profile and optimize bottlenecks

### Tasks
1. **Code Optimization**
   - [ ] Profile training and inference
   - [ ] Optimize attention computation (use PyTorch efficient attention)
   - [ ] Batch processing for backtesting
   - [ ] Cache context set encodings

2. **Memory Optimization**
   - [ ] Use mixed precision training (FP16)
   - [ ] Gradient checkpointing for long sequences
   - [ ] Efficient data loading (don't load all data at once)

3. **Parallelization**
   - [ ] Parallelize hyperparameter search
   - [ ] Parallelize 10-seed ensemble training
   - [ ] Parallelize expanding window backtests

4. **GPU Utilization**
   - [ ] Maximize batch size for GPU
   - [ ] Pin memory for data loading
   - [ ] Use DataParallel for multi-GPU if available

### Visual Completion Criteria

✅ **You should see:**
```python
# Profiling results
Training time per epoch: 45.2s → 12.3s  (3.7x speedup)
Inference time (1 day): 0.23s → 0.05s  (4.6x speedup)
Memory usage: 8.2GB → 3.1GB  (2.6x reduction)

# Bottleneck analysis
Top 3 bottlenecks:
1. Cross-attention: 45% of time → optimized with torch.nn.MultiheadAttention
2. LSTM forward pass: 30% of time → batched processing
3. CPD segmentation: 15% of time → cached regime boundaries

# Parallel training
10-seed ensemble training time:
- Sequential: 8.5 hours
- Parallel (8 cores): 1.3 hours  (6.5x speedup)
```

**Final benchmarks:**
- Full backtest (1990-2023) completes in < 2 hours
- Training single model completes in < 30 minutes
- Inference for all assets for 1 day < 100ms

---

## Summary Checklist

### Core Functionality
- [ ] All 50 assets load correctly
- [ ] Features calculated correctly (returns, MACD, volatility)
- [ ] CPD segments time series into regimes
- [ ] Context sets constructed with proper causality
- [ ] Cross-attention mechanism works
- [ ] Model outputs positions in (-1, 1) range
- [ ] Sharpe loss optimizes properly
- [ ] Episodic learning (few-shot & zero-shot) implemented

### Performance Targets (Few-Shot, 2018-2023)
- [ ] X-Trend Sharpe in range 2.1-3.1 (paper reference: ~2.65-2.70)
- [ ] Baseline DMN Sharpe in range 1.9-2.9 (paper reference: ~2.27)
- [ ] TSMOM Sharpe in range 0.2-1.0 (paper reference: ~0.23-1.01 depending on period)
- [ ] COVID-19 recovery faster than baseline (paper shows ~1.5-2x improvement)
- [ ] Maximum drawdown during COVID in reasonable range (paper: ~26%, expect 20-35%)

### Performance Targets (Zero-Shot, 2018-2023)
- [ ] X-Trend-G Sharpe positive and >0.3 (paper reference: ~0.47)
- [ ] Baseline DMN struggles with unseen assets (paper reference: -0.11 to 0.02)
- [ ] X-Trend demonstrates generalization (loss-making baseline → profitable)

### Interpretability
- [ ] Attention weights interpretable (like Figure 9)
- [ ] Top-3 attended contexts make intuitive sense
- [ ] Predictive distribution vs position relationship clear
- [ ] Regime segmentation aligns with market events

### Code Quality
- [ ] All tests passing
- [ ] No data leakage (causality enforced)
- [ ] Reproducible results (seed control)
- [ ] Well-documented code
- [ ] Configuration-driven (not hardcoded)

---

## Dependencies Between Phases

```
Phase 1 (Data)
    ↓
Phase 2 (CPD) ← Phase 3 (Architecture)
    ↓              ↓
Phase 4 (Context Set)
    ↓
Phase 5 (Cross-Attention)
    ↓
Phase 6 (Decoder & Loss)
    ↓
Phase 7 (Episodic Learning) ← Phase 8 (Training)
    ↓
Phase 9 (Backtesting)
    ↓
Phase 10 (Optimization)
```

**Recommended order:**
1. Start with Phase 1 (data foundation)
2. Implement Phase 3 (base architecture) in parallel with Phase 2 (CPD)
3. Complete Phases 4-6 sequentially (build up model)
4. Implement Phases 7-8 together (training framework)
5. Run Phase 9 (evaluation)
6. Optimize with Phase 10 as needed

---

## Quick Reference: Key Equations

| Component | Equation | Description |
|-----------|----------|-------------|
| Returns | `r_t = (p_t - p_{t-1}) / p_{t-1}` | Daily returns |
| Normalized returns | `r̂ = r / (σ_t √t')` | Multi-scale normalized returns |
| MACD | `(EWMA_S - EWMA_L) / std` | Moving average convergence divergence |
| Volatility targeting | `σ_tgt / σ_t` | Leverage factor |
| Portfolio returns | `R_port = (1/N) Σ z_i * σ_tgt/σ_i * r_i` | With volatility targeting |
| CPD severity | `ν = L_C / (L_M + L_C)` | Change-point detection threshold |
| VSN | `Σ w_j * FFN_j(x_j)` | Variable selection network |
| Attention | `softmax(Q K^T / √d) V` | Scaled dot-product attention |
| Sharpe loss | `-√252 * mean(R) / std(R)` | Differentiable Sharpe ratio |
| MLE loss | `-log p(r_{t+1} | x, C)` | Maximum likelihood |

---

## Notes

- **Transaction costs**: Paper uses C=0 to focus on predictive power. Add realistic costs later.
- **COVID-19**: Critical test period. Model should recover 2x faster than baseline.
- **Ensemble**: Always use 10-seed ensemble for final results (matches paper).
- **Hyperparameters**: Use paper's values (Tables 3, 4) as starting point, then search.
- **Attention interpretability**: Figure 9 quality - should match intuition about similar market regimes.

---

## Recent Progress

### Completed (2025-11-16)
- ✅ **Bloomberg Data Visualization** - Built interactive Streamlit web app
  - Multi-symbol price visualization with normalization
  - Date range filtering and interactive charts
  - Summary statistics and data quality checks
  - Modular architecture ready for analysis features (returns, volatility, correlations)
  - Location: `scripts/bloomberg_explorer.py`, `scripts/bloomberg_viz/`
  - Documentation: `scripts/README.md`, `docs/plans/2025-11-16-bloomberg-streamlit-explorer-design.md`

### In Progress
- [ ] Returns calculation at multiple timescales
- [ ] MACD feature engineering
- [ ] Volatility targeting implementation

---

**Last Updated:** 2025-11-16
**Paper:** [Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies](https://arxiv.org/abs/2310.10500)
**Code:** https://github.com/kieranjwood/x-trend
