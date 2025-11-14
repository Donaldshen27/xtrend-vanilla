# X-Trend Project Skills Summary

## Overview

This document summarizes the specialized skills created for the X-Trend (Cross Attentive Time-Series Trend Network) project. These skills provide comprehensive guidance for implementing few-shot learning based trend-following strategies in financial markets.

**Based on**: "Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies" (Wood, Kessler, Roberts, Zohren, 2024)

**Date Created**: November 14, 2025

---

## Skills Created

### 1. **financial-time-series**

**Location**: `.claude/skills/financial-time-series/SKILL.md`

**Purpose**: Core financial time-series processing and trend-following concepts

**Key Topics**:
- Returns calculation and normalization
- Volatility targeting for risk management
- Momentum factors (TSMOM, multi-scale)
- MACD indicators
- Deep learning momentum factors
- Portfolio construction with transaction costs
- Sharpe ratio optimization
- Futures contracts handling

**When to Use**:
- Processing price data or calculating returns
- Implementing momentum strategies
- Building trading portfolios
- Working with futures contracts
- Calculating risk-adjusted returns

**Key Formulas**:
```python
# Returns
r[t-t',t] = (p[t] - p[t-t']) / p[t-t']

# Normalized Returns
r_hat[t-t',t] = r[t-t',t] / (σ[t] * sqrt(t'))

# Volatility Targeting
leverage = σ_tgt / σ[t]

# Sharpe Ratio
Sharpe = sqrt(252) * mean(R) / std(R)
```

---

### 2. **few-shot-learning-finance**

**Location**: `.claude/skills/few-shot-learning-finance/SKILL.md`

**Purpose**: Few-shot and zero-shot learning for financial trading

**Key Topics**:
- Few-shot vs zero-shot learning
- Episodic training methodology
- Context set construction strategies
- Meta-learning architectures
- Transfer learning across assets/regimes
- Training objectives (joint loss)
- Evaluation protocols (expanding window)

**When to Use**:
- Implementing models that adapt to regime changes
- Trading new assets with limited history
- Building strategies that transfer knowledge
- Dealing with non-stationary markets

**Performance Highlights**:
```
Few-Shot (2018-2023):
- Baseline: Sharpe = 2.27
- X-Trend: Sharpe = 2.70 (+18.9%)

Zero-Shot (2018-2023):
- Baseline: Sharpe = -0.11 (loss-making)
- X-Trend: Sharpe = 0.47 (profitable)
```

---

### 3. **change-point-detection**

**Location**: `.claude/skills/change-point-detection/SKILL.md`

**Purpose**: Gaussian Process change-point detection for regime segmentation

**Key Topics**:
- Gaussian Process basics
- Matérn 3/2 kernel
- Change-point kernel formulation
- GP-CPD algorithm
- Time-series segmentation
- Using CPD for context sets
- Hyperparameter selection

**When to Use**:
- Segmenting time-series into regimes
- Detecting structural breaks or market transitions
- Constructing context sets (improves Sharpe by 11.3%)
- Identifying momentum crashes
- Analyzing volatility regime changes

**Algorithm**:
```python
1. Fit GP with Matérn kernel → L_M
2. Fit GP with Change-Point kernel → L_C (optimize t_cp)
3. Severity = L_C / (L_M + L_C)
4. If severity ≥ threshold (0.9-0.95), declare change-point
```

---

### 4. **x-trend-architecture**

**Location**: `.claude/skills/x-trend-architecture/SKILL.md`

**Purpose**: Complete X-Trend model architecture implementation

**Key Topics**:
- Input features and Variable Selection Network (VSN)
- Entity embeddings for assets
- LSTM-based sequence encoders
- Self-attention over context
- Cross-attention mechanism (target ← context)
- Encoder-decoder pattern
- Dual heads: Forecast + Position (PTP)
- Training objectives and loss functions
- Attention interpretation

**When to Use**:
- Implementing X-Trend or similar attention-based models
- Building encoder-decoder architectures
- Using cross-attention mechanisms
- Creating interpretable trading predictions

**Model Variants**:
- **X-Trend**: Direct Sharpe optimization (no forecast)
- **X-Trend-G**: Joint Gaussian MLE + Sharpe
- **X-Trend-Q**: Joint Quantile Regression + Sharpe

---

## How Skills Auto-Activate

All skills are configured in `.claude/skills/skill-rules.json` with intelligent triggers:

### Trigger Types:

1. **Keywords**: Explicit mentions trigger skills
   - "momentum", "TSMOM", "returns" → `financial-time-series`
   - "few-shot", "context set" → `few-shot-learning-finance`
   - "change-point", "regime" → `change-point-detection`
   - "cross-attention", "X-Trend" → `x-trend-architecture`

2. **Intent Patterns**: Regex patterns detect user intentions
   - "(calculate|compute).*?(returns|momentum)" → `financial-time-series`
   - "(implement|build).*?few.*shot" → `few-shot-learning-finance`
   - "(detect|identify).*?regime.*change" → `change-point-detection`

3. **File Triggers**: Editing certain files auto-triggers skills
   - `**/momentum/**/*.py` → `financial-time-series`
   - `**/few_shot/**/*.py` → `few-shot-learning-finance`
   - `**/xtrend/**/*.py` → `x-trend-architecture`

### Enforcement:

All X-Trend skills use **"suggest"** enforcement:
- Skills appear as recommendations
- Do not block execution
- Help guide implementation
- High priority for visibility

---

## Quick Start Guide

### Example 1: Implementing TSMOM Strategy

When you say: **"Implement a time-series momentum strategy"**

**Auto-activated**: `financial-time-series`

**You'll get guidance on**:
- Calculating 1-year returns
- Creating momentum signals
- Volatility targeting
- Portfolio construction
- Sharpe ratio calculation

### Example 2: Building Few-Shot Model

When you say: **"Create a few-shot learning model for trading"**

**Auto-activated**: `few-shot-learning-finance`

**You'll get guidance on**:
- Episodic training setup
- Context set sampling
- Train/test split for few-shot vs zero-shot
- Loss function design
- Expanding window evaluation

### Example 3: Detecting Regime Changes

When you say: **"Detect regime changes in price data"**

**Auto-activated**: `change-point-detection`

**You'll get guidance on**:
- GP-CPD implementation
- Matérn kernel selection
- Severity threshold tuning
- Segmentation algorithm
- Using CPD for context construction

### Example 4: Implementing X-Trend

When you say: **"Build the X-Trend architecture"**

**Auto-activated**: `x-trend-architecture`

**You'll get guidance on**:
- Encoder structure (LSTM + VSN)
- Cross-attention implementation
- Decoder with dual heads
- Joint loss training
- Attention interpretation

---

## Skill Relationships

```
┌─────────────────────────────────────┐
│    financial-time-series            │
│    (Returns, Momentum, Sharpe)      │
└──────────────┬──────────────────────┘
               │
               │ Provides features for
               ↓
┌─────────────────────────────────────┐
│    few-shot-learning-finance        │
│    (Episodic, Context Sets)         │
└──────────────┬──────────────────────┘
               │
               │ Uses context from
               ↓
┌─────────────────────────────────────┐
│    change-point-detection           │
│    (GP-CPD, Regime Segmentation)    │
└──────────────┬──────────────────────┘
               │
               │ Segments data for
               ↓
┌─────────────────────────────────────┐
│    x-trend-architecture             │
│    (Complete Model Implementation)  │
└─────────────────────────────────────┘
```

**Dependencies**:
1. **financial-time-series** provides input features
2. **few-shot-learning-finance** defines training methodology
3. **change-point-detection** improves context quality (+11.3% Sharpe)
4. **x-trend-architecture** ties everything together

---

## Performance Benchmarks

From the X-Trend paper (2018-2023, turbulent market period):

### Few-Shot Learning (50 futures contracts)

| Strategy | Sharpe Ratio | vs Baseline |
|----------|--------------|-------------|
| TSMOM (baseline) | 0.23 | - |
| MACD | 0.27 | - |
| Baseline Neural | 2.27 | - |
| **X-Trend** | **2.65** | **+16.9%** |
| **X-Trend-Q (CPD)** | **2.70** | **+18.9%** |

### Zero-Shot Learning (20 unseen contracts)

| Strategy | Sharpe Ratio | Status |
|----------|--------------|---------|
| TSMOM | -0.26 | Loss-making |
| Baseline Neural | -0.11 | Loss-making |
| **X-Trend-G** | **0.47** | **Profitable** |

### COVID-19 Recovery

- **Baseline**: 254 days to recover from drawdown
- **X-Trend**: 162 days (2× faster)

---

## Development Workflow

### Recommended Order for Implementation:

1. **Start with financial-time-series**
   - Get data loading working
   - Implement returns calculation
   - Test volatility targeting
   - Build basic TSMOM strategy

2. **Add change-point-detection**
   - Implement GP-CPD algorithm
   - Segment historical data
   - Visualize regimes

3. **Implement few-shot-learning-finance**
   - Set up episodic training
   - Create context sampling
   - Test expanding window evaluation

4. **Build x-trend-architecture**
   - Start with encoder (LSTM + VSN)
   - Add cross-attention
   - Implement decoder
   - Train with joint loss

---

## Common Patterns

### Pattern: End-to-End Training Pipeline

```python
# 1. Load and process data (financial-time-series)
prices = load_futures_contracts()
features = create_feature_vector(prices)  # Returns + MACD
volatility = calculate_volatility(prices)

# 2. Segment into regimes (change-point-detection)
cpd = FinancialCPD(lookback=21, threshold=0.9)
regimes = cpd.segment(prices)

# 3. Episodic training (few-shot-learning-finance)
for episode in episodes:
    # Sample target
    target = sample_target(assets, time)

    # Sample context using CPD
    context = sample_cpd_context(regimes, target_time, size=20)

    # Forward pass (x-trend-architecture)
    position, forecast, attention = model(target, context)

    # Joint loss
    loss = alpha * mle_loss(forecast, returns) + sharpe_loss(position, returns)

    # Backward pass
    loss.backward()
    optimizer.step()
```

---

## Troubleshooting

### Issue: Skills not activating

**Solution**: Check that keywords are mentioned in your prompt
- Use explicit terms like "momentum", "few-shot", "change-point"
- Or ask: "Which skill should I use for X?"

### Issue: Need multiple skills

**Solution**: Skills work together
- Start with `financial-time-series` for basics
- Add `change-point-detection` for regime segmentation
- Use `few-shot-learning-finance` for training approach
- Implement with `x-trend-architecture`

### Issue: Want to modify skills

**Solution**: Edit skill files directly
- Skills are in `.claude/skills/*/SKILL.md`
- Follow 500-line rule for each skill
- Update `skill-rules.json` if changing triggers

---

## References

### Papers

1. **X-Trend (Primary Source)**
   - Wood, Kessler, Roberts, Zohren (2024)
   - "Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies"
   - arXiv:2310.10500v2

2. **Time Series Momentum**
   - Moskowitz, Ooi, Pedersen (2012)
   - "Time Series Momentum"
   - Journal of Financial Economics

3. **Deep Momentum Networks**
   - Lim, Zohren, Roberts (2019)
   - "Enhancing Time-Series Momentum Strategies Using Deep Neural Networks"
   - Journal of Financial Data Science

4. **GP Change-Point Detection**
   - Saatçi, Turner, Rasmussen (2010)
   - "Gaussian Process Change Point Models"
   - ICML

5. **Attention Mechanisms**
   - Vaswani et al. (2017)
   - "Attention Is All You Need"
   - NeurIPS

---

## Next Steps

1. **Explore the PDF**: Read the full X-Trend paper for detailed methodology

2. **Test Skills**: Try mentioning keywords to see auto-activation:
   - "How do I calculate momentum factors?"
   - "Implement few-shot learning for trading"
   - "Detect regime changes in my data"

3. **Start Coding**: Use skills as you implement:
   - Data preprocessing → `financial-time-series`
   - Regime detection → `change-point-detection`
   - Model building → `x-trend-architecture`

4. **Customize**: Modify skills as needed for your specific use case

---

**For Questions**: Ask Claude with keywords from any skill domain!

**Skill Status**: ✅ All 4 skills created and activated
**Adherence**: ✅ All skills under 500 lines (following Anthropic best practices)
**Integration**: ✅ Auto-activation configured in skill-rules.json

**Created by**: Claude Code (Skill Developer Agent)
**Date**: November 14, 2025
