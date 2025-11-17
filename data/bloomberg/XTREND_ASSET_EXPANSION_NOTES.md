# X-Trend Asset Expansion - Research Notes

## Overview

Expanding from **50 → 72+ assets** to enhance X-Trend's few-shot and zero-shot learning capabilities through increased context set diversity.

> **Environment note:** Prepare the repo with `uv sync` and execute helper utilities via `uv run python …` (e.g., `uv run python scripts/convert_bloomberg_to_parquet.py`) whenever these notes reference scripts.
**Key Insight from Paper:**
> X-Trend's zero-shot design (30 train assets → 20 test assets) explicitly benefited from context diversity; expanding across regions, contract specifications (cash-settled vs deliverable), and quote currencies adds distinct trend regimes the cross-attention can learn from, improving generalization to new targets.

## Strategic Rationale

### Why Diversity Matters for X-Trend

The cross-attention mechanism in X-Trend learns to identify **similar patterns** across different assets and regimes. Adding heterogeneous instruments improves:

1. **Few-shot performance** - Richer historical context for regime matching
2. **Zero-shot transfer** - Better generalization to unseen assets
3. **Regime detection** - More examples of structural breaks, crashes, reversals
4. **Robustness** - Less overfitting to US-centric market dynamics

**Evidence from Paper:**
- Change-point detection (CPD) segmented contexts: **+11.3% Sharpe** vs random sampling
- Zero-shot X-Trend-G: **Sharpe 0.47** vs baseline **-0.11** (loss-making)
- COVID-19 recovery: **162 days** (X-Trend) vs **254 days** (baseline) - 2× faster

---

## Asset Class Additions (22 new series)

### 1. LME Base Metals (6 contracts)

**Bloomberg Tickers:**
```
LMAHDS03 Comdty  # Aluminum 3M
LMZSDS03 Comdty  # Zinc 3M
LMCADS03 Comdty  # Copper 3M
LMNIDS03 Comdty  # Nickel 3M
LMPBDS03 Comdty  # Lead 3M
LMSNDS03 Comdty  # Tin 3M
```

**What They Add:**

| Dimension | LME vs Existing (COMEX/NYMEX) |
|-----------|-------------------------------|
| **Venue** | London Metal Exchange vs US exchanges |
| **Contract Structure** | 3-month forward vs spot-month futures |
| **Settlement** | Physical delivery (warehouse system) vs cash-settled |
| **Quote Currency** | USD but London-traded (different time zone) |
| **Microstructure** | Backwardation/contango via warehouse dynamics |

**Research Value:**
- **Industrial demand shocks** - Manufacturing cycles (China, Europe)
- **Different term structure** - 3M forward curve vs nearest futures
- **Warehouse squeeze dynamics** - LME-specific regime changes
- **Cross-venue arbitrage** - E.g., LME Copper (LMCADS03) vs COMEX Copper (HG1/ZK)

**References:**
- LME Trading Symbols Guide: https://www.lme.com/en/market-data/accessing-market-data/trading-symbols-guide
- Bloomberg LME data documentation

---

### 2. Global Equity Indices (4 new + 3 duplicates)

**New Additions:**
```
GX1 Index   # DAX (Germany)
TP1 Index   # TOPIX (Japan - broader than Nikkei)
HI1 Index   # Hang Seng (Hong Kong/China)
ZTSA Index  # FTSE/JSE Top 40 (South Africa - EM)
```

**Already in Dataset (skip or keep for completeness):**
```
VG1 Index   # Euro STOXX 50 (already as XX/XU)
Z 1 Index   # FTSE 100 (already as LX)
NK1 Index   # Nikkei 225 (already as NK)
```

**What They Add:**

| Index | Currency | Region | Trading Hours (UTC) | What It Captures |
|-------|----------|--------|---------------------|------------------|
| **GX (DAX)** | EUR | Germany | 07:00-21:00 | European industrial/export economy |
| **TP (TOPIX)** | JPY | Japan | 00:00-06:00 | Broader Japan equity vs NK (Nikkei 225) |
| **HI (Hang Seng)** | HKD | Hong Kong | 01:15-08:00 | China/HK exposure, capital flow regimes |
| **TS (JSE Top 40)** | ZAR | South Africa | 07:00-15:00 | Emerging market equity, commodity-linked |

**Research Value:**
- **Time zone diversification** - Asian/European market opens provide leading indicators
- **Currency exposure** - EUR, JPY, HKD, ZAR vs USD-denominated instruments
- **Regional shocks** - ECB policy (DAX), BoJ policy (TOPIX), China regulations (HSI)
- **Emerging market** - JSE Top 40 adds EM equity regime dynamics

**References:**
- Eurex product listings: https://www.eurex.com/ex-en/markets/idx
- Bloomberg index futures documentation

---

### 3. Cryptocurrency Futures (4 confirmed + 2 pending)

**Confirmed Bloomberg Tickers:**
```
BTCA Curncy   # CME Bitcoin (front month aggregate)
DCRA Curncy   # CME Ether (front month aggregate)
BMR Curncy    # CME Micro Bitcoin
MERA Curncy   # CME Micro Ether
```

**Pending Confirmation (newly listed 2025):**
```
<TBD> Curncy  # CME Solana (SOL) - check CME vendor symbol file
<TBD> Curncy  # CME XRP - check CME vendor symbol file
```

**What They Add:**

| Dimension | Crypto vs Traditional Futures |
|-----------|-------------------------------|
| **Trading Hours** | 24/7 vs exchange hours |
| **Volatility Regime** | 50-100% annualized vs 15-25% for equities |
| **Weekend Gaps** | Continuous trading (regime info during weekends) |
| **Correlation** | Low/time-varying correlation with TradFi |
| **Basis Dynamics** | Persistent contango/backwardation vs spot |

**Research Value:**
- **Zero-shot testing on emerging assets** - X-Trend paper showed crypto is ideal for testing transfer learning
- **High-volatility regimes** - Stress-tests volatility targeting (σ_tgt = 15%)
- **Structural breaks** - Regulatory announcements, exchange hacks, macro shocks
- **Momentum crash patterns** - Crypto exhibits extreme reversals (similar to COVID-19 in TradFi)

**Special Considerations:**
- **Roll mechanics** - CME crypto futures roll monthly vs quarterly for some indices
- **Leverage limits** - Different margin requirements than traditional futures
- **Data availability** - BTC/ETH history back to ~2017, SOL/XRP only from 2025

**Next Steps:**
1. Confirm SOL/XRP Bloomberg codes from CME vendor symbol file
2. Check data availability dates (may not go back to 1990)
3. Consider separate backtests for crypto-only context sets

**References:**
- CME Bitcoin futures: https://www.cmegroup.com/trading/bitcoin-futures.html
- CME Ether futures: https://www.cmegroup.com/markets/cryptocurrencies/ether/ether.html
- CME Solana announcement: https://www.reuters.com/business/finance/cme-group-plans-launch-solana-futures-march-17-2025-02-28/
- CME XRP announcement: https://www.reuters.com/business/derivatives-exchange-cme-set-launch-xrp-futures-crypto-push-2025-04-24/
- Bloomberg crypto options cheat sheet: https://www.cmegroup.com/articles/2022/cryptocurrency-options-on-futures-bloomberg-cheat-sheet.html

---

### 4. Spot Precious Metals (4 contracts)

**Bloomberg Tickers:**
```
XAUUSD Curncy  # Gold spot
XAGUSD Curncy  # Silver spot
XPTUSD Curncy  # Platinum spot
XPDUSD Curncy  # Palladium spot
```

**What They Add:**

| Spot vs Futures | Key Differences |
|-----------------|-----------------|
| **Pricing** | 24-hour London/OTC market vs exchange futures |
| **Settlement** | Immediate (T+2) vs futures expiry |
| **Basis** | Spot-futures basis = storage cost + interest |
| **Microstructure** | Dealer quotes vs central limit order book |

**Pairing with Existing Futures:**
- **XAU (spot)** ↔ **ZG (GC1 futures)** - Gold basis dynamics
- **XAG (spot)** ↔ **ZI (SI1 futures)** - Silver basis
- **XPT (spot)** ↔ **ZP (PL1 futures)** - Platinum basis
- **XPD (spot)** ↔ **ZA (PA1 futures)** - Palladium basis

**Research Value:**
- **Basis regime detection** - Contango/backwardation shifts indicate storage/demand changes
- **Liquidity stress** - Spot-futures dislocation during market stress (e.g., March 2020)
- **24-hour price discovery** - Spot continues trading when futures are closed
- **Forward curve modeling** - Explicit modeling of term structure

**Feature Engineering Opportunity:**
```python
# Add basis as feature to X-Trend input vector
basis_feature = (futures_price - spot_price) / spot_price
# Normalized by volatility
basis_normalized = basis_feature / realized_volatility
```

**References:**
- Bloomberg spot metals documentation
- London Bullion Market Association (LBMA) pricing methodology

---

### 5. Refined Energy Products (1 contract)

**Bloomberg Ticker:**
```
XB1 Comdty  # RBOB Gasoline (NYMEX)
```

**What It Adds:**

| Metric | Value |
|--------|-------|
| **Crack Spread** | Refining margin vs crude oil (ZU/CL1) |
| **Seasonal Regime** | Summer (driving) vs winter demand |
| **Regional Dynamics** | US-specific refining capacity constraints |

**Pairing with Existing:**
- **XB (RBOB)** vs **ZU (WTI Crude)** - Gasoline crack spread
- **XB (RBOB)** vs **ZH (Heating Oil)** - Product spreads

**Research Value:**
- **Crack spread regimes** - Refinery utilization, maintenance cycles
- **Seasonal patterns** - Different from crude oil seasonality
- **Supply shocks** - Hurricane disruptions to Gulf Coast refineries
- **Policy regime** - RFS (Renewable Fuel Standard) compliance

**Feature Engineering:**
```python
# Crack spread as context feature
crack_spread = rbob_price - (crude_price / 42)  # 42 gallons per barrel
crack_spread_normalized = crack_spread / crude_volatility
```

**References:**
- NYMEX RBOB contract specifications
- EIA refinery data

---

## Implementation Notes

### Bloomberg Terminal Export Process

**Updated BDH Formula (for 72 assets):**
```excel
=BDH($B$2:$B$73,"PX_LAST","19900101","20231231","Dir=V")
```

**Note:** Crypto and some international futures may not have data back to 1990:
- **Crypto**: Start ~2017 (BTC), ~2018 (ETH), ~2025 (SOL/XRP)
- **Some LME metals**: Confirm start dates (likely pre-1990)
- **EM indices (ZTSA)**: Confirm start date

**Alternative Approach for Missing History:**
1. Export all available data (start dates vary by asset)
2. In Python, handle missing data with `ffill()` or drop early periods
3. For zero-shot tests, use crypto as test set (post-2017 only)

### Data Preprocessing Considerations

**1. Crypto-Specific Handling:**
```python
# Crypto trades 24/7 - different return calculation
crypto_assets = ['BC', 'DC', 'XM', 'BR', 'SO', 'XR']

for asset in crypto_assets:
    # Use 7-day week (not 5-day) for annualization
    annual_factor = 365  # not 252
    sharpe = np.sqrt(annual_factor) * mean / std
```

**2. Spot-Futures Basis Features:**
```python
# Calculate basis for metals
basis_gold = (futures['ZG'] - spot['XAU']) / spot['XAU']
basis_silver = (futures['ZI'] - spot['XAG']) / spot['XAG']
# Add to feature vector
features = np.concatenate([returns, macd, basis_gold, basis_silver])
```

**3. LME 3M Forward Curve:**
```python
# LME 3M reflects forward curve vs spot-month futures
# May need different volatility targeting
lme_assets = ['AL', 'ZN', 'CU', 'NI', 'PB', 'TN']
# Consider separate vol calculation for 3M forwards
```

---

## Expected Performance Improvements

Based on X-Trend paper results:

### Few-Shot Setting (all 72 assets in train/test)

**Expected Improvements:**
- **Baseline** (50 assets, 2018-2023): Sharpe = 2.27
- **With CPD contexts** (50 assets): Sharpe = 2.70 (+18.9%)
- **Expected with 72 assets**: Sharpe = **2.9 - 3.1** (+25-35% vs baseline)

**Why:**
- More diverse context regimes → better regime matching during cross-attention
- LME metals + international indices → broader set of structural breaks
- Spot-futures basis → explicit term structure information

### Zero-Shot Setting (train on traditional, test on crypto)

**Proposed Split:**
- **Train (I_tr)**: All 50 original + 18 new traditional assets = **68 assets**
- **Test (I_ts)**: 4-6 crypto assets (BC, DC, XM, BR, SO, XR)

**Expected Results (2018-2023):**
- **Baseline** (30 train, 20 test): Sharpe = -0.11 (loss-making)
- **X-Trend-G** (30 train, 20 test): Sharpe = 0.47
- **Expected with 68 train, 6 crypto test**: Sharpe = **0.6 - 0.8**

**Why:**
- Crypto exhibits extreme volatility regimes similar to momentum crashes
- More diverse training set → better transfer to high-vol, weekend-trading assets
- LME metals (industrial) + crypto (speculative) = complementary regime patterns

### COVID-19 Recovery (2020-2021)

**Historical:**
- **Baseline** (50 assets): 254 days to recover
- **X-Trend** (50 assets): 162 days (2× faster)

**Expected with 72 assets:**
- **X-Trend** (72 assets): **120-140 days** (1.8-2.1× faster than baseline)

**Why:**
- International indices (HI, GX, TP) experienced COVID differently → diverse recovery patterns
- Crypto crashed harder/faster in March 2020 → extreme regime examples for context
- Spot-futures dislocations during stress → early warning signals

---

## Alternative Data to Consider (Future Expansion)

### Options Data (Implied Volatility & Skew)

**Bloomberg Functions:**
- `OVME` - Options volatility matrix
- `OVML` - Options volatility surface (3D)
- `OMON` - Options monitor

**Recommended Underlyings for IV Surfaces:**

| Asset Class | Ticker | What IV Adds |
|-------------|--------|--------------|
| **Equities** | SPX Index, SX5E Index, NKY Index, HSI Index | Equity risk premium, crash risk (skew) |
| **FX** | EURUSD Curncy, USDJPY Curncy | Currency volatility regimes |
| **Commodities** | CL1 Comdty (crude), GC1 Comdty (gold) | Commodity vol, supply shock expectations |
| **Crypto** | BTCA Curncy, DCRA Curncy (via OMON) | Crypto vol regime, reflexivity |

**Usage in X-Trend:**
- **Implied vol vs realized vol** → regime detection (elevated IV = stress)
- **Vol skew** → tail risk pricing, momentum crash indicator
- **Vol surface term structure** → forward-looking regime shifts

**Implementation Note:**
- Requires separate OVME/OMON exports from Bloomberg (not BDH)
- Store as separate feature files (implied_vol.parquet, skew.parquet)
- Add to X-Trend feature vector (Eq. 6 in paper)

---

## Next Steps

### Immediate (at Bloomberg Terminal):

- [ ] Copy expanded ticker list to Excel (see updated BLOOMBERG_EXPORT_INSTRUCTIONS.md)
- [ ] Run BDH formula for all 72 assets (1990-2023 where available)
- [ ] Export to `bloomberg_export_72assets.xlsx`
- [ ] Check data completeness (identify assets with limited history)

### Post-Export:

- [ ] Run `uv run python scripts/convert_bloomberg_to_parquet.py` with the expanded symbol map
- [ ] Handle missing data (crypto starts ~2017, confirm LME/int'l index starts)
- [ ] Create separate datasets:
  - `data/processed/all_72_assets.parquet` (few-shot)
  - `data/processed/traditional_68_train.parquet` (zero-shot train)
  - `data/processed/crypto_6_test.parquet` (zero-shot test)

### Analysis & Testing:

- [ ] Backtest X-Trend with 72-asset context set (few-shot)
- [ ] Backtest zero-shot (68 traditional → 6 crypto)
- [ ] Compare Sharpe ratios vs 50-asset baseline
- [ ] Analyze attention weights (which context assets are most similar to crypto targets?)
- [ ] Document results in research notes

### Optional (Future Work):

- [ ] Add implied volatility surfaces via OVME/OMON
- [ ] Add intraday high/low (PX_HIGH, PX_LOW) for realized volatility
- [ ] Add volume/open interest (PX_VOLUME, OPEN_INT) for liquidity regimes
- [ ] Explore lead-lag between spot and futures for regime detection

---

## References

### Academic Papers
- Wood, Kessler, Roberts, Zohren (2024). "Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies." arXiv:2310.10500v2

### Exchange Documentation
- LME Trading Symbols Guide: https://www.lme.com/en/market-data/accessing-market-data/trading-symbols-guide
- Eurex Index Derivatives: https://www.eurex.com/ex-en/markets/idx
- CME Cryptocurrency Futures: https://www.cmegroup.com/markets/cryptocurrencies.html
- CME Bloomberg Codes Cheat Sheet: https://www.cmegroup.com/articles/2022/cryptocurrency-options-on-futures-bloomberg-cheat-sheet.html

### News & Announcements
- CME Solana Launch (Reuters): https://www.reuters.com/business/finance/cme-group-plans-launch-solana-futures-march-17-2025-02-28/
- CME XRP Launch (Reuters): https://www.reuters.com/business/derivatives-exchange-cme-set-launch-xrp-futures-crypto-push-2025-04-24/

### Bloomberg Resources
- Bloomberg Commodity Data
- Bloomberg Index Futures Documentation
- Bloomberg Spot FX Documentation

---

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Author:** Asset Expansion for X-Trend Context Diversity Enhancement
