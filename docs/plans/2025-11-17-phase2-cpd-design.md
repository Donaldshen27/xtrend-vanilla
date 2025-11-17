# Phase 2: GP Change-Point Detection - Design Document

**Date:** 2025-11-17
**Author:** Claude Code + Codex Review
**Status:** Design Complete - Ready for Implementation

---

## Executive Summary

This document describes the complete design for Phase 2 of the X-Trend implementation: Gaussian Process-based Change-Point Detection (GP-CPD) for segmenting financial time-series into regimes.

**Approach:** Paper-faithful GP implementation using GPyTorch
**Key Innovation:** Corrected severity formula using log Bayes factor (not likelihood ratio)
**Integration:** Extends existing Bloomberg Streamlit explorer with interactive regime visualization
**Validation:** Three-pronged (statistical tests + known events + visual inspection)

---

## Table of Contents

1. [Overall Architecture](#1-overall-architecture)
2. [GPyTorch Kernels](#2-gpytorch-kernels)
3. [GP Fitting & Likelihood](#3-gp-fitting--likelihood)
4. [Segmentation Algorithm](#4-segmentation-algorithm)
5. [Validation Strategy](#5-validation-strategy)
6. [Streamlit Integration](#6-streamlit-integration)
7. [Testing Strategy](#7-testing-strategy)
8. [Implementation Checklist](#8-implementation-checklist)

---

## 1. Overall Architecture

### Component Structure

```
xtrend/cpd/
├── __init__.py          # Public API exports
├── kernels.py           # Custom GPyTorch kernels (if needed)
├── gp_fitter.py         # GP model fitting and likelihood computation
├── segmenter.py         # GPCPDSegmenter class (main API)
├── validation.py        # Statistical tests and known-event validation
└── types.py             # RegimeSegments, CPDConfig dataclasses
```

### Core Classes

**1. `CPDConfig` (dataclass):**
```python
@dataclass
class CPDConfig:
    lookback: int = 21          # Lookback window for CPD
    threshold: float = 0.9      # Severity threshold
    min_length: int = 5         # Minimum regime length
    max_length: int = 21        # Maximum regime length
```

**2. `RegimeSegment` (NamedTuple):**
```python
class RegimeSegment(NamedTuple):
    start_idx: int              # Start index in series
    end_idx: int                # End index (inclusive)
    severity: float             # Detection severity [0, 1]
    start_date: pd.Timestamp    # Start date
    end_date: pd.Timestamp      # End date
```

**3. `RegimeSegments` (dataclass):**
```python
@dataclass
class RegimeSegments:
    segments: List[RegimeSegment]
    config: CPDConfig

    # Built-in validation methods
    def validate_statistics(self, prices) -> ValidationReport
    def validate_known_events(self, events) -> EventValidation
    def plot(self, prices) -> plt.Figure
```

**4. `GPCPDSegmenter`:**
```python
class GPCPDSegmenter:
    def __init__(self, config: CPDConfig):
        self.config = config
        self.fitter = GPFitter()

    def fit_segment(self, prices: pd.Series) -> RegimeSegments:
        """Main API: segment price series into regimes."""
```

### Design Decisions

- **GPyTorch for GP operations**: Leverage existing kernel infrastructure, GPU-ready
- **Offline preprocessing**: Run CPD once, cache results (not in training loop)
- **Validation built-in**: RegimeSegments has validation methods
- **Class-based API**: Reusable across 50+ assets with consistent config

---

## 2. GPyTorch Kernels

### Option 1: Use Built-in ChangePointKernel (Recommended)

**Codex Review Insight:** GPyTorch already provides `gpytorch.kernels.ChangePointKernel`!

```python
from gpytorch.kernels import MaternKernel, ChangePointKernel, ScaleKernel

# Base kernels for before/after change-point
base_kernel1 = ScaleKernel(MaternKernel(nu=1.5))
base_kernel2 = ScaleKernel(MaternKernel(nu=1.5))

# Built-in change-point kernel with proper parameter handling
changepoint_kernel = ChangePointKernel(
    base_kernel1,
    base_kernel2,
    changepoint_loc_prior=...,  # Constrain to valid time range
)
```

**Strategy:**
1. Start with built-in `ChangePointKernel`
2. Validate against paper's algorithm
3. Only implement custom kernel if built-in doesn't match exactly

### Option 2: Custom Kernel (Fallback)

If built-in doesn't match paper exactly:

```python
class PaperChangePointKernel(gpytorch.kernels.Kernel):
    """
    Soft transition between two GPs at change-point location.

    k_CP(x1, x2) = σ(x1) * σ(x2) * k1(x1, x2)
                 + (1-σ(x1)) * (1-σ(x2)) * k2(x1, x2)

    where σ(x) = sigmoid((x - t_cp) / sigma)
    """

    def __init__(self, base_kernel1, base_kernel2, t_cp_init, time_bounds):
        super().__init__(has_lengthscale=False)

        # Properly register learnable parameters (Codex recommendation)
        self.register_parameter(
            "raw_t_cp",
            torch.nn.Parameter(torch.tensor(t_cp_init))
        )
        self.register_constraint(
            "raw_t_cp",
            gpytorch.constraints.Interval(time_bounds[0], time_bounds[1])
        )

        # Transition smoothness (learnable with lower bound)
        self.register_parameter(
            "raw_sigma",
            torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint(
            "raw_sigma",
            gpytorch.constraints.Positive(lower_bound=0.1)
        )

        self.k1 = base_kernel1
        self.k2 = base_kernel2

    def forward(self, x1, x2, diag=False):
        # Compute sigmoid weights
        s1 = torch.sigmoid((x1 - self.t_cp) / self.sigma)
        s2 = torch.sigmoid((x2 - self.t_cp) / self.sigma)

        # Weighted combination of two kernels
        k1_mat = self.k1(x1, x2, diag=diag)
        k2_mat = self.k2(x1, x2, diag=diag)

        return s1 * s2 * k1_mat + (1 - s1) * (1 - s2) * k2_mat
```

**Key Points (from Codex review):**
- Must properly register parameters with constraints
- Use `torch.nn.Parameter` for learnable t_cp
- Ensure proper broadcasting for batch operations
- Sigma controls identifiability (learnable with regularization)

---

## 3. GP Fitting & Likelihood

### 🚨 Critical Fix: Severity Formula

**INCORRECT (original):**
```python
# WRONG! L_M and L_C are log-likelihoods, not likelihoods
severity = L_C / (L_M + L_C)  # ❌
```

**CORRECT (Codex-reviewed):**
```python
# Use log Bayes factor
delta = L_C - L_M  # Log Bayes factor
severity = torch.sigmoid(torch.tensor(delta)).item()  # ✅

# Interpretation:
# severity ≈ 0.5: No evidence for CP
# severity ≥ 0.9: Strong evidence (delta ≥ 2.2)
```

### GPFitter Implementation

```python
class GPFitter:
    """Fits GPs and computes marginal likelihoods for CPD."""

    def fit_stationary_gp(self, x: Tensor, y: Tensor) -> Tuple[GP, float]:
        """
        Fit single Matérn GP (no change-point).
        Returns: (fitted_gp, log_marginal_likelihood)
        """
        kernel = ScaleKernel(MaternKernel(nu=1.5))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x, y, likelihood, kernel)

        # Optimize with convergence loop (NOT fixed iterations!)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        prev_loss = float('inf')
        patience_count = 0

        for i in range(200):  # Max iterations
            optimizer.zero_grad()
            loss = -mll(model(x), y)
            loss.backward()
            optimizer.step()

            # Convergence check (Codex recommendation)
            if abs(loss.item() - prev_loss) < 1e-3:
                patience_count += 1
                if patience_count >= 5:
                    break
            else:
                patience_count = 0
            prev_loss = loss.item()

        log_mll = mll(model(x), y).item()
        return model, log_mll

    def fit_changepoint_gp(self, x: Tensor, y: Tensor,
                           stationary_model: GP) -> Tuple[GP, float, float]:
        """
        Fit change-point GP with gradient-based t_cp optimization.
        Warm-start from stationary model hyperparameters (Codex rec).
        """
        # Initialize with stationary hyperparameters (WARM START!)
        cp_kernel = ChangePointKernel(
            ScaleKernel(MaternKernel(nu=1.5)),
            ScaleKernel(MaternKernel(nu=1.5)),
            changepoint_loc=len(x) // 2  # Initial guess: middle
        )

        # Warm-start: copy hyperparameters from stationary model
        cp_kernel.k1.outputscale = stationary_model.covar_module.outputscale
        cp_kernel.k1.base_kernel.lengthscale = stationary_model.covar_module.base_kernel.lengthscale
        cp_kernel.k2.outputscale = stationary_model.covar_module.outputscale
        cp_kernel.k2.base_kernel.lengthscale = stationary_model.covar_module.base_kernel.lengthscale

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = stationary_model.likelihood.noise  # Warm-start noise

        model = ExactGPModel(x, y, likelihood, cp_kernel)

        # Optimize jointly: hyperparameters + t_cp location (gradient-based!)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # ... convergence loop similar to stationary ...

        log_mll = mll(model(x), y).item()
        best_t_cp = cp_kernel.t_cp.item()

        return model, log_mll, best_t_cp

    def compute_severity(self, log_mll_stationary: float,
                        log_mll_changepoint: float) -> float:
        """
        Compute severity using log Bayes factor (CORRECT formula).

        Δ = L_C - L_M (log Bayes factor)
        severity = sigmoid(Δ) ∈ [0, 1]
        """
        delta = log_mll_changepoint - log_mll_stationary
        severity = torch.sigmoid(torch.tensor(delta)).item()
        return severity
```

**Key Improvements (from Codex):**
1. ✅ Convergence loop (not fixed 50 iterations)
2. ✅ Gradient-based t_cp optimization (not grid search)
3. ✅ Warm-start hyperparameters from stationary model
4. ✅ Correct severity formula using log Bayes factor
5. ✅ Learning rate decay and early stopping

### GPyTorch Gotchas (from Codex)
- Always call `model.eval()` for predictions
- Wrap predictions in `torch.no_grad()`
- Add jitter: `gpytorch.settings.cholesky_jitter(1e-4)`
- Keep tensors on same device/dtype
- Monitor for NaN in parameters

---

## 4. Segmentation Algorithm

### Recursive Backward Segmentation

**Algorithm (from X-Trend paper Algorithm 1):**

1. Start from end of series (current_end = len(prices) - 1)
2. Look back `lookback` days
3. Test for change-point in window
4. If severity ≥ threshold:
   - Mark regime from change-point to current_end
   - Move current_end to before change-point, repeat
5. If severity < threshold:
   - No change-point detected
   - Jump by max_length (avoid wasteful iteration)
6. Continue until current_end < min_length

### Implementation with Edge Case Handling

```python
class GPCPDSegmenter:
    def fit_segment(self, prices: pd.Series) -> RegimeSegments:
        """Segment entire price series with proper edge case handling."""
        segments = []
        current_end = len(prices) - 1

        while current_end >= self.config.min_length:
            window_start = max(0, current_end - self.config.lookback + 1)
            window = prices.iloc[window_start:current_end + 1]

            # Handle leftover stub at beginning
            if len(window) < self.config.min_length:
                if current_end + 1 >= self.config.min_length:
                    segments.append(RegimeSegment(
                        start_idx=0,
                        end_idx=current_end,
                        severity=0.0,
                        start_date=prices.index[0],
                        end_date=prices.index[current_end]
                    ))
                break

            # Detect change-point in window
            x = torch.arange(len(window)).float().unsqueeze(-1)
            y = torch.tensor(window.values).float()

            stat_model, log_mll_M = self.fitter.fit_stationary_gp(x, y)
            cp_model, log_mll_C, t_cp_relative = self.fitter.fit_changepoint_gp(
                x, y, stat_model
            )
            severity = self.fitter.compute_severity(log_mll_M, log_mll_C)

            if severity >= self.config.threshold:
                # Change-point detected!
                # Round t_cp (don't truncate) - Codex recommendation
                t_cp_absolute = window_start + round(t_cp_relative)

                # Validate CP creates valid regimes on both sides
                regime_length = current_end - t_cp_absolute + 1
                left_length = t_cp_absolute - window_start

                if (regime_length >= self.config.min_length and
                    left_length >= self.config.min_length):
                    # Valid CP
                    segments.append(RegimeSegment(
                        start_idx=t_cp_absolute,
                        end_idx=current_end,
                        severity=severity,
                        start_date=prices.index[t_cp_absolute],
                        end_date=prices.index[current_end]
                    ))
                    current_end = t_cp_absolute - 1
                else:
                    # CP too close to edge, treat as no CP
                    # Jump by max_length (performance optimization)
                    regime_len = min(self.config.max_length, current_end - window_start + 1)
                    segments.append(RegimeSegment(
                        start_idx=current_end - regime_len + 1,
                        end_idx=current_end,
                        severity=severity,
                        start_date=prices.index[current_end - regime_len + 1],
                        end_date=prices.index[current_end]
                    ))
                    current_end -= regime_len
            else:
                # No change-point detected
                # Jump by max_length immediately (don't waste GP fits!)
                regime_len = min(self.config.max_length, current_end - window_start + 1)

                if regime_len >= self.config.min_length:
                    segments.append(RegimeSegment(
                        start_idx=current_end - regime_len + 1,
                        end_idx=current_end,
                        severity=severity,
                        start_date=prices.index[current_end - regime_len + 1],
                        end_date=prices.index[current_end]
                    ))
                    current_end -= regime_len
                else:
                    current_end -= 1

        # Reverse (built backward)
        segments.reverse()

        return RegimeSegments(segments=segments, config=self.config)
```

**Key Improvements (from Codex review):**
1. ✅ Stub handling at beginning (merge into final segment)
2. ✅ CP boundary check (ensure valid regimes on both sides)
3. ✅ Performance: Jump by max_length when no CP (avoid wasteful GP fits)
4. ✅ Rounding: Use `round()` instead of `int()` for t_cp
5. ✅ Min/max enforcement: All regimes satisfy constraints

---

## 5. Validation Strategy

### Three-Pronged Approach

1. **Statistical Tests** (automated)
2. **Known-Event Validation** (COVID, Brexit, etc.)
3. **Visual Validation** (human inspection)

### Enhanced Statistical Validation

```python
@dataclass
class RegimeSegments:
    def validate_statistics(self, prices: pd.Series) -> ValidationReport:
        """Enhanced statistical validation with dispersion and quality metrics."""
        checks = []
        lengths = [seg.end_idx - seg.start_idx + 1 for seg in self.segments]

        # 1. Length statistics (mean + dispersion)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        cv_length = std_length / mean_length  # Coefficient of variation

        checks.append(ValidationCheck(
            name="Mean regime length",
            expected_range=(10, 15),
            actual=mean_length,
            passed=(10 <= mean_length <= 15)
        ))

        checks.append(ValidationCheck(
            name="Length dispersion (CV)",
            expected_range=(0.3, 0.7),
            actual=cv_length,
            passed=(0.3 <= cv_length <= 0.7)
        ))

        # 2. Min/max constraints satisfied
        all_within_bounds = all(
            self.config.min_length <= length <= self.config.max_length
            for length in lengths
        )

        # 3. No gaps or overlaps
        sorted_segs = sorted(self.segments, key=lambda s: s.start_idx)
        no_gaps = all(
            sorted_segs[i].end_idx + 1 == sorted_segs[i+1].start_idx
            for i in range(len(sorted_segs) - 1)
        )

        # 4. Within vs across regime return dispersion (Codex suggestion)
        from scipy.stats import levene

        regime_return_groups = [
            prices.iloc[seg.start_idx:seg.end_idx + 1].pct_change().dropna()
            for seg in self.segments if seg.end_idx - seg.start_idx >= 5
        ]

        if len(regime_return_groups) >= 3:
            levene_stat, levene_p = levene(*regime_return_groups)
            checks.append(ValidationCheck(
                name="Regime volatility heterogeneity",
                expected="Regimes have different volatility (p < 0.05)",
                actual=f"p={levene_p:.4f}",
                passed=(levene_p < 0.05)
            ))

        # 5. Within-regime stationarity (ADF test)
        from statsmodels.tsa.stattools import adfuller

        stationary_count = 0
        total_testable = 0

        for seg in self.segments:
            if seg.end_idx - seg.start_idx >= 10:  # Need ≥10 points for ADF
                regime_returns = prices.iloc[seg.start_idx:seg.end_idx + 1].pct_change().dropna()
                adf_result = adfuller(regime_returns, regression='c')
                if adf_result[1] < 0.05:  # p-value < 0.05 = stationary
                    stationary_count += 1
                total_testable += 1

        stationarity_pct = stationary_count / total_testable if total_testable > 0 else 0
        checks.append(ValidationCheck(
            name="Within-regime stationarity",
            expected_range=(0.6, 1.0),
            actual=stationarity_pct,
            passed=(stationarity_pct >= 0.6)
        ))

        # 6. Severity calibration (empirical, not hard-coded - Codex rec)
        severities = [seg.severity for seg in self.segments]
        severity_p90 = np.percentile(severities, 90)

        checks.append(ValidationCheck(
            name="Severity 90th percentile",
            expected_range=(0.85, 0.95),
            actual=severity_p90,
            passed=(0.85 <= severity_p90 <= 0.95)
        ))

        return ValidationReport(checks=checks)
```

### Known-Event Validation

```python
def validate_known_events(self, events: List[KnownEvent]) -> EventValidation:
    """
    Validate against known regime changes using event windows.
    Codex recommendation: Use overlap, not single-date matching.
    """
    results = []

    for event in events:
        # Find regime boundaries that overlap event window
        overlapping_cps = []

        for seg in self.segments:
            cp_date = seg.start_date
            event_window = pd.date_range(
                event.start_date - pd.Timedelta(days=event.tolerance_days),
                event.end_date + pd.Timedelta(days=event.tolerance_days)
            )

            if cp_date in event_window:
                overlapping_cps.append(seg)

        detected = len(overlapping_cps) > 0

        if detected:
            # Find closest CP to event mid-point
            event_mid = event.start_date + (event.end_date - event.start_date) / 2
            closest_cp = min(overlapping_cps,
                           key=lambda s: abs((s.start_date - event_mid).days))
            days_off = abs((closest_cp.start_date - event_mid).days)
            severity = closest_cp.severity
        else:
            days_off = float('inf')
            severity = 0.0

        results.append(EventValidationResult(
            event_name=event.name,
            event_window=(event.start_date, event.end_date),
            detected=detected,
            days_off=days_off,
            severity=severity,
            passed=(detected and days_off <= event.tolerance_days)
        ))

    return EventValidation(results=results)
```

**Known Events to Test:**
- COVID crash: Feb 24 - Mar 23, 2020 (tolerance: ±7 days)
- Brexit vote: Jun 23 - Jul 15, 2016 (tolerance: ±7 days)
- Taper tantrum: May 22 - Jun 24, 2013 (tolerance: ±10 days)

---

## 6. Streamlit Integration

### Extend Existing Bloomberg Explorer

Add new "Phase 2: Regimes" tab to `bloomberg_explorer.py`.

**File Structure:**
```
scripts/
├── bloomberg_explorer.py        # Main app (add new tab)
└── bloomberg_viz/
    ├── __init__.py
    ├── prices_tab.py            # Existing
    ├── returns_tab.py           # Phase 1
    └── regimes_tab.py           # NEW - Phase 2
```

### Implementation with Proper Caching

**Codex recommendations incorporated:**
- Use `@st.cache_data` with proper cache key (not session_state)
- Show "Last run" badge to indicate stale results
- Progress indicators for long-running computations
- Downsample large datasets for performance

```python
# scripts/bloomberg_viz/regimes_tab.py

import streamlit as st
import plotly.graph_objects as go
from xtrend.cpd import GPCPDSegmenter, CPDConfig

@st.cache_data
def run_cpd_cached(asset: str, start_date, end_date,
                   lookback: int, threshold: float,
                   min_length: int, max_length: int):
    """
    Run GP-CPD with proper caching.
    Cache key = (asset, dates, hyperparams)
    """
    from xtrend.data import BloombergParquetSource

    data_source = BloombergParquetSource()
    prices = data_source.load_symbol(asset, start_date, end_date)['Close']

    config = CPDConfig(
        lookback=lookback,
        threshold=threshold,
        min_length=min_length,
        max_length=max_length
    )

    segmenter = GPCPDSegmenter(config)
    segments = segmenter.fit_segment(prices)

    return segments, prices, config

def render_regimes_tab(data_source):
    """Render the Regimes tab with proper caching and UX."""

    st.header("Phase 2: Regime Detection & Validation")

    # Sidebar: Configuration
    with st.sidebar:
        st.subheader("CPD Configuration")
        lookback = st.slider("Lookback window", 10, 63, 21)
        threshold = st.slider("Severity threshold", 0.5, 0.99, 0.9, 0.01)
        min_length = st.slider("Min regime length", 3, 10, 5)
        max_length = st.slider("Max regime length", 21, 126, 21)

    # Asset selection
    selected_asset = st.selectbox(
        "Select asset for regime analysis",
        data_source.available_symbols()
    )

    date_range = st.date_input(
        "Date range",
        value=(pd.Timestamp('2019-01-01'), pd.Timestamp('2023-12-31'))
    )

    # Run CPD button
    run_button = st.button("🔍 Detect Regimes", type="primary")

    # Show last run info if available
    if 'last_run_params' in st.session_state:
        last_run = st.session_state['last_run_params']
        with st.expander("ℹ️ Last Run Info", expanded=False):
            st.write(f"**Asset:** {last_run['asset']}")
            st.write(f"**Date Range:** {last_run['start']} to {last_run['end']}")

            current_params = {
                'asset': selected_asset,
                'start': date_range[0],
                'end': date_range[1],
                'lookback': lookback,
                'threshold': threshold
            }

            if current_params != last_run:
                st.warning("⚠️ Settings differ from last run")

    # Execute if button clicked
    if run_button:
        with st.status("Running GP-CPD...", state="running") as status:
            try:
                status.update(label="Loading price data...", state="running")

                segments, prices, config = run_cpd_cached(
                    selected_asset, date_range[0], date_range[1],
                    lookback, threshold, min_length, max_length
                )

                status.update(label="Segmentation complete!", state="complete")

                st.session_state['current_results'] = {
                    'segments': segments,
                    'prices': prices,
                    'asset': selected_asset
                }

                st.session_state['last_run_params'] = {
                    'asset': selected_asset,
                    'start': date_range[0],
                    'end': date_range[1],
                    'lookback': lookback,
                    'threshold': threshold
                }

                st.toast(f"✅ Detected {len(segments.segments)} regimes")

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {str(e)}")
                return

    # Display results
    if 'current_results' in st.session_state:
        results = st.session_state['current_results']
        segments = results['segments']
        prices = results['prices']

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Regimes", len(segments.segments))
        with col2:
            avg_len = np.mean([s.end_idx - s.start_idx + 1 for s in segments.segments])
            st.metric("Avg Length", f"{avg_len:.1f} days")
        with col3:
            high_sev = sum(s.severity >= 0.9 for s in segments.segments) / len(segments.segments)
            st.metric("High Severity %", f"{high_sev:.1%}")

        # Three sub-tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Visualization",
            "✅ Validation",
            "🎯 Known Events"
        ])

        with tab1:
            render_regime_chart(prices, segments, results['asset'])
        with tab2:
            render_validation_results(segments, prices)
        with tab3:
            render_known_events(segments)
    else:
        st.info("👆 Click 'Detect Regimes' to begin")

def render_regime_chart(prices, segments, asset):
    """Interactive Plotly chart with regime coloring."""

    # Downsample if needed (Codex rec)
    if len(prices) > 2000:
        prices_plot = prices.resample('W').last()
        st.caption(f"ℹ️ Downsampled to {len(prices_plot)} points")
    else:
        prices_plot = prices

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=prices_plot.index, y=prices_plot.values,
        mode='lines', name='Price',
        line=dict(color='black', width=1.5), opacity=0.7
    ))

    # Colored regime segments
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3

    for i, seg in enumerate(segments.segments):
        color = colors[i % len(colors)]

        fig.add_vrect(
            x0=seg.start_date, x1=seg.end_date,
            fillcolor=color,
            opacity=0.15 + 0.25 * seg.severity,
            layer="below", line_width=0,
            annotation_text=f"R{i+1}",
            annotation_position="top left"
        )

        if i > 0:
            fig.add_vline(
                x=seg.start_date,
                line_dash="dash", line_color="red",
                line_width=1.5, opacity=seg.severity
            )

    fig.update_layout(
        title=f"{asset} - {len(segments.segments)} Regimes",
        xaxis_title="Date", yaxis_title="Price",
        hovermode='x unified', height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Regime details table
    regime_df = pd.DataFrame([
        {
            'ID': i + 1,
            'Start': seg.start_date.strftime('%Y-%m-%d'),
            'End': seg.end_date.strftime('%Y-%m-%d'),
            'Days': seg.end_idx - seg.start_idx + 1,
            'Severity': f"{seg.severity:.3f}",
            'Status': '🔴 High' if seg.severity >= 0.9 else '🟡 Med' if seg.severity >= 0.7 else '🟢 Low'
        }
        for i, seg in enumerate(segments.segments)
    ])

    st.dataframe(regime_df, use_container_width=True, hide_index=True)
```

**Key UX Features:**
- ✅ Proper caching with `@st.cache_data`
- ✅ "Last run" badge shows parameter staleness
- ✅ Progress indicators (`st.status`)
- ✅ Toast notifications
- ✅ Downsampling for large datasets
- ✅ Responsive layout
- ✅ Three sub-tabs for organization

---

## 7. Testing Strategy

### Test Organization

```
tests/
├── cpd/
│   ├── __init__.py
│   ├── conftest.py              # Fixtures
│   ├── test_kernels.py          # Unit: Kernels
│   ├── test_gp_fitter.py        # Unit: GP fitting
│   ├── test_segmenter.py        # Integration: Full pipeline
│   ├── test_validation.py       # Validation methods
│   └── test_data/
│       ├── synthetic_regimes.csv
│       └── known_events.yaml
└── integration/
    └── test_phase2_pipeline.py  # End-to-end Phase 2
```

### Unit Tests: Kernels

```python
# tests/cpd/test_kernels.py

class TestChangePointKernel:
    def test_kernel_shape(self):
        """Kernel output has correct shape."""
        cp_kernel = ChangePointKernel(...)
        x = torch.arange(20).float().unsqueeze(-1)
        K = cp_kernel(x, x).evaluate()

        assert K.shape == (20, 20)
        assert torch.allclose(K, K.t())  # Symmetric

    def test_parameter_learnable(self):
        """t_cp is properly registered as learnable."""
        cp_kernel = ChangePointKernel(...)
        params = list(cp_kernel.parameters())
        assert any('t_cp' in name for name, _ in cp_kernel.named_parameters())
```

### Unit Tests: GP Fitting

```python
# tests/cpd/test_gp_fitter.py

class TestGPFitter:
    @pytest.fixture
    def synthetic_changepoint_data(self):
        """Generate synthetic data with known change-point at t=15."""
        torch.manual_seed(42)
        x = torch.arange(30).float().unsqueeze(-1)

        # Two different regimes (Codex: add variants!)
        y1 = torch.sin(x[:15] / 5.0) + 0.1 * torch.randn(15)
        y2 = -torch.sin(x[15:] / 5.0) + 2.0 + 0.1 * torch.randn(15)
        y = torch.cat([y1, y2])

        return x, y, 15  # True change-point

    def test_changepoint_detection_on_synthetic(self, synthetic_changepoint_data):
        """Detects obvious change-point in synthetic data."""
        x, y, true_cp = synthetic_changepoint_data

        fitter = GPFitter()
        stat_model, log_mll_M = fitter.fit_stationary_gp(x, y)
        cp_model, log_mll_C, detected_cp = fitter.fit_changepoint_gp(x, y, stat_model)

        # Change-point model should fit better
        assert log_mll_C > log_mll_M

        # Should detect CP near true location (±3 points = 10% tolerance)
        assert abs(detected_cp - true_cp) <= 3

        # Severity should be high
        severity = fitter.compute_severity(log_mll_M, log_mll_C)
        assert severity >= 0.8

    def test_severity_formula_correct(self):
        """Severity uses correct log Bayes factor formula."""
        fitter = GPFitter()

        # Equal likelihoods → severity ≈ 0.5
        assert fitter.compute_severity(0.0, 0.0) == pytest.approx(0.5, abs=0.01)

        # Strong evidence → severity ≥ 0.9
        assert fitter.compute_severity(0.0, 2.2) >= 0.9

    def test_convergence_checks(self):
        """GP optimization converges properly (Codex rec)."""
        # Test gradient norm, hyperparameter stability across seeds
        pass
```

### Integration Tests: Full Segmentation

```python
# tests/cpd/test_segmenter.py

class TestGPCPDSegmenter:
    @pytest.fixture
    def sample_prices(self):
        """Load sample price data for testing (Codex: multiple assets!)."""
        from xtrend.data import BloombergParquetSource
        source = BloombergParquetSource()

        # Test on ES (equity), CL (commodity), and EC (FX) - Codex rec
        prices_es = source.load_symbol('ES', '2020-01-01', '2020-12-31')['Close']
        return prices_es

    def test_no_gaps_or_overlaps(self, sample_prices):
        """Segmentation produces contiguous coverage."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)
        segments = segmenter.fit_segment(sample_prices)

        sorted_segs = sorted(segments.segments, key=lambda s: s.start_idx)

        # No gaps
        for i in range(len(sorted_segs) - 1):
            assert sorted_segs[i].end_idx + 1 == sorted_segs[i+1].start_idx

    def test_covid_crash_detected(self, sample_prices):
        """Detects COVID crash in Feb-Mar 2020 (±10 days tolerance)."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)
        segments = segmenter.fit_segment(sample_prices)

        covid_date = pd.Timestamp('2020-02-24')

        nearby_cps = [
            seg for seg in segments.segments
            if abs((seg.start_date - covid_date).days) <= 10
        ]

        assert len(nearby_cps) > 0, "Should detect COVID crash"
        assert nearby_cps[0].severity >= 0.85

    def test_flat_series_no_cp(self):
        """Flat series returns no high-severity change-points (Codex rec)."""
        flat_prices = pd.Series(100.0, index=pd.date_range('2020-01-01', periods=100))
        # ... should not detect spurious CPs ...
```

### Property-Based Tests (Codex Recommendation)

```python
# tests/cpd/test_properties.py

from hypothesis import given, strategies as st

class TestSegmentationProperties:
    @given(
        st.integers(min_value=5, max_value=10),  # min_length
        st.integers(min_value=15, max_value=30)  # max_length
    )
    def test_length_constraints_always_satisfied(self, min_len, max_len):
        """Property: All segments satisfy [min_length, max_length]."""
        # Generate random price series
        # Run segmentation
        # Assert all lengths in bounds
        pass

    def test_coverage_complete(self):
        """Property: Sum of segment lengths = series length."""
        # For any valid segmentation, no gaps
        pass
```

### Test Coverage Goals

- **Unit tests**: 80%+ coverage on core modules
- **Integration tests**: Full pipeline on sample data (ES, CL, EC)
- **Known-event tests**: COVID, Brexit, Taper Tantrum
- **Synthetic data**: Multiple break types (amplitude, variance, SNR)
- **Degradation tests**: Flat series, missing data, outliers

### Running Tests

```bash
# All Phase 2 tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/ -v

# With coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/ --cov=xtrend.cpd --cov-report=term-missing

# Known-event tests only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/ -k "covid or brexit"

# Multi-asset tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/ -k "multi_asset"
```

---

## 8. Implementation Checklist

### Phase 2A: Core Implementation

- [ ] **Kernels** (`xtrend/cpd/kernels.py`)
  - [ ] Test built-in `gpytorch.kernels.ChangePointKernel`
  - [ ] If needed, implement custom `PaperChangePointKernel`
  - [ ] Unit tests for kernel properties

- [ ] **GP Fitter** (`xtrend/cpd/gp_fitter.py`)
  - [ ] `fit_stationary_gp` with convergence loop
  - [ ] `fit_changepoint_gp` with warm-start
  - [ ] `compute_severity` using log Bayes factor (CORRECT formula!)
  - [ ] Unit tests on synthetic data

- [ ] **Segmenter** (`xtrend/cpd/segmenter.py`)
  - [ ] `GPCPDSegmenter` class
  - [ ] Recursive backward segmentation
  - [ ] Edge case handling (stubs, boundaries)
  - [ ] Integration tests on real data

- [ ] **Types** (`xtrend/cpd/types.py`)
  - [ ] `CPDConfig` dataclass
  - [ ] `RegimeSegment` NamedTuple
  - [ ] `RegimeSegments` with validation methods

### Phase 2B: Validation & Visualization

- [ ] **Validation** (`xtrend/cpd/validation.py`)
  - [ ] Statistical tests (length, dispersion, Levene, ADF)
  - [ ] Known-event validation (COVID, Brexit, Taper Tantrum)
  - [ ] Visual validation (matplotlib plots)

- [ ] **Streamlit Tab** (`scripts/bloomberg_viz/regimes_tab.py`)
  - [ ] `run_cpd_cached` with `@st.cache_data`
  - [ ] `render_regimes_tab` main function
  - [ ] Three sub-tabs (Visualization, Validation, Events)
  - [ ] Integration with existing `bloomberg_explorer.py`

### Phase 2C: Testing

- [ ] **Unit Tests**
  - [ ] `tests/cpd/test_kernels.py`
  - [ ] `tests/cpd/test_gp_fitter.py`
  - [ ] `tests/cpd/test_validation.py`

- [ ] **Integration Tests**
  - [ ] `tests/cpd/test_segmenter.py`
  - [ ] Multi-asset tests (ES, CL, EC)
  - [ ] Known-event tests
  - [ ] Synthetic data tests (multiple break types)

- [ ] **Property Tests**
  - [ ] Hypothesis-based invariant checks
  - [ ] Coverage completeness
  - [ ] Constraint satisfaction

### Phase 2D: Documentation & Review

- [ ] Generate visualizations matching Figure 3 from paper
- [ ] Run validation on all 50 assets
- [ ] Code review with `code-reviewer` subagent
- [ ] Update `phases.md` with completion status

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **GP Library** | GPyTorch | PyTorch ecosystem, GPU-ready, built-in ChangePointKernel |
| **Severity Formula** | `sigmoid(L_C - L_M)` | Correct log Bayes factor (Codex fix) |
| **t_cp Optimization** | Gradient-based | Better than grid search (Codex rec) |
| **Hyperparameter Init** | Warm-start from stationary | Faster convergence (Codex rec) |
| **API Design** | Class-based with caching | Reusable across 50+ assets |
| **Performance** | Jump by max_length when no CP | Avoid wasteful GP fits (Codex rec) |
| **Edge Cases** | Proper stub handling | No constraint violations (Codex rec) |
| **Validation** | 3-pronged (stats + events + visual) | Comprehensive confidence |
| **Streamlit** | `@st.cache_data` with cache key | Proper caching (Codex rec) |
| **Testing** | Multi-asset + synthetic + property | Robust coverage (Codex rec) |

---

## References

1. **X-Trend Paper**: Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies (Wood et al. 2024)
2. **GPyTorch Documentation**: https://gpytorch.ai/
3. **Codex Reviews**: 5 comprehensive reviews incorporated throughout design
4. **Change-Point Detection Skill**: `.claude/skills/change-point-detection/`

---

## Appendix: Codex Review Summary

### Review 1: Kernel Design
- ✅ Use built-in `gpytorch.kernels.ChangePointKernel`
- ✅ Proper parameter registration with constraints
- ✅ Sigma should be learnable with regularization
- ✅ Grid search for t_cp initialization if needed

### Review 2: GP Fitting
- 🚨 **Critical fix**: Severity formula was wrong! Use log Bayes factor
- ✅ Convergence loop (not fixed 50 iterations)
- ✅ Warm-start hyperparameters from stationary model
- ✅ Learning rate decay and early stopping

### Review 3: Segmentation Algorithm
- ✅ Proper edge case handling (stubs at beginning)
- ✅ CP boundary validation (both sides must be valid)
- ✅ Performance: Jump by max_length when no CP
- ✅ Round t_cp (don't truncate)

### Review 4: Validation Strategy
- ✅ Add dispersion checks (CV of lengths)
- ✅ Return dispersion tests (Levene)
- ✅ Within-regime stationarity (ADF)
- ✅ Event window matching (not single-date)
- ✅ Severity calibration using percentiles

### Review 5: Streamlit Integration
- ✅ Use `@st.cache_data` with proper cache key
- ✅ "Last run" badge for staleness
- ✅ Progress indicators
- ✅ Downsample large datasets

### Review 6: Testing Strategy
- ✅ Synthetic data variants (multiple break types)
- ✅ Multi-asset testing (ES, CL, EC)
- ✅ Degradation tests (flat series, outliers)
- ✅ GP convergence checks (gradient norm)
- ✅ Hypothesis property testing

---

**Design Status:** ✅ Complete and Codex-Reviewed
**Ready for Implementation:** Yes
**Next Step:** Execute implementation plan with subagents
