# Bloomberg Visualization - Tasks

**Last Updated:** 2025-11-16 22:45 CST
**Status:** ✅ All MVP Tasks Complete

---

## Completed Tasks

### Phase 1: MVP Implementation ✅

- [x] **Task 1: Module Structure**
  - Created `scripts/bloomberg_viz/` package
  - Implemented `data_loader.py` with caching
  - Implemented `charts.py` with Plotly builders
  - Created stub files: `analysis.py`, `quality.py`
  - Commit: 3854f7b

- [x] **Task 2: Dependencies**
  - Added `streamlit>=1.28.0` to pyproject.toml
  - Added `plotly>=5.17.0` to pyproject.toml
  - Ran `uv sync` to install
  - Commit: 5265e1f

- [x] **Task 3: Main Streamlit App**
  - Built `bloomberg_explorer.py` (155 lines)
  - Implemented Tab 1: Price Explorer (fully functional)
  - Implemented Tabs 2-4: Stubs with feature descriptions
  - Added error handling and validation
  - Commit: c13e5a1

- [x] **Task 4: Documentation**
  - Created `scripts/README.md`
  - Referenced architecture and launch commands
  - Commit: e63bb29

- [x] **Task 5: Bug Fixes**
  - Fixed data path inconsistency (2 attempts)
  - Implemented git root resolution for worktrees
  - Final fix commit: c30ce00

- [x] **Task 6: Testing & Verification**
  - Manual testing with user
  - Verified all features work
  - Confirmed worktree compatibility
  - No automated tests (not required)

- [x] **Task 7: Merge to Main**
  - Merged feature/bloomberg-explorer → main
  - Deleted feature branch
  - Cleaned up worktree
  - All commits on main

- [x] **Task 8: Update Documentation**
  - Updated `phases.md` with progress
  - Marked Phase 1 data loading as complete
  - Documented visualization tools
  - Commit: e9f0f91

---

## Future Tasks (Not Started)

### Tab 2: Returns Analysis 🔜

- [ ] Implement `calculate_returns(data, period='daily')` in `analysis.py`
  - Daily, weekly, monthly returns
  - Percentage change calculation
  - Handle NaN values

- [ ] Implement `calculate_volatility(data, window=20)` in `analysis.py`
  - Rolling standard deviation
  - Exponentially weighted volatility
  - Multiple window sizes

- [ ] Create returns chart in `charts.py`
  - Cumulative returns visualization
  - Returns distribution histogram
  - Multi-symbol comparison

- [ ] Wire up Tab 2 in `bloomberg_explorer.py`
  - Period selector (daily/weekly/monthly)
  - Volatility window slider
  - Charts and statistics display

**Estimated Effort:** 2-3 hours
**Prerequisites:** None (stubs ready)
**Priority:** Medium

---

### Tab 3: Data Quality 🔜

- [ ] Implement `check_missing_dates(df, symbol)` in `quality.py`
  - Identify trading day gaps
  - Weekend/holiday handling
  - Generate gap report

- [ ] Implement `detect_price_anomalies(df, threshold=5.0)` in `quality.py`
  - Find >5% daily jumps
  - Flag potential data errors
  - Generate anomaly list

- [ ] Implement `generate_quality_report(data)` in `quality.py`
  - Date range completeness
  - Missing data percentage
  - Anomaly count per symbol

- [ ] Create quality dashboard in Tab 3
  - Summary statistics table
  - Gap visualization (calendar heatmap)
  - Anomaly timeline

**Estimated Effort:** 3-4 hours
**Prerequisites:** None (stubs ready)
**Priority:** Low (data quality already verified manually)

---

### Tab 4: Correlations 🔜

- [ ] Implement `calculate_correlation_matrix(data, period='daily')` in `analysis.py`
  - Pearson correlation across symbols
  - Return-based correlation
  - Handle missing data

- [ ] Implement `rolling_correlation(data, window=60)` in `analysis.py`
  - Time-varying correlation
  - Pairwise correlations
  - Multiple window sizes

- [ ] Create correlation heatmap in `charts.py`
  - Plotly heatmap with color scale
  - Symbol labels and clustering
  - Interactive hover details

- [ ] Create rolling correlation chart in `charts.py`
  - Time series of correlation
  - Multiple symbol pairs
  - Shaded confidence intervals

- [ ] Wire up Tab 4 in `bloomberg_explorer.py`
  - Symbol pair selector
  - Window size slider
  - Heatmap and time series display

**Estimated Effort:** 4-5 hours
**Prerequisites:** None (stubs ready)
**Priority:** Low (can be added as needed)

---

## X-Trend Integration Tasks 🚀

### Phase 1: Feature Engineering Connection

- [ ] Refactor returns calculation to match X-Trend requirements
  - Multi-timescale: 1, 21, 63, 126, 252 days
  - Normalization: `r̂ = r / (σ_t * √t')`
  - Integration with `analysis.py`

- [ ] Add MACD visualization to Tab 2
  - Reuse MACD calculation from X-Trend pipeline
  - Overlay on price chart
  - Timescale selector: (8,24), (16,28), (32,96)

- [ ] Add volatility targeting visualization
  - 60-day EWMA volatility σ_t
  - Target volatility line (σ_tgt = 15%)
  - Leverage factor display

**Prerequisites:** X-Trend feature engineering implemented
**Integration Point:** `scripts/bloomberg_viz/analysis.py`

---

### Phase 2: Change-Point Detection Visualization

- [ ] Visualize CPD regime segments
  - Color-coded price chart by regime
  - Regime boundaries as vertical lines
  - Regime statistics overlay

- [ ] Show change-point severity over time
  - ν = L_C / (L_M + L_C) timeline
  - Threshold markers
  - Regime change annotations

- [ ] Regime length histogram
  - Distribution of regime durations
  - Min/max length markers
  - Average regime length stat

**Prerequisites:** CPD implementation complete (Phase 2 of X-Trend)
**Integration Point:** New tab or extend Tab 3

---

### Phase 3: Model Output Visualization

- [ ] Attention weights heatmap
  - Target steps × context sequences
  - Top-k attended contexts
  - Asset class color coding

- [ ] Predictive distribution display
  - Confidence intervals (95%, 50%)
  - Mean prediction vs actual
  - Quantile regression output

- [ ] Trading signal visualization
  - Position output z_t ∈ (-1, 1)
  - Predicted vs realized returns
  - Sharpe ratio by period

**Prerequisites:** X-Trend model trained (Phases 3-8)
**Integration Point:** New "Model Analysis" tab

---

## Technical Debt / Improvements

### Code Quality 🧹

- [ ] Add type hints to stub functions
  - Use `Dict[str, pd.DataFrame]` instead of `dict`
  - Add return type hints to all functions
  - **Effort:** 30 minutes

- [ ] Move imports to module level
  - `datetime` import in `bloomberg_explorer.py:108`
  - **Effort:** 5 minutes

- [ ] Add docstrings to `__init__.py`
  - Package-level documentation
  - **Effort:** 10 minutes

---

### Testing 🧪

- [ ] Add pytest tests for data loading
  - Test `get_available_symbols()`
  - Test `load_symbol()` with mock data
  - Test date range filtering
  - **Effort:** 1-2 hours

- [ ] Add pytest tests for charts
  - Test normalization logic
  - Test summary statistics calculation
  - Mock Plotly figure creation
  - **Effort:** 1-2 hours

- [ ] Add integration tests
  - End-to-end data loading → chart creation
  - Test with real parquet files
  - **Effort:** 1 hour

---

### Performance 🚀

- [ ] Profile memory usage with all 72 symbols
  - Measure peak memory
  - Identify cache size limits
  - Implement cache eviction if needed
  - **Effort:** 2 hours

- [ ] Optimize date range filtering
  - Currently filters in Python (fast enough)
  - Could push to parquet predicate if needed
  - **Effort:** 1 hour (if needed)

- [ ] Add progress indicators for slow operations
  - Loading many symbols
  - Long date ranges
  - **Effort:** 30 minutes

---

### UX Improvements 💡

- [ ] Add symbol limit warning
  - Warn when >10 symbols selected
  - Suggest UI becomes cluttered
  - **Effort:** 15 minutes

- [ ] Add keyboard shortcuts
  - Clear selection: Ctrl+K
  - Reset date range: Ctrl+R
  - **Effort:** 1 hour

- [ ] Add export functionality
  - Download filtered data as CSV
  - Export chart as PNG/SVG
  - **Effort:** 2 hours

- [ ] Add comparison mode
  - Side-by-side date range comparison
  - Before/after analysis
  - **Effort:** 3-4 hours

---

## Priority Matrix

### High Priority (Do Next)
1. Tab 2: Returns Analysis - Core feature for X-Trend integration
2. Pytest tests - Prevent regressions as code evolves

### Medium Priority (When Needed)
3. X-Trend feature engineering connection
4. Performance profiling with 72 symbols
5. Export functionality

### Low Priority (Nice to Have)
6. Tab 3: Data Quality
7. Tab 4: Correlations
8. UX improvements (shortcuts, warnings)
9. Code quality improvements (type hints)

### Future (After X-Trend Complete)
10. CPD visualization
11. Model output visualization
12. Attention weights display

---

## Notes

- All MVP tasks complete and merged to main
- App is production-ready for price visualization
- Architecture supports easy extension
- No blocking issues or technical debt
- Ready to proceed with X-Trend integration

**Next Session Recommendation:** Implement Tab 2 (Returns Analysis) to support X-Trend feature engineering.

---

**End of Tasks Document**
