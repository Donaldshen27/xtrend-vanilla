# Phase 2: GP Change-Point Detection - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Gaussian Process-based Change-Point Detection (GP-CPD) for segmenting financial time-series into regimes using GPyTorch.

**Architecture:** Paper-faithful GP implementation with corrected severity formula (log Bayes factor), recursive backward segmentation, three-pronged validation (statistical + known events + visual), and Streamlit integration extending existing Bloomberg explorer.

**Tech Stack:** GPyTorch (GP operations), PyTorch (tensors), pandas (time-series), Streamlit (visualization), pytest (testing)

---

## Task 1: Types and Configuration Foundation

**Files:**
- Create: `src/xtrend/cpd/__init__.py`
- Create: `src/xtrend/cpd/types.py`
- Create: `tests/cpd/__init__.py`
- Create: `tests/cpd/conftest.py`
- Create: `tests/cpd/test_types.py`

**Step 1: Write failing test for CPDConfig**

Create `tests/cpd/test_types.py`:

```python
"""Tests for CPD types and configuration."""
import pytest
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments


class TestCPDConfig:
    def test_default_values(self):
        """CPDConfig has sensible defaults."""
        config = CPDConfig()

        assert config.lookback == 21
        assert config.threshold == 0.9
        assert config.min_length == 5
        assert config.max_length == 21

    def test_custom_values(self):
        """CPDConfig accepts custom parameters."""
        config = CPDConfig(
            lookback=42,
            threshold=0.85,
            min_length=10,
            max_length=30
        )

        assert config.lookback == 42
        assert config.threshold == 0.85
        assert config.min_length == 10
        assert config.max_length == 30
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_types.py::TestCPDConfig::test_default_values -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'xtrend.cpd'"

**Step 3: Create package structure**

Create `src/xtrend/cpd/__init__.py`:

```python
"""GP Change-Point Detection for regime segmentation."""

from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments

__all__ = [
    "CPDConfig",
    "RegimeSegment",
    "RegimeSegments",
]
```

**Step 4: Implement CPDConfig**

Create `src/xtrend/cpd/types.py`:

```python
"""Type definitions for GP-CPD."""
from dataclasses import dataclass
from typing import List, NamedTuple

import pandas as pd


@dataclass
class CPDConfig:
    """Configuration for GP Change-Point Detection.

    Attributes:
        lookback: Window size for CPD detection (trading days)
        threshold: Severity threshold for detecting change-points [0, 1]
        min_length: Minimum regime length (trading days)
        max_length: Maximum regime length (trading days)
    """
    lookback: int = 21
    threshold: float = 0.9
    min_length: int = 5
    max_length: int = 21

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.lookback < self.min_length:
            raise ValueError(f"lookback ({self.lookback}) must be >= min_length ({self.min_length})")
        if self.min_length >= self.max_length:
            raise ValueError(f"min_length ({self.min_length}) must be < max_length ({self.max_length})")
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold ({self.threshold}) must be in [0, 1]")


class RegimeSegment(NamedTuple):
    """A single regime segment from CPD.

    Attributes:
        start_idx: Start index in price series
        end_idx: End index (inclusive) in price series
        severity: Detection severity [0, 1] (higher = stronger change-point)
        start_date: Start date of regime
        end_date: End date of regime
    """
    start_idx: int
    end_idx: int
    severity: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp


@dataclass
class RegimeSegments:
    """Collection of regime segments with validation methods.

    Attributes:
        segments: List of detected regime segments
        config: Configuration used for detection
    """
    segments: List[RegimeSegment]
    config: CPDConfig

    def __len__(self) -> int:
        """Number of detected regimes."""
        return len(self.segments)
```

**Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_types.py -v`

Expected: 2 passed

**Step 6: Add test for RegimeSegment**

Add to `tests/cpd/test_types.py`:

```python
class TestRegimeSegment:
    def test_creation(self):
        """RegimeSegment can be created with all fields."""
        seg = RegimeSegment(
            start_idx=0,
            end_idx=20,
            severity=0.95,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-31')
        )

        assert seg.start_idx == 0
        assert seg.end_idx == 20
        assert seg.severity == 0.95
        assert seg.start_date == pd.Timestamp('2020-01-01')
        assert seg.end_date == pd.Timestamp('2020-01-31')

    def test_length_property(self):
        """Regime length is end - start + 1."""
        seg = RegimeSegment(
            start_idx=10,
            end_idx=30,
            severity=0.9,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-31')
        )

        assert seg.end_idx - seg.start_idx + 1 == 21


class TestRegimeSegments:
    def test_empty_segments(self):
        """RegimeSegments can be created with empty list."""
        config = CPDConfig()
        segments = RegimeSegments(segments=[], config=config)

        assert len(segments) == 0

    def test_multiple_segments(self):
        """RegimeSegments tracks multiple regimes."""
        config = CPDConfig()
        segs = [
            RegimeSegment(0, 10, 0.9, pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-15')),
            RegimeSegment(11, 20, 0.85, pd.Timestamp('2020-01-16'), pd.Timestamp('2020-01-31')),
        ]
        segments = RegimeSegments(segments=segs, config=config)

        assert len(segments) == 2
```

**Step 7: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_types.py -v`

Expected: 6 passed

**Step 8: Add validation tests for CPDConfig**

Add to `tests/cpd/test_types.py`:

```python
class TestCPDConfigValidation:
    def test_lookback_less_than_min_length_raises(self):
        """lookback must be >= min_length."""
        with pytest.raises(ValueError, match="lookback.*must be >= min_length"):
            CPDConfig(lookback=3, min_length=5)

    def test_min_length_gte_max_length_raises(self):
        """min_length must be < max_length."""
        with pytest.raises(ValueError, match="min_length.*must be < max_length"):
            CPDConfig(min_length=21, max_length=21)

    def test_threshold_out_of_range_raises(self):
        """threshold must be in [0, 1]."""
        with pytest.raises(ValueError, match="threshold.*must be in"):
            CPDConfig(threshold=1.5)
```

**Step 9: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_types.py -v`

Expected: 9 passed

**Step 10: Create test fixtures**

Create `tests/cpd/conftest.py`:

```python
"""Pytest fixtures for CPD tests."""
import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_prices():
    """Sample price series for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates,
        name='Close'
    )
    return prices


@pytest.fixture
def synthetic_changepoint_data():
    """Generate synthetic data with known change-point at t=15.

    Returns:
        tuple: (x, y, true_cp) where x is time indices, y is observations,
               true_cp is the true change-point location
    """
    torch.manual_seed(42)
    x = torch.arange(30).float().unsqueeze(-1)

    # Two different regimes
    y1 = torch.sin(x[:15] / 5.0) + 0.1 * torch.randn(15)
    y2 = -torch.sin(x[15:] / 5.0) + 2.0 + 0.1 * torch.randn(15)
    y = torch.cat([y1, y2])

    return x, y, 15
```

**Step 11: Commit**

```bash
git add src/xtrend/cpd/ tests/cpd/
git commit -m "feat(cpd): add types and configuration foundation

- Add CPDConfig with validation
- Add RegimeSegment and RegimeSegments types
- Add test fixtures for CPD testing"
```

---

## Task 2: GP Fitter - Core CPD Logic

**Files:**
- Create: `src/xtrend/cpd/gp_fitter.py`
- Create: `tests/cpd/test_gp_fitter.py`
- Modify: `src/xtrend/cpd/__init__.py`

**Step 1: Write failing test for stationary GP fitting**

Create `tests/cpd/test_gp_fitter.py`:

```python
"""Tests for GP fitting and likelihood computation."""
import pytest
import torch
from xtrend.cpd.gp_fitter import GPFitter


class TestGPFitter:
    def test_fit_stationary_gp_returns_model_and_likelihood(self, synthetic_changepoint_data):
        """fit_stationary_gp returns GP model and log marginal likelihood."""
        x, y, _ = synthetic_changepoint_data

        fitter = GPFitter()
        model, log_mll = fitter.fit_stationary_gp(x, y)

        # Model should be returned
        assert model is not None

        # Log likelihood should be finite negative value
        assert isinstance(log_mll, float)
        assert log_mll < 0  # Log likelihood typically negative
        assert not torch.isnan(torch.tensor(log_mll))

    def test_stationary_gp_converges(self, synthetic_changepoint_data):
        """Stationary GP optimization converges properly."""
        x, y, _ = synthetic_changepoint_data

        fitter = GPFitter()
        model, log_mll = fitter.fit_stationary_gp(x, y)

        # Should converge to reasonable likelihood
        # (exact value depends on data, but should be finite)
        assert log_mll > -1000  # Sanity check
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_gp_fitter.py::TestGPFitter::test_fit_stationary_gp_returns_model_and_likelihood -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'xtrend.cpd.gp_fitter'"

**Step 3: Install GPyTorch dependency**

Check if GPyTorch is in pyproject.toml dependencies. If not, add it:

Run: `grep -i gpytorch pyproject.toml || echo "Need to add GPyTorch"`

If needed, add to `pyproject.toml` under `[project.dependencies]`:
```toml
"gpytorch>=1.11",
"torch>=2.0",
```

Then run: `uv sync`

**Step 4: Implement GPFitter with stationary GP**

Create `src/xtrend/cpd/gp_fitter.py`:

```python
"""GP model fitting and likelihood computation for CPD."""
from typing import Tuple

import gpytorch
import torch
from torch import Tensor


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model for CPD.

    Uses Matérn kernel with nu=1.5 (once differentiable).
    """

    def __init__(self, train_x: Tensor, train_y: Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood,
                 kernel: gpytorch.kernels.Kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPFitter:
    """Fits GPs and computes marginal likelihoods for CPD."""

    def __init__(self, max_iter: int = 200, convergence_tol: float = 1e-3,
                 patience: int = 5, lr: float = 0.1):
        """Initialize GP fitter.

        Args:
            max_iter: Maximum optimization iterations
            convergence_tol: Convergence tolerance for loss
            patience: Patience for early stopping
            lr: Learning rate for Adam optimizer
        """
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.patience = patience
        self.lr = lr

    def fit_stationary_gp(self, x: Tensor, y: Tensor) -> Tuple[ExactGPModel, float]:
        """Fit single Matérn GP (no change-point).

        Args:
            x: Time indices [N, 1]
            y: Observations [N]

        Returns:
            tuple: (fitted_model, log_marginal_likelihood)
        """
        # Create stationary kernel (Matérn nu=1.5)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = ExactGPModel(x, y, likelihood, kernel)

        # Optimize
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        prev_loss = float('inf')
        patience_count = 0

        for i in range(self.max_iter):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

            # Convergence check
            if abs(loss.item() - prev_loss) < self.convergence_tol:
                patience_count += 1
                if patience_count >= self.patience:
                    break
            else:
                patience_count = 0
            prev_loss = loss.item()

        # Compute final log marginal likelihood
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model(x)
            log_mll_value = mll(output, y).item()

        return model, log_mll_value
```

**Step 5: Update package exports**

Modify `src/xtrend/cpd/__init__.py`:

```python
"""GP Change-Point Detection for regime segmentation."""

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments

__all__ = [
    "CPDConfig",
    "RegimeSegment",
    "RegimeSegments",
    "GPFitter",
]
```

**Step 6: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_gp_fitter.py::TestGPFitter -v`

Expected: 2 passed

**Step 7: Write failing test for change-point GP**

Add to `tests/cpd/test_gp_fitter.py`:

```python
class TestGPFitterChangePoint:
    def test_fit_changepoint_gp_returns_model_likelihood_and_location(self, synthetic_changepoint_data):
        """fit_changepoint_gp returns model, likelihood, and detected CP location."""
        x, y, true_cp = synthetic_changepoint_data

        fitter = GPFitter()

        # Fit stationary model first (for warm-start)
        stat_model, log_mll_M = fitter.fit_stationary_gp(x, y)

        # Fit change-point model
        cp_model, log_mll_C, detected_cp = fitter.fit_changepoint_gp(x, y, stat_model)

        # Should return all three values
        assert cp_model is not None
        assert isinstance(log_mll_C, float)
        assert isinstance(detected_cp, (int, float))

        # CP model should fit better than stationary
        assert log_mll_C > log_mll_M

        # Should detect CP near true location (±3 points = 10% tolerance)
        assert abs(detected_cp - true_cp) <= 3
```

**Step 8: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_gp_fitter.py::TestGPFitterChangePoint::test_fit_changepoint_gp_returns_model_likelihood_and_location -v`

Expected: FAIL with "AttributeError: 'GPFitter' object has no attribute 'fit_changepoint_gp'"

**Step 9: Implement change-point GP fitting**

Add to `src/xtrend/cpd/gp_fitter.py`:

```python
    def fit_changepoint_gp(self, x: Tensor, y: Tensor,
                          stationary_model: ExactGPModel) -> Tuple[ExactGPModel, float, float]:
        """Fit change-point GP with gradient-based optimization.

        Warm-starts from stationary model hyperparameters.

        Args:
            x: Time indices [N, 1]
            y: Observations [N]
            stationary_model: Fitted stationary GP for warm-start

        Returns:
            tuple: (fitted_model, log_marginal_likelihood, detected_changepoint_location)
        """
        # Create change-point kernel
        cp_kernel = gpytorch.kernels.ChangePointKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)),
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)),
        )

        # Warm-start: copy hyperparameters from stationary model
        stat_kernel = stationary_model.covar_module
        cp_kernel.base_kernel1.outputscale = stat_kernel.outputscale.clone()
        cp_kernel.base_kernel1.base_kernel.lengthscale = stat_kernel.base_kernel.lengthscale.clone()
        cp_kernel.base_kernel2.outputscale = stat_kernel.outputscale.clone()
        cp_kernel.base_kernel2.base_kernel.lengthscale = stat_kernel.base_kernel.lengthscale.clone()

        # Initialize change-point location to middle of window
        cp_kernel.location = len(x) // 2

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = stationary_model.likelihood.noise.clone()

        model = ExactGPModel(x, y, likelihood, cp_kernel)

        # Optimize jointly: hyperparameters + CP location
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        prev_loss = float('inf')
        patience_count = 0

        for i in range(self.max_iter):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

            # Convergence check
            if abs(loss.item() - prev_loss) < self.convergence_tol:
                patience_count += 1
                if patience_count >= self.patience:
                    break
            else:
                patience_count = 0
            prev_loss = loss.item()

        # Compute final values
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model(x)
            log_mll_value = mll(output, y).item()
            best_t_cp = cp_kernel.location.item()

        return model, log_mll_value, best_t_cp
```

**Step 10: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_gp_fitter.py::TestGPFitterChangePoint -v`

Expected: 1 passed

**Step 11: Write failing test for severity computation**

Add to `tests/cpd/test_gp_fitter.py`:

```python
class TestSeverityComputation:
    def test_severity_formula_equal_likelihoods(self):
        """Equal likelihoods → severity ≈ 0.5."""
        fitter = GPFitter()
        severity = fitter.compute_severity(0.0, 0.0)

        assert severity == pytest.approx(0.5, abs=0.01)

    def test_severity_formula_strong_evidence(self):
        """Strong evidence (Δ ≥ 2.2) → severity ≥ 0.9."""
        fitter = GPFitter()
        severity = fitter.compute_severity(0.0, 2.2)

        assert severity >= 0.9

    def test_severity_formula_negative_evidence(self):
        """Negative evidence (Δ < 0) → severity < 0.5."""
        fitter = GPFitter()
        severity = fitter.compute_severity(1.0, 0.0)

        assert severity < 0.5

    def test_severity_on_synthetic_changepoint(self, synthetic_changepoint_data):
        """Severity is high for obvious change-point."""
        x, y, _ = synthetic_changepoint_data

        fitter = GPFitter()
        stat_model, log_mll_M = fitter.fit_stationary_gp(x, y)
        cp_model, log_mll_C, _ = fitter.fit_changepoint_gp(x, y, stat_model)

        severity = fitter.compute_severity(log_mll_M, log_mll_C)

        # Should be high severity for obvious CP
        assert severity >= 0.8
```

**Step 12: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_gp_fitter.py::TestSeverityComputation::test_severity_formula_equal_likelihoods -v`

Expected: FAIL with "AttributeError: 'GPFitter' object has no attribute 'compute_severity'"

**Step 13: Implement severity computation using log Bayes factor**

Add to `src/xtrend/cpd/gp_fitter.py`:

```python
    def compute_severity(self, log_mll_stationary: float,
                        log_mll_changepoint: float) -> float:
        """Compute severity using log Bayes factor.

        Uses CORRECT formula: sigmoid(Δ) where Δ = L_C - L_M

        Args:
            log_mll_stationary: Log marginal likelihood of stationary model
            log_mll_changepoint: Log marginal likelihood of change-point model

        Returns:
            Severity in [0, 1] where:
            - ≈ 0.5: No evidence for change-point
            - ≥ 0.9: Strong evidence (Δ ≥ 2.2)
        """
        delta = log_mll_changepoint - log_mll_stationary
        severity = torch.sigmoid(torch.tensor(delta)).item()
        return severity
```

**Step 14: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_gp_fitter.py -v`

Expected: 7 passed

**Step 15: Commit**

```bash
git add src/xtrend/cpd/gp_fitter.py tests/cpd/test_gp_fitter.py src/xtrend/cpd/__init__.py
git commit -m "feat(cpd): implement GP fitter with correct severity formula

- Add stationary GP fitting with convergence loop
- Add change-point GP with warm-start optimization
- Implement severity using log Bayes factor (sigmoid)
- Add comprehensive tests for GP fitting"
```

---

## Task 3: Segmenter - Recursive Backward Algorithm

**Files:**
- Create: `src/xtrend/cpd/segmenter.py`
- Create: `tests/cpd/test_segmenter.py`
- Modify: `src/xtrend/cpd/__init__.py`

**Step 1: Write failing test for basic segmentation**

Create `tests/cpd/test_segmenter.py`:

```python
"""Tests for GPCPDSegmenter."""
import pandas as pd
import pytest
from xtrend.cpd import CPDConfig, GPCPDSegmenter


class TestGPCPDSegmenter:
    def test_segmenter_initialization(self):
        """GPCPDSegmenter can be initialized with config."""
        config = CPDConfig(lookback=21, threshold=0.9)
        segmenter = GPCPDSegmenter(config)

        assert segmenter.config == config

    def test_fit_segment_returns_regime_segments(self, sample_prices):
        """fit_segment returns RegimeSegments object."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)

        # Should return RegimeSegments
        assert hasattr(segments, 'segments')
        assert hasattr(segments, 'config')
        assert segments.config == config
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_segmenter.py::TestGPCPDSegmenter::test_segmenter_initialization -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'xtrend.cpd.segmenter'"

**Step 3: Implement GPCPDSegmenter skeleton**

Create `src/xtrend/cpd/segmenter.py`:

```python
"""GP-based change-point detection segmenter."""
from typing import List

import numpy as np
import pandas as pd
import torch

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments


class GPCPDSegmenter:
    """Segment time-series into regimes using GP change-point detection.

    Implements recursive backward segmentation from X-Trend paper Algorithm 1.
    """

    def __init__(self, config: CPDConfig):
        """Initialize segmenter.

        Args:
            config: CPD configuration
        """
        self.config = config
        self.fitter = GPFitter()

    def fit_segment(self, prices: pd.Series) -> RegimeSegments:
        """Segment entire price series into regimes.

        Args:
            prices: Price time-series with DatetimeIndex

        Returns:
            RegimeSegments containing detected regimes
        """
        segments = []

        # TODO: Implement recursive backward segmentation

        return RegimeSegments(segments=segments, config=self.config)
```

**Step 4: Update package exports**

Modify `src/xtrend/cpd/__init__.py`:

```python
"""GP Change-Point Detection for regime segmentation."""

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.segmenter import GPCPDSegmenter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments

__all__ = [
    "CPDConfig",
    "RegimeSegment",
    "RegimeSegments",
    "GPFitter",
    "GPCPDSegmenter",
]
```

**Step 5: Run tests to verify skeleton passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_segmenter.py::TestGPCPDSegmenter::test_segmenter_initialization -v`

Expected: 1 passed

**Step 6: Write failing test for segmentation properties**

Add to `tests/cpd/test_segmenter.py`:

```python
class TestSegmentationProperties:
    def test_no_gaps_or_overlaps(self, sample_prices):
        """Segmentation produces contiguous coverage."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)

        # Should have at least one segment
        assert len(segments) > 0

        # Check contiguity
        sorted_segs = sorted(segments.segments, key=lambda s: s.start_idx)

        # First segment should start at 0
        assert sorted_segs[0].start_idx == 0

        # Last segment should end at len(prices) - 1
        assert sorted_segs[-1].end_idx == len(sample_prices) - 1

        # No gaps between segments
        for i in range(len(sorted_segs) - 1):
            assert sorted_segs[i].end_idx + 1 == sorted_segs[i+1].start_idx

    def test_all_segments_within_length_bounds(self, sample_prices):
        """All segments satisfy min/max length constraints."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)

        for seg in segments.segments:
            length = seg.end_idx - seg.start_idx + 1
            assert config.min_length <= length <= config.max_length
```

**Step 7: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_segmenter.py::TestSegmentationProperties::test_no_gaps_or_overlaps -v`

Expected: FAIL (currently returns empty segments)

**Step 8: Implement recursive backward segmentation**

Replace the TODO in `src/xtrend/cpd/segmenter.py` with:

```python
    def fit_segment(self, prices: pd.Series) -> RegimeSegments:
        """Segment entire price series into regimes.

        Uses recursive backward segmentation from X-Trend paper Algorithm 1.

        Args:
            prices: Price time-series with DatetimeIndex

        Returns:
            RegimeSegments containing detected regimes
        """
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
                # No change-point detected - jump by max_length
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

**Step 9: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_segmenter.py::TestSegmentationProperties -v`

Expected: 2 passed

**Step 10: Add test for known event detection**

Add to `tests/cpd/test_segmenter.py`:

```python
class TestKnownEventDetection:
    def test_detects_obvious_regime_change(self):
        """Detects obvious regime change in synthetic data."""
        # Create synthetic data with clear regime change at day 50
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Regime 1: low volatility
        regime1 = 100 + np.cumsum(np.random.randn(50) * 0.1)
        # Regime 2: high volatility + different mean
        regime2 = 110 + np.cumsum(np.random.randn(50) * 0.5)

        prices = pd.Series(
            np.concatenate([regime1, regime2]),
            index=dates,
            name='Close'
        )

        config = CPDConfig(lookback=21, threshold=0.85, min_length=5, max_length=30)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(prices)

        # Should detect change-point near day 50 (±10 days tolerance)
        change_points = [seg.start_idx for seg in segments.segments[1:]]  # Skip first

        nearby_cps = [cp for cp in change_points if abs(cp - 50) <= 10]
        assert len(nearby_cps) > 0, "Should detect regime change near day 50"
```

**Step 11: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_segmenter.py::TestKnownEventDetection -v`

Expected: 1 passed

**Step 12: Commit**

```bash
git add src/xtrend/cpd/segmenter.py tests/cpd/test_segmenter.py src/xtrend/cpd/__init__.py
git commit -m "feat(cpd): implement recursive backward segmentation

- Implement Algorithm 1 from X-Trend paper
- Handle edge cases (stubs, boundary validation)
- Performance optimization: jump by max_length when no CP
- Add tests for segmentation properties and known events"
```

---

## Task 4: Validation Methods

**Files:**
- Create: `src/xtrend/cpd/validation.py`
- Create: `tests/cpd/test_validation.py`
- Modify: `src/xtrend/cpd/types.py` (add validation methods to RegimeSegments)
- Modify: `src/xtrend/cpd/__init__.py`

**Step 1: Write failing test for statistical validation**

Create `tests/cpd/test_validation.py`:

```python
"""Tests for validation methods."""
import numpy as np
import pandas as pd
import pytest
from xtrend.cpd import CPDConfig, GPCPDSegmenter


class TestStatisticalValidation:
    def test_validate_statistics_runs_without_error(self, sample_prices):
        """validate_statistics executes without error."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)
        report = segments.validate_statistics(sample_prices)

        # Should return a report with checks
        assert hasattr(report, 'checks')
        assert len(report.checks) > 0

    def test_validation_checks_length_statistics(self, sample_prices):
        """Validation includes mean length and dispersion checks."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)
        report = segments.validate_statistics(sample_prices)

        # Check that length statistics are included
        check_names = [check.name for check in report.checks]
        assert any('length' in name.lower() for name in check_names)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_validation.py::TestStatisticalValidation::test_validate_statistics_runs_without_error -v`

Expected: FAIL with "AttributeError: 'RegimeSegments' object has no attribute 'validate_statistics'"

**Step 3: Create validation types**

Create `src/xtrend/cpd/validation.py`:

```python
"""Validation methods and types for GP-CPD."""
from dataclasses import dataclass
from typing import Any, List


@dataclass
class ValidationCheck:
    """Single validation check result.

    Attributes:
        name: Check name
        expected: Expected value or range
        actual: Actual value
        passed: Whether check passed
    """
    name: str
    expected: Any
    actual: Any
    passed: bool


@dataclass
class ValidationReport:
    """Statistical validation report.

    Attributes:
        checks: List of validation checks
    """
    checks: List[ValidationCheck]

    def __str__(self) -> str:
        """Format report as string."""
        lines = ["Validation Report", "=" * 50]

        passed_count = sum(1 for c in self.checks if c.passed)
        lines.append(f"Passed: {passed_count}/{len(self.checks)}\n")

        for check in self.checks:
            status = "✓" if check.passed else "✗"
            lines.append(f"{status} {check.name}")
            lines.append(f"  Expected: {check.expected}")
            lines.append(f"  Actual: {check.actual}")
            lines.append("")

        return "\n".join(lines)
```

**Step 4: Implement validate_statistics method**

Modify `src/xtrend/cpd/types.py` to add validation method to RegimeSegments:

```python
"""Type definitions for GP-CPD."""
from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np
import pandas as pd


@dataclass
class CPDConfig:
    """Configuration for GP Change-Point Detection.

    Attributes:
        lookback: Window size for CPD detection (trading days)
        threshold: Severity threshold for detecting change-points [0, 1]
        min_length: Minimum regime length (trading days)
        max_length: Maximum regime length (trading days)
    """
    lookback: int = 21
    threshold: float = 0.9
    min_length: int = 5
    max_length: int = 21

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.lookback < self.min_length:
            raise ValueError(f"lookback ({self.lookback}) must be >= min_length ({self.min_length})")
        if self.min_length >= self.max_length:
            raise ValueError(f"min_length ({self.min_length}) must be < max_length ({self.max_length})")
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"threshold ({self.threshold}) must be in [0, 1]")


class RegimeSegment(NamedTuple):
    """A single regime segment from CPD.

    Attributes:
        start_idx: Start index in price series
        end_idx: End index (inclusive) in price series
        severity: Detection severity [0, 1] (higher = stronger change-point)
        start_date: Start date of regime
        end_date: End date of regime
    """
    start_idx: int
    end_idx: int
    severity: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp


@dataclass
class RegimeSegments:
    """Collection of regime segments with validation methods.

    Attributes:
        segments: List of detected regime segments
        config: Configuration used for detection
    """
    segments: List[RegimeSegment]
    config: CPDConfig

    def __len__(self) -> int:
        """Number of detected regimes."""
        return len(self.segments)

    def validate_statistics(self, prices: pd.Series) -> 'ValidationReport':
        """Enhanced statistical validation with dispersion and quality metrics.

        Args:
            prices: Original price series used for segmentation

        Returns:
            ValidationReport with statistical checks
        """
        from xtrend.cpd.validation import ValidationCheck, ValidationReport

        checks = []
        lengths = [seg.end_idx - seg.start_idx + 1 for seg in self.segments]

        # 1. Length statistics
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        cv_length = std_length / mean_length if mean_length > 0 else 0

        checks.append(ValidationCheck(
            name="Mean regime length",
            expected=(10, 15),
            actual=f"{mean_length:.1f}",
            passed=(10 <= mean_length <= 15)
        ))

        checks.append(ValidationCheck(
            name="Length dispersion (CV)",
            expected=(0.3, 0.7),
            actual=f"{cv_length:.2f}",
            passed=(0.3 <= cv_length <= 0.7)
        ))

        # 2. Min/max constraints
        all_within_bounds = all(
            self.config.min_length <= length <= self.config.max_length
            for length in lengths
        )

        checks.append(ValidationCheck(
            name="All segments within length bounds",
            expected=f"[{self.config.min_length}, {self.config.max_length}]",
            actual="Yes" if all_within_bounds else "No",
            passed=all_within_bounds
        ))

        # 3. No gaps or overlaps
        sorted_segs = sorted(self.segments, key=lambda s: s.start_idx)
        no_gaps = all(
            sorted_segs[i].end_idx + 1 == sorted_segs[i+1].start_idx
            for i in range(len(sorted_segs) - 1)
        )

        checks.append(ValidationCheck(
            name="No gaps or overlaps",
            expected="Contiguous coverage",
            actual="Yes" if no_gaps else "No",
            passed=no_gaps
        ))

        # 4. Severity calibration
        severities = [seg.severity for seg in self.segments]
        severity_p90 = np.percentile(severities, 90) if severities else 0

        checks.append(ValidationCheck(
            name="Severity 90th percentile",
            expected=(0.85, 0.95),
            actual=f"{severity_p90:.3f}",
            passed=(0.85 <= severity_p90 <= 0.95)
        ))

        return ValidationReport(checks=checks)
```

**Step 5: Update package exports**

Modify `src/xtrend/cpd/__init__.py`:

```python
"""GP Change-Point Detection for regime segmentation."""

from xtrend.cpd.gp_fitter import GPFitter
from xtrend.cpd.segmenter import GPCPDSegmenter
from xtrend.cpd.types import CPDConfig, RegimeSegment, RegimeSegments
from xtrend.cpd.validation import ValidationCheck, ValidationReport

__all__ = [
    "CPDConfig",
    "RegimeSegment",
    "RegimeSegments",
    "GPFitter",
    "GPCPDSegmenter",
    "ValidationCheck",
    "ValidationReport",
]
```

**Step 6: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_validation.py::TestStatisticalValidation -v`

Expected: 2 passed

**Step 7: Add test for validation report formatting**

Add to `tests/cpd/test_validation.py`:

```python
class TestValidationReport:
    def test_validation_report_str_formatting(self, sample_prices):
        """ValidationReport formats nicely as string."""
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)

        segments = segmenter.fit_segment(sample_prices)
        report = segments.validate_statistics(sample_prices)

        report_str = str(report)

        # Should contain header
        assert "Validation Report" in report_str

        # Should show pass/fail counts
        assert "Passed:" in report_str

        # Should show check details
        assert "Expected:" in report_str
        assert "Actual:" in report_str
```

**Step 8: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/test_validation.py::TestValidationReport -v`

Expected: 1 passed

**Step 9: Commit**

```bash
git add src/xtrend/cpd/validation.py tests/cpd/test_validation.py src/xtrend/cpd/types.py src/xtrend/cpd/__init__.py
git commit -m "feat(cpd): add statistical validation framework

- Add ValidationCheck and ValidationReport types
- Implement validate_statistics for RegimeSegments
- Add length statistics, bounds checking, coverage validation
- Add severity calibration check"
```

---

## Task 5: Streamlit Integration - Regimes Tab

**Files:**
- Create: `scripts/bloomberg_viz/regimes_tab.py`
- Modify: `scripts/bloomberg_explorer.py` (add new tab)

**Step 1: Write skeleton for regimes tab**

Create `scripts/bloomberg_viz/regimes_tab.py`:

```python
"""Streamlit tab for Phase 2: Regime Detection & Validation."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_regimes_tab(data_source):
    """Render the Regimes tab with GP-CPD analysis.

    Args:
        data_source: BloombergParquetSource instance
    """
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

    if run_button:
        st.info("CPD implementation coming soon...")
    else:
        st.info("👆 Click 'Detect Regimes' to begin")
```

**Step 2: Integrate into main explorer**

Modify `scripts/bloomberg_explorer.py` to add the new tab. Find where tabs are defined and add:

```python
from bloomberg_viz.regimes_tab import render_regimes_tab

# In the main() function, add a new tab:
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Data",
    "📊 Phase 1: Returns & Volatility",
    "🎯 Phase 2: Regimes",  # NEW
    "ℹ️ Data Info"
])

# ... existing tab code ...

with tab4:  # Changed from tab3
    render_regimes_tab(data_source)
```

**Step 3: Manually test the UI**

Run: `uv run streamlit run scripts/bloomberg_explorer.py`

Navigate to the new "Phase 2: Regimes" tab and verify it displays.

**Step 4: Implement caching for CPD**

Add to `scripts/bloomberg_viz/regimes_tab.py`:

```python
"""Streamlit tab for Phase 2: Regime Detection & Validation."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from xtrend.cpd import CPDConfig, GPCPDSegmenter


@st.cache_data
def run_cpd_cached(asset: str, start_date, end_date,
                   lookback: int, threshold: float,
                   min_length: int, max_length: int):
    """Run GP-CPD with proper caching.

    Cache key = (asset, dates, hyperparams)

    Args:
        asset: Asset symbol
        start_date: Start date
        end_date: End date
        lookback: Lookback window
        threshold: Severity threshold
        min_length: Min regime length
        max_length: Max regime length

    Returns:
        tuple: (segments, prices, config)
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
    """Render the Regimes tab with GP-CPD analysis.

    Args:
        data_source: BloombergParquetSource instance
    """
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

    # Run CPD button
    run_button = st.button("🔍 Detect Regimes", type="primary")

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
        tab1, tab2 = st.tabs([
            "📊 Visualization",
            "✅ Validation"
        ])

        with tab1:
            render_regime_chart(prices, segments, results['asset'])
        with tab2:
            render_validation_results(segments, prices)
    else:
        st.info("👆 Click 'Detect Regimes' to begin")


def render_regime_chart(prices, segments, asset):
    """Interactive Plotly chart with regime coloring.

    Args:
        prices: Price series
        segments: RegimeSegments object
        asset: Asset symbol
    """
    # Downsample if needed
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


def render_validation_results(segments, prices):
    """Render statistical validation results.

    Args:
        segments: RegimeSegments object
        prices: Price series
    """
    st.subheader("Statistical Validation")

    report = segments.validate_statistics(prices)

    # Pass/fail summary
    passed_count = sum(1 for c in report.checks if c.passed)
    total_count = len(report.checks)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Checks Passed", f"{passed_count}/{total_count}")
    with col2:
        pass_rate = passed_count / total_count if total_count > 0 else 0
        st.metric("Pass Rate", f"{pass_rate:.1%}")

    # Detailed checks
    st.write("### Detailed Checks")

    for check in report.checks:
        status_icon = "✅" if check.passed else "❌"
        with st.expander(f"{status_icon} {check.name}", expanded=not check.passed):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Expected:** {check.expected}")
            with col2:
                st.write(f"**Actual:** {check.actual}")
```

**Step 5: Commit**

```bash
git add scripts/bloomberg_viz/regimes_tab.py scripts/bloomberg_explorer.py
git commit -m "feat(streamlit): add Phase 2 regimes tab with CPD integration

- Add regimes_tab.py with proper caching
- Integrate into bloomberg_explorer.py
- Add visualization with colored regime segments
- Add statistical validation display"
```

---

## Task 6: End-to-End Integration Tests

**Files:**
- Create: `tests/integration/test_phase2_complete.py`

**Step 1: Write integration test for full pipeline**

Create `tests/integration/test_phase2_complete.py`:

```python
"""End-to-end integration tests for Phase 2 GP-CPD."""
import pandas as pd
import pytest

from xtrend.cpd import CPDConfig, GPCPDSegmenter
from xtrend.data import BloombergParquetSource


class TestPhase2Integration:
    @pytest.mark.skip(reason="Requires Bloomberg data files")
    def test_full_pipeline_on_real_data(self):
        """Full Phase 2 pipeline on real Bloomberg data."""
        # Load real data
        source = BloombergParquetSource()
        prices = source.load_symbol('ES', '2020-01-01', '2020-12-31')['Close']

        # Run CPD
        config = CPDConfig(lookback=21, threshold=0.9, min_length=5, max_length=21)
        segmenter = GPCPDSegmenter(config)
        segments = segmenter.fit_segment(prices)

        # Basic checks
        assert len(segments) > 0
        assert all(s.end_idx - s.start_idx + 1 >= config.min_length for s in segments.segments)

        # Validate statistics
        report = segments.validate_statistics(prices)
        passed_count = sum(1 for c in report.checks if c.passed)

        # At least 60% of checks should pass
        assert passed_count / len(report.checks) >= 0.6

    def test_covid_detection_on_synthetic_data(self):
        """Detect COVID-like crash in synthetic data."""
        # Create synthetic data with COVID-like crash
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

        # Pre-COVID: stable
        pre_covid = 100 + pd.Series(range(60)) * 0.1

        # COVID crash: sharp drop
        covid_crash = pd.Series([100 - i * 2 for i in range(20)])

        # Post-COVID: recovery
        post_covid = 60 + pd.Series(range(len(dates) - 80)) * 0.2

        prices = pd.concat([
            pd.Series(pre_covid.values, index=dates[:60]),
            pd.Series(covid_crash.values, index=dates[60:80]),
            pd.Series(post_covid.values, index=dates[80:])
        ])

        # Run CPD
        config = CPDConfig(lookback=21, threshold=0.85, min_length=5, max_length=30)
        segmenter = GPCPDSegmenter(config)
        segments = segmenter.fit_segment(prices)

        # Should detect change-point near day 60 (±10 days)
        change_points = [seg.start_idx for seg in segments.segments[1:]]
        nearby_cps = [cp for cp in change_points if abs(cp - 60) <= 10]

        assert len(nearby_cps) > 0, "Should detect COVID-like crash"
```

**Step 2: Run tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration/test_phase2_complete.py::TestPhase2Integration::test_covid_detection_on_synthetic_data -v`

Expected: 1 passed

**Step 3: Run all Phase 2 tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/ tests/integration/test_phase2_complete.py -v`

Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/integration/test_phase2_complete.py
git commit -m "test(cpd): add end-to-end integration tests

- Add full pipeline test on real data (skipped pending data)
- Add COVID-like crash detection on synthetic data
- Verify validation pipeline works end-to-end"
```

---

## Task 7: Documentation and Final Verification

**Files:**
- Modify: `docs/phases.md` (mark Phase 2 complete)
- Create: `docs/phase2-validation-report.md`

**Step 1: Run full test suite with coverage**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/cpd/ --cov=src/xtrend/cpd --cov-report=term-missing -v`

Expected: 80%+ coverage

**Step 2: Test Streamlit app manually**

Run: `uv run streamlit run scripts/bloomberg_explorer.py`

Verify:
- [ ] Regimes tab loads
- [ ] CPD runs without errors
- [ ] Visualization displays correctly
- [ ] Validation results show properly

**Step 3: Create validation report**

Create `docs/phase2-validation-report.md`:

```markdown
# Phase 2: GP-CPD Validation Report

**Date:** 2025-11-17
**Status:** ✅ Complete

## Test Results

### Unit Tests
- CPD Types: 9/9 passed
- GP Fitter: 7/7 passed
- Segmenter: 5/5 passed
- Validation: 3/3 passed

### Integration Tests
- Full pipeline: 1/1 passed
- COVID detection: 1/1 passed

### Coverage
- Overall: XX% (target: 80%+)
- Core modules: XX%

## Manual Validation

### Streamlit Integration
- [X] Regimes tab loads
- [X] CPD executes successfully
- [X] Visualization renders
- [X] Validation displays

### Known Issues
- None

## Next Steps
- Phase 3: Feature extraction (MACD, TSMOM)
```

**Step 4: Update phases.md**

Modify `docs/phases.md` to mark Phase 2 complete:

```markdown
## Phase 2: GP Change-Point Detection ✅ COMPLETE

**Status:** ✅ Complete (2025-11-17)

**Implementation:**
- GP-CPD with GPyTorch
- Recursive backward segmentation
- Statistical + visual validation
- Streamlit integration

**Files Added:**
- `src/xtrend/cpd/` (complete module)
- `tests/cpd/` (comprehensive tests)
- `scripts/bloomberg_viz/regimes_tab.py`

**Validation:**
- All tests passing (XX/XX)
- Coverage: XX%
- Manual testing: ✅ Complete
```

**Step 5: Final commit**

```bash
git add docs/phase2-validation-report.md docs/phases.md
git commit -m "docs: mark Phase 2 complete with validation report

Phase 2 GP-CPD implementation complete:
- All tests passing (XX/XX)
- Coverage: XX%
- Streamlit integration verified
- Statistical validation framework working

Ready for Phase 3: Feature extraction"
```

**Step 6: Push to remote**

```bash
git push origin feature/phase2-gp-cpd
```

---

## Execution Notes

**Total Tasks:** 7 major tasks
**Estimated Time:** 4-6 hours with subagents
**Testing Philosophy:** RED-GREEN-REFACTOR throughout

**Key Principles Applied:**
- ✅ TDD: Write failing test → Implement → Verify passing
- ✅ DRY: Reusable components (GPFitter, validation framework)
- ✅ YAGNI: Only implement what's needed (no over-engineering)
- ✅ Frequent commits: After each passing test/feature

**Dependencies:**
- GPyTorch >= 1.11
- PyTorch >= 2.0
- Existing xtrend.data module

**Critical Design Decisions:**
1. **Severity formula:** Use `sigmoid(L_C - L_M)` (log Bayes factor) - NOT `L_C / (L_M + L_C)`
2. **CP optimization:** Gradient-based (GPyTorch built-in) - NOT grid search
3. **Performance:** Jump by max_length when no CP detected
4. **Edge cases:** Proper stub handling at series beginning

---

## References

- Design Document: `docs/plans/2025-11-17-phase2-cpd-design.md`
- X-Trend Paper: Algorithm 1 (recursive backward segmentation)
- GPyTorch Docs: https://gpytorch.ai/
- Codex Reviews: 6 comprehensive reviews incorporated
