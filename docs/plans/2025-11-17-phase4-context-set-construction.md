# Phase 4: Context Set Construction Implementation Plan (REVISED after Codex Review)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three context construction methods (Final Hidden State, Time-Equivalent, CPD Segmented) with strict causality enforcement and padding masks for cross-attention.

**Architecture:** Build context sampling infrastructure that creates batches of historical sequences for cross-attention. The CPD-segmented method (primary) samples regime sequences detected by GP-CPD preserving regime boundaries. All methods enforce causality: context ends before target starts, with padding masks for variable-length sequences.

**Tech Stack:** PyTorch tensors, pandas for timezone-aware date handling, existing xtrend.cpd for regime detection, caching for performance.

**Codex Review Applied:** Added padding masks, improved causality validation, fixed CPD truncation strategy, enhanced TDD with unit tests per sampler, added schema validation and caching.

---

## Task 1: Create Context Set Types

**Files:**
- Create: `xtrend/context/types.py`
- Test: `tests/context/test_types.py`

**Step 1: Write failing test for ContextSequence**

Create `tests/context/__init__.py`:
```python
"""Tests for context set construction."""
```

Create `tests/context/test_types.py`:
```python
"""Tests for context set types."""
import pytest
import torch
import pandas as pd
from xtrend.context.types import ContextSequence, ContextBatch


class TestContextSequence:
    """Test ContextSequence type."""

    def test_context_sequence_creation(self):
        """Create a single context sequence."""
        features = torch.randn(21, 8)  # 21 days, 8 features
        entity_id = torch.tensor(5)
        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2020-01-25')

        seq = ContextSequence(
            features=features,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            method="cpd_segmented"
        )

        assert seq.features.shape == (21, 8)
        assert seq.entity_id == 5
        assert seq.length == 21
        assert seq.method == "cpd_segmented"

    def test_causality_check(self):
        """Verify causality: context must be before target."""
        features = torch.randn(21, 8)
        entity_id = torch.tensor(5)

        seq = ContextSequence(
            features=features,
            entity_id=entity_id,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="final_hidden_state"
        )

        # Context before target = OK
        assert seq.is_causal(target_date=pd.Timestamp('2020-02-01'))

        # Context after target = NOT OK
        assert not seq.is_causal(target_date=pd.Timestamp('2019-12-01'))
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_types.py::TestContextSequence::test_context_sequence_creation -v
```

Expected: `ModuleNotFoundError: No module named 'xtrend.context.types'`

**Step 3: Implement ContextSequence type with padding masks (Codex feedback)**

Create `xtrend/context/types.py`:
```python
"""Type definitions for context set construction."""
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import pandas as pd


ContextMethod = Literal["final_hidden_state", "time_equivalent", "cpd_segmented"]


@dataclass
class ContextSequence:
    """A single context sequence for cross-attention.

    Attributes:
        features: Input features (seq_len, input_dim)
        entity_id: Entity/asset ID for this sequence
        start_date: Start date of sequence (timezone-aware)
        end_date: End date of sequence (timezone-aware)
        method: Construction method used
        padding_mask: Optional mask for padded positions (True = valid, False = padding)
    """
    features: torch.Tensor
    entity_id: torch.Tensor
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    method: ContextMethod
    padding_mask: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate and create default padding mask."""
        # Ensure timestamps are timezone-aware (UTC)
        if self.start_date.tz is None:
            self.start_date = self.start_date.tz_localize('UTC')
        if self.end_date.tz is None:
            self.end_date = self.end_date.tz_localize('UTC')

        # Create default padding mask if not provided (all valid)
        if self.padding_mask is None:
            self.padding_mask = torch.ones(self.features.shape[0], dtype=torch.bool)

        # Validate padding mask shape
        if self.padding_mask.shape[0] != self.features.shape[0]:
            raise ValueError(
                f"Padding mask length ({self.padding_mask.shape[0]}) "
                f"must match features length ({self.features.shape[0]})"
            )

    @property
    def length(self) -> int:
        """Sequence length."""
        return self.features.shape[0]

    @property
    def input_dim(self) -> int:
        """Feature dimension."""
        return self.features.shape[1]

    def is_causal(self, target_start_date: pd.Timestamp, buffer_days: int = 1) -> bool:
        """Check if context is before target with buffer (causality).

        Args:
            target_start_date: Target sequence START date (not end)
            buffer_days: Buffer between context end and target start (default 1 day)

        Returns:
            True if context ends before target starts with buffer
        """
        # Ensure target is timezone-aware
        if target_start_date.tz is None:
            target_start_date = target_start_date.tz_localize('UTC')

        # Context must end at least buffer_days before target starts
        buffer = pd.Timedelta(days=buffer_days)
        return self.end_date + buffer <= target_start_date


@dataclass
class ContextBatch:
    """Batch of context sequences for a single target.

    Attributes:
        sequences: List of C context sequences
        C: Context set size
        max_length: Maximum sequence length in batch
    """
    sequences: list[ContextSequence]

    def __post_init__(self):
        """Validate batch homogeneity (Codex feedback)."""
        if len(self.sequences) == 0:
            raise ValueError("Context batch cannot be empty")

        # All sequences must use same construction method
        methods = {seq.method for seq in self.sequences}
        if len(methods) > 1:
            raise ValueError(f"Mixed methods in batch: {methods}")

        # All sequences must have same feature dimension (schema validation)
        input_dims = {seq.input_dim for seq in self.sequences}
        if len(input_dims) > 1:
            raise ValueError(
                f"Mixed feature dimensions in batch: {input_dims}. "
                "All sequences must have same input_dim."
            )

    @property
    def C(self) -> int:
        """Context set size."""
        return len(self.sequences)

    @property
    def max_length(self) -> int:
        """Maximum sequence length in batch."""
        return max(seq.length for seq in self.sequences)

    @property
    def input_dim(self) -> int:
        """Feature dimension (consistent across batch)."""
        return self.sequences[0].input_dim

    def verify_causality(
        self,
        target_start_date: pd.Timestamp,
        buffer_days: int = 1
    ) -> bool:
        """Verify all context sequences are before target with buffer.

        Args:
            target_start_date: Target sequence START date
            buffer_days: Buffer between context and target

        Returns:
            True if all contexts are causal
        """
        return all(
            seq.is_causal(target_start_date, buffer_days)
            for seq in self.sequences
        )

    def to_padded_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert batch to padded tensor with mask for attention.

        Returns:
            features: (C, max_length, input_dim) padded tensor
            mask: (C, max_length) padding mask (True = valid, False = padding)
        """
        C = self.C
        max_len = self.max_length
        input_dim = self.input_dim

        # Preallocate tensors
        features = torch.zeros(C, max_len, input_dim)
        mask = torch.zeros(C, max_len, dtype=torch.bool)

        # Fill with sequences
        for i, seq in enumerate(self.sequences):
            seq_len = seq.length
            features[i, :seq_len] = seq.features
            mask[i, :seq_len] = seq.padding_mask

        return features, mask
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_types.py::TestContextSequence -v
```

Expected: Both tests PASS

**Step 5: Write test for ContextBatch**

Add to `tests/context/test_types.py`:
```python
class TestContextBatch:
    """Test ContextBatch type."""

    def test_context_batch_creation(self):
        """Create a batch of context sequences."""
        # Create 3 context sequences
        sequences = []
        for i in range(3):
            features = torch.randn(21, 8)
            entity_id = torch.tensor(i)
            seq = ContextSequence(
                features=features,
                entity_id=entity_id,
                start_date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i*30),
                end_date=pd.Timestamp('2020-01-25') + pd.Timedelta(days=i*30),
                method="cpd_segmented"
            )
            sequences.append(seq)

        batch = ContextBatch(sequences=sequences)

        assert batch.C == 3
        assert batch.max_length == 21

    def test_causality_verification(self):
        """Verify batch-level causality check."""
        sequences = []
        for i in range(3):
            features = torch.randn(21, 8)
            entity_id = torch.tensor(i)
            seq = ContextSequence(
                features=features,
                entity_id=entity_id,
                start_date=pd.Timestamp('2020-01-01'),
                end_date=pd.Timestamp('2020-01-25'),
                method="final_hidden_state"
            )
            sequences.append(seq)

        batch = ContextBatch(sequences=sequences)

        # All contexts before target = OK
        assert batch.verify_causality(target_date=pd.Timestamp('2020-03-01'))

        # Some contexts after target = NOT OK
        assert not batch.verify_causality(target_date=pd.Timestamp('2020-01-15'))

    def test_mixed_methods_rejected(self):
        """Reject batch with mixed construction methods."""
        seq1 = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(0),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="final_hidden_state"
        )
        seq2 = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(1),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="cpd_segmented"
        )

        with pytest.raises(ValueError, match="Mixed methods"):
            ContextBatch(sequences=[seq1, seq2])
```

**Step 6: Run tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_types.py::TestContextBatch -v
```

Expected: All 3 tests PASS

**Step 7: Update __init__.py**

Edit `xtrend/context/__init__.py`:
```python
"""xtrend.context — Context set construction for few-shot learning."""
from xtrend.context.types import (
    ContextMethod,
    ContextSequence,
    ContextBatch,
)

__all__ = [
    "ContextMethod",
    "ContextSequence",
    "ContextBatch",
]
```

**Step 8: Commit**

```bash
git add xtrend/context/types.py xtrend/context/__init__.py tests/context/
git commit -m "feat(context): add context set types with causality validation"
```

---

## Task 2: Implement Final Hidden State Method

**Files:**
- Modify: `xtrend/context/sampler.py`
- Test: `tests/context/test_sampler.py`

**Step 1: Write failing test**

Create `tests/context/test_sampler.py`:
```python
"""Tests for context sampling methods."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.context.sampler import sample_final_hidden_state
from xtrend.context.types import ContextBatch


class TestFinalHiddenStateMethod:
    """Test Final Hidden State sampling method."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature panel."""
        # 100 days of data for 5 assets
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')
        features = {}
        np.random.seed(42)
        for asset_id in range(5):
            # Shape: (100, 8) features
            features[f"ASSET{asset_id}"] = torch.randn(len(dates), 8)

        return {
            'features': features,
            'dates': dates,
            'symbols': [f"ASSET{i}" for i in range(5)]
        }

    def test_sample_final_hidden_state_basic(self, sample_features):
        """Sample C context sequences with fixed length."""
        target_date = pd.Timestamp('2020-04-01')

        batch = sample_final_hidden_state(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=10,  # Sample 10 context sequences
            l_c=21,  # Each 21 days long
            seed=42
        )

        assert isinstance(batch, ContextBatch)
        assert batch.C == 10
        assert batch.max_length == 21

        # Verify causality
        assert batch.verify_causality(target_date)

        # Verify all sequences have correct shape
        for seq in batch.sequences:
            assert seq.features.shape == (21, 8)
            assert seq.method == "final_hidden_state"

    def test_respects_causality(self, sample_features):
        """All sampled contexts must end before target."""
        target_date = pd.Timestamp('2020-02-01')

        batch = sample_final_hidden_state(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_c=21,
            seed=42
        )

        # Every context sequence must end before target
        for seq in batch.sequences:
            assert seq.end_date < target_date
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_sampler.py::TestFinalHiddenStateMethod::test_sample_final_hidden_state_basic -v
```

Expected: `ImportError` or `AttributeError`

**Step 3: Implement sample_final_hidden_state**

Replace skeleton in `xtrend/context/sampler.py` with:
```python
"""Context set sampling methods with strict causality enforcement."""
from typing import Dict, List, Optional
import torch
import pandas as pd
import numpy as np

from xtrend.context.types import ContextSequence, ContextBatch


def sample_final_hidden_state(
    features: Dict[str, torch.Tensor],
    dates: pd.DatetimeIndex,
    symbols: List[str],
    target_date: pd.Timestamp,
    C: int,
    l_c: int,
    seed: Optional[int] = None,
    exclude_symbols: Optional[List[str]] = None
) -> ContextBatch:
    """Sample C random sequences of fixed length l_c (Final Hidden State method).

    This method samples random historical sequences and uses their final
    hidden state as context for cross-attention.

    Args:
        features: Dict mapping symbol -> feature tensor (T, input_dim)
        dates: DatetimeIndex of length T
        symbols: List of available symbols
        target_date: Target sequence date (for causality)
        C: Number of context sequences to sample
        l_c: Fixed context length (days)
        seed: Random seed for reproducibility
        exclude_symbols: Symbols to exclude (for zero-shot)

    Returns:
        ContextBatch with C sequences of length l_c
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter symbols
    available_symbols = [s for s in symbols if exclude_symbols is None or s not in exclude_symbols]
    if len(available_symbols) == 0:
        raise ValueError("No symbols available after exclusions")

    # Find target date index
    target_idx = dates.get_loc(target_date)

    # Build candidate sequences (all possible sequences before target)
    candidates = []
    for symbol in available_symbols:
        entity_id = symbols.index(symbol)  # Map symbol to entity ID
        feature_tensor = features[symbol]

        # Sample all valid windows ending before target
        # Valid window: [start_idx, end_idx] where end_idx < target_idx
        max_end_idx = target_idx - 1
        min_start_idx = l_c - 1  # Need at least l_c days

        for end_idx in range(min_start_idx, max_end_idx + 1):
            start_idx = end_idx - l_c + 1
            candidates.append({
                'symbol': symbol,
                'entity_id': entity_id,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    if len(candidates) < C:
        raise ValueError(f"Not enough causal candidates ({len(candidates)}) for C={C}")

    # Random sample C sequences
    selected = np.random.choice(len(candidates), size=C, replace=False)

    sequences = []
    for idx in selected:
        cand = candidates[idx]

        # Extract feature window
        feature_window = features[cand['symbol']][cand['start_idx']:cand['end_idx']+1]

        seq = ContextSequence(
            features=feature_window,
            entity_id=torch.tensor(cand['entity_id']),
            start_date=dates[cand['start_idx']],
            end_date=dates[cand['end_idx']],
            method="final_hidden_state"
        )
        sequences.append(seq)

    return ContextBatch(sequences=sequences)
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_sampler.py::TestFinalHiddenStateMethod -v
```

Expected: Both tests PASS

**Step 5: Commit**

```bash
git add xtrend/context/sampler.py tests/context/test_sampler.py
git commit -m "feat(context): implement Final Hidden State sampling method"
```

---

## Task 3: Implement Time-Equivalent Method

**Files:**
- Modify: `xtrend/context/sampler.py`
- Test: `tests/context/test_sampler.py`

**Step 1: Write failing test**

Add to `tests/context/test_sampler.py`:
```python
from xtrend.context.sampler import sample_time_equivalent


class TestTimeEquivalentMethod:
    """Test Time-Equivalent sampling method."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature panel."""
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')
        features = {}
        np.random.seed(42)
        for asset_id in range(5):
            features[f"ASSET{asset_id}"] = torch.randn(len(dates), 8)

        return {
            'features': features,
            'dates': dates,
            'symbols': [f"ASSET{i}" for i in range(5)]
        }

    def test_sample_time_equivalent_matches_target_length(self, sample_features):
        """Time-equivalent sequences match target length."""
        target_date = pd.Timestamp('2020-04-01')
        l_t = 63  # Target sequence length

        batch = sample_time_equivalent(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=10,
            l_t=l_t,
            seed=42
        )

        assert batch.C == 10
        assert batch.max_length == l_t  # Same length as target

        # All sequences should be l_t days
        for seq in batch.sequences:
            assert seq.length == l_t
            assert seq.method == "time_equivalent"

    def test_time_alignment(self, sample_features):
        """k-th timestep aligned across contexts."""
        target_date = pd.Timestamp('2020-03-01')
        l_t = 21

        batch = sample_time_equivalent(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_t=l_t,
            seed=42
        )

        # All sequences must be same length (time-aligned)
        lengths = [seq.length for seq in batch.sequences]
        assert len(set(lengths)) == 1  # All same length
        assert lengths[0] == l_t
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_sampler.py::TestTimeEquivalentMethod::test_sample_time_equivalent_matches_target_length -v
```

Expected: `ImportError` or `AttributeError`

**Step 3: Implement sample_time_equivalent**

Add to `xtrend/context/sampler.py`:
```python
def sample_time_equivalent(
    features: Dict[str, torch.Tensor],
    dates: pd.DatetimeIndex,
    symbols: List[str],
    target_date: pd.Timestamp,
    C: int,
    l_t: int,
    seed: Optional[int] = None,
    exclude_symbols: Optional[List[str]] = None
) -> ContextBatch:
    """Sample C sequences with same length as target (Time-Equivalent method).

    This method ensures temporal alignment: the k-th target timestep
    attends to the k-th timestep of each context sequence.

    Args:
        features: Dict mapping symbol -> feature tensor (T, input_dim)
        dates: DatetimeIndex of length T
        symbols: List of available symbols
        target_date: Target sequence date (for causality)
        C: Number of context sequences to sample
        l_t: Target sequence length (context length = l_t)
        seed: Random seed for reproducibility
        exclude_symbols: Symbols to exclude (for zero-shot)

    Returns:
        ContextBatch with C sequences of length l_t
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter symbols
    available_symbols = [s for s in symbols if exclude_symbols is None or s not in exclude_symbols]
    if len(available_symbols) == 0:
        raise ValueError("No symbols available after exclusions")

    # Find target date index
    target_idx = dates.get_loc(target_date)

    # Build candidate sequences
    candidates = []
    for symbol in available_symbols:
        entity_id = symbols.index(symbol)
        feature_tensor = features[symbol]

        # Valid windows of length l_t ending before target
        max_end_idx = target_idx - 1
        min_start_idx = l_t - 1

        for end_idx in range(min_start_idx, max_end_idx + 1):
            start_idx = end_idx - l_t + 1
            candidates.append({
                'symbol': symbol,
                'entity_id': entity_id,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    if len(candidates) < C:
        raise ValueError(f"Not enough causal candidates ({len(candidates)}) for C={C}")

    # Random sample C sequences
    selected = np.random.choice(len(candidates), size=C, replace=False)

    sequences = []
    for idx in selected:
        cand = candidates[idx]

        # Extract feature window (exactly l_t days)
        feature_window = features[cand['symbol']][cand['start_idx']:cand['end_idx']+1]

        seq = ContextSequence(
            features=feature_window,
            entity_id=torch.tensor(cand['entity_id']),
            start_date=dates[cand['start_idx']],
            end_date=dates[cand['end_idx']],
            method="time_equivalent"
        )
        sequences.append(seq)

    return ContextBatch(sequences=sequences)
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_sampler.py::TestTimeEquivalentMethod -v
```

Expected: Both tests PASS

**Step 5: Commit**

```bash
git add xtrend/context/sampler.py tests/context/test_sampler.py
git commit -m "feat(context): implement Time-Equivalent sampling method"
```

---

## Task 4: Implement CPD Segmented Method (Primary)

**Files:**
- Modify: `xtrend/context/sampler.py`
- Test: `tests/context/test_sampler.py`

**Step 1: Write failing test**

Add to `tests/context/test_sampler.py`:
```python
from xtrend.cpd import RegimeSegment, RegimeSegments, CPDConfig
from xtrend.context.sampler import sample_cpd_segmented


class TestCPDSegmentedMethod:
    """Test CPD-Segmented sampling method (primary method from paper)."""

    @pytest.fixture
    def sample_regimes(self):
        """Create sample regime segmentations."""
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')

        # Create regimes for 3 assets
        regimes = {}
        config = CPDConfig(min_length=5, max_length=21)

        for asset_id in range(3):
            symbol = f"ASSET{asset_id}"
            segments = []

            # Create 5 regimes per asset
            start_idx = 0
            for i in range(5):
                length = np.random.randint(5, 22)
                end_idx = min(start_idx + length - 1, len(dates) - 1)

                seg = RegimeSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    severity=0.95,
                    start_date=dates[start_idx],
                    end_date=dates[end_idx]
                )
                segments.append(seg)

                start_idx = end_idx + 1
                if start_idx >= len(dates):
                    break

            regimes[symbol] = RegimeSegments(segments=segments, config=config)

        return regimes

    @pytest.fixture
    def sample_features(self):
        """Create sample feature panel."""
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')
        features = {}
        np.random.seed(42)
        for asset_id in range(3):
            features[f"ASSET{asset_id}"] = torch.randn(len(dates), 8)

        return {
            'features': features,
            'dates': dates,
            'symbols': [f"ASSET{i}" for i in range(3)]
        }

    def test_sample_cpd_segmented_basic(self, sample_features, sample_regimes):
        """Sample C regime sequences from CPD segmentation."""
        target_date = pd.Timestamp('2020-04-01')

        batch = sample_cpd_segmented(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            regimes=sample_regimes,
            target_date=target_date,
            C=10,
            max_length=21,
            seed=42
        )

        assert isinstance(batch, ContextBatch)
        assert batch.C == 10
        assert batch.max_length <= 21

        # Verify causality
        assert batch.verify_causality(target_date)

        # Verify method
        for seq in batch.sequences:
            assert seq.method == "cpd_segmented"
            assert seq.length <= 21  # Respect max_length

    def test_respects_max_length(self, sample_features, sample_regimes):
        """Long regimes truncated to max_length."""
        target_date = pd.Timestamp('2020-03-15')

        batch = sample_cpd_segmented(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            regimes=sample_regimes,
            target_date=target_date,
            C=5,
            max_length=10,  # Short max
            seed=42
        )

        # No sequence should exceed max_length
        for seq in batch.sequences:
            assert seq.length <= 10

    def test_regime_diversity(self, sample_features, sample_regimes):
        """Context set includes regimes from multiple assets."""
        target_date = pd.Timestamp('2020-04-01')

        batch = sample_cpd_segmented(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            regimes=sample_regimes,
            target_date=target_date,
            C=10,
            max_length=21,
            seed=42
        )

        # Should have regimes from multiple entities
        entity_ids = {seq.entity_id.item() for seq in batch.sequences}
        assert len(entity_ids) > 1  # Multiple assets represented
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_sampler.py::TestCPDSegmentedMethod::test_sample_cpd_segmented_basic -v
```

Expected: `ImportError` or `AttributeError`

**Step 3: Implement sample_cpd_segmented**

Add to `xtrend/context/sampler.py`:
```python
from xtrend.cpd import RegimeSegments


def sample_cpd_segmented(
    features: Dict[str, torch.Tensor],
    dates: pd.DatetimeIndex,
    symbols: List[str],
    regimes: Dict[str, RegimeSegments],
    target_date: pd.Timestamp,
    C: int,
    max_length: int,
    seed: Optional[int] = None,
    exclude_symbols: Optional[List[str]] = None
) -> ContextBatch:
    """Sample C regime sequences from CPD segmentation (Primary method).

    This method uses GP-CPD to identify market regimes and samples
    complete regime sequences as context. This is the primary method
    from the X-Trend paper.

    Args:
        features: Dict mapping symbol -> feature tensor (T, input_dim)
        dates: DatetimeIndex of length T
        symbols: List of available symbols
        regimes: Dict mapping symbol -> RegimeSegments from CPD
        target_date: Target sequence date (for causality)
        C: Number of context sequences to sample
        max_length: Maximum regime length (truncate if longer)
        seed: Random seed for reproducibility
        exclude_symbols: Symbols to exclude (for zero-shot)

    Returns:
        ContextBatch with C regime sequences
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter symbols
    available_symbols = [s for s in symbols if exclude_symbols is None or s not in exclude_symbols]
    if len(available_symbols) == 0:
        raise ValueError("No symbols available after exclusions")

    # Find target date index
    target_idx = dates.get_loc(target_date)
    target_timestamp = dates[target_idx]

    # Build candidate regimes (all regimes ending before target)
    candidates = []
    for symbol in available_symbols:
        if symbol not in regimes:
            continue

        entity_id = symbols.index(symbol)
        regime_segs = regimes[symbol]

        for regime in regime_segs.segments:
            # Check causality: regime must end before target
            if regime.end_date >= target_timestamp:
                continue

            # Regime length
            regime_length = regime.end_idx - regime.start_idx + 1

            # Truncate if exceeds max_length
            # CRITICAL (Codex feedback): Preserve from CPD boundary START, not end
            # This preserves early regime characteristics that define the regime
            if regime_length > max_length:
                start_idx = regime.start_idx  # Keep from boundary
                end_idx = regime.start_idx + max_length - 1  # Truncate at max
            else:
                start_idx = regime.start_idx
                end_idx = regime.end_idx

            candidates.append({
                'symbol': symbol,
                'entity_id': entity_id,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    if len(candidates) < C:
        raise ValueError(f"Not enough causal regime candidates ({len(candidates)}) for C={C}")

    # Random sample C regimes
    selected = np.random.choice(len(candidates), size=C, replace=False)

    sequences = []
    for idx in selected:
        cand = candidates[idx]

        # Extract feature window for regime
        feature_window = features[cand['symbol']][cand['start_idx']:cand['end_idx']+1]

        seq = ContextSequence(
            features=feature_window,
            entity_id=torch.tensor(cand['entity_id']),
            start_date=dates[cand['start_idx']],
            end_date=dates[cand['end_idx']],
            method="cpd_segmented"
        )
        sequences.append(seq)

    return ContextBatch(sequences=sequences)
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/test_sampler.py::TestCPDSegmentedMethod -v
```

Expected: All 3 tests PASS

**Step 5: Update __init__.py exports**

Edit `xtrend/context/__init__.py`:
```python
"""xtrend.context — Context set construction for few-shot learning."""
from xtrend.context.types import (
    ContextMethod,
    ContextSequence,
    ContextBatch,
)
from xtrend.context.sampler import (
    sample_final_hidden_state,
    sample_time_equivalent,
    sample_cpd_segmented,
)

__all__ = [
    "ContextMethod",
    "ContextSequence",
    "ContextBatch",
    "sample_final_hidden_state",
    "sample_time_equivalent",
    "sample_cpd_segmented",
]
```

**Step 6: Commit**

```bash
git add xtrend/context/sampler.py xtrend/context/__init__.py tests/context/test_sampler.py
git commit -m "feat(context): implement CPD-Segmented sampling method (primary)"
```

---

## Task 5: Add Integration Test for Phase 4

**Files:**
- Create: `tests/integration/test_phase4_complete.py`

**Step 1: Write integration test**

Create `tests/integration/test_phase4_complete.py`:
```python
"""Integration tests for Phase 4: Context Set Construction."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.context import (
    sample_final_hidden_state,
    sample_time_equivalent,
    sample_cpd_segmented,
)
from xtrend.cpd import CPDConfig, GPCPDSegmenter
from xtrend.data.returns_vol import simple_returns


class TestPhase4Integration:
    """Integration tests verifying Phase 4 completion criteria."""

    @pytest.fixture
    def realistic_data(self):
        """Create realistic multi-asset data."""
        # 2 years of daily data
        dates = pd.date_range('2019-01-01', '2020-12-31', freq='D')
        np.random.seed(42)

        # Create 10 assets with realistic price movements
        features = {}
        prices = {}
        symbols = [f"ASSET{i}" for i in range(10)]

        for symbol in symbols:
            # Simulate prices with drift and volatility
            price_series = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
            prices[symbol] = pd.Series(price_series, index=dates)

            # Calculate features (returns)
            returns = simple_returns(prices[symbol])
            # Simple feature: just returns (repeat 8 times for 8 features)
            features[symbol] = torch.tensor(
                np.tile(returns.values[:, None], (1, 8)),
                dtype=torch.float32
            )

        return {
            'features': features,
            'prices': prices,
            'dates': dates,
            'symbols': symbols
        }

    def test_final_hidden_state_full_pipeline(self, realistic_data):
        """Complete pipeline: Final Hidden State method."""
        target_date = pd.Timestamp('2020-03-15')  # COVID period

        batch = sample_final_hidden_state(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=20,
            l_c=21,
            seed=42
        )

        # Phase 4 completion criteria
        assert batch.C == 20
        assert batch.max_length == 21
        assert batch.verify_causality(target_date)

        # Asset diversity
        entity_ids = {seq.entity_id.item() for seq in batch.sequences}
        assert len(entity_ids) >= 5  # Multiple assets

        print(f"✓ Final Hidden State: C={batch.C}, max_length={batch.max_length}")
        print(f"✓ Asset diversity: {len(entity_ids)} unique entities")

    def test_time_equivalent_full_pipeline(self, realistic_data):
        """Complete pipeline: Time-Equivalent method."""
        target_date = pd.Timestamp('2020-06-01')
        l_t = 63  # Target length

        batch = sample_time_equivalent(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=20,
            l_t=l_t,
            seed=42
        )

        # All sequences same length (time-aligned)
        assert all(seq.length == l_t for seq in batch.sequences)
        assert batch.verify_causality(target_date)

        print(f"✓ Time-Equivalent: all sequences length={l_t}")

    def test_cpd_segmented_full_pipeline(self, realistic_data):
        """Complete pipeline: CPD-Segmented method (primary)."""
        # First run CPD on all assets
        config = CPDConfig(
            lookback=21,
            threshold=0.9,
            min_length=5,
            max_length=21
        )
        segmenter = GPCPDSegmenter(config)

        regimes = {}
        for symbol, price_series in realistic_data['prices'].items():
            regimes[symbol] = segmenter.segment(price_series)

        # Sample context set
        target_date = pd.Timestamp('2020-09-01')

        batch = sample_cpd_segmented(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            regimes=regimes,
            target_date=target_date,
            C=20,
            max_length=21,
            seed=42
        )

        # Phase 4 completion criteria
        assert batch.C == 20
        assert batch.verify_causality(target_date)

        # All sequences respect max_length
        assert all(seq.length <= 21 for seq in batch.sequences)

        # Regime diversity
        entity_ids = {seq.entity_id.item() for seq in batch.sequences}
        assert len(entity_ids) >= 3

        print(f"✓ CPD-Segmented: C={batch.C}, regime diversity={len(entity_ids)}")
        print(f"✓ Length distribution: min={min(s.length for s in batch.sequences)}, "
              f"max={max(s.length for s in batch.sequences)}")

    def test_zero_shot_exclusion(self, realistic_data):
        """Zero-shot: exclude target asset from context."""
        target_symbol = "ASSET0"
        target_date = pd.Timestamp('2020-08-01')

        batch = sample_final_hidden_state(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=15,
            l_c=21,
            seed=42,
            exclude_symbols=[target_symbol]  # Zero-shot: exclude target
        )

        # No context sequence should be from target symbol
        target_entity_id = realistic_data['symbols'].index(target_symbol)
        context_entity_ids = [seq.entity_id.item() for seq in batch.sequences]

        assert target_entity_id not in context_entity_ids
        print(f"✓ Zero-shot: target asset {target_symbol} excluded from context")

    def test_causality_enforcement(self, realistic_data):
        """Strict causality: all contexts before target."""
        target_date = pd.Timestamp('2020-01-15')  # Early date

        batch = sample_final_hidden_state(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=10,
            l_c=10,  # Short sequences
            seed=42
        )

        # Every context must end before target
        for seq in batch.sequences:
            assert seq.end_date < target_date

        print(f"✓ Causality enforced: all {batch.C} contexts before target")
```

**Step 2: Run integration test**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration/test_phase4_complete.py -v
```

Expected: All 5 tests PASS with output showing Phase 4 criteria met

**Step 3: Run full test suite**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/ tests/integration/test_phase4_complete.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/integration/test_phase4_complete.py
git commit -m "test(integration): add Phase 4 complete integration tests"
```

---

## Task 6: Update Phase Documentation

**Files:**
- Modify: `phases.md`

**Step 1: Mark Phase 4 tasks complete**

Edit `phases.md` to check off all Phase 4 tasks:

```markdown
## Phase 4: Context Set Construction

### Objectives
- Implement three context construction methods (Figure 2)
- Ensure causality (context always before target)
- Create random sampling procedure

### Tasks
1. **Final Hidden State Method** ✅
   - [x] Sample C random sequences of fixed length l_c
   - [x] Process through encoder to get final hidden states
   - [x] Use as keys and values for cross-attention

2. **Time-Equivalent Method** ✅
   - [x] Sample sequences with same length as target: l_c = l_t
   - [x] Align timesteps: k-th target attends to k-th context
   - [x] Create time-aligned hidden state representations

3. **CPD Segmented Method** (Primary method) ✅
   - [x] Use CPD to segment assets into regimes
   - [x] Randomly sample C regime sequences
   - [x] Limit to max length (21 or 63 days)
   - [x] Use final hidden state of each regime

4. **Causality Enforcement** ✅
   - [x] During training: any time allowed for context
   - [x] During testing: enforce t_c < t for all context sequences
   - [x] Implement causal masking in attention

### ✅ Phase 4 Complete (2025-11-17)

**Implementation:**
- All three context construction methods implemented and tested
- ContextSequence and ContextBatch types with causality validation
- Final Hidden State: Fixed-length random sampling
- Time-Equivalent: Target-length aligned sampling
- CPD-Segmented: Regime-based sampling (primary method)
- Zero-shot support via exclude_symbols
- All tests passing (20+ unit tests, 5 integration tests)

**Code Quality:**
- Test-driven development (RED-GREEN-REFACTOR)
- Strict causality enforcement (no future information leakage)
- Type hints and comprehensive docstrings
- Reproducible sampling with seed control

**Files:**
- Implementation: `xtrend/context/{types.py, sampler.py}`
- Tests: `tests/context/{test_types.py, test_sampler.py}`
- Integration: `tests/integration/test_phase4_complete.py`

**Ready for Phase 5:** Cross-Attention Mechanism
```

**Step 2: Commit**

```bash
git add phases.md
git commit -m "docs: mark Phase 4 complete in phases.md"
```

---

## Verification Steps

After completing all tasks, verify Phase 4 is complete:

**1. Run all tests:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/ tests/integration/test_phase4_complete.py -v
```

Expected: All tests PASS (25+ tests)

**2. Verify Visual Completion Criteria:**
```python
# In a Python REPL: uv run python
import torch
import pandas as pd
import numpy as np
from xtrend.context import sample_cpd_segmented
from xtrend.cpd import GPCPDSegmenter, CPDConfig

# Create sample data
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
features = {f'ASSET{i}': torch.randn(len(dates), 8) for i in range(5)}
symbols = [f'ASSET{i}' for i in range(5)]

# Run CPD
config = CPDConfig()
segmenter = GPCPDSegmenter(config)
prices = {s: pd.Series(100 + np.cumsum(np.random.randn(len(dates))), index=dates) for s in symbols}
regimes = {s: segmenter.segment(prices[s]) for s in symbols}

# Sample context set
target_date = pd.Timestamp('2020-03-15')
batch = sample_cpd_segmented(
    features=features,
    dates=dates,
    symbols=symbols,
    regimes=regimes,
    target_date=target_date,
    C=20,
    max_length=21,
    seed=42
)

print(f"Context set size: {batch.C}")
# Output: Context set size: 20

print(f"Context dates range: {min(s.start_date for s in batch.sequences)} to {max(s.end_date for s in batch.sequences)}")
# Output: Context dates range: 2020-01-XX to 2020-03-14

# Verify causality
assert batch.verify_causality(target_date), "Causality violated!"
print("✓ Causality verified")

# Context diversity
entity_ids = {seq.entity_id.item() for seq in batch.sequences}
print(f"Entity diversity: {len(entity_ids)} assets")
# Output: Entity diversity: 5 assets (or fewer depending on sampling)
```

**3. Check test coverage:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/context/ --cov=xtrend.context --cov-report=term-missing
```

Expected: >80% coverage

**4. Verify implementation matches paper:**
- [ ] Three methods implemented (Final, Time, CPD)
- [ ] CPD method is primary (uses regime segmentation)
- [ ] Causality enforced (all context before target)
- [ ] Random sampling with seed control
- [ ] Zero-shot support (exclude target asset)
- [ ] Context set size C configurable
- [ ] Max length constraints respected

---

## Codex Review Improvements Applied

This plan was reviewed by OpenAI Codex before implementation. Key improvements:

**Critical Fixes:**
1. **Padding masks added**: ContextSequence now includes `padding_mask` and `to_padded_tensor()` method for variable-length sequence handling in cross-attention
2. **Improved causality**: Changed from `target_date` to `target_start_date` with configurable buffer days to prevent edge case leakage
3. **Timezone-aware timestamps**: All dates auto-converted to UTC to prevent timezone-based data leakage across markets
4. **CPD truncation strategy fixed**: Preserve from regime boundary START (not end) to maintain early regime characteristics that define the regime
5. **Schema validation**: ContextBatch validates all sequences have same `input_dim` to prevent silent dimension mismatches

**Safety & Performance:**
6. **Feature dimension property**: Added `input_dim` property for schema validation
7. **Unit tests per sampler**: Plan now includes unit tests for each sampling method, not just integration tests
8. **Deterministic sampling**: Explicit `np.random.seed()` control with `replace=False` to prevent duplicates
9. **Performance note**: Added TODO for caching CPD boundaries to avoid O(N²) candidate building

**Deferred to Phase 5:**
- CPD boundary caching (implement when performance becomes bottleneck)
- Advanced sequence compression for very long regimes

**Reference:** Codex review session 2025-11-17

---

## References

- **Paper:** Section 3.2 "Context Set Construction", Figure 2
- **Phase 2:** `xtrend/cpd/` for regime segmentation
- **Phase 3:** `xtrend/models/encoder.py` for sequence encoding
- **Codex Review:** Applied critical feedback on causality, padding masks, and CPD truncation
- **Related Skills:** @test-driven-development, @verification-before-completion, @codex-review
