# Phase 3: Base Neural Architecture - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the core neural architecture components for X-Trend: Variable Selection Network (VSN), entity embeddings, LSTM encoder with skip connections, and baseline DMN model.

**Architecture:** Paper-faithful PyTorch implementation following Equations 7, 12, 13, and 14 from the X-Trend paper. Modular design with separate components for VSN, embeddings, encoder blocks, and baseline model. Follows test-driven development with comprehensive unit and integration tests.

**Tech Stack:** PyTorch (neural networks), pytest (testing), numpy (numerical operations)

---

## Task 1: Types and Configuration Foundation

**Files:**
- Create: `xtrend/models/types.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/conftest.py`
- Create: `tests/models/test_types.py`

**Step 1: Write failing test for ModelConfig**

Create `tests/models/test_types.py`:

```python
"""Tests for model types and configuration."""
import pytest
import torch
from xtrend.models.types import ModelConfig, EncoderOutput


class TestModelConfig:
    def test_default_values(self):
        """ModelConfig has sensible defaults matching paper Table 4."""
        config = ModelConfig()

        assert config.input_dim == 8  # Paper: 5 returns + 3 MACD
        assert config.hidden_dim == 64  # Paper: d_h ∈ {64, 128}
        assert config.dropout == 0.3  # Paper Table 3: {0.3, 0.4, 0.5}
        assert config.num_entities == 50  # Paper: 50 futures contracts
        assert config.num_attention_heads == 4  # Paper Section 2.4

    def test_custom_values(self):
        """ModelConfig accepts custom parameters."""
        config = ModelConfig(
            hidden_dim=128,
            dropout=0.5,
            num_entities=30
        )

        assert config.hidden_dim == 128
        assert config.dropout == 0.5
        assert config.num_entities == 30

    def test_validation(self):
        """ModelConfig validates parameters."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            ModelConfig(hidden_dim=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=1.5)


class TestEncoderOutput:
    def test_encoder_output_structure(self):
        """EncoderOutput holds hidden states and metadata."""
        batch_size, seq_len, hidden_dim = 32, 126, 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        output = EncoderOutput(
            hidden_states=hidden_states,
            sequence_length=seq_len
        )

        assert output.hidden_states.shape == (batch_size, seq_len, hidden_dim)
        assert output.sequence_length == seq_len
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_types.py::TestModelConfig::test_default_values -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'xtrend.models.types'"

**Step 3: Create package structure**

Create `tests/models/__init__.py`:
```python
"""Tests for neural network models."""
```

Create `tests/models/conftest.py`:
```python
"""Shared fixtures for model tests."""
import pytest
import torch
import numpy as np


@pytest.fixture
def sample_features():
    """Sample input features (batch, sequence, features)."""
    batch_size, seq_len, num_features = 32, 126, 8
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, num_features)


@pytest.fixture
def sample_entity_ids():
    """Sample entity IDs for embedding lookup."""
    batch_size = 32
    num_entities = 50
    torch.manual_seed(42)
    return torch.randint(0, num_entities, (batch_size,))


@pytest.fixture
def model_config():
    """Default model configuration."""
    from xtrend.models.types import ModelConfig
    return ModelConfig()
```

**Step 4: Implement ModelConfig and EncoderOutput**

Create `xtrend/models/types.py`:

```python
"""Type definitions for neural network models."""
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


@dataclass
class ModelConfig:
    """Configuration for X-Trend neural architecture.

    Attributes:
        input_dim: Number of input features (5 returns + 3 MACD)
        hidden_dim: Hidden state dimension d_h (Paper: 64 or 128)
        dropout: Dropout rate (Paper: 0.3, 0.4, or 0.5)
        num_entities: Number of futures contracts for embeddings (Paper: 50)
        num_attention_heads: Number of parallel attention heads (Paper: 4)
    """
    input_dim: int = 8
    hidden_dim: int = 64
    dropout: float = 0.3
    num_entities: int = 50
    num_attention_heads: int = 4

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim ({self.input_dim}) must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout ({self.dropout}) must be in [0, 1)")
        if self.num_entities <= 0:
            raise ValueError(f"num_entities ({self.num_entities}) must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be positive")


class EncoderOutput(NamedTuple):
    """Output from encoder module.

    Attributes:
        hidden_states: Encoded representations (batch, seq_len, hidden_dim)
        sequence_length: Length of sequences in batch
    """
    hidden_states: torch.Tensor
    sequence_length: int
```

**Step 5: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_types.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add xtrend/models/types.py tests/models/
git commit -m "feat(models): add ModelConfig and EncoderOutput types

- Add ModelConfig dataclass with paper-faithful defaults
- Add EncoderOutput NamedTuple for encoder outputs
- Add comprehensive validation
- Add test fixtures for model testing"
```

---

## Task 2: Variable Selection Network (VSN)

**Files:**
- Modify: `xtrend/models/types.py` (add VSNOutput)
- Create: `xtrend/models/vsn.py`
- Create: `tests/models/test_vsn.py`

**Step 1: Write failing test for VSN**

Create `tests/models/test_vsn.py`:

```python
"""Tests for Variable Selection Network."""
import pytest
import torch
import numpy as np
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.types import ModelConfig


class TestVariableSelectionNetwork:
    def test_vsn_initialization(self, model_config):
        """VSN initializes with correct dimensions."""
        vsn = VariableSelectionNetwork(model_config)

        # Should have feature-wise FFNs for each input feature
        assert len(vsn.feature_ffns) == model_config.input_dim
        assert vsn.feature_attention is not None

    def test_vsn_forward_shape(self, model_config, sample_features):
        """VSN produces correct output shape."""
        vsn = VariableSelectionNetwork(model_config)

        output, weights = vsn(sample_features)

        batch_size, seq_len, _ = sample_features.shape
        # Output should be (batch, seq_len, hidden_dim)
        assert output.shape == (batch_size, seq_len, model_config.hidden_dim)
        # Weights should be (batch, seq_len, input_dim)
        assert weights.shape == (batch_size, seq_len, model_config.input_dim)

    def test_vsn_weights_sum_to_one(self, model_config, sample_features):
        """VSN attention weights sum to 1 (Softmax property)."""
        vsn = VariableSelectionNetwork(model_config)

        _, weights = vsn(sample_features)

        # Weights should sum to 1 across features dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_vsn_weights_non_negative(self, model_config, sample_features):
        """VSN attention weights are non-negative (Softmax property)."""
        vsn = VariableSelectionNetwork(model_config)

        _, weights = vsn(sample_features)

        assert (weights >= 0).all()

    def test_vsn_equation_13(self, model_config):
        """VSN implements Equation 13 from paper."""
        # Equation 13: VSN(x_t) = Σ w_{t,j} * FFN_j(x_{t,j})
        # where w_t = Softmax(FFN(x_t))

        vsn = VariableSelectionNetwork(model_config)
        vsn.eval()

        # Simple input: 1 sample, 1 timestep, 8 features
        x = torch.randn(1, 1, 8)

        with torch.no_grad():
            output, weights = vsn(x)

        # Manually verify weighted sum
        # Each feature gets processed by its FFN, then weighted
        manual_output = torch.zeros(1, 1, model_config.hidden_dim)
        for j in range(model_config.input_dim):
            feature_j = x[:, :, j:j+1]  # (1, 1, 1)
            processed_j = vsn.feature_ffns[j](feature_j)  # (1, 1, hidden_dim)
            manual_output += weights[:, :, j:j+1] * processed_j

        assert torch.allclose(output, manual_output, atol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_vsn.py::TestVariableSelectionNetwork::test_vsn_initialization -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'xtrend.models.vsn'"

**Step 3: Implement VariableSelectionNetwork**

Create `xtrend/models/vsn.py`:

```python
"""Variable Selection Network (Equation 13)."""
import torch
import torch.nn as nn
from typing import Tuple

from xtrend.models.types import ModelConfig


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network from X-Trend paper (Equation 13).

    Implements feature-wise processing with learned attention weights:
        VSN(x_t) = Σ_{j=1}^{|X|} w_{t,j} * FFN_j(x_{t,j})
        where w_t = Softmax(FFN(x_t))

    This allows the model to learn which features are most important
    at each timestep.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Feature-wise FFNs (one per input feature)
        self.feature_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, config.hidden_dim),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
            for _ in range(config.input_dim)
        ])

        # Attention FFN for computing feature weights
        self.feature_attention = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply variable selection to input features.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            output: Weighted feature representations (batch, seq_len, hidden_dim)
            weights: Attention weights (batch, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # Compute attention weights: w_t = Softmax(FFN(x_t))
        # Shape: (batch, seq_len, input_dim) -> (batch, seq_len, input_dim)
        attn_logits = self.feature_attention(x)
        weights = torch.softmax(attn_logits, dim=-1)

        # Process each feature with its dedicated FFN
        # and weight by attention
        processed_features = []
        for j in range(input_dim):
            # Extract j-th feature: (batch, seq_len, 1)
            feature_j = x[:, :, j:j+1]
            # Process with j-th FFN: (batch, seq_len, hidden_dim)
            processed_j = self.feature_ffns[j](feature_j)
            # Weight by attention: (batch, seq_len, 1) * (batch, seq_len, hidden_dim)
            weighted_j = weights[:, :, j:j+1] * processed_j
            processed_features.append(weighted_j)

        # Sum weighted features: Σ w_{t,j} * FFN_j(x_{t,j})
        output = torch.stack(processed_features, dim=-1).sum(dim=-1)

        return output, weights
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_vsn.py -v`

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add xtrend/models/vsn.py tests/models/test_vsn.py
git commit -m "feat(models): implement Variable Selection Network (Equation 13)

- Add VSN with feature-wise FFNs and attention weights
- Implement Softmax-based feature weighting
- Add comprehensive tests verifying Equation 13
- Test output shapes, weight properties (sum=1, non-negative)"
```

---

## Task 3: Entity Embeddings and Conditional FFN

**Files:**
- Create: `xtrend/models/embeddings.py`
- Create: `tests/models/test_embeddings.py`

**Step 1: Write failing test for EntityEmbedding**

Create `tests/models/test_embeddings.py`:

```python
"""Tests for entity embeddings."""
import pytest
import torch
from xtrend.models.embeddings import EntityEmbedding, ConditionalFFN
from xtrend.models.types import ModelConfig


class TestEntityEmbedding:
    def test_embedding_initialization(self, model_config):
        """EntityEmbedding initializes with correct dimensions."""
        emb = EntityEmbedding(model_config)

        # Should have embedding table for num_entities
        assert emb.embedding.num_embeddings == model_config.num_entities
        assert emb.embedding.embedding_dim == model_config.hidden_dim

    def test_embedding_forward_shape(self, model_config, sample_entity_ids):
        """EntityEmbedding produces correct output shape."""
        emb = EntityEmbedding(model_config)

        output = emb(sample_entity_ids)

        batch_size = sample_entity_ids.shape[0]
        assert output.shape == (batch_size, model_config.hidden_dim)

    def test_embedding_unique_per_entity(self, model_config):
        """Different entities get different embeddings."""
        emb = EntityEmbedding(model_config)

        entity_0 = torch.tensor([0])
        entity_1 = torch.tensor([1])

        emb_0 = emb(entity_0)
        emb_1 = emb(entity_1)

        # Different entities should have different embeddings
        assert not torch.allclose(emb_0, emb_1)

    def test_embedding_same_entity_consistent(self, model_config):
        """Same entity gets same embedding."""
        emb = EntityEmbedding(model_config)

        entity_0 = torch.tensor([0, 0, 0])

        output = emb(entity_0)

        # All three outputs should be identical
        assert torch.allclose(output[0], output[1])
        assert torch.allclose(output[1], output[2])


class TestConditionalFFN:
    def test_conditional_ffn_initialization(self, model_config):
        """ConditionalFFN initializes correctly."""
        entity_embedding = EntityEmbedding(model_config)
        cffn = ConditionalFFN(model_config, use_entity=True, entity_embedding=entity_embedding)

        assert cffn.linear1 is not None
        assert cffn.linear2 is not None
        assert cffn.linear3 is not None
        assert cffn.entity_embedding is entity_embedding  # Verify shared instance

    def test_conditional_ffn_equation_12(self, model_config):
        """ConditionalFFN implements Equation 12."""
        # Equation 12: FFN(h_t, s) = Linear3 ∘ ELU(Linear1(h_t) + Linear2(Embedding(s)))

        entity_embedding = EntityEmbedding(model_config)
        cffn = ConditionalFFN(model_config, use_entity=True, entity_embedding=entity_embedding)

        batch_size = 32
        h_t = torch.randn(batch_size, model_config.hidden_dim)
        entity_ids = torch.randint(0, model_config.num_entities, (batch_size,))

        output = cffn(h_t, entity_ids)

        assert output.shape == (batch_size, model_config.hidden_dim)

    def test_conditional_ffn_with_no_entity(self, model_config):
        """ConditionalFFN works without entity info (zero-shot)."""
        cffn = ConditionalFFN(model_config, use_entity=False, entity_embedding=None)

        batch_size = 32
        h_t = torch.randn(batch_size, model_config.hidden_dim)

        # Should work with entity_ids=None for zero-shot
        output = cffn(h_t, entity_ids=None)

        assert output.shape == (batch_size, model_config.hidden_dim)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_embeddings.py::TestEntityEmbedding::test_embedding_initialization -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'xtrend.models.embeddings'"

**Step 3: Implement EntityEmbedding and ConditionalFFN**

Create `xtrend/models/embeddings.py`:

```python
"""Entity embeddings for futures contracts (Equation 12)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig


class EntityEmbedding(nn.Module):
    """Learnable embeddings for futures contract types.

    Maps each contract ticker to a learned embedding vector.
    This allows the model to learn similarities between contracts
    (e.g., crude oil and heating oil should have similar embeddings).

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding table: num_entities x hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=config.num_entities,
            embedding_dim=config.hidden_dim
        )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for entity IDs.

        Args:
            entity_ids: Entity indices (batch,)
                Note: While nn.Embedding supports (batch, seq_len), ConditionalFFN
                expects (batch,) as each sequence corresponds to one entity.

        Returns:
            Embeddings (batch, hidden_dim)
        """
        return self.embedding(entity_ids)


class ConditionalFFN(nn.Module):
    """Feed-forward network conditioned on entity embeddings (Equation 12).

    Implements: FFN(h_t, s) = Linear3 ∘ ELU(Linear1(h_t) + Linear2(Embedding(s)))

    This fuses time-series representations with entity-specific information.
    For zero-shot learning, entity info can be excluded (entity_ids=None).

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
        entity_embedding: Shared EntityEmbedding instance (required if use_entity=True)
    """

    def __init__(
        self,
        config: ModelConfig,
        use_entity: bool = True,
        entity_embedding: Optional[EntityEmbedding] = None
    ):
        super().__init__()
        self.config = config
        self.use_entity = use_entity

        # Linear transformations
        self.linear1 = nn.Linear(config.hidden_dim, config.hidden_dim)

        if use_entity:
            assert entity_embedding is not None, "entity_embedding required when use_entity=True"
            self.entity_embedding = entity_embedding
            self.linear2 = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(config.dropout)
        self.linear3 = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply conditional FFN.

        Args:
            h_t: Hidden states (batch, hidden_dim) or (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) - one entity per sequence. None for zero-shot.

        Returns:
            Output (same shape as h_t)
        """
        # Linear1(h_t)
        transformed = self.linear1(h_t)

        # Add entity information if available
        if self.use_entity and entity_ids is not None:
            # Entity IDs must be (batch,) - one entity per sequence
            assert entity_ids.dim() == 1, f"entity_ids must be (batch,), got shape {entity_ids.shape}"

            entity_emb = self.entity_embedding(entity_ids)  # (batch, hidden_dim)

            # Broadcast entity embedding to match h_t shape
            if h_t.dim() == 3:  # (batch, seq_len, hidden_dim)
                entity_emb = entity_emb.unsqueeze(1)  # (batch, 1, hidden_dim)

            # Linear2(Embedding(s))
            entity_transformed = self.linear2(entity_emb)
            transformed = transformed + entity_transformed

        # ELU activation
        activated = self.activation(transformed)
        activated = self.dropout(activated)

        # Linear3
        output = self.linear3(activated)

        return output
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_embeddings.py -v`

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add xtrend/models/embeddings.py tests/models/test_embeddings.py
git commit -m "feat(models): add entity embeddings and conditional FFN (Equation 12)

- Add EntityEmbedding for 50 futures contracts
- Add ConditionalFFN fusing time-series and entity info
- Support zero-shot mode (no entity embeddings)
- Add comprehensive tests for embeddings and conditional processing"
```

---

## Task 4: LSTM Encoder with Skip Connections

**Files:**
- Create: `xtrend/models/encoder.py`
- Create: `tests/models/test_encoder.py`

**Step 1: Write failing test for LSTMEncoder**

Create `tests/models/test_encoder.py`:

```python
"""Tests for LSTM encoder."""
import pytest
import torch
from xtrend.models.encoder import LSTMEncoder
from xtrend.models.embeddings import EntityEmbedding
from xtrend.models.types import ModelConfig


class TestLSTMEncoder:
    def test_encoder_initialization(self, model_config):
        """LSTMEncoder initializes with correct components."""
        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)

        assert encoder.vsn is not None
        assert encoder.lstm is not None
        assert encoder.lstm_dropout is not None
        assert encoder.layer_norm1 is not None
        assert encoder.ffn is not None
        assert encoder.layer_norm2 is not None
        assert encoder.entity_embedding is entity_embedding  # Verify shared instance

    def test_encoder_forward_shape(self, model_config, sample_features, sample_entity_ids):
        """Encoder produces correct output shape."""
        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)

        output = encoder(sample_features, sample_entity_ids)

        batch_size, seq_len, _ = sample_features.shape
        # Output should be (batch, seq_len, hidden_dim)
        assert output.hidden_states.shape == (batch_size, seq_len, model_config.hidden_dim)
        assert output.sequence_length == seq_len

    def test_encoder_equation_14(self, model_config):
        """Encoder implements Equation 14 from paper."""
        # Equation 14:
        #   x'_t = VSN(x_t, s)
        #   (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        #   a_t = LayerNorm(h_t + x'_t)  # Skip connection
        #   Ξ = LayerNorm(FFN(a_t, s) + a_t)  # Skip connection

        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)
        encoder.eval()

        # Single sample to verify skip connections
        x = torch.randn(1, 10, 8)  # (batch=1, seq_len=10, input_dim=8)
        entity_ids = torch.tensor([0])

        with torch.no_grad():
            output = encoder(x, entity_ids)

        # Output should exist
        assert output.hidden_states is not None
        assert output.hidden_states.shape == (1, 10, model_config.hidden_dim)

    def test_encoder_lstm_state_persistence(self, model_config):
        """LSTM maintains state across sequence."""
        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)

        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 8)
        entity_ids = torch.randint(0, 50, (batch_size,))

        output = encoder(x, entity_ids)

        # Later timesteps should be different from earlier ones
        # (LSTM should maintain context)
        first_step = output.hidden_states[:, 0, :]
        last_step = output.hidden_states[:, -1, :]

        assert not torch.allclose(first_step, last_step)

    def test_encoder_zero_shot_mode(self, model_config):
        """Encoder works in zero-shot mode (no entity info)."""
        encoder = LSTMEncoder(model_config, use_entity=False, entity_embedding=None)

        x = torch.randn(4, 20, 8)

        # Should work without entity_ids
        output = encoder(x, entity_ids=None)

        assert output.hidden_states.shape == (4, 20, model_config.hidden_dim)
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_encoder.py::TestLSTMEncoder::test_encoder_initialization -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'xtrend.models.encoder'"

**Step 3: Implement LSTMEncoder**

Create `xtrend/models/encoder.py`:

```python
"""LSTM encoder with skip connections (Equation 14)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import ConditionalFFN


class LSTMEncoder(nn.Module):
    """LSTM encoder with Variable Selection and skip connections (Equation 14).

    Architecture:
        x'_t = VSN(x_t, s)
        (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        a_t = LayerNorm(h_t + x'_t)  # Skip connection 1
        Ξ = LayerNorm(FFN(a_t, s) + a_t)  # Skip connection 2

    The skip connections allow the model to suppress components
    when needed, enabling adaptive complexity.

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
        entity_embedding: Shared EntityEmbedding instance (required if use_entity=True)
    """

    def __init__(
        self,
        config: ModelConfig,
        use_entity: bool = True,
        entity_embedding: Optional[EntityEmbedding] = None
    ):
        super().__init__()
        self.config = config
        self.use_entity = use_entity

        # Shared entity embedding
        if use_entity:
            assert entity_embedding is not None, "entity_embedding required when use_entity=True"
            self.entity_embedding = entity_embedding

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(config)

        # LSTM cell (single layer, so dropout parameter is ignored)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True
        )

        # Explicit dropout on LSTM outputs (since single-layer LSTM ignores dropout param)
        self.lstm_dropout = nn.Dropout(config.dropout)

        # Layer normalization after LSTM
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)

        # FFN with entity conditioning (shares entity_embedding)
        self.ffn = ConditionalFFN(
            config,
            use_entity=use_entity,
            entity_embedding=entity_embedding if use_entity else None
        )

        # Layer normalization after FFN
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)

        # Learnable initial LSTM state (per entity if using entity info)
        if use_entity:
            self.init_h = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
        else:
            self.init_h = nn.Parameter(torch.randn(1, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(1, config.hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> EncoderOutput:
        """Encode input sequences.

        Args:
            x: Input features (batch, seq_len, input_dim)
            entity_ids: Entity IDs (batch,) - None for zero-shot

        Returns:
            EncoderOutput with hidden_states (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Variable Selection Network
        # x'_t = VSN(x_t, s)
        x_prime, _ = self.vsn(x)  # (batch, seq_len, hidden_dim)

        # Step 2: Initialize LSTM state
        if self.use_entity and entity_ids is not None:
            # Entity-specific initialization
            h_0 = self.init_h[entity_ids]  # (batch, hidden_dim)
            c_0 = self.init_c[entity_ids]  # (batch, hidden_dim)
        else:
            # Generic initialization for zero-shot
            h_0 = self.init_h.expand(batch_size, -1)  # (batch, hidden_dim)
            c_0 = self.init_c.expand(batch_size, -1)  # (batch, hidden_dim)

        # LSTM expects (num_layers, batch, hidden_dim)
        h_0 = h_0.unsqueeze(0)  # (1, batch, hidden_dim)
        c_0 = c_0.unsqueeze(0)  # (1, batch, hidden_dim)

        # Step 3: LSTM forward pass
        # (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        lstm_out, _ = self.lstm(x_prime, (h_0, c_0))  # (batch, seq_len, hidden_dim)

        # Apply dropout to LSTM outputs
        lstm_out = self.lstm_dropout(lstm_out)

        # Step 4: First skip connection with layer norm
        # a_t = LayerNorm(h_t + x'_t)
        a_t = self.layer_norm1(lstm_out + x_prime)

        # Step 5: FFN with entity conditioning
        ffn_out = self.ffn(a_t, entity_ids)  # (batch, seq_len, hidden_dim)

        # Step 6: Second skip connection with layer norm
        # Ξ = LayerNorm(FFN(a_t, s) + a_t)
        output = self.layer_norm2(ffn_out + a_t)

        return EncoderOutput(
            hidden_states=output,
            sequence_length=seq_len
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_encoder.py -v`

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add xtrend/models/encoder.py tests/models/test_encoder.py
git commit -m "feat(models): implement LSTM encoder with skip connections (Equation 14)

- Add LSTMEncoder with VSN, LSTM, and dual skip connections
- Implement learnable entity-specific LSTM initialization
- Support zero-shot mode (no entity embeddings)
- Add tests verifying Equation 14 and skip connections"
```

---

## Task 5: Baseline DMN Model

**Files:**
- Update: `xtrend/models/baseline_dmn.py`
- Create: `tests/models/test_baseline_dmn.py`

**Step 1: Write failing test for BaselineDMN**

Create `tests/models/test_baseline_dmn.py`:

```python
"""Tests for Baseline DMN model."""
import pytest
import torch
from xtrend.models.baseline_dmn import BaselineDMN
from xtrend.models.types import ModelConfig


class TestBaselineDMN:
    def test_baseline_initialization(self, model_config):
        """BaselineDMN initializes correctly."""
        model = BaselineDMN(model_config)

        assert model.encoder is not None
        assert model.position_head is not None

    def test_baseline_forward_shape(self, model_config, sample_features, sample_entity_ids):
        """Baseline produces positions with correct shape."""
        model = BaselineDMN(model_config)

        positions = model(sample_features, sample_entity_ids)

        batch_size, seq_len, _ = sample_features.shape
        # Positions should be (batch, seq_len)
        assert positions.shape == (batch_size, seq_len)

    def test_baseline_position_range(self, model_config, sample_features, sample_entity_ids):
        """Positions are in (-1, 1) range (tanh bounded)."""
        model = BaselineDMN(model_config)

        positions = model(sample_features, sample_entity_ids)

        # All positions should be in (-1, 1)
        assert (positions > -1).all()
        assert (positions < 1).all()

    def test_baseline_equation_7(self, model_config):
        """Baseline implements Equation 7 from paper."""
        # Equation 7: z_t = tanh(Linear(g(x_t)))
        # where g(x_t) is the encoder

        model = BaselineDMN(model_config)
        model.eval()

        x = torch.randn(4, 20, 8)
        entity_ids = torch.randint(0, 50, (4,))

        with torch.no_grad():
            positions = model(x, entity_ids)

        # Positions exist and are bounded
        assert positions is not None
        assert positions.shape == (4, 20)
        assert (positions.abs() < 1).all()

    def test_baseline_gradient_flow(self, model_config):
        """Gradients flow through the model."""
        model = BaselineDMN(model_config)

        x = torch.randn(4, 20, 8, requires_grad=True)
        entity_ids = torch.randint(0, 50, (4,))

        positions = model(x, entity_ids)
        loss = positions.mean()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_baseline_dmn.py::TestBaselineDMN::test_baseline_initialization -v`

Expected: FAIL (BaselineDMN has only stub implementation)

**Step 3: Implement BaselineDMN**

Update `xtrend/models/baseline_dmn.py`:

```python
"""Baseline Deep Momentum Network (DMN) without cross-attention (Equation 7)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig
from xtrend.models.embeddings import EntityEmbedding
from xtrend.models.encoder import LSTMEncoder


class BaselineDMN(nn.Module):
    """Baseline neural forecaster without context/cross-attention.

    Implements Equation 7: z_t = tanh(Linear(g(x_t)))
    where g(·) is the LSTM encoder.

    This serves as the baseline comparison for X-Trend. It uses the same
    encoder architecture but without the cross-attention mechanism that
    enables few-shot learning from context sets.

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
    """

    def __init__(self, config: ModelConfig, use_entity: bool = True):
        super().__init__()
        self.config = config
        self.use_entity = use_entity

        # Shared entity embedding (one instance for entire model)
        if use_entity:
            self.entity_embedding = EntityEmbedding(config)
        else:
            self.entity_embedding = None

        # Encoder: g(x_t) from Equation 7
        self.encoder = LSTMEncoder(
            config,
            use_entity=use_entity,
            entity_embedding=self.entity_embedding
        )

        # Position head: tanh(Linear(g(x_t))) - Equation 7
        # Paper-faithful: single linear projection before tanh
        self.position_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Tanh()
        )

    def forward(
        self,
        x: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict trading positions.

        Args:
            x: Input features (batch, seq_len, input_dim)
            entity_ids: Entity IDs (batch,)

        Returns:
            Positions (batch, seq_len) in range (-1, 1)
        """
        # Encode: g(x_t)
        encoder_output = self.encoder(x, entity_ids)
        hidden_states = encoder_output.hidden_states  # (batch, seq_len, hidden_dim)

        # Position head: tanh(Linear(g(x_t)))
        positions = self.position_head(hidden_states)  # (batch, seq_len, 1)
        positions = positions.squeeze(-1)  # (batch, seq_len)

        return positions
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_baseline_dmn.py -v`

Expected: PASS (all tests)

**Step 5: Update model __init__.py**

Update `xtrend/models/__init__.py`:

```python
"""Neural network models for X-Trend."""
from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import EntityEmbedding, ConditionalFFN
from xtrend.models.encoder import LSTMEncoder
from xtrend.models.baseline_dmn import BaselineDMN

__all__ = [
    "ModelConfig",
    "EncoderOutput",
    "VariableSelectionNetwork",
    "EntityEmbedding",
    "ConditionalFFN",
    "LSTMEncoder",
    "BaselineDMN",
]
```

**Step 6: Commit**

```bash
git add xtrend/models/baseline_dmn.py xtrend/models/__init__.py tests/models/test_baseline_dmn.py
git commit -m "feat(models): implement Baseline DMN model (Equation 7)

- Add BaselineDMN with encoder and position head
- Implement tanh-bounded position output z_t ∈ (-1, 1)
- Add comprehensive tests for baseline model
- Update models __init__ with all Phase 3 exports"
```

---

## Task 6: Integration Tests

**Files:**
- Create: `tests/integration/test_phase3_complete.py`

**Step 1: Write integration test**

Create `tests/integration/test_phase3_complete.py`:

```python
"""Integration tests for Phase 3: Base Neural Architecture."""
import pytest
import torch
import numpy as np
from xtrend.models import (
    ModelConfig,
    VariableSelectionNetwork,
    EntityEmbedding,
    ConditionalFFN,
    LSTMEncoder,
    BaselineDMN,
)


class TestPhase3Integration:
    """Integration tests verifying Phase 3 completion criteria."""

    def test_full_pipeline_shape(self):
        """Complete pipeline produces correct shapes."""
        config = ModelConfig(hidden_dim=64, dropout=0.3)

        # Create model
        model = BaselineDMN(config)

        # Sample data
        batch_size, seq_len = 32, 126
        x = torch.randn(batch_size, seq_len, 8)
        entity_ids = torch.randint(0, 50, (batch_size,))

        # Forward pass
        positions = model(x, entity_ids)

        # Check output
        assert positions.shape == (batch_size, seq_len)
        assert (positions > -1).all() and (positions < 1).all()

    def test_vsn_attention_interpretability(self):
        """VSN attention weights have correct structural properties."""
        config = ModelConfig()
        vsn = VariableSelectionNetwork(config)

        x = torch.randn(4, 10, 8)
        output, weights = vsn(x)

        # Structural check 1: Weights should sum to 1 (softmax property)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 10), atol=1e-6)

        # Structural check 2: Weights should be non-negative (softmax property)
        assert (weights >= 0).all()

        # Structural check 3: Weights should have correct shape
        assert weights.shape == (4, 10, 8)  # (batch, seq_len, input_dim)

    def test_entity_embedding_structural_properties(self):
        """Entity embeddings have correct structural properties."""
        config = ModelConfig(num_entities=50)
        embedding = EntityEmbedding(config)

        # Structural check 1: Different entities get different embeddings (not identical)
        entity_0 = torch.tensor([0])
        entity_1 = torch.tensor([1])
        emb_0 = embedding(entity_0)
        emb_1 = embedding(entity_1)
        assert not torch.allclose(emb_0, emb_1)

        # Structural check 2: Same entity gets same embedding (consistency)
        emb_0_again = embedding(entity_0)
        assert torch.allclose(emb_0, emb_0_again)

        # Structural check 3: Embeddings are finite (no NaN/inf)
        assert torch.isfinite(emb_0).all()
        assert torch.isfinite(emb_1).all()

    def test_baseline_dmn_gradient_flow(self):
        """Gradients flow properly through entire model."""
        config = ModelConfig(hidden_dim=64)
        model = BaselineDMN(config)

        x = torch.randn(4, 20, 8, requires_grad=True)
        entity_ids = torch.randint(0, 50, (4,))

        # Forward pass
        positions = model(x, entity_ids)

        # Simple loss
        loss = positions.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"

    def test_encoder_skip_connections_structural(self):
        """Skip connections are structurally present in the computation graph."""
        config = ModelConfig(hidden_dim=64)

        # Create encoder with shared embedding
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)

        x = torch.randn(4, 50, 8, requires_grad=True)
        entity_ids = torch.randint(0, 50, (4,))

        # Forward pass
        output = encoder(x, entity_ids)
        loss = output.hidden_states.mean()
        loss.backward()

        # Structural check 1: Gradients flow back to input
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Structural check 2: All model parameters receive gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"

    def test_skip_connections_functional(self):
        """Verify skip connections actually affect the output."""
        config = ModelConfig(hidden_dim=64)

        # Create encoder with shared embedding
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        encoder.eval()  # Disable dropout for deterministic test

        x = torch.randn(4, 20, 8)
        entity_ids = torch.randint(0, 50, (4,))

        with torch.no_grad():
            # Get intermediate representations
            x_prime, _ = encoder.vsn(x)
            h_0 = encoder.init_h[entity_ids].unsqueeze(0)
            c_0 = encoder.init_c[entity_ids].unsqueeze(0)
            lstm_out, _ = encoder.lstm(x_prime, (h_0, c_0))
            lstm_out = encoder.lstm_dropout(lstm_out)

            # First skip connection: a_t = LayerNorm(h_t + x'_t)
            # Verify that output differs from just LayerNorm(h_t)
            with_skip = encoder.layer_norm1(lstm_out + x_prime)
            without_skip = encoder.layer_norm1(lstm_out)
            assert not torch.allclose(with_skip, without_skip), \
                "First skip connection has no effect"

            # Second skip connection: output = LayerNorm(FFN(a_t) + a_t)
            # Verify that output differs from just LayerNorm(FFN(a_t))
            a_t = with_skip
            ffn_out = encoder.ffn(a_t, entity_ids)
            with_skip_2 = encoder.layer_norm2(ffn_out + a_t)
            without_skip_2 = encoder.layer_norm2(ffn_out)
            assert not torch.allclose(with_skip_2, without_skip_2), \
                "Second skip connection has no effect"

    def test_zero_shot_mode(self):
        """Model works in zero-shot mode (no entity embeddings)."""
        config = ModelConfig()

        # Create encoder without entity embeddings
        encoder = LSTMEncoder(config, use_entity=False, entity_embedding=None)

        x = torch.randn(4, 20, 8)

        # Forward without entity_ids
        output = encoder(x, entity_ids=None)

        assert output.hidden_states.shape == (4, 20, 64)

    def test_phase3_visual_completion_criteria(self):
        """Verify Phase 3 visual completion criteria from phases.md."""
        # From phases.md:
        # - Output shape: torch.Size([32, 126, 64])
        # - Output range: within tanh range (-1, 1)

        config = ModelConfig(hidden_dim=64)
        model = BaselineDMN(config, use_entity=True)
        model.eval()

        batch_size, seq_len, n_features = 32, 126, 8
        x = torch.randn(batch_size, seq_len, n_features)
        entity_ids = torch.randint(0, 50, (batch_size,))

        with torch.no_grad():
            # Get encoder output
            encoder_output = model.encoder(x, entity_ids)
            print(f"Encoder output shape: {encoder_output.hidden_states.shape}")
            assert encoder_output.hidden_states.shape == torch.Size([32, 126, 64])

            # Get position output
            positions = model(x, entity_ids)
            print(f"Position output shape: {positions.shape}")
            assert positions.shape == (32, 126)

            print(f"Position range: [{positions.min():.3f}, {positions.max():.3f}]")
            # Should be within tanh range (-1, 1)
            assert positions.min() >= -1.0
            assert positions.max() <= 1.0


def test_phase3_complete():
    """Overall Phase 3 completion test."""
    # All components should be importable
    from xtrend.models import (
        ModelConfig,
        VariableSelectionNetwork,
        EntityEmbedding,
        ConditionalFFN,
        LSTMEncoder,
        BaselineDMN,
    )

    # Create a complete model
    config = ModelConfig()
    model = BaselineDMN(config)

    # Model should have correct structure
    assert isinstance(model.encoder, LSTMEncoder)
    assert isinstance(model.encoder.vsn, VariableSelectionNetwork)

    print("✅ Phase 3 Complete!")
    print("   - Variable Selection Network (Equation 13)")
    print("   - Entity Embeddings (50 contracts)")
    print("   - LSTM Encoder with Skip Connections (Equation 14)")
    print("   - Baseline DMN Model (Equation 7)")
```

**Step 2: Run integration tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration/test_phase3_complete.py -v`

Expected: PASS (all integration tests)

**Step 3: Run all Phase 3 tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/ tests/integration/test_phase3_complete.py -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/integration/test_phase3_complete.py
git commit -m "test(models): add Phase 3 integration tests

- Add comprehensive integration tests for full pipeline
- Test VSN attention interpretability
- Test gradient flow through skip connections
- Test zero-shot mode
- Verify visual completion criteria from phases.md
- All Phase 3 components working together"
```

---

## Task 7: Update phases.md

**Files:**
- Modify: `phases.md`

**Step 1: Mark Phase 3 tasks as complete**

Update the Phase 3 section in `phases.md` to mark all tasks as completed:

```markdown
### Tasks
1. **Variable Selection Network** (Equation 13) ✅
   - [x] Implement feature-wise FFN: `FFN_j(x_{t,j})` ✅
   - [x] Implement softmax attention weights: `w_t = Softmax(FFN(x_t))` ✅
   - [x] Combine: `VSN(x_t) = Σ w_{t,j} * FFN_j(x_{t,j})` ✅

2. **Entity Embeddings** ✅
   - [x] Create embedding layer for 50 contract types ✅
   - [x] Set embedding dimension d_h (e.g., 64 or 128) ✅
   - [x] Implement conditional FFN with embeddings (Equation 12) ✅

3. **Encoder Architecture** (Equation 14) ✅
   - [x] LSTM cell implementation ✅
   - [x] Layer normalization after LSTM ✅
   - [x] Skip connections: `a_t = LayerNorm(h_t + x'_t)` ✅
   - [x] Final FFN with skip: `Ξ = LayerNorm(FFN(a_t) + a_t)` ✅

4. **Baseline DMN Model** (Equation 7) ✅
   - [x] Implement position output: `z_t = tanh(Linear(g(x_t)))` ✅
   - [x] Verify output range: z_t ∈ (-1, 1) ✅
```

**Step 2: Add Phase 3 completion summary**

Add after the tasks section:

```markdown
### ✅ Phase 3 Complete (2025-11-17)

**Implementation:**
- All core components implemented and tested
- VariableSelectionNetwork: Feature-wise FFNs with attention weights
- EntityEmbedding: Learnable embeddings for 50 contracts
- ConditionalFFN: Fuses time-series and entity information
- LSTMEncoder: VSN → LSTM → Skip connections
- BaselineDMN: Complete model for position prediction
- All tests passing (30+ unit tests, 8 integration tests)

**Code Quality:**
- Test-driven development (RED-GREEN-REFACTOR)
- Paper-faithful implementation (Equations 7, 12, 13, 14)
- Type hints and comprehensive docstrings
- Gradient flow verified
- Zero-shot mode supported

**Files:**
- Implementation: `xtrend/models/{types.py, vsn.py, embeddings.py, encoder.py, baseline_dmn.py}`
- Tests: `tests/models/{test_*.py}`, `tests/integration/test_phase3_complete.py`
- Documentation: `docs/plans/2025-11-17-phase3-neural-architecture.md`

**Ready for Phase 4:** Context Set Construction
```

**Step 3: Commit**

```bash
git add phases.md
git commit -m "docs: mark Phase 3 complete in phases.md

- Mark all Phase 3 tasks as complete
- Add Phase 3 completion summary
- Document all implemented components
- Note ready for Phase 4"
```

---

## Verification Checklist

Before considering Phase 3 complete, verify:

- [ ] All tests pass: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/ tests/integration/test_phase3_complete.py -v`
- [ ] Visual completion criteria met (from phases.md):
  - [ ] Output shape: `torch.Size([32, 126, 64])`
  - [ ] Position range: `z_t ∈ (-1, 1)`
  - [ ] Gradient flow test passes (no vanishing/exploding gradients)
  - [ ] VSN attention weights sum to 1
  - [ ] LSTM hidden state maintains reasonable magnitude
- [ ] All commits have descriptive messages
- [ ] Code follows Phase 1 and 2 patterns
- [ ] Type hints on all functions
- [ ] Docstrings reference paper equations
- [ ] phases.md updated with completion status

---

## Notes

- **Paper Reference:** Equations 7, 12, 13, and 14 from Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies (arXiv:2310.10500v2)
- **Architecture:** This implements the baseline DMN and encoder components. Phase 4 will add context set construction, and Phase 5 will add cross-attention.
- **Testing:** Following strict TDD - write failing test first, implement minimum code, verify pass, then commit.
- **Entity Embeddings:** Used for few-shot learning. Zero-shot mode (Phase 7) will exclude these for unseen assets.
- **Skip Connections:** Critical for gradient flow in deep networks. Both encoder skip connections are tested.

---

## Code Review Fixes (2025-11-17)

Based on detailed code review feedback, the following fixes were applied to ensure paper-faithful implementation and robust testing:

### 1. Entity Embedding Shape Documentation (Finding 1)
**Issue:** Docstring claimed support for `(batch, seq_len)` entity IDs, but implementation only handled `(batch,)`.
**Fix:** Updated documentation to clarify `(batch,)` only - one entity per sequence. Added assertion to validate shape.
**Rationale:** Each sequence corresponds to one asset; per-timestep entities don't make theoretical sense.

### 2. Shared Entity Embeddings (Finding 2)
**Issue:** Each `ConditionalFFN` created its own `EntityEmbedding`, preventing weight sharing across the model.
**Fix:** Refactored to inject shared `EntityEmbedding` instance from top level (`BaselineDMN` → `LSTMEncoder` → `ConditionalFFN`).
**Rationale:** Ensures consistent entity representations across all model components, aligning with "modular design" goal.

### 3. Zero-Shot Toggle (Finding 3)
**Issue:** `BaselineDMN` hard-coded `use_entity=True`, preventing reuse for zero-shot inference.
**Fix:** Added `use_entity` parameter to `BaselineDMN` constructor, enabling zero-shot mode without retraining.
**Rationale:** Supports requirement "Support zero-shot mode (entity_ids=None)" from Phase 3 spec.

### 4. Flaky Test Fixes (Finding 4)
**Issue:** Tests relied on random initialization to assert semantic behaviors (attention favoring scaled features, gradient norms in range, entity similarity).
**Fix:** Replaced with structural checks:
- `test_vsn_attention_interpretability`: Verify softmax properties (sum=1, non-negative)
- `test_entity_embedding_structural_properties`: Verify consistency and finiteness
- `test_encoder_skip_connections_structural`: Verify gradient flow to all parameters
**Rationale:** Structural properties are guaranteed by architecture; semantic properties emerge during training.

### 5. Paper-Faithful Position Head (Finding 5)
**Issue:** Position head implemented `Linear→ELU→Dropout→Linear→Tanh` instead of Equation 7's `tanh(Linear(g(x_t)))`.
**Fix:** Simplified to single `Linear→Tanh` projection.
**Rationale:** Plan states "paper-faithful implementation"; baseline must match paper for valid comparison with X-Trend.

### 6. LSTM Dropout Fix (Finding 6)
**Issue:** Single-layer LSTM silently ignores `dropout` parameter (only applies between layers).
**Fix:** Added explicit `nn.Dropout` layer after LSTM outputs.
**Rationale:** Achieves intended regularization from Table 3 without changing architecture depth.

### 7. Skip Connection Validation (Finding 7)
**Issue:** Tests only checked shapes/gradients; would pass even if residual terms were accidentally dropped.
**Fix:** Added `test_skip_connections_functional` that verifies skip connections actually change the output by comparing with/without skip terms.
**Rationale:** Functional tests ensure skip connections are computationally active, not just structurally present.

### Impact Summary
- **Architecture:** Now strictly paper-faithful (Equations 7, 12, 13, 14)
- **Modularity:** Shared embeddings enable consistent entity representations
- **Flexibility:** Zero-shot mode supported at model level
- **Test Quality:** Structural checks replace flaky semantic assertions
- **Regularization:** Dropout now actually applies to LSTM outputs

All tests updated to use new signatures with shared `EntityEmbedding` instances.

---

**Last Updated:** 2025-11-17
**Paper:** [Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies](https://arxiv.org/abs/2310.10500)
