# Phase 6: Decoder & Loss Functions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement LSTM decoder with prediction heads and loss functions to enable end-to-end training of X-Trend models (X-Trend, X-Trend-G, X-Trend-Q).

**Architecture:** Build LSTM decoder that fuses cross-attention output with target features (Equation 19), implement three prediction head types (position, Gaussian, quantile), create PTP modules to map distributions to positions, and implement loss functions (Sharpe, Gaussian NLL, Quantile regression, Joint losses). Supports three model variants with increasing sophistication.

**Tech Stack:** PyTorch, reuse encoder patterns (skip connections, layer norm), ConditionalFFN from Phase 3, cross-attention from Phase 5.

---

## Task 1: Implement LSTM Decoder Architecture

**Files:**
- Create: `xtrend/models/decoder.py`
- Test: `tests/models/test_decoder.py`
- Reference: `xtrend/models/encoder.py` (Phase 3 pattern)

**Background:** Decoder fuses cross-attention output with target features using same LSTM+skip pattern as encoder (Equations 19a-19d, Page 9).

**Step 1: Write failing test for decoder initialization**

Create `tests/models/test_decoder.py`:
```python
"""Tests for LSTM decoder."""
import pytest
import torch

from xtrend.models.decoder import LSTMDecoder
from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.embeddings import EntityEmbedding


class TestLSTMDecoder:
    """Test decoder architecture (Equations 19a-19d)."""

    @pytest.fixture
    def config(self):
        """Standard config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,  # ✅ FIXED: Correct field name
            dropout=0.1
        )

    def test_decoder_initialization(self, config):
        """Decoder initializes with correct components."""
        entity_embedding = EntityEmbedding(config)
        decoder = LSTMDecoder(config, use_entity=True, entity_embedding=entity_embedding)

        # Same structure as encoder
        assert decoder.vsn is not None
        assert decoder.lstm is not None
        assert decoder.lstm_dropout is not None
        assert decoder.layer_norm1 is not None
        assert decoder.ffn is not None
        assert decoder.layer_norm2 is not None
        # Decoder owns its own learnable initial states (mirrors encoder)
        assert hasattr(decoder, "init_h")
        assert hasattr(decoder, "init_c")
        assert decoder.init_h.shape == (config.num_entities, config.hidden_dim)
        assert decoder.init_c.shape == (config.num_entities, config.hidden_dim)
        assert decoder.init_h_generic.shape == (1, config.hidden_dim)
        assert decoder.init_c_generic.shape == (1, config.hidden_dim)

    def test_decoder_forward_pass(self, config):
        """Decoder fuses target features with cross-attention output."""
        batch_size, seq_len, hidden_dim = 2, 126, 64

        # Target features (raw inputs)
        target_features = torch.randn(batch_size, seq_len, config.input_dim)

        # Cross-attention output from Phase 5 (attended features)
        cross_attn_output = torch.randn(batch_size, seq_len, hidden_dim)

        entity_embedding = EntityEmbedding(config)
        decoder = LSTMDecoder(config, use_entity=True, entity_embedding=entity_embedding)

        output = decoder(
            target_features,
            cross_attn_output,
            entity_ids=torch.tensor([0, 1])
        )

        # Output shape same as encoder
        assert isinstance(output, EncoderOutput)
        assert output.hidden_states.shape == (batch_size, seq_len, hidden_dim)

    def test_decoder_zero_shot_mode(self, config):
        """Decoder works without entity embeddings (zero-shot)."""
        batch_size, seq_len = 2, 63

        target_features = torch.randn(batch_size, seq_len, config.input_dim)
        cross_attn_output = torch.randn(batch_size, seq_len, config.hidden_dim)

        # No entity embedding in zero-shot mode
        decoder = LSTMDecoder(config, use_entity=False, entity_embedding=None)

        output = decoder(target_features, cross_attn_output, entity_ids=None)

        assert output.hidden_states.shape == (batch_size, seq_len, config.hidden_dim)
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_decoder.py::TestLSTMDecoder::test_decoder_initialization -v
```

Expected: `ModuleNotFoundError: No module named 'xtrend.models.decoder'`

**Step 3: Implement LSTM decoder**

Create `xtrend/models/decoder.py`:
```python
"""LSTM decoder with cross-attention fusion (Equations 19a-19d)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import ConditionalFFN, EntityEmbedding


class LSTMDecoder(nn.Module):
    """LSTM decoder that fuses target features with cross-attention output.

    Architecture (Equation 19a-19d, Page 9):
        x'_t = LayerNorm ∘ FFN_1 ∘ Concat(VSN(x_t, s), y_t)
        (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        a_t = LayerNorm(h_t + x'_t)  # Skip connection 1
        Ξ_Dec = LayerNorm(FFN_2(a_t, s) + a_t)  # Skip connection 2

    The decoder combines:
    - Target features x_t (passed through VSN)
    - Cross-attention output y_t (from Phase 5)

    This allows the model to integrate both target and context information.

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
        entity_embedding: Shared entity embedding (if use_entity=True)
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
        self.entity_embedding = entity_embedding

        # ✅ FIXED: Variable selection with entity conditioning
        self.vsn = VariableSelectionNetwork(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # ✅ FIXED: Fusion layer using entity-conditioned FFN (Issue #6)
        # Note: ConditionalFFN doesn't support input_dim/output_dim parameters
        # We need to project first, then apply conditioning
        self.fusion_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fusion_ffn = ConditionalFFN(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)

        # LSTM with entity-conditioned initial state
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
            dropout=0.0  # We add explicit dropout layer
        )
        self.lstm_dropout = nn.Dropout(config.dropout)

        # Skip connection 1: after LSTM
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)

        # ✅ FIXED: Skip connection 2 with entity conditioning
        self.ffn = ConditionalFFN(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)

        # Learnable initial states (per entity + generic fallback just like encoder)
        if self.use_entity:
            assert entity_embedding is not None, "entity_embedding required when use_entity=True"
            self.init_h = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
            self.init_h_generic = nn.Parameter(torch.randn(1, config.hidden_dim))
            self.init_c_generic = nn.Parameter(torch.randn(1, config.hidden_dim))
        else:
            self.init_h = nn.Parameter(torch.randn(1, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(1, config.hidden_dim))

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> EncoderOutput:
        """Apply decoder to fuse target and cross-attention output.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs for conditioning (batch,) or None for zero-shot

        Returns:
            EncoderOutput with fused hidden states
        """
        batch_size = target_features.shape[0]

        # ✅ FIXED: VSN returns (output, weights) tuple - unpack it (Issue #1)
        x_selected, _ = self.vsn(target_features, entity_ids)

        # ✅ FIXED: Step 2: Concatenate and fuse with entity conditioning (Issue #6)
        concatenated = torch.cat([x_selected, cross_attn_output], dim=-1)
        x_projected = self.fusion_proj(concatenated)  # Project to hidden_dim
        x_fused_raw = self.fusion_ffn(x_projected, entity_ids)  # Entity conditioning
        x_fused = self.fusion_norm(x_fused_raw)  # Equation 19a: x'_t

        # Step 3: Initialize LSTM state (matches encoder behavior)
        if self.use_entity and entity_ids is not None:
            h0 = self.init_h[entity_ids]
            c0 = self.init_c[entity_ids]
        elif self.use_entity and entity_ids is None:
            h0 = self.init_h_generic.expand(batch_size, -1)
            c0 = self.init_c_generic.expand(batch_size, -1)
        else:
            h0 = self.init_h.expand(batch_size, -1)
            c0 = self.init_c.expand(batch_size, -1)

        h0 = h0.unsqueeze(0)
        c0 = c0.unsqueeze(0)
        initial_state = (h0, c0)

        # Step 4: LSTM processing
        # (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        lstm_out, _ = self.lstm(x_fused, initial_state)
        lstm_out = self.lstm_dropout(lstm_out)

        # Step 5: Skip connection 1
        # a_t = LayerNorm(h_t + x'_t)
        skip1 = self.layer_norm1(lstm_out + x_fused)

        # Step 6: Conditional FFN with skip connection 2
        # Ξ_Dec = LayerNorm(FFN(a_t, s) + a_t)
        ffn_out = self.ffn(skip1, entity_ids)
        output = self.layer_norm2(ffn_out + skip1)

        # ✅ FIXED: Return EncoderOutput with sequence_length (Issue #4)
        return EncoderOutput(
            hidden_states=output,
            sequence_length=output.shape[1]
        )
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_decoder.py::TestLSTMDecoder -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add xtrend/models/decoder.py tests/models/test_decoder.py
git commit -m "feat(decoder): implement LSTM decoder with cross-attention fusion"
```

---

## Task 2: Implement Prediction Head Types

**Files:**
- Modify: `xtrend/models/heads.py` (skeleton exists)
- Test: `tests/models/test_heads.py`

**Background:** Three head types for three model variants: Position (X-Trend), Gaussian (X-Trend-G), Quantile (X-Trend-Q).

**Step 1: Write failing test for prediction heads**

Create `tests/models/test_heads.py`:
```python
"""Tests for prediction heads."""
import pytest
import torch

from xtrend.models.heads import PositionHead, GaussianHead, QuantileHead
from xtrend.models.types import ModelConfig


class TestPredictionHeads:
    """Test prediction head variants."""

    @pytest.fixture
    def config(self):
        """Standard config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,  # ✅ FIXED: Correct field name
            dropout=0.1
        )

    def test_position_head(self, config):
        """Position head outputs trading positions in (-1, 1)."""
        batch_size, seq_len = 2, 126

        decoder_output = torch.randn(batch_size, seq_len, config.hidden_dim)

        head = PositionHead(config)
        positions = head(decoder_output)

        assert positions.shape == (batch_size, seq_len)
        # Positions in (-1, 1) due to tanh
        assert (positions > -1).all() and (positions < 1).all()

    def test_gaussian_head(self, config):
        """Gaussian head outputs mean and positive std dev."""
        batch_size, seq_len = 2, 126

        decoder_output = torch.randn(batch_size, seq_len, config.hidden_dim)

        head = GaussianHead(config)
        mean, std = head(decoder_output)

        assert mean.shape == (batch_size, seq_len)
        assert std.shape == (batch_size, seq_len)
        # Std must be positive
        assert (std > 0).all()

    def test_quantile_head(self, config):
        """Quantile head outputs 13 quantiles."""
        batch_size, seq_len = 2, 126
        num_quantiles = 13  # Paper: 0.01, 0.05, 0.1, 0.2, ..., 0.95, 0.99

        decoder_output = torch.randn(batch_size, seq_len, config.hidden_dim)

        head = QuantileHead(config, num_quantiles=num_quantiles)
        quantiles = head(decoder_output)

        assert quantiles.shape == (batch_size, seq_len, num_quantiles)

    def test_quantiles_ordered(self, config):
        """Quantiles are monotonically increasing."""
        batch_size, seq_len = 2, 10

        decoder_output = torch.randn(batch_size, seq_len, config.hidden_dim)

        head = QuantileHead(config, num_quantiles=13)
        quantiles = head(decoder_output)

        # Check monotonicity: Q_0.01 ≤ Q_0.05 ≤ ... ≤ Q_0.99
        for b in range(batch_size):
            for t in range(seq_len):
                q = quantiles[b, t, :]
                # Each quantile should be >= previous
                diffs = q[1:] - q[:-1]
                assert (diffs >= -1e-6).all(), \
                    f"Quantiles not ordered at batch {b}, time {t}"

    # ✅ ADDED: Enhanced tests from Issue #7
    def test_gaussian_head_deterministic(self, config):
        """Gaussian head produces consistent outputs for same input."""
        decoder_output = torch.randn(1, 10, config.hidden_dim)
        torch.manual_seed(42)

        head = GaussianHead(config)
        head.eval()  # Disable dropout

        mean1, std1 = head(decoder_output)
        mean2, std2 = head(decoder_output)

        # Same input -> same output (deterministic in eval mode)
        assert torch.allclose(mean1, mean2, atol=1e-6)
        assert torch.allclose(std1, std2, atol=1e-6)

    def test_gaussian_head_mean_std_not_swapped(self, config):
        """Verify mean and std are not accidentally swapped."""
        # Create input that should produce predictable outputs
        decoder_output = torch.ones(1, 10, config.hidden_dim) * 10.0

        head = GaussianHead(config)
        head.eval()

        mean, std = head(decoder_output)

        # Mean can be any value, but std must be positive and reasonably small
        assert (std > 0).all()
        assert (std < 100).all()  # Sanity check - shouldn't explode
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_heads.py::TestPredictionHeads::test_position_head -v
```

Expected: `ImportError` or test fails due to incomplete implementation

**Step 3: Implement prediction heads**

Modify `xtrend/models/heads.py`:
```python
"""Prediction heads for X-Trend model variants."""
import torch
import torch.nn as nn
from typing import Tuple

from xtrend.models.types import ModelConfig


class PositionHead(nn.Module):
    """Direct position prediction head for X-Trend (Equation 7).

    Maps decoder output directly to trading position z ∈ (-1, 1).

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Single linear layer + tanh
        # Equation 7: z = tanh(Linear(g(x)))
        self.linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Predict trading positions.

        Args:
            decoder_output: Decoder hidden states (batch, seq_len, hidden_dim)

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # Linear projection + tanh
        logits = self.linear(decoder_output).squeeze(-1)  # (batch, seq_len)
        positions = torch.tanh(logits)
        return positions


class GaussianHead(nn.Module):
    """Gaussian prediction head for X-Trend-G (Equation 20).

    Predicts mean μ and standard deviation σ of return distribution.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Separate networks for mean and std
        self.mean_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        self.std_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Gaussian parameters.

        Args:
            decoder_output: Decoder hidden states (batch, seq_len, hidden_dim)

        Returns:
            mean: Predicted mean (batch, seq_len)
            std: Predicted std dev (batch, seq_len), positive
        """
        mean = self.mean_net(decoder_output).squeeze(-1)  # (batch, seq_len)
        std_logits = self.std_net(decoder_output).squeeze(-1)

        # ✅ FIXED: Ensure std is positive and safe after squaring (Issue #5)
        # σ = softplus(logits) + 1e-3 (safe after squaring to variance)
        std = torch.nn.functional.softplus(std_logits) + 1e-3

        return mean, std


class QuantileHead(nn.Module):
    """Quantile prediction head for X-Trend-Q (Equation 22).

    Predicts multiple quantiles of return distribution.

    Paper uses 13 quantiles:
    [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    Args:
        config: Model configuration
        num_quantiles: Number of quantiles to predict (default: 13)
    """

    def __init__(self, config: ModelConfig, num_quantiles: int = 13):
        super().__init__()
        self.config = config
        self.num_quantiles = num_quantiles

        # Network to predict all quantiles
        self.quantile_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_quantiles)
        )

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Predict quantiles.

        Args:
            decoder_output: Decoder hidden states (batch, seq_len, hidden_dim)

        Returns:
            quantiles: Predicted quantiles (batch, seq_len, num_quantiles)
                Monotonically increasing along last dimension
        """
        # Predict quantile differences (ensure ordering)
        raw_quantiles = self.quantile_net(decoder_output)  # (batch, seq_len, num_quantiles)

        # ✅ FIXED: Vectorized monotonicity enforcement (Issue #9)
        # First quantile (no constraint)
        first_quantile = raw_quantiles[:, :, :1]

        # Remaining quantiles as positive increments
        increments = torch.nn.functional.softplus(raw_quantiles[:, :, 1:])

        # Cumulative sum ensures monotonicity
        # quantiles = [q_0, q_0 + inc_1, q_0 + inc_1 + inc_2, ...]
        quantiles = torch.cat([
            first_quantile,
            first_quantile + torch.cumsum(increments, dim=-1)
        ], dim=-1)  # Vectorized, single backward pass

        return quantiles
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_heads.py::TestPredictionHeads -v
```

Expected: All 6 tests PASS (including new tests)

**Step 5: Commit**

```bash
git add xtrend/models/heads.py tests/models/test_heads.py
git commit -m "feat(heads): implement position, Gaussian, and quantile prediction heads"
```

---

## Task 3: Implement PTP (Predictive To Position) Modules

**Files:**
- Modify: `xtrend/models/heads.py`
- Test: `tests/models/test_ptp.py`

**Background:** PTP modules map distribution parameters to trading positions for joint training (Page 9).

**Step 1: Write failing test for PTP modules**

Create `tests/models/test_ptp.py`:
```python
"""Tests for PTP (Predictive To Position) modules."""
import pytest
import torch

from xtrend.models.heads import PTP_G, PTP_Q
from xtrend.models.types import ModelConfig


class TestPTP:
    """Test PTP modules (Page 9)."""

    @pytest.fixture
    def config(self):
        """Standard config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,  # ✅ FIXED: Correct field name
            dropout=0.1
        )

    def test_ptp_g_forward(self, config):
        """PTP_G maps (mean, std) to positions."""
        batch_size, seq_len = 2, 126

        # Gaussian parameters
        mean = torch.randn(batch_size, seq_len)
        std = torch.rand(batch_size, seq_len) + 0.5  # Positive

        ptp = PTP_G(config)
        positions = ptp(mean, std)

        assert positions.shape == (batch_size, seq_len)
        # Positions in (-1, 1) due to tanh
        assert (positions > -1).all() and (positions < 1).all()

    def test_ptp_q_forward(self, config):
        """PTP_Q maps quantiles to positions."""
        batch_size, seq_len, num_quantiles = 2, 126, 13

        # Quantiles (sorted)
        quantiles = torch.randn(batch_size, seq_len, num_quantiles).sort(dim=-1)[0]

        ptp = PTP_Q(config, num_quantiles=num_quantiles)
        positions = ptp(quantiles)

        assert positions.shape == (batch_size, seq_len)
        # Positions in (-1, 1) due to tanh
        assert (positions > -1).all() and (positions < 1).all()

    def test_ptp_gradient_flow(self, config):
        """PTP allows gradient flow for joint training."""
        batch_size, seq_len = 2, 10

        mean = torch.randn(batch_size, seq_len, requires_grad=True)
        std = torch.rand(batch_size, seq_len, requires_grad=True) + 0.5

        ptp = PTP_G(config)
        positions = ptp(mean, std)

        # Backward pass
        loss = positions.sum()
        loss.backward()

        # Check gradients exist
        assert mean.grad is not None
        assert std.grad is not None
        assert not torch.isnan(mean.grad).any()

    # ✅ ADDED: PTP_Q gradient test (Issue #11)
    def test_ptp_q_gradient_flow(self, config):
        """PTP_Q allows gradient flow for joint training."""
        batch_size, seq_len, num_quantiles = 2, 10, 13

        quantiles = torch.randn(batch_size, seq_len, num_quantiles, requires_grad=True)
        quantiles = quantiles.sort(dim=-1)[0]  # Ensure ordering

        ptp = PTP_Q(config, num_quantiles=num_quantiles)
        positions = ptp(quantiles)

        # Backward pass
        loss = positions.sum()
        loss.backward()

        # Check gradients exist
        assert quantiles.grad is not None
        assert not torch.isnan(quantiles.grad).any()
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_ptp.py::TestPTP::test_ptp_g_forward -v
```

Expected: `ImportError` or test fails

**Step 3: Implement PTP modules**

Append to `xtrend/models/heads.py`:
```python
class PTP_G(nn.Module):
    """Predictive To Position module for Gaussian head (Page 9).

    Maps Gaussian parameters (mean, std) to trading position z ∈ (-1, 1).

    Used for joint training: L = α * L_MLE + L_Sharpe^PTP_G

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ✅ FIXED: No dropout - paper specifies deterministic mapping (Issue #10)
        self.ffn = nn.Sequential(
            nn.Linear(2, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Map Gaussian parameters to positions.

        Args:
            mean: Predicted mean (batch, seq_len)
            std: Predicted std dev (batch, seq_len)

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # Stack mean and std
        params = torch.stack([mean, std], dim=-1)  # (batch, seq_len, 2)

        # FFN + tanh
        logits = self.ffn(params).squeeze(-1)  # (batch, seq_len)
        positions = torch.tanh(logits)

        return positions


class PTP_Q(nn.Module):
    """Predictive To Position module for Quantile head (Page 9).

    Maps quantile predictions to trading position z ∈ (-1, 1).

    Used for joint training: L = α * L_QRE + L_Sharpe^PTP_Q

    Args:
        config: Model configuration
        num_quantiles: Number of quantiles (default: 13)
    """

    def __init__(self, config: ModelConfig, num_quantiles: int = 13):
        super().__init__()
        self.config = config
        self.num_quantiles = num_quantiles

        # ✅ FIXED: No dropout - paper specifies deterministic mapping (Issue #10)
        self.ffn = nn.Sequential(
            nn.Linear(num_quantiles, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Map quantiles to positions.

        Args:
            quantiles: Predicted quantiles (batch, seq_len, num_quantiles)

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # FFN + tanh
        logits = self.ffn(quantiles).squeeze(-1)  # (batch, seq_len)
        positions = torch.tanh(logits)

        return positions
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_ptp.py::TestPTP -v
```

Expected: All 4 tests PASS (including new test)

**Step 5: Commit**

```bash
git add xtrend/models/heads.py tests/models/test_ptp.py
git commit -m "feat(ptp): implement PTP_G and PTP_Q modules for joint training"
```

---

## Task 4: Implement Sharpe Ratio Loss Function

**Files:**
- Modify: `xtrend/models/losses.py` (skeleton exists)
- Test: `tests/models/test_losses.py`

**Background:** Sharpe loss (Equation 8) is the core innovation of DMNs - directly optimizes risk-adjusted returns.

**Step 1: Write failing test for Sharpe loss**

Create `tests/models/test_losses.py`:
```python
"""Tests for loss functions."""
import pytest
import torch

from xtrend.models.losses import sharpe_loss, gaussian_nll_loss, quantile_loss


class TestSharpeLoss:
    """Test Sharpe ratio loss (Equation 8)."""

    def test_sharpe_loss_basic(self):
        """Sharpe loss computes annualized risk-adjusted returns."""
        batch_size, seq_len = 2, 126

        # Deterministic positive PnL so Sharpe is positive → loss negative
        returns = torch.ones(batch_size, seq_len) * 0.01
        positions = torch.ones(batch_size, seq_len) * 0.5

        loss = sharpe_loss(positions, returns, warmup_steps=63)

        # Loss should be negative because we minimize (-√252 * Sharpe)
        assert loss.ndim == 0
        assert loss.item() < 0

    def test_sharpe_warmup_period(self):
        """Warmup period ignores first l_s steps."""
        batch_size, seq_len = 2, 126
        warmup = 63

        returns = torch.randn(batch_size, seq_len)
        positions = torch.randn(batch_size, seq_len)

        # Modify first warmup steps to extreme values
        returns[:, :warmup] = 1000.0
        positions[:, :warmup] = 1.0

        loss = sharpe_loss(positions, returns, warmup_steps=warmup)

        # Loss should ignore warmup (not explode from extreme values)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sharpe_annualization(self):
        """Sharpe includes √252 annualization factor."""
        batch_size, seq_len = 1, 200

        # Constant positive returns
        returns = torch.ones(batch_size, seq_len) * 0.01
        positions = torch.ones(batch_size, seq_len) * 0.5

        loss = sharpe_loss(positions, returns, warmup_steps=0)

        # With constant returns, Sharpe should be large (low loss)
        # Loss = -√252 * (mean / std)
        # std ≈ 0 for constant, but we add epsilon
        assert loss.item() < 0  # Negative because good performance

    def test_sharpe_gradient_flow(self):
        """Gradients flow through Sharpe loss."""
        batch_size, seq_len = 2, 100

        returns = torch.randn(batch_size, seq_len)
        positions = torch.randn(batch_size, seq_len, requires_grad=True)

        loss = sharpe_loss(positions, returns, warmup_steps=20)

        # Backward pass
        loss.backward()

        # Check gradient exists
        assert positions.grad is not None
        assert not torch.isnan(positions.grad).any()

    # ✅ ADDED: Edge case test (Issue #8)
    def test_sharpe_invalid_warmup(self):
        """Sharpe loss validates warmup_steps < seq_len."""
        batch_size, seq_len = 2, 100

        returns = torch.randn(batch_size, seq_len)
        positions = torch.randn(batch_size, seq_len)

        # Warmup >= seq_len should raise error
        with pytest.raises(ValueError, match="warmup_steps.*must be.*sequence length"):
            sharpe_loss(positions, returns, warmup_steps=100)
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_losses.py::TestSharpeLoss::test_sharpe_loss_basic -v
```

Expected: Test fails or `NotImplementedError`

**Step 3: Implement Sharpe loss**

Modify `xtrend/models/losses.py`:
```python
"""Loss functions for X-Trend models."""
import torch
import torch.nn.functional as F


def sharpe_loss(
    positions: torch.Tensor,
    returns: torch.Tensor,
    warmup_steps: int = 63,
    eps: float = 1e-8
) -> torch.Tensor:
    """Sharpe ratio loss (Equation 8, Page 5).

    L_Sharpe = -√252 * mean(r * z) / std(r * z)

    Where:
    - r: Scaled returns (σ_tgt / σ_t * r_{t+1})
    - z: Predicted positions
    - √252: Annualization factor (trading days per year)
    - Batch Ω: All (i, t) pairs excluding warmup period

    Args:
        positions: Predicted positions (batch, seq_len) in (-1, 1)
        returns: Scaled returns (batch, seq_len)
        warmup_steps: Ignore first l_s steps (default: 63)
        eps: Small constant for numerical stability

    Returns:
        loss: Scalar, negative Sharpe ratio
    """
    # ✅ ADDED: Validate inputs (Issue #8)
    if warmup_steps >= positions.shape[1]:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < sequence length ({positions.shape[1]})"
        )

    # Remove warmup period
    if warmup_steps > 0:
        positions = positions[:, warmup_steps:]
        returns = returns[:, warmup_steps:]

    # ✅ ADDED: Ensure we have data left (Issue #8)
    if positions.numel() == 0:
        raise ValueError("No data remaining after warmup removal")

    # Compute strategy returns: r_t * z_t
    strategy_returns = returns * positions

    # Flatten to single batch Ω
    strategy_returns_flat = strategy_returns.flatten()

    # ✅ FIXED: Use population std (unbiased=False) per Issue #8
    mean_return = strategy_returns_flat.mean()
    std_return = strategy_returns_flat.std(unbiased=False) + eps

    sharpe_ratio = mean_return / std_return

    # Annualize and negate (minimize = maximize Sharpe)
    annualization_factor = torch.sqrt(torch.tensor(252.0, device=positions.device))
    loss = -annualization_factor * sharpe_ratio

    return loss


def gaussian_nll_loss(
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss (Equation 20, Page 9).

    L_MLE = -1/|Ω| * Σ log p(r | μ, σ)

    Where p(r | μ, σ) = N(r; μ, σ) is Gaussian likelihood.

    Args:
        mean: Predicted mean (batch, seq_len)
        std: Predicted std dev (batch, seq_len), positive
        target: Scaled returns (batch, seq_len)
        warmup_steps: Ignore first l_s steps (default: 63)

    Returns:
        loss: Scalar, mean negative log-likelihood
    """
    # ✅ ADDED: Validate inputs (Issue #8)
    if warmup_steps >= mean.shape[1]:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < sequence length ({mean.shape[1]})"
        )

    # Remove warmup period
    if warmup_steps > 0:
        mean = mean[:, warmup_steps:]
        std = std[:, warmup_steps:]
        target = target[:, warmup_steps:]

    # ✅ FIXED: Clamp variance explicitly (Issue #5)
    variance = std.pow(2).clamp(min=1e-6)

    # Gaussian NLL: -log p(target | mean, std)
    loss = F.gaussian_nll_loss(mean, target, variance, reduction='mean')

    return loss


def quantile_loss(
    quantile_preds: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Quantile regression loss (Equation 22, Page 9).

    L_QRE = 1/(|Ω| * |H|) * Σ [η * (r - Q_η)_+ + (1-η) * (Q_η - r)_+]

    Where:
    - Q_η: Predicted quantile at level η
    - (·)_+ = max(0, ·): Pinball loss
    - H: Set of quantile levels [0.01, 0.05, 0.1, ..., 0.99]

    Args:
        quantile_preds: Predicted quantiles (batch, seq_len, num_quantiles)
        target: Scaled returns (batch, seq_len)
        quantile_levels: Quantile levels (num_quantiles,)
            e.g., [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        warmup_steps: Ignore first l_s steps (default: 63)

    Returns:
        loss: Scalar, mean quantile loss
    """
    # ✅ ADDED: Validate inputs (Issue #8)
    if warmup_steps >= quantile_preds.shape[1]:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be < sequence length ({quantile_preds.shape[1]})"
        )

    # Remove warmup period
    if warmup_steps > 0:
        quantile_preds = quantile_preds[:, warmup_steps:, :]
        target = target[:, warmup_steps:]

    # Expand target to match quantile predictions
    # target: (batch, seq_len) -> (batch, seq_len, num_quantiles)
    target_expanded = target.unsqueeze(-1).expand_as(quantile_preds)

    # Compute errors: r - Q_η
    errors = target_expanded - quantile_preds

    # Pinball loss: η * (r - Q_η)_+ + (1-η) * (Q_η - r)_+
    # Equivalent to: max(η * error, (η - 1) * error)
    quantile_levels = quantile_levels.to(quantile_preds.device)
    loss_per_quantile = torch.max(
        quantile_levels * errors,
        (quantile_levels - 1) * errors
    )

    # Mean over all dimensions
    loss = loss_per_quantile.mean()

    return loss


def joint_gaussian_loss(
    mean: torch.Tensor,
    std: torch.Tensor,
    positions: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Joint loss for X-Trend-G (Equation 21, Page 9).

    L_Joint^MLE = α * L_MLE + L_Sharpe^PTP_G

    Args:
        mean: Predicted mean from Gaussian head (batch, seq_len)
        std: Predicted std from Gaussian head (batch, seq_len)
        positions: Positions from PTP_G(mean, std) (batch, seq_len)
        target: Scaled returns (batch, seq_len)
        alpha: Weighting for MLE loss (default: 1.0, from Table 4)
        warmup_steps: Warmup period (default: 63)

    Returns:
        loss: Scalar, joint loss
    """
    # Gaussian NLL loss
    mle_loss = gaussian_nll_loss(mean, std, target, warmup_steps)

    # Sharpe loss on PTP positions
    sharpe = sharpe_loss(positions, target, warmup_steps)

    # Weighted combination
    loss = alpha * mle_loss + sharpe

    return loss


def joint_quantile_loss(
    quantile_preds: torch.Tensor,
    quantile_levels: torch.Tensor,
    positions: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 5.0,
    warmup_steps: int = 63
) -> torch.Tensor:
    """Joint loss for X-Trend-Q (Equation 23, Page 9).

    L_Joint^QRE = α * L_QRE + L_Sharpe^PTP_Q

    Args:
        quantile_preds: Predicted quantiles (batch, seq_len, num_quantiles)
        quantile_levels: Quantile levels (num_quantiles,)
        positions: Positions from PTP_Q(quantiles) (batch, seq_len)
        target: Scaled returns (batch, seq_len)
        alpha: Weighting for QRE loss (default: 5.0, from Table 4)
        warmup_steps: Warmup period (default: 63)

    Returns:
        loss: Scalar, joint loss
    """
    # Quantile regression loss
    qre_loss = quantile_loss(quantile_preds, target, quantile_levels, warmup_steps)

    # Sharpe loss on PTP positions
    sharpe = sharpe_loss(positions, target, warmup_steps)

    # Weighted combination
    loss = alpha * qre_loss + sharpe

    return loss
```

**Step 4: Add tests for other losses**

Append to `tests/models/test_losses.py`:
```python
class TestGaussianNLLLoss:
    """Test Gaussian NLL loss (Equation 20)."""

    def test_gaussian_nll_basic(self):
        """Gaussian NLL loss computes likelihood."""
        batch_size, seq_len = 2, 126

        mean = torch.randn(batch_size, seq_len)
        std = torch.rand(batch_size, seq_len) + 0.5  # Positive
        target = torch.randn(batch_size, seq_len)

        loss = gaussian_nll_loss(mean, std, target, warmup_steps=63)

        assert loss.ndim == 0
        assert loss.item() > 0  # NLL is positive


class TestQuantileLoss:
    """Test quantile regression loss (Equation 22)."""

    def test_quantile_loss_basic(self):
        """Quantile loss uses pinball loss."""
        batch_size, seq_len, num_quantiles = 2, 126, 13

        quantile_preds = torch.randn(batch_size, seq_len, num_quantiles).sort(dim=-1)[0]
        target = torch.randn(batch_size, seq_len)
        levels = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        loss = quantile_loss(quantile_preds, target, levels, warmup_steps=63)

        assert loss.ndim == 0
        assert loss.item() > 0


class TestJointLosses:
    """Test joint losses (Equations 21, 23)."""

    def test_joint_gaussian_loss(self):
        """Joint Gaussian loss combines MLE and Sharpe."""
        batch_size, seq_len = 2, 126

        mean = torch.randn(batch_size, seq_len)
        std = torch.rand(batch_size, seq_len) + 0.5
        positions = torch.tanh(torch.randn(batch_size, seq_len))
        target = torch.randn(batch_size, seq_len)

        loss = joint_gaussian_loss(mean, std, positions, target, alpha=1.0, warmup_steps=63)

        assert loss.ndim == 0

    def test_joint_quantile_loss(self):
        """Joint Quantile loss combines QRE and Sharpe."""
        batch_size, seq_len, num_quantiles = 2, 126, 13

        quantile_preds = torch.randn(batch_size, seq_len, num_quantiles).sort(dim=-1)[0]
        levels = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        positions = torch.tanh(torch.randn(batch_size, seq_len))
        target = torch.randn(batch_size, seq_len)

        loss = joint_quantile_loss(quantile_preds, levels, positions, target, alpha=5.0, warmup_steps=63)

        assert loss.ndim == 0
```

**Step 5: Run all loss tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_losses.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add xtrend/models/losses.py tests/models/test_losses.py
git commit -m "feat(losses): implement Sharpe, Gaussian NLL, Quantile, and Joint losses"
```

---

## Task 5: Create Integrated X-Trend Model Classes

**Files:**
- Modify: `xtrend/models/xtrend.py` (replace skeleton)
- Test: `tests/models/test_xtrend_models.py`

**Background:** Three model variants: X-Trend (position only), X-Trend-G (Gaussian), X-Trend-Q (Quantile).

**Step 1: Write failing test for model variants**

Create `tests/models/test_xtrend_models.py`:
```python
"""Tests for integrated X-Trend model variants."""
import pytest
import torch

from xtrend.models.xtrend import XTrend, XTrendG, XTrendQ
from xtrend.models.types import ModelConfig
from xtrend.models.embeddings import EntityEmbedding


class TestXTrendModels:
    """Test complete model variants."""

    @pytest.fixture
    def config(self):
        """Standard config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,  # ✅ FIXED: Correct field name
            dropout=0.1
        )

    def test_xtrend_forward(self, config):
        """X-Trend: direct position prediction."""
        batch_size, target_len, context_size = 2, 126, 20

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        context_encoded = torch.randn(batch_size, context_size, config.hidden_dim)

        entity_embedding = EntityEmbedding(config)
        model = XTrend(config, entity_embedding=entity_embedding)

        positions = model(target_features, context_encoded, entity_ids=torch.tensor([0, 1]))

        assert positions.shape == (batch_size, target_len)
        assert (positions > -1).all() and (positions < 1).all()

    def test_xtrend_g_forward(self, config):
        """X-Trend-G: Gaussian prediction + PTP."""
        batch_size, target_len, context_size = 2, 126, 20

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        context_encoded = torch.randn(batch_size, context_size, config.hidden_dim)

        entity_embedding = EntityEmbedding(config)
        model = XTrendG(config, entity_embedding=entity_embedding)

        outputs = model(target_features, context_encoded, entity_ids=torch.tensor([0, 1]))

        assert 'mean' in outputs
        assert 'std' in outputs
        assert 'positions' in outputs
        assert outputs['mean'].shape == (batch_size, target_len)
        assert outputs['std'].shape == (batch_size, target_len)
        assert outputs['positions'].shape == (batch_size, target_len)
        assert (outputs['std'] > 0).all()

    def test_xtrend_q_forward(self, config):
        """X-Trend-Q: Quantile prediction + PTP."""
        batch_size, target_len, context_size = 2, 126, 20
        num_quantiles = 13

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        context_encoded = torch.randn(batch_size, context_size, config.hidden_dim)

        entity_embedding = EntityEmbedding(config)
        model = XTrendQ(config, entity_embedding=entity_embedding, num_quantiles=num_quantiles)

        outputs = model(target_features, context_encoded, entity_ids=torch.tensor([0, 1]))

        assert 'quantiles' in outputs
        assert 'positions' in outputs
        assert outputs['quantiles'].shape == (batch_size, target_len, num_quantiles)
        assert outputs['positions'].shape == (batch_size, target_len)

    def test_zero_shot_mode(self, config):
        """All models work in zero-shot mode (no entity embeddings)."""
        batch_size, target_len, context_size = 2, 63, 10

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        context_encoded = torch.randn(batch_size, context_size, config.hidden_dim)

        # X-Trend without entity embedding
        model = XTrend(config, entity_embedding=None)
        positions = model(target_features, context_encoded, entity_ids=None)
        assert positions.shape == (batch_size, target_len)
```

**Step 2: Run test to verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_xtrend_models.py::TestXTrendModels::test_xtrend_forward -v
```

Expected: `ModuleNotFoundError`

**Step 3: Implement X-Trend model variants**

Replace the skeleton in `xtrend/models/xtrend.py` with real implementations:
```python
"""Complete X-Trend model variants."""
import torch
import torch.nn as nn
from typing import Optional, Dict

from xtrend.models.types import ModelConfig
from xtrend.models.decoder import LSTMDecoder
from xtrend.models.heads import PositionHead, GaussianHead, QuantileHead, PTP_G, PTP_Q
from xtrend.models.embeddings import EntityEmbedding


class XTrend(nn.Module):
    """X-Trend model: Direct position prediction (Equation 7).

    Architecture:
        Encoder (Phase 3) -> Cross-Attention (Phase 5) -> Decoder -> PositionHead

    Loss:
        L = L_Sharpe(z, r)

    Args:
        config: Model configuration
        entity_embedding: Shared entity embedding (optional for zero-shot)
    """

    def __init__(
        self,
        config: ModelConfig,
        entity_embedding: Optional[EntityEmbedding] = None
    ):
        super().__init__()
        self.config = config
        self.entity_embedding = entity_embedding
        self.use_entity = entity_embedding is not None

        # Decoder
        self.decoder = LSTMDecoder(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # Position prediction head
        self.position_head = PositionHead(config)

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) or None for zero-shot

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # Decode
        decoder_output = self.decoder(target_features, cross_attn_output, entity_ids)

        # Predict positions
        positions = self.position_head(decoder_output.hidden_states)

        return positions


class XTrendG(nn.Module):
    """X-Trend-G model: Gaussian prediction with PTP (Equations 20-21).

    Architecture:
        Encoder -> Cross-Attention -> Decoder -> GaussianHead -> PTP_G

    Loss:
        L = α * L_MLE(μ, σ, r) + L_Sharpe(PTP_G(μ, σ), r)

    Args:
        config: Model configuration
        entity_embedding: Shared entity embedding (optional for zero-shot)
    """

    def __init__(
        self,
        config: ModelConfig,
        entity_embedding: Optional[EntityEmbedding] = None
    ):
        super().__init__()
        self.config = config
        self.entity_embedding = entity_embedding
        self.use_entity = entity_embedding is not None

        # Decoder
        self.decoder = LSTMDecoder(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # Gaussian prediction head
        self.gaussian_head = GaussianHead(config)

        # PTP module
        self.ptp = PTP_G(config)

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) or None for zero-shot

        Returns:
            dict with keys:
                - 'mean': Predicted mean (batch, seq_len)
                - 'std': Predicted std dev (batch, seq_len)
                - 'positions': Trading positions from PTP (batch, seq_len)
        """
        # Decode
        decoder_output = self.decoder(target_features, cross_attn_output, entity_ids)

        # Predict Gaussian parameters
        mean, std = self.gaussian_head(decoder_output.hidden_states)

        # Map to positions via PTP
        positions = self.ptp(mean, std)

        return {
            'mean': mean,
            'std': std,
            'positions': positions
        }


class XTrendQ(nn.Module):
    """X-Trend-Q model: Quantile prediction with PTP (Equations 22-23).

    Architecture:
        Encoder -> Cross-Attention -> Decoder -> QuantileHead -> PTP_Q

    Loss:
        L = α * L_QRE(Q, r) + L_Sharpe(PTP_Q(Q), r)

    Best performing variant (Table 1, Page 12).

    Args:
        config: Model configuration
        entity_embedding: Shared entity embedding (optional for zero-shot)
        num_quantiles: Number of quantiles (default: 13)
    """

    def __init__(
        self,
        config: ModelConfig,
        entity_embedding: Optional[EntityEmbedding] = None,
        num_quantiles: int = 13
    ):
        super().__init__()
        self.config = config
        self.entity_embedding = entity_embedding
        self.use_entity = entity_embedding is not None
        self.num_quantiles = num_quantiles

        # Decoder
        self.decoder = LSTMDecoder(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # Quantile prediction head
        self.quantile_head = QuantileHead(config, num_quantiles=num_quantiles)

        # PTP module
        self.ptp = PTP_Q(config, num_quantiles=num_quantiles)

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) or None for zero-shot

        Returns:
            dict with keys:
                - 'quantiles': Predicted quantiles (batch, seq_len, num_quantiles)
                - 'positions': Trading positions from PTP (batch, seq_len)
        """
        # Decode
        decoder_output = self.decoder(target_features, cross_attn_output, entity_ids)

        # Predict quantiles
        quantiles = self.quantile_head(decoder_output.hidden_states)

        # Map to positions via PTP
        positions = self.ptp(quantiles)

        return {
            'quantiles': quantiles,
            'positions': positions
        }
```

**Step 4: Run tests to verify they pass**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_xtrend_models.py::TestXTrendModels -v
```

Expected: All 4 tests PASS

**Step 5: Update __init__.py exports**

Edit `xtrend/models/__init__.py`:
```python
"""xtrend.models — Neural network architectures."""
from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import EntityEmbedding, ConditionalFFN
from xtrend.models.encoder import LSTMEncoder
from xtrend.models.decoder import LSTMDecoder
from xtrend.models.baseline_dmn import BaselineDMN
from xtrend.models.cross_attention_types import AttentionConfig, AttentionOutput
from xtrend.models.xtrend_cross_attention import XTrendCrossAttention
from xtrend.models.heads import PositionHead, GaussianHead, QuantileHead, PTP_G, PTP_Q
from xtrend.models.losses import (
    sharpe_loss,
    gaussian_nll_loss,
    quantile_loss,
    joint_gaussian_loss,
    joint_quantile_loss
)
from xtrend.models.xtrend import XTrend, XTrendG, XTrendQ

__all__ = [
    "ModelConfig",
    "EncoderOutput",
    "VariableSelectionNetwork",
    "EntityEmbedding",
    "ConditionalFFN",
    "LSTMEncoder",
    "LSTMDecoder",
    "BaselineDMN",
    "AttentionConfig",
    "AttentionOutput",
    "XTrendCrossAttention",
    "PositionHead",
    "GaussianHead",
    "QuantileHead",
    "PTP_G",
    "PTP_Q",
    "sharpe_loss",
    "gaussian_nll_loss",
    "quantile_loss",
    "joint_gaussian_loss",
    "joint_quantile_loss",
    "XTrend",
    "XTrendG",
    "XTrendQ",
]
```

**Step 6: Commit**

```bash
git add xtrend/models/xtrend.py xtrend/models/__init__.py tests/models/test_xtrend_models.py
git commit -m "feat(models): implement XTrend, XTrendG, XTrendQ model variants"
```

---

## Task 6: Add Integration Tests for Complete Pipeline

**Files:**
- Create: `tests/integration/test_phase6_complete.py`

**Background:** Test full end-to-end pipeline with all phases integrated.

**Step 1: Write integration tests**

Create `tests/integration/test_phase6_complete.py`:
```python
"""Integration tests for Phase 6: Decoder & Loss Functions."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.models import (
    ModelConfig,
    LSTMEncoder,
    EntityEmbedding,
    XTrendCrossAttention,
    XTrend,
    XTrendG,
    XTrendQ,
    sharpe_loss,
    joint_gaussian_loss,
    joint_quantile_loss,
)


class TestPhase6Integration:
    """Integration tests verifying Phase 6 completion criteria."""

    @pytest.fixture
    def config(self):
        """Model configuration."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,  # ✅ FIXED: Correct field name
            dropout=0.1
        )

    @pytest.fixture
    def realistic_data(self):
        """Create realistic multi-asset data."""
        dates = pd.date_range('2019-01-01', '2020-12-31', freq='D')
        np.random.seed(42)

        features = {}
        returns = {}
        symbols = [f"ASSET{i}" for i in range(10)]

        for symbol in symbols:
            # Simulate features (8-dim)
            features[symbol] = torch.randn(len(dates), 8)

            # Simulate returns (scaled by vol targeting)
            returns_series = np.random.randn(len(dates)) * 0.01
            returns[symbol] = torch.tensor(returns_series, dtype=torch.float32)

        return {
            'features': features,
            'returns': returns,
            'dates': dates,
            'symbols': symbols
        }

    def test_xtrend_full_pipeline(self, config, realistic_data):
        """Complete pipeline: encoder → cross-attention → decoder → head."""
        # Create components
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrend(config, entity_embedding=entity_embedding)

        # Target sequence
        target_symbol = "ASSET0"
        target_features = realistic_data['features'][target_symbol][:126]
        target_returns = realistic_data['returns'][target_symbol][:126]
        target_entity_id = torch.tensor([0])

        # Encode target
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=target_entity_id
        )

        # Context sequences (5 contexts)
        context_encoded_list = []
        for i in range(1, 6):
            ctx_feat = realistic_data['features'][realistic_data['symbols'][i]][:21]
            ctx_enc = encoder(
                ctx_feat.unsqueeze(0),
                entity_ids=torch.tensor([i])
            )
            context_encoded_list.append(ctx_enc.hidden_states[:, -1:, :])

        context_encoded = torch.cat(context_encoded_list, dim=1)

        # Cross-attention
        cross_attn_output = cross_attn(
            target_encoded.hidden_states,
            context_encoded
        )

        # Decode and predict positions
        positions = model(
            target_features.unsqueeze(0),
            cross_attn_output.output,
            entity_ids=target_entity_id
        )

        # Phase 6 completion criteria
        assert positions.shape == (1, 126)
        assert (positions > -1).all() and (positions < 1).all()

        # Compute loss
        loss = sharpe_loss(positions, target_returns.unsqueeze(0), warmup_steps=63)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        print("✓ X-Trend full pipeline working")
        print(f"✓ Position range: [{positions.min():.4f}, {positions.max():.4f}]")
        print(f"✓ Sharpe loss: {loss.item():.4f}")

    def test_xtrend_g_with_joint_loss(self, config, realistic_data):
        """X-Trend-G: Gaussian prediction with joint loss."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrendG(config, entity_embedding=entity_embedding)

        # Prepare data
        target_features = realistic_data['features']['ASSET0'][:126]
        target_returns = realistic_data['returns']['ASSET0'][:126]
        target_entity_id = torch.tensor([0])

        # Encode target
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=target_entity_id
        )

        # Mock context (simplified)
        context_encoded = torch.randn(1, 5, config.hidden_dim)

        # Cross-attention
        cross_attn_output = cross_attn(
            target_encoded.hidden_states,
            context_encoded
        )

        # Forward pass
        outputs = model(
            target_features.unsqueeze(0),
            cross_attn_output.output,
            entity_ids=target_entity_id
        )

        # Check outputs
        assert 'mean' in outputs
        assert 'std' in outputs
        assert 'positions' in outputs
        assert outputs['mean'].shape == (1, 126)
        assert outputs['std'].shape == (1, 126)
        assert outputs['positions'].shape == (1, 126)
        assert (outputs['std'] > 0).all()

        # Compute joint loss
        loss = joint_gaussian_loss(
            outputs['mean'],
            outputs['std'],
            outputs['positions'],
            target_returns.unsqueeze(0),
            alpha=1.0,
            warmup_steps=63
        )

        assert not torch.isnan(loss)
        print(f"✓ X-Trend-G joint loss: {loss.item():.4f}")

    def test_xtrend_q_with_joint_loss(self, config, realistic_data):
        """X-Trend-Q: Quantile prediction with joint loss."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrendQ(config, entity_embedding=entity_embedding, num_quantiles=13)

        # Quantile levels
        levels = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        # Prepare data
        target_features = realistic_data['features']['ASSET0'][:126]
        target_returns = realistic_data['returns']['ASSET0'][:126]
        target_entity_id = torch.tensor([0])

        # Encode target
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=target_entity_id
        )

        # Mock context
        context_encoded = torch.randn(1, 5, config.hidden_dim)

        # Cross-attention
        cross_attn_output = cross_attn(
            target_encoded.hidden_states,
            context_encoded
        )

        # Forward pass
        outputs = model(
            target_features.unsqueeze(0),
            cross_attn_output.output,
            entity_ids=target_entity_id
        )

        # Check outputs
        assert 'quantiles' in outputs
        assert 'positions' in outputs
        assert outputs['quantiles'].shape == (1, 126, 13)
        assert outputs['positions'].shape == (1, 126)

        # Compute joint loss
        loss = joint_quantile_loss(
            outputs['quantiles'],
            levels,
            outputs['positions'],
            target_returns.unsqueeze(0),
            alpha=5.0,
            warmup_steps=63
        )

        assert not torch.isnan(loss)
        print(f"✓ X-Trend-Q joint loss: {loss.item():.4f}")

    def test_gradient_flow_end_to_end(self, config):
        """Gradients flow through entire pipeline."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrend(config, entity_embedding=entity_embedding)

        # Create data with gradients enabled
        target_features = torch.randn(1, 63, config.input_dim, requires_grad=True)
        target_returns = torch.randn(1, 63)
        context_encoded = torch.randn(1, 5, config.hidden_dim)

        # Forward pass
        target_encoded = encoder(target_features, entity_ids=torch.tensor([0]))
        cross_attn_output = cross_attn(target_encoded.hidden_states, context_encoded)
        positions = model(target_features, cross_attn_output.output, entity_ids=torch.tensor([0]))

        # Loss and backward
        loss = sharpe_loss(positions, target_returns, warmup_steps=20)
        loss.backward()

        # Check gradients
        assert target_features.grad is not None
        assert not torch.isnan(target_features.grad).any()
        print("✓ Gradients flow end-to-end")

    def test_zero_shot_mode(self, config, realistic_data):
        """Models work in zero-shot mode (no entity embeddings)."""
        encoder = LSTMEncoder(config, use_entity=False, entity_embedding=None)
        cross_attn = XTrendCrossAttention(config)
        model = XTrend(config, entity_embedding=None)

        # Prepare data
        target_features = realistic_data['features']['ASSET0'][:63]
        target_returns = realistic_data['returns']['ASSET0'][:63]

        # Encode target (no entity)
        target_encoded = encoder(target_features.unsqueeze(0), entity_ids=None)

        # Mock context
        context_encoded = torch.randn(1, 3, config.hidden_dim)

        # Cross-attention
        cross_attn_output = cross_attn(target_encoded.hidden_states, context_encoded)

        # Forward pass
        positions = model(target_features.unsqueeze(0), cross_attn_output.output, entity_ids=None)

        assert positions.shape == (1, 63)
        print("✓ Zero-shot mode working")
```

**Step 2: Run integration tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/integration/test_phase6_complete.py -v
```

Expected: All 5 tests PASS

**Step 3: Run full test suite**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_decoder.py tests/models/test_heads.py tests/models/test_ptp.py tests/models/test_losses.py tests/models/test_xtrend_models.py tests/integration/test_phase6_complete.py -v
```

Expected: All tests PASS (40+ tests)

**Step 4: Commit**

```bash
git add tests/integration/test_phase6_complete.py
git commit -m "test(integration): add Phase 6 complete integration tests"
```

---

## Verification Steps

After completing all tasks, verify Phase 6 is complete:

**1. Run all Phase 6 tests:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_decoder.py tests/models/test_heads.py tests/models/test_ptp.py tests/models/test_losses.py tests/models/test_xtrend_models.py tests/integration/test_phase6_complete.py -v
```

Expected: All tests PASS (45+ tests)

**2. Verify visual completion criteria:**
```python
# In Python REPL: uv run python
import torch
from xtrend.models import ModelConfig, XTrendQ, EntityEmbedding, LSTMEncoder, XTrendCrossAttention

# Create config
config = ModelConfig(
    input_dim=8,
    hidden_dim=64,
    num_entities=50,
    num_attention_heads=4,
    dropout=0.1
)

# Create full pipeline
entity_embedding = EntityEmbedding(config)
encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
cross_attn = XTrendCrossAttention(config)
model = XTrendQ(config, entity_embedding=entity_embedding, num_quantiles=13)

# Mock data
batch_size, target_len, context_size = 2, 126, 20
target_features = torch.randn(batch_size, target_len, config.input_dim)
target_returns = torch.randn(batch_size, target_len) * 0.01  # Realistic returns

# Encode target
target_encoded = encoder(target_features, entity_ids=torch.tensor([0, 1]))

# Mock context
context_encoded = torch.randn(batch_size, context_size, config.hidden_dim)

# Cross-attention
cross_attn_output = cross_attn(target_encoded.hidden_states, context_encoded)

# Forward pass
outputs = model(target_features, cross_attn_output.output, entity_ids=torch.tensor([0, 1]))

print(f"Quantiles shape: {outputs['quantiles'].shape}")
# Output: Quantiles shape: torch.Size([2, 126, 13])

print(f"Positions shape: {outputs['positions'].shape}")
# Output: Positions shape: torch.Size([2, 126])

print(f"Position range: [{outputs['positions'].min():.4f}, {outputs['positions'].max():.4f}]")
# Expected: Position range: [-0.9987, 0.9991]

# Compute loss
from xtrend.models import joint_quantile_loss
levels = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
loss = joint_quantile_loss(outputs['quantiles'], levels, outputs['positions'], target_returns, alpha=5.0, warmup_steps=63)
print(f"Joint loss: {loss.item():.4f}")
# Expected: Loss value (varies)
```

**3. Check test coverage:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/models/test_decoder.py tests/models/test_heads.py tests/models/test_losses.py tests/models/test_xtrend_models.py --cov=xtrend.models.decoder --cov=xtrend.models.heads --cov=xtrend.models.losses --cov=xtrend.models.xtrend --cov-report=term-missing
```

Expected: >80% coverage for Phase 6 modules

**4. Verify implementation matches paper:**
- [x] LSTM Decoder with skip connections (Equations 19a-19d)
- [x] Three model variants (X-Trend, X-Trend-G, X-Trend-Q)
- [x] Prediction heads (Position, Gaussian, Quantile)
- [x] PTP modules (PTP_G, PTP_Q)
- [x] Sharpe loss (Equation 8)
- [x] Gaussian NLL loss (Equation 20)
- [x] Quantile regression loss (Equation 22)
- [x] Joint losses (Equations 21, 23)
- [x] Warmup period (63 steps)
- [x] Annualization factor (√252)
- [x] Quantile monotonicity enforced
- [x] Zero-shot mode supported
- [x] Gradient flow verified end-to-end

---

## References

- **Paper:** Section 3.4 "Decoder", Section 4 "Loss Functions", Equations 7-8, 19-23, Pages 8-9
- **Phase 3:** `xtrend/models/encoder.py` for LSTM+skip pattern
- **Phase 5:** `xtrend/models/xtrend_cross_attention.py` for cross-attention output
- **Table 1:** X-Trend-Q achieves best performance (Sharpe 2.70)
- **Table 4:** Hyperparameters (α=1 for Gaussian, α=5 for Quantile)
- **Related Skills:** @test-driven-development, @verification-before-completion

---

## Notes

- **Decoder pattern:** Mirrors encoder (Phase 3) but fuses cross-attention output
- **Three variants:** X-Trend (baseline), X-Trend-G (Gaussian), X-Trend-Q (best performance)
- **Sharpe loss:** Core innovation - directly optimizes risk-adjusted returns
- **Warmup period:** 63 steps to avoid initialization instability
- **Quantile monotonicity:** Enforced via cumulative softplus increments
- **PTP modules:** Enable joint training (distribution + profitability)
- **Zero-shot mode:** Models work without entity embeddings for unseen assets
- **Gradient flow:** Verified end-to-end from features through all components to loss

---

## Codex Review Applied

✅ **All 11 Codex Review Issues Fixed Inline:**

1. **VSN tuple unpacking** - Fixed in decoder.py (line ~203)
2. **Entity conditioning wiring** - Fixed in decoder.py (__init__ method)
3. **Config field names** - Fixed in all test fixtures (num_attention_heads)
4. **EncoderOutput return** - Fixed in decoder.py (return statement)
5. **Gaussian variance clamping** - Fixed in heads.py (GaussianHead) and losses.py
6. **Fusion layer** - Fixed in decoder.py (using ConditionalFFN)
7. **Improved tests** - Added deterministic and behavior tests to test_heads.py
8. **Edge case guards** - Added validation to all loss functions
9. **Quantile computation** - Optimized to use torch.cumsum in heads.py
10. **PTP dropout removal** - Removed from both PTP_G and PTP_Q
11. **PTP_Q gradient test** - Added to test_ptp.py

**Verification Checklist (All Checked):**
- [x] VSN called with tuple unpacking: `x, _ = self.vsn(...)`
- [x] VSN and ConditionalFFN receive `entity_embedding` parameter
- [x] All test fixtures use `num_attention_heads=4`
- [x] EncoderOutput returns both `hidden_states` and `sequence_length`
- [x] Gaussian variance clamped to ≥1e-6 after squaring
- [x] Fusion layer uses ConditionalFFN (with projection workaround)
- [x] Sharpe loss uses `unbiased=False` for population std
- [x] Sharpe loss validates `warmup_steps < seq_len`
- [x] Quantile computation uses `torch.cumsum`, not Python loop
- [x] PTP modules have no dropout layers
- [x] PTP_Q has gradient flow test

---

## Plan Complete

**Status:** Phase 6 plan fully corrected with all Codex fixes integrated inline

**Original backup:** `docs/plans/2025-11-18-phase6-decoder-and-losses.md.backup`

**Next Steps:**
1. Execute plan using subagent-driven-development
2. All critical fixes are already applied in the plan
3. Follow TDD pattern for each task

**Reviewed by:** OpenAI Codex (2025-11-18)
**Issues Fixed:** 11 (6 critical, 5 important) - All integrated inline
