"""Tests for integrated X-Trend cross-attention module."""
import pytest
import torch

from xtrend.models.xtrend_cross_attention import XTrendCrossAttention
from xtrend.models.types import ModelConfig
from xtrend.models.cross_attention_types import AttentionOutput


class TestXTrendCrossAttention:
    """Test complete cross-attention module integrating all components."""

    @pytest.fixture
    def config(self):
        """Standard model config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=50,
            num_attention_heads=4,
            dropout=0.1
        )

    def test_full_pipeline(self, config):
        """Complete pipeline: target + context -> attended output."""
        batch_size = 2
        target_len, context_size = 126, 20
        hidden_dim = 64

        # Encoded target states (from Phase 3 encoder)
        target_encoded = torch.randn(batch_size, target_len, hidden_dim)

        # Encoded context states (from Phase 3 encoder)
        context_encoded = torch.randn(batch_size, context_size, hidden_dim)

        xtrend_attn = XTrendCrossAttention(config)
        output = xtrend_attn(target_encoded, context_encoded)

        assert isinstance(output, AttentionOutput)
        assert output.output.shape == (batch_size, target_len, hidden_dim)
        assert output.attention_weights.shape == (batch_size, config.num_attention_heads, target_len, context_size)

    def test_with_context_padding_mask(self, config):
        """Handle variable-length context sequences."""
        batch_size = 2
        target_len, max_context_size = 126, 20
        hidden_dim = 64

        target_encoded = torch.randn(batch_size, target_len, hidden_dim)
        context_encoded = torch.randn(batch_size, max_context_size, hidden_dim)

        # Variable-length contexts
        context_mask = torch.ones(batch_size, max_context_size, dtype=torch.bool)
        context_mask[0, 15:] = False  # First: 15 valid contexts
        context_mask[1, 18:] = False  # Second: 18 valid contexts

        xtrend_attn = XTrendCrossAttention(config)
        output = xtrend_attn(
            target_encoded,
            context_encoded,
            context_padding_mask=context_mask
        )

        # Output shape unchanged
        assert output.output.shape == (batch_size, target_len, hidden_dim)

        # No attention to padded contexts
        assert torch.allclose(
            output.attention_weights[0, :, :, 15:],
            torch.zeros_like(output.attention_weights[0, :, :, 15:]),
            atol=1e-6
        )

    def test_gradient_flow(self, config):
        """Gradients flow through attention mechanism."""
        batch_size = 2
        target_len, context_size = 10, 5
        hidden_dim = 64

        target_encoded = torch.randn(batch_size, target_len, hidden_dim, requires_grad=True)
        context_encoded = torch.randn(batch_size, context_size, hidden_dim, requires_grad=True)

        xtrend_attn = XTrendCrossAttention(config)
        output = xtrend_attn(target_encoded, context_encoded)

        # Backward pass
        loss = output.output.sum()
        loss.backward()

        # Check gradients exist
        assert target_encoded.grad is not None
        assert context_encoded.grad is not None
        assert not torch.isnan(target_encoded.grad).any()
        assert not torch.isnan(context_encoded.grad).any()

    def test_interpretable_attention_weights(self, config):
        """Attention weights available for interpretability."""
        batch_size = 1
        target_len, context_size = 5, 10
        hidden_dim = 64

        target_encoded = torch.randn(batch_size, target_len, hidden_dim)
        context_encoded = torch.randn(batch_size, context_size, hidden_dim)

        xtrend_attn = XTrendCrossAttention(config)
        xtrend_attn.eval()  # Set to eval mode to disable dropout
        output = xtrend_attn(target_encoded, context_encoded)

        # Can access attention weights for each head
        weights = output.attention_weights  # (1, 4, 5, 10)

        # Each head's attention sums to 1 across contexts
        for head_idx in range(config.num_attention_heads):
            head_weights = weights[0, head_idx, :, :]  # (5, 10)
            row_sums = head_weights.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
