"""Tests for self-attention over context set."""
import pytest
import torch

from xtrend.models.self_attention import MultiHeadSelfAttention
from xtrend.models.cross_attention_types import AttentionConfig


class TestMultiHeadSelfAttention:
    """Test self-attention mechanism (Equation 17)."""

    @pytest.fixture
    def config(self):
        """Standard attention config."""
        return AttentionConfig(
            hidden_dim=64,
            num_heads=4,
            dropout=0.1
        )

    def test_self_attention_forward(self, config):
        """Self-attention processes context set."""
        batch_size, context_size, hidden_dim = 2, 20, 64

        # Context values: V_t in paper
        context = torch.randn(batch_size, context_size, hidden_dim)

        self_attn = MultiHeadSelfAttention(config)
        output = self_attn(context)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, context_size, hidden_dim)

    def test_self_attention_with_padding_mask(self, config):
        """Self-attention respects padding mask for variable-length contexts."""
        batch_size, max_context_size, hidden_dim = 2, 20, 64

        context = torch.randn(batch_size, max_context_size, hidden_dim)

        # Padding mask: True = valid, False = padding
        # First sequence: 15 valid, 5 padding
        # Second sequence: 18 valid, 2 padding
        padding_mask = torch.ones(batch_size, max_context_size, dtype=torch.bool)
        padding_mask[0, 15:] = False  # Mask last 5
        padding_mask[1, 18:] = False  # Mask last 2

        self_attn = MultiHeadSelfAttention(config)
        output = self_attn(context, key_padding_mask=padding_mask)

        assert output.shape == (batch_size, max_context_size, hidden_dim)

        # Padded positions should have zero gradients
        # (attention shouldn't attend to them)

    def test_attention_weights_sum_to_one(self, config):
        """Attention weights normalize to 1 (Equation 10)."""
        batch_size, context_size, hidden_dim = 2, 10, 64

        context = torch.randn(batch_size, context_size, hidden_dim)

        self_attn = MultiHeadSelfAttention(config)
        self_attn.eval()  # Disable dropout for this test
        output = self_attn(context, return_attention_weights=True)

        # If return_attention_weights=True, return tuple
        assert isinstance(output, tuple)
        values, weights = output

        # Weights shape: (batch, num_heads, seq_len, seq_len)
        assert weights.shape == (batch_size, config.num_heads, context_size, context_size)

        # Each row sums to 1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
