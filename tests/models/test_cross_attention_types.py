"""Tests for cross-attention types."""
import pytest
import torch

from xtrend.models.cross_attention_types import AttentionOutput, AttentionConfig


class TestAttentionTypes:
    """Test attention type definitions."""

    def test_attention_output_creation(self):
        """Create AttentionOutput with values and weights."""
        batch_size, seq_len, hidden_dim = 2, 10, 64
        num_heads, context_size = 4, 20

        output = torch.randn(batch_size, seq_len, hidden_dim)
        weights = torch.randn(batch_size, num_heads, seq_len, context_size)

        attn_output = AttentionOutput(
            output=output,
            attention_weights=weights
        )

        assert attn_output.output.shape == (batch_size, seq_len, hidden_dim)
        assert attn_output.attention_weights.shape == (batch_size, num_heads, seq_len, context_size)

    def test_attention_config_defaults(self):
        """AttentionConfig with paper defaults."""
        config = AttentionConfig(
            hidden_dim=64,
            num_heads=4,
            dropout=0.3
        )

        assert config.hidden_dim == 64
        assert config.num_heads == 4
        assert config.dropout == 0.3
        assert config.hidden_dim % config.num_heads == 0
