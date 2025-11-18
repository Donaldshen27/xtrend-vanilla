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

    def test_attention_output_wrong_rank(self):
        """AttentionOutput rejects wrong tensor ranks."""
        output_2d = torch.randn(2, 10)  # Wrong rank
        weights = torch.randn(2, 4, 10, 20)

        with pytest.raises(ValueError, match="must be rank-3"):
            AttentionOutput(output=output_2d, attention_weights=weights)

    def test_attention_output_shape_mismatch(self):
        """AttentionOutput detects batch/seq_len mismatch."""
        output = torch.randn(2, 10, 64)
        weights = torch.randn(3, 4, 10, 20)  # Different batch size

        with pytest.raises(ValueError, match="Shape mismatch"):
            AttentionOutput(output=output, attention_weights=weights)

    def test_attention_output_head_constraint(self):
        """AttentionOutput validates hidden_dim divisible by num_heads."""
        output = torch.randn(2, 10, 63)  # 63 not divisible by 4
        weights = torch.randn(2, 4, 10, 20)

        with pytest.raises(ValueError, match="must be divisible"):
            AttentionOutput(output=output, attention_weights=weights)

    def test_attention_config_invalid_dimensions(self):
        """AttentionConfig rejects invalid dimensions."""
        with pytest.raises(ValueError, match="must be positive"):
            AttentionConfig(hidden_dim=-64, num_heads=4, dropout=0.3)

        with pytest.raises(ValueError, match="must be positive"):
            AttentionConfig(hidden_dim=64, num_heads=0, dropout=0.3)

    def test_attention_config_invalid_dropout(self):
        """AttentionConfig validates dropout range."""
        with pytest.raises(ValueError, match="must be in"):
            AttentionConfig(hidden_dim=64, num_heads=4, dropout=1.5)

        with pytest.raises(ValueError, match="must be in"):
            AttentionConfig(hidden_dim=64, num_heads=4, dropout=-0.1)

    def test_attention_config_head_divisibility(self):
        """AttentionConfig requires hidden_dim divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            AttentionConfig(hidden_dim=63, num_heads=4, dropout=0.3)
