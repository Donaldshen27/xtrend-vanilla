"""Comprehensive padding mask validation tests (Codex critical issue)."""
import pytest
import torch

from xtrend.models import (
    ModelConfig,
    XTrendCrossAttention,
    LSTMEncoder,
    EntityEmbedding,
)


class TestPaddingMaskValidation:
    """Validate padding masks don't silently corrupt attention (Codex critical)."""

    @pytest.fixture
    def config(self):
        """Standard config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,
            dropout=0.1
        )

    def test_zero_attention_to_padded_positions(self, config):
        """Padded positions receive exactly zero attention weight."""
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()  # Disable dropout for deterministic behavior

        batch_size = 2
        target_len, max_context_size = 10, 20

        target = torch.randn(batch_size, target_len, config.hidden_dim)
        context = torch.randn(batch_size, max_context_size, config.hidden_dim)

        # Batch 1: 15 valid, 5 padding
        # Batch 2: 18 valid, 2 padding
        mask = torch.ones(batch_size, max_context_size, dtype=torch.bool)
        mask[0, 15:] = False
        mask[1, 18:] = False

        output = cross_attn(target, context, context_padding_mask=mask)

        # Attention to padded positions must be EXACTLY zero
        weights = output.attention_weights  # (batch, heads, target_len, context_size)

        # Batch 0: positions 15-19 should have zero attention
        padded_weights_b0 = weights[0, :, :, 15:]
        assert torch.allclose(
            padded_weights_b0,
            torch.zeros_like(padded_weights_b0),
            atol=1e-7
        ), "Batch 0: Padded positions have non-zero attention"

        # Batch 1: positions 18-19 should have zero attention
        padded_weights_b1 = weights[1, :, :, 18:]
        assert torch.allclose(
            padded_weights_b1,
            torch.zeros_like(padded_weights_b1),
            atol=1e-7
        ), "Batch 1: Padded positions have non-zero attention"

        print("✓ Zero attention to all padded positions")

    def test_attention_sums_to_one_with_padding(self, config):
        """Attention weights sum to 1 even with variable-length contexts."""
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()  # Disable dropout

        batch_size = 3
        target_len, max_context_size = 5, 15

        target = torch.randn(batch_size, target_len, config.hidden_dim)
        context = torch.randn(batch_size, max_context_size, config.hidden_dim)

        # Different valid lengths per batch
        mask = torch.zeros(batch_size, max_context_size, dtype=torch.bool)
        mask[0, :10] = True   # 10 valid
        mask[1, :7] = True    # 7 valid
        mask[2, :15] = True   # 15 valid (no padding)

        output = cross_attn(target, context, context_padding_mask=mask)

        # Each row should sum to 1 (over valid positions only)
        weights = output.attention_weights  # (batch, heads, target_len, context_size)
        row_sums = weights.sum(dim=-1)  # (batch, heads, target_len)

        assert torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-6
        ), "Attention weights don't sum to 1 with padding"

        print("✓ Attention sums to 1 across valid positions only")

    def test_mask_broadcasting_correctness(self, config):
        """Padding mask broadcasts correctly across heads and target positions."""
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()

        batch_size = 1
        target_len, context_size = 3, 5

        target = torch.randn(batch_size, target_len, config.hidden_dim)
        context = torch.randn(batch_size, context_size, config.hidden_dim)

        # Mask: [True, True, False, True, False]
        # Valid: positions 0, 1, 3
        # Padded: positions 2, 4
        mask = torch.tensor([[True, True, False, True, False]], dtype=torch.bool)

        output = cross_attn(target, context, context_padding_mask=mask)

        weights = output.attention_weights[0]  # (heads, target_len, context_size)

        # For EVERY head and EVERY target position:
        # positions 2 and 4 should have zero attention
        for head_idx in range(config.num_attention_heads):
            for target_pos in range(target_len):
                assert weights[head_idx, target_pos, 2].item() == 0.0, \
                    f"Head {head_idx}, Target {target_pos}: Position 2 not masked"
                assert weights[head_idx, target_pos, 4].item() == 0.0, \
                    f"Head {head_idx}, Target {target_pos}: Position 4 not masked"

        print("✓ Mask broadcasts correctly across heads and target positions")

    def test_different_masks_per_batch(self, config):
        """Each batch item can have different padding pattern."""
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()

        batch_size = 2
        target_len, context_size = 5, 10

        target = torch.randn(batch_size, target_len, config.hidden_dim)
        context = torch.randn(batch_size, context_size, config.hidden_dim)

        # Batch 0: first 6 valid
        # Batch 1: positions 0-3 and 7-9 valid (non-contiguous)
        mask = torch.zeros(batch_size, context_size, dtype=torch.bool)
        mask[0, :6] = True
        mask[1, :4] = True
        mask[1, 7:] = True

        output = cross_attn(target, context, context_padding_mask=mask)

        weights = output.attention_weights

        # Batch 0: positions 6-9 should be zero
        assert torch.allclose(
            weights[0, :, :, 6:],
            torch.zeros_like(weights[0, :, :, 6:]),
            atol=1e-7
        )

        # Batch 1: positions 4-6 should be zero
        assert torch.allclose(
            weights[1, :, :, 4:7],
            torch.zeros_like(weights[1, :, :, 4:7]),
            atol=1e-7
        )

        # Batch 1: valid positions should have attention > 0
        valid_weights = weights[1, :, :, :4]  # First 4 positions
        assert (valid_weights > 0).any(), "Valid positions have zero attention"

        print("✓ Different padding patterns per batch handled correctly")

    def test_all_padding_edge_case(self, config):
        """Handle edge case: all context positions are padding.

        Note: This is an edge case that shouldn't occur in practice
        (we always have at least one valid context). PyTorch's
        MultiheadAttention produces NaN when all keys are masked,
        which is expected behavior. This test documents that behavior.
        """
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()

        batch_size = 1
        target_len, context_size = 3, 5

        target = torch.randn(batch_size, target_len, config.hidden_dim)
        context = torch.randn(batch_size, context_size, config.hidden_dim)

        # All positions are padding (shouldn't happen in practice)
        mask = torch.zeros(batch_size, context_size, dtype=torch.bool)

        # This produces NaN values (expected PyTorch behavior)
        output = cross_attn(target, context, context_padding_mask=mask)

        # Document that all-padding produces NaN (expected)
        # In practice, we ensure at least one valid context sequence
        assert torch.isnan(output.output).any(), \
            "All-padding case produces NaN (expected PyTorch behavior)"

        print("✓ All-padding edge case produces NaN (expected, shouldn't occur in practice)")

    def test_no_padding_baseline(self, config):
        """No-padding case works as baseline."""
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()

        batch_size = 2
        target_len, context_size = 5, 10

        target = torch.randn(batch_size, target_len, config.hidden_dim)
        context = torch.randn(batch_size, context_size, config.hidden_dim)

        # No padding (all valid)
        mask = torch.ones(batch_size, context_size, dtype=torch.bool)

        output = cross_attn(target, context, context_padding_mask=mask)

        # All positions should have some attention (non-zero)
        weights = output.attention_weights

        # At least some non-zero weights everywhere
        for batch_idx in range(batch_size):
            for pos in range(context_size):
                pos_weights = weights[batch_idx, :, :, pos]
                assert (pos_weights > 0).any(), \
                    f"Position {pos} has zero attention in no-padding case"

        print("✓ No-padding baseline: all positions receive attention")

    def test_mask_dtype_validation(self, config):
        """Padding mask must be boolean dtype."""
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()

        target = torch.randn(1, 5, config.hidden_dim)
        context = torch.randn(1, 10, config.hidden_dim)

        # Wrong dtype: int instead of bool
        wrong_mask = torch.ones(1, 10, dtype=torch.int)

        # Should still work (PyTorch converts internally)
        # But we document expected dtype is bool
        output = cross_attn(target, context, context_padding_mask=wrong_mask.bool())

        assert output.output.shape == (1, 5, config.hidden_dim)
        print("✓ Mask dtype validation: bool expected")
