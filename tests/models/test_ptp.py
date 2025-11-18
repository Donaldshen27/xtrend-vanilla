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
        std = torch.rand(batch_size, seq_len) + 0.5
        std.requires_grad = True

        ptp = PTP_G(config)
        positions = ptp(mean, std)

        # Backward pass
        loss = positions.sum()
        loss.backward()

        # Check gradients exist (both mean and std are leaf tensors)
        assert mean.grad is not None
        assert std.grad is not None
        assert not torch.isnan(mean.grad).any()
        assert not torch.isnan(std.grad).any()

    # ✅ ADDED: PTP_Q gradient test (Issue #11)
    def test_ptp_q_gradient_flow(self, config):
        """PTP_Q allows gradient flow for joint training."""
        batch_size, seq_len, num_quantiles = 2, 10, 13

        # Create sorted quantiles directly (leaf tensor)
        quantiles = torch.randn(batch_size, seq_len, num_quantiles)
        quantiles, _ = quantiles.sort(dim=-1)
        quantiles.requires_grad = True

        ptp = PTP_Q(config, num_quantiles=num_quantiles)
        positions = ptp(quantiles)

        # Backward pass
        loss = positions.sum()
        loss.backward()

        # Check gradients exist (quantiles is now a leaf tensor)
        assert quantiles.grad is not None
        assert not torch.isnan(quantiles.grad).any()
