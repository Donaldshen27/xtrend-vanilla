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
