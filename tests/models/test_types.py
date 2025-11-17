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
        with pytest.raises(ValueError, match=r"hidden_dim \(0\) must be positive"):
            ModelConfig(hidden_dim=0)

        with pytest.raises(ValueError, match=r"dropout \(1\.5\) must be in \[0, 1\)"):
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
