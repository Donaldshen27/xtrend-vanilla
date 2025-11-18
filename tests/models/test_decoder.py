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

    def test_decoder_gradient_flow(self, config):
        """Gradients flow from decoder output to both target_features and cross_attn_output."""
        batch_size, seq_len = 2, 63

        # Create inputs with gradient tracking enabled
        target_features = torch.randn(batch_size, seq_len, config.input_dim, requires_grad=True)
        cross_attn_output = torch.randn(batch_size, seq_len, config.hidden_dim, requires_grad=True)

        entity_embedding = EntityEmbedding(config)
        decoder = LSTMDecoder(config, use_entity=True, entity_embedding=entity_embedding)

        # Forward pass
        output = decoder(
            target_features,
            cross_attn_output,
            entity_ids=torch.tensor([0, 1])
        )

        # Compute loss and backward pass
        loss = output.hidden_states.sum()
        loss.backward()

        # Verify gradients exist and are not NaN
        assert target_features.grad is not None, "No gradient for target_features"
        assert cross_attn_output.grad is not None, "No gradient for cross_attn_output"
        assert not torch.isnan(target_features.grad).any(), "NaN gradients in target_features"
        assert not torch.isnan(cross_attn_output.grad).any(), "NaN gradients in cross_attn_output"

        # Verify gradients are non-zero (gradients should flow)
        assert target_features.grad.abs().sum() > 0, "Zero gradients for target_features"
        assert cross_attn_output.grad.abs().sum() > 0, "Zero gradients for cross_attn_output"
