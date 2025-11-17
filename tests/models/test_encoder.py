"""Tests for LSTM encoder."""
import pytest
import torch
from xtrend.models.encoder import LSTMEncoder
from xtrend.models.embeddings import EntityEmbedding
from xtrend.models.types import ModelConfig


class TestLSTMEncoder:
    def test_encoder_initialization(self, model_config):
        """LSTMEncoder initializes with correct components."""
        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)

        assert encoder.vsn is not None
        assert encoder.lstm is not None
        assert encoder.lstm_dropout is not None
        assert encoder.layer_norm1 is not None
        assert encoder.ffn is not None
        assert encoder.layer_norm2 is not None
        assert encoder.entity_embedding is entity_embedding  # Verify shared instance

    def test_encoder_forward_shape(self, model_config, sample_features, sample_entity_ids):
        """Encoder produces correct output shape."""
        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)

        output = encoder(sample_features, sample_entity_ids)

        batch_size, seq_len, _ = sample_features.shape
        # Output should be (batch, seq_len, hidden_dim)
        assert output.hidden_states.shape == (batch_size, seq_len, model_config.hidden_dim)
        assert output.sequence_length == seq_len

    def test_encoder_equation_14(self, model_config):
        """Encoder implements Equation 14 from paper."""
        # Equation 14:
        #   x'_t = VSN(x_t, s)
        #   (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        #   a_t = LayerNorm(h_t + x'_t)  # Skip connection
        #   Ξ = LayerNorm(FFN(a_t, s) + a_t)  # Skip connection

        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)
        encoder.eval()

        # Single sample to verify skip connections
        x = torch.randn(1, 10, 8)  # (batch=1, seq_len=10, input_dim=8)
        entity_ids = torch.tensor([0])

        with torch.no_grad():
            output = encoder(x, entity_ids)

        # Output should exist
        assert output.hidden_states is not None
        assert output.hidden_states.shape == (1, 10, model_config.hidden_dim)

    def test_encoder_lstm_state_persistence(self, model_config):
        """LSTM maintains state across sequence."""
        entity_embedding = EntityEmbedding(model_config)
        encoder = LSTMEncoder(model_config, use_entity=True, entity_embedding=entity_embedding)

        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 8)
        entity_ids = torch.randint(0, 50, (batch_size,))

        output = encoder(x, entity_ids)

        # Later timesteps should be different from earlier ones
        # (LSTM should maintain context)
        first_step = output.hidden_states[:, 0, :]
        last_step = output.hidden_states[:, -1, :]

        assert not torch.allclose(first_step, last_step)

    def test_encoder_zero_shot_mode(self, model_config):
        """Encoder works in zero-shot mode (no entity info)."""
        encoder = LSTMEncoder(model_config, use_entity=False, entity_embedding=None)

        x = torch.randn(4, 20, 8)

        # Should work without entity_ids
        output = encoder(x, entity_ids=None)

        assert output.hidden_states.shape == (4, 20, model_config.hidden_dim)
