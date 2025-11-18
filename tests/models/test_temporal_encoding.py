"""Tests for temporal order preservation in encoder."""
import pytest
import torch

from xtrend.models import LSTMEncoder, EntityEmbedding, ModelConfig


class TestTemporalEncoding:
    """Verify LSTM preserves temporal order (Codex critical issue)."""

    @pytest.fixture
    def config(self):
        """Standard config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,
            dropout=0.0  # CRITICAL: No dropout for deterministic testing
        )

    def test_lstm_preserves_temporal_order(self, config):
        """LSTM hidden states encode temporal position."""
        torch.manual_seed(42)  # Deterministic initialization
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        encoder.eval()  # CRITICAL: Disable dropout for deterministic testing

        # Create sequences with temporal patterns
        batch_size, seq_len = 2, 10

        # Sequence 1: increasing trend
        seq1 = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)
        seq1 = seq1.expand(1, seq_len, config.input_dim)

        # Sequence 2: reversed (decreasing trend)
        seq2 = torch.linspace(1, 0, seq_len).unsqueeze(0).unsqueeze(-1)
        seq2 = seq2.expand(1, seq_len, config.input_dim)

        # Encode both
        out1 = encoder(seq1, entity_ids=torch.tensor([0]))
        out2 = encoder(seq2, entity_ids=torch.tensor([0]))

        # Hidden states should be different (temporal order matters)
        # Compare final hidden states
        final1 = out1.hidden_states[:, -1, :]
        final2 = out2.hidden_states[:, -1, :]

        # Should NOT be identical (LSTM captures sequence order)
        assert not torch.allclose(final1, final2, atol=0.01)

        print("✓ LSTM preserves temporal order in hidden states")

    def test_permutation_sensitivity(self, config):
        """Permuting sequence changes encoding (temporal sensitivity)."""
        torch.manual_seed(42)  # Deterministic initialization
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        encoder.eval()  # CRITICAL: Disable dropout for deterministic testing

        seq_len = 20

        # Original sequence
        original = torch.randn(1, seq_len, config.input_dim)

        # Rotated sequence (keeps last element different to test earlier positions)
        rotated = torch.roll(original, shifts=seq_len//2, dims=1)

        # Encode both
        out_orig = encoder(original, entity_ids=torch.tensor([0]))
        out_rot = encoder(rotated, entity_ids=torch.tensor([0]))

        # Measure baseline noise (same sequence encoded twice with dropout=0)
        out_baseline = encoder(original, entity_ids=torch.tensor([0]))
        baseline_diff = (out_orig.hidden_states[0] - out_baseline.hidden_states[0]).norm()

        # Compare ALL hidden states, not just final (to test earlier positions)
        hidden_orig = out_orig.hidden_states[0]  # (seq_len, hidden_dim)
        hidden_rot = out_rot.hidden_states[0]

        distance = (hidden_orig - hidden_rot).norm()

        # Require distance >> baseline (10x margin for robustness)
        assert distance > baseline_diff * 10, f"Distance {distance:.4f} not >> baseline {baseline_diff:.4f}"

        print(f"✓ Rotation distance: {distance:.4f} vs baseline: {baseline_diff:.4f} (temporal order matters)")

    def test_positional_encoding_not_needed(self, config):
        """Verify LSTM alone provides temporal info (no explicit pos encoding needed)."""
        # This test documents our design decision:
        # LSTMs have built-in temporal modeling via recurrence
        # Unlike Transformers, we don't need additive positional encodings

        torch.manual_seed(42)  # Deterministic initialization
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        encoder.eval()  # CRITICAL: Disable dropout for deterministic testing

        # Same input at different positions should produce different hidden states
        batch_size, seq_len = 1, 10
        constant_value = torch.ones(batch_size, seq_len, config.input_dim)

        output = encoder(constant_value, entity_ids=torch.tensor([0]))
        hidden = output.hidden_states[0]  # (seq_len, hidden_dim)

        # Compare first and last hidden states
        # Even with constant input, LSTM state evolves over time
        h_first = hidden[0]
        h_last = hidden[-1]

        # Should be different (LSTM state accumulates)
        assert not torch.allclose(h_first, h_last, atol=0.1)

        print("✓ LSTM provides temporal encoding without explicit positional embeddings")
