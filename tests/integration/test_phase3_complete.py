"""Integration tests for Phase 3: Base Neural Architecture."""
import pytest
import torch
import numpy as np
from xtrend.models import (
    ModelConfig,
    VariableSelectionNetwork,
    EntityEmbedding,
    ConditionalFFN,
    LSTMEncoder,
    BaselineDMN,
)


class TestPhase3Integration:
    """Integration tests verifying Phase 3 completion criteria."""

    def test_full_pipeline_shape(self):
        """Complete pipeline produces correct shapes."""
        config = ModelConfig(hidden_dim=64, dropout=0.3)

        # Create model
        model = BaselineDMN(config)

        # Sample data
        batch_size, seq_len = 32, 126
        x = torch.randn(batch_size, seq_len, 8)
        entity_ids = torch.randint(0, 50, (batch_size,))

        # Forward pass
        positions = model(x, entity_ids)

        # Check output
        assert positions.shape == (batch_size, seq_len)
        assert (positions > -1).all() and (positions < 1).all()

    def test_vsn_attention_interpretability(self):
        """VSN attention weights have correct structural properties."""
        config = ModelConfig()
        vsn = VariableSelectionNetwork(config)

        x = torch.randn(4, 10, 8)
        output, weights = vsn(x)

        # Structural check 1: Weights should sum to 1 (softmax property)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 10), atol=1e-6)

        # Structural check 2: Weights should be non-negative (softmax property)
        assert (weights >= 0).all()

        # Structural check 3: Weights should have correct shape
        assert weights.shape == (4, 10, 8)  # (batch, seq_len, input_dim)

    def test_entity_embedding_structural_properties(self):
        """Entity embeddings have correct structural properties."""
        config = ModelConfig(num_entities=50)
        embedding = EntityEmbedding(config)

        # Structural check 1: Different entities get different embeddings (not identical)
        entity_0 = torch.tensor([0])
        entity_1 = torch.tensor([1])
        emb_0 = embedding(entity_0)
        emb_1 = embedding(entity_1)
        assert not torch.allclose(emb_0, emb_1)

        # Structural check 2: Same entity gets same embedding (consistency)
        emb_0_again = embedding(entity_0)
        assert torch.allclose(emb_0, emb_0_again)

        # Structural check 3: Embeddings are finite (no NaN/inf)
        assert torch.isfinite(emb_0).all()
        assert torch.isfinite(emb_1).all()

    def test_baseline_dmn_gradient_flow(self):
        """Gradients flow properly through entire model."""
        config = ModelConfig(hidden_dim=64)
        model = BaselineDMN(config)

        x = torch.randn(4, 20, 8, requires_grad=True)
        entity_ids = torch.randint(0, 50, (4,))

        # Forward pass
        positions = model(x, entity_ids)

        # Simple loss
        loss = positions.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Skip generic LSTM states when entity_ids are provided (not used in forward)
                if 'generic' in name:
                    continue
                assert param.grad is not None, f"{name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"

    def test_encoder_skip_connections_structural(self):
        """Skip connections are structurally present in the computation graph."""
        config = ModelConfig(hidden_dim=64)

        # Create encoder with shared embedding
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)

        x = torch.randn(4, 50, 8, requires_grad=True)
        entity_ids = torch.randint(0, 50, (4,))

        # Forward pass
        output = encoder(x, entity_ids)
        loss = output.hidden_states.mean()
        loss.backward()

        # Structural check 1: Gradients flow back to input
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Structural check 2: All model parameters receive gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                # Skip generic LSTM states when entity_ids are provided (not used in forward)
                if 'generic' in name:
                    continue
                assert param.grad is not None, f"{name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"

    def test_skip_connections_functional(self):
        """Verify skip connections actually affect the output."""
        config = ModelConfig(hidden_dim=64)

        # Create encoder with shared embedding
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        encoder.eval()  # Disable dropout for deterministic test

        x = torch.randn(4, 20, 8)
        entity_ids = torch.randint(0, 50, (4,))

        with torch.no_grad():
            # Get intermediate representations
            x_prime, _ = encoder.vsn(x)
            h_0 = encoder.init_h[entity_ids].unsqueeze(0)
            c_0 = encoder.init_c[entity_ids].unsqueeze(0)
            lstm_out, _ = encoder.lstm(x_prime, (h_0, c_0))
            lstm_out = encoder.lstm_dropout(lstm_out)

            # First skip connection: a_t = LayerNorm(h_t + x'_t)
            # Verify that output differs from just LayerNorm(h_t)
            with_skip = encoder.layer_norm1(lstm_out + x_prime)
            without_skip = encoder.layer_norm1(lstm_out)
            assert not torch.allclose(with_skip, without_skip), \
                "First skip connection has no effect"

            # Second skip connection: output = LayerNorm(FFN(a_t) + a_t)
            # Verify that output differs from just LayerNorm(FFN(a_t))
            a_t = with_skip
            ffn_out = encoder.ffn(a_t, entity_ids)
            with_skip_2 = encoder.layer_norm2(ffn_out + a_t)
            without_skip_2 = encoder.layer_norm2(ffn_out)
            assert not torch.allclose(with_skip_2, without_skip_2), \
                "Second skip connection has no effect"

    def test_zero_shot_mode(self):
        """Model works in zero-shot mode (no entity embeddings)."""
        config = ModelConfig()

        # Create encoder without entity embeddings
        encoder = LSTMEncoder(config, use_entity=False, entity_embedding=None)

        x = torch.randn(4, 20, 8)

        # Forward without entity_ids
        output = encoder(x, entity_ids=None)

        assert output.hidden_states.shape == (4, 20, 64)

    def test_phase3_visual_completion_criteria(self):
        """Verify Phase 3 visual completion criteria from phases.md."""
        # From phases.md:
        # - Output shape: torch.Size([32, 126, 64])
        # - Output range: within tanh range (-1, 1)

        config = ModelConfig(hidden_dim=64)
        model = BaselineDMN(config, use_entity=True)
        model.eval()

        batch_size, seq_len, n_features = 32, 126, 8
        x = torch.randn(batch_size, seq_len, n_features)
        entity_ids = torch.randint(0, 50, (batch_size,))

        with torch.no_grad():
            # Get encoder output
            encoder_output = model.encoder(x, entity_ids)
            print(f"Encoder output shape: {encoder_output.hidden_states.shape}")
            assert encoder_output.hidden_states.shape == torch.Size([32, 126, 64])

            # Get position output
            positions = model(x, entity_ids)
            print(f"Position output shape: {positions.shape}")
            assert positions.shape == (32, 126)

            print(f"Position range: [{positions.min():.3f}, {positions.max():.3f}]")
            # Should be within tanh range (-1, 1)
            assert positions.min() >= -1.0
            assert positions.max() <= 1.0


def test_phase3_complete():
    """Overall Phase 3 completion test."""
    # All components should be importable
    from xtrend.models import (
        ModelConfig,
        VariableSelectionNetwork,
        EntityEmbedding,
        ConditionalFFN,
        LSTMEncoder,
        BaselineDMN,
    )

    # Create a complete model
    config = ModelConfig()
    model = BaselineDMN(config)

    # Model should have correct structure
    assert isinstance(model.encoder, LSTMEncoder)
    assert isinstance(model.encoder.vsn, VariableSelectionNetwork)

    print("✅ Phase 3 Complete!")
    print("   - Variable Selection Network (Equation 13)")
    print("   - Entity Embeddings (50 contracts)")
    print("   - LSTM Encoder with Skip Connections (Equation 14)")
    print("   - Baseline DMN Model (Equation 7)")
