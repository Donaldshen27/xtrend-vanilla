"""Tests for Variable Selection Network."""
import pytest
import torch
import numpy as np
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import EntityEmbedding
from xtrend.models.types import ModelConfig


class TestVariableSelectionNetwork:
    def test_vsn_initialization(self, model_config):
        """VSN initializes with correct dimensions."""
        vsn = VariableSelectionNetwork(model_config)

        # Should have feature-wise FFNs for each input feature
        assert len(vsn.feature_ffns) == model_config.input_dim
        # Should have attention layers
        assert vsn.linear1 is not None
        assert vsn.linear3 is not None

    def test_vsn_forward_shape(self, model_config, sample_features):
        """VSN produces correct output shape."""
        vsn = VariableSelectionNetwork(model_config)

        output, weights = vsn(sample_features)

        batch_size, seq_len, _ = sample_features.shape
        # Output should be (batch, seq_len, hidden_dim)
        assert output.shape == (batch_size, seq_len, model_config.hidden_dim)
        # Weights should be (batch, seq_len, input_dim)
        assert weights.shape == (batch_size, seq_len, model_config.input_dim)

    def test_vsn_weights_sum_to_one(self, model_config, sample_features):
        """VSN attention weights sum to 1 (Softmax property)."""
        vsn = VariableSelectionNetwork(model_config)

        _, weights = vsn(sample_features)

        # Weights should sum to 1 across features dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_vsn_weights_non_negative(self, model_config, sample_features):
        """VSN attention weights are non-negative (Softmax property)."""
        vsn = VariableSelectionNetwork(model_config)

        _, weights = vsn(sample_features)

        assert (weights >= 0).all()

    def test_vsn_equation_13(self, model_config):
        """VSN implements Equation 13 from paper."""
        # Equation 13: VSN(x_t) = Σ w_{t,j} * FFN_j(x_{t,j})
        # where w_t = Softmax(FFN(x_t))

        vsn = VariableSelectionNetwork(model_config)
        vsn.eval()

        # Simple input: 1 sample, 1 timestep, 8 features
        x = torch.randn(1, 1, 8)

        with torch.no_grad():
            output, weights = vsn(x)

        # Manually verify weighted sum
        # Each feature gets processed by its FFN, then weighted
        manual_output = torch.zeros(1, 1, model_config.hidden_dim)
        for j in range(model_config.input_dim):
            feature_j = x[:, :, j:j+1]  # (1, 1, 1)
            processed_j = vsn.feature_ffns[j](feature_j)  # (1, 1, hidden_dim)
            manual_output += weights[:, :, j:j+1] * processed_j

        assert torch.allclose(output, manual_output, atol=1e-5)

    def test_vsn_entity_conditioning_different_weights(self, model_config):
        """VSN produces different attention weights for different entities (Equation 14)."""
        # Equation 14: x'_t = VSN(x_t, s) where s is the entity
        entity_embedding = EntityEmbedding(model_config)
        vsn = VariableSelectionNetwork(model_config, use_entity=True, entity_embedding=entity_embedding)
        vsn.eval()

        # Same input features for both samples
        x = torch.randn(2, 10, 8)
        entity_0 = torch.tensor([0, 0])  # Both samples are entity 0
        entity_1 = torch.tensor([1, 1])  # Both samples are entity 1

        with torch.no_grad():
            _, weights_0 = vsn(x, entity_0)
            _, weights_1 = vsn(x, entity_1)

        # Different entities should produce different attention patterns
        # This is the core requirement of Equation 14
        assert not torch.allclose(weights_0, weights_1, atol=1e-3), \
            "VSN must produce different attention weights for different entities"

    def test_vsn_entity_conditioning_same_weights(self, model_config):
        """VSN produces same attention weights for same entity."""
        entity_embedding = EntityEmbedding(model_config)
        vsn = VariableSelectionNetwork(model_config, use_entity=True, entity_embedding=entity_embedding)
        vsn.eval()

        # Same input features
        x = torch.randn(2, 10, 8)
        entity_same = torch.tensor([0, 0])  # Both samples are same entity

        with torch.no_grad():
            _, weights_1 = vsn(x, entity_same)
            _, weights_2 = vsn(x, entity_same)

        # Same entity should produce identical attention patterns
        assert torch.allclose(weights_1, weights_2, atol=1e-6)

    def test_vsn_without_entity(self, model_config, sample_features):
        """VSN works without entity conditioning (zero-shot mode)."""
        vsn = VariableSelectionNetwork(model_config, use_entity=False)

        output, weights = vsn(sample_features)

        batch_size, seq_len, _ = sample_features.shape
        assert output.shape == (batch_size, seq_len, model_config.hidden_dim)
        assert weights.shape == (batch_size, seq_len, model_config.input_dim)

    def test_vsn_entity_required_when_use_entity_true(self, model_config):
        """VSN requires entity_embedding when use_entity=True."""
        with pytest.raises(AssertionError, match="entity_embedding required"):
            VariableSelectionNetwork(model_config, use_entity=True, entity_embedding=None)
