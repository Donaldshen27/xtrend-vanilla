"""Tests for Query/Key/Value projection networks."""
import pytest
import torch

from xtrend.models.qkv_projections import QKVProjections
from xtrend.models.types import ModelConfig


class TestQKVProjections:
    """Test Q/K/V projection networks (Equations 15-16)."""

    @pytest.fixture
    def config(self):
        """Standard model config."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=50,
            num_attention_heads=4,
            dropout=0.1
        )

    def test_qkv_projections_creation(self, config):
        """Create separate Q/K/V projection networks."""
        qkv = QKVProjections(config)

        # Should have 3 separate networks
        assert hasattr(qkv, 'query_proj')
        assert hasattr(qkv, 'key_proj')
        assert hasattr(qkv, 'value_proj')

    def test_query_projection_from_target(self, config):
        """Project target sequence to queries (Equation 15)."""
        batch_size, seq_len, hidden_dim = 2, 126, 64

        # Target encoded states from encoder
        target_states = torch.randn(batch_size, seq_len, hidden_dim)

        qkv = QKVProjections(config)
        queries = qkv.project_query(target_states)

        # Output should maintain shape
        assert queries.shape == (batch_size, seq_len, hidden_dim)

    def test_key_projection_from_context(self, config):
        """Project context to keys (Equation 16)."""
        batch_size, context_size, hidden_dim = 2, 20, 64

        # Context encoded states (after self-attention)
        context_states = torch.randn(batch_size, context_size, hidden_dim)

        qkv = QKVProjections(config)
        keys = qkv.project_key(context_states)

        assert keys.shape == (batch_size, context_size, hidden_dim)

    def test_value_projection_from_context(self, config):
        """Project context to values (Equation 16)."""
        batch_size, context_size, hidden_dim = 2, 20, 64

        context_states = torch.randn(batch_size, context_size, hidden_dim)

        qkv = QKVProjections(config)
        values = qkv.project_value(context_states)

        assert values.shape == (batch_size, context_size, hidden_dim)

    def test_separate_networks(self, config):
        """Q/K/V use separate parameter networks."""
        qkv = QKVProjections(config)

        # Get parameters from each network
        query_params = list(qkv.query_proj.parameters())
        key_params = list(qkv.key_proj.parameters())
        value_params = list(qkv.value_proj.parameters())

        # Should have different parameters
        assert len(query_params) > 0
        assert len(key_params) > 0
        assert len(value_params) > 0

        # Parameters should not be shared
        assert id(query_params[0]) != id(key_params[0])
        assert id(key_params[0]) != id(value_params[0])
