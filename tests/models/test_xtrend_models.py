"""Tests for integrated X-Trend model variants."""
import pytest
import torch

from xtrend.models.xtrend import XTrend, XTrendG, XTrendQ
from xtrend.models.types import ModelConfig
from xtrend.models.embeddings import EntityEmbedding


class TestXTrendModels:
    """Test complete model variants."""

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

    def test_xtrend_forward(self, config):
        """X-Trend: direct position prediction."""
        batch_size, target_len = 2, 126

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        # Cross-attention output has same seq_len as target
        cross_attn_output = torch.randn(batch_size, target_len, config.hidden_dim)

        entity_embedding = EntityEmbedding(config)
        model = XTrend(config, entity_embedding=entity_embedding)

        positions = model(target_features, cross_attn_output, entity_ids=torch.tensor([0, 1]))

        assert positions.shape == (batch_size, target_len)
        assert (positions > -1).all() and (positions < 1).all()

    def test_xtrend_g_forward(self, config):
        """X-Trend-G: Gaussian prediction + PTP."""
        batch_size, target_len = 2, 126

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        # Cross-attention output has same seq_len as target
        cross_attn_output = torch.randn(batch_size, target_len, config.hidden_dim)

        entity_embedding = EntityEmbedding(config)
        model = XTrendG(config, entity_embedding=entity_embedding)

        outputs = model(target_features, cross_attn_output, entity_ids=torch.tensor([0, 1]))

        assert 'mean' in outputs
        assert 'std' in outputs
        assert 'positions' in outputs
        assert outputs['mean'].shape == (batch_size, target_len)
        assert outputs['std'].shape == (batch_size, target_len)
        assert outputs['positions'].shape == (batch_size, target_len)
        assert (outputs['std'] > 0).all()

    def test_xtrend_q_forward(self, config):
        """X-Trend-Q: Quantile prediction + PTP."""
        batch_size, target_len = 2, 126
        num_quantiles = 13

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        # Cross-attention output has same seq_len as target
        cross_attn_output = torch.randn(batch_size, target_len, config.hidden_dim)

        entity_embedding = EntityEmbedding(config)
        model = XTrendQ(config, entity_embedding=entity_embedding, num_quantiles=num_quantiles)

        outputs = model(target_features, cross_attn_output, entity_ids=torch.tensor([0, 1]))

        assert 'quantiles' in outputs
        assert 'positions' in outputs
        assert outputs['quantiles'].shape == (batch_size, target_len, num_quantiles)
        assert outputs['positions'].shape == (batch_size, target_len)

    def test_zero_shot_mode(self, config):
        """All models work in zero-shot mode (no entity embeddings)."""
        batch_size, target_len = 2, 63

        target_features = torch.randn(batch_size, target_len, config.input_dim)
        # Cross-attention output has same seq_len as target
        cross_attn_output = torch.randn(batch_size, target_len, config.hidden_dim)

        # X-Trend without entity embedding
        model = XTrend(config, entity_embedding=None)
        positions = model(target_features, cross_attn_output, entity_ids=None)
        assert positions.shape == (batch_size, target_len)
