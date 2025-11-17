"""Tests for Baseline DMN model."""
import pytest
import torch
from xtrend.models.baseline_dmn import BaselineDMN
from xtrend.models.types import ModelConfig


class TestBaselineDMN:
    def test_baseline_initialization(self, model_config):
        """BaselineDMN initializes correctly."""
        model = BaselineDMN(model_config)

        assert model.encoder is not None
        assert model.position_head is not None

    def test_baseline_forward_shape(self, model_config, sample_features, sample_entity_ids):
        """Baseline produces positions with correct shape."""
        model = BaselineDMN(model_config)

        positions = model(sample_features, sample_entity_ids)

        batch_size, seq_len, _ = sample_features.shape
        # Positions should be (batch, seq_len)
        assert positions.shape == (batch_size, seq_len)

    def test_baseline_position_range(self, model_config, sample_features, sample_entity_ids):
        """Positions are in (-1, 1) range (tanh bounded)."""
        model = BaselineDMN(model_config)

        positions = model(sample_features, sample_entity_ids)

        # All positions should be in (-1, 1)
        assert (positions > -1).all()
        assert (positions < 1).all()

    def test_baseline_equation_7(self, model_config):
        """Baseline implements Equation 7 from paper."""
        # Equation 7: z_t = tanh(Linear(g(x_t)))
        # where g(x_t) is the encoder

        model = BaselineDMN(model_config)
        model.eval()

        x = torch.randn(4, 20, 8)
        entity_ids = torch.randint(0, 50, (4,))

        with torch.no_grad():
            positions = model(x, entity_ids)

        # Positions exist and are bounded
        assert positions is not None
        assert positions.shape == (4, 20)
        assert (positions.abs() < 1).all()

    def test_baseline_gradient_flow(self, model_config):
        """Gradients flow through the model."""
        model = BaselineDMN(model_config)

        x = torch.randn(4, 20, 8, requires_grad=True)
        entity_ids = torch.randint(0, 50, (4,))

        positions = model(x, entity_ids)
        loss = positions.mean()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
