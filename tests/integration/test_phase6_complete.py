"""Integration tests for Phase 6: Decoder & Loss Functions."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.models import (
    ModelConfig,
    LSTMEncoder,
    EntityEmbedding,
    XTrendCrossAttention,
    XTrend,
    XTrendG,
    XTrendQ,
    sharpe_loss,
    joint_gaussian_loss,
    joint_quantile_loss,
)


class TestPhase6Integration:
    """Integration tests verifying Phase 6 completion criteria."""

    @pytest.fixture
    def config(self):
        """Model configuration."""
        return ModelConfig(
            input_dim=8,
            hidden_dim=64,
            num_entities=10,
            num_attention_heads=4,
            dropout=0.1
        )

    @pytest.fixture
    def realistic_data(self):
        """Create realistic multi-asset data."""
        dates = pd.date_range('2019-01-01', '2020-12-31', freq='D')
        np.random.seed(42)

        features = {}
        returns = {}
        symbols = [f"ASSET{i}" for i in range(10)]

        for symbol in symbols:
            # Simulate features (8-dim)
            features[symbol] = torch.randn(len(dates), 8)

            # Simulate returns (scaled by vol targeting)
            returns_series = np.random.randn(len(dates)) * 0.01
            returns[symbol] = torch.tensor(returns_series, dtype=torch.float32)

        return {
            'features': features,
            'returns': returns,
            'dates': dates,
            'symbols': symbols
        }

    def test_xtrend_full_pipeline(self, config, realistic_data):
        """Complete pipeline: encoder → cross-attention → decoder → head."""
        # Create components
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrend(config, entity_embedding=entity_embedding)

        # Target sequence
        target_symbol = "ASSET0"
        target_features = realistic_data['features'][target_symbol][:126]
        target_returns = realistic_data['returns'][target_symbol][:126]
        target_entity_id = torch.tensor([0])

        # Encode target
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=target_entity_id
        )

        # Context sequences (5 contexts)
        context_encoded_list = []
        for i in range(1, 6):
            ctx_feat = realistic_data['features'][realistic_data['symbols'][i]][:21]
            ctx_enc = encoder(
                ctx_feat.unsqueeze(0),
                entity_ids=torch.tensor([i])
            )
            context_encoded_list.append(ctx_enc.hidden_states[:, -1:, :])

        context_encoded = torch.cat(context_encoded_list, dim=1)

        # Cross-attention
        cross_attn_output = cross_attn(
            target_encoded.hidden_states,
            context_encoded
        )

        # Decode and predict positions
        positions = model(
            target_features.unsqueeze(0),
            cross_attn_output.output,
            entity_ids=target_entity_id
        )

        # Phase 6 completion criteria
        assert positions.shape == (1, 126)
        assert (positions > -1).all() and (positions < 1).all()

        # Compute loss
        loss = sharpe_loss(positions, target_returns.unsqueeze(0), warmup_steps=63)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        print("✓ X-Trend full pipeline working")
        print(f"✓ Position range: [{positions.min():.4f}, {positions.max():.4f}]")
        print(f"✓ Sharpe loss: {loss.item():.4f}")

    def test_xtrend_g_with_joint_loss(self, config, realistic_data):
        """X-Trend-G: Gaussian prediction with joint loss."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrendG(config, entity_embedding=entity_embedding)

        # Prepare data
        target_features = realistic_data['features']['ASSET0'][:126]
        target_returns = realistic_data['returns']['ASSET0'][:126]
        target_entity_id = torch.tensor([0])

        # Encode target
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=target_entity_id
        )

        # Mock context (simplified)
        context_encoded = torch.randn(1, 5, config.hidden_dim)

        # Cross-attention
        cross_attn_output = cross_attn(
            target_encoded.hidden_states,
            context_encoded
        )

        # Forward pass
        outputs = model(
            target_features.unsqueeze(0),
            cross_attn_output.output,
            entity_ids=target_entity_id
        )

        # Check outputs
        assert 'mean' in outputs
        assert 'std' in outputs
        assert 'positions' in outputs
        assert outputs['mean'].shape == (1, 126)
        assert outputs['std'].shape == (1, 126)
        assert outputs['positions'].shape == (1, 126)
        assert (outputs['std'] > 0).all()

        # Compute joint loss
        loss = joint_gaussian_loss(
            outputs['mean'],
            outputs['std'],
            outputs['positions'],
            target_returns.unsqueeze(0),
            alpha=1.0,
            warmup_steps=63
        )

        assert not torch.isnan(loss)
        print(f"✓ X-Trend-G joint loss: {loss.item():.4f}")

    def test_xtrend_q_with_joint_loss(self, config, realistic_data):
        """X-Trend-Q: Quantile prediction with joint loss."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrendQ(config, entity_embedding=entity_embedding, num_quantiles=13)

        # Quantile levels
        levels = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        # Prepare data
        target_features = realistic_data['features']['ASSET0'][:126]
        target_returns = realistic_data['returns']['ASSET0'][:126]
        target_entity_id = torch.tensor([0])

        # Encode target
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=target_entity_id
        )

        # Mock context
        context_encoded = torch.randn(1, 5, config.hidden_dim)

        # Cross-attention
        cross_attn_output = cross_attn(
            target_encoded.hidden_states,
            context_encoded
        )

        # Forward pass
        outputs = model(
            target_features.unsqueeze(0),
            cross_attn_output.output,
            entity_ids=target_entity_id
        )

        # Check outputs
        assert 'quantiles' in outputs
        assert 'positions' in outputs
        assert outputs['quantiles'].shape == (1, 126, 13)
        assert outputs['positions'].shape == (1, 126)

        # Compute joint loss
        loss = joint_quantile_loss(
            outputs['quantiles'],
            levels,
            outputs['positions'],
            target_returns.unsqueeze(0),
            alpha=5.0,
            warmup_steps=63
        )

        assert not torch.isnan(loss)
        print(f"✓ X-Trend-Q joint loss: {loss.item():.4f}")

    def test_gradient_flow_end_to_end(self, config):
        """Gradients flow through entire pipeline."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)
        model = XTrend(config, entity_embedding=entity_embedding)

        # Create data with gradients enabled
        target_features = torch.randn(1, 63, config.input_dim, requires_grad=True)
        target_returns = torch.randn(1, 63)
        context_encoded = torch.randn(1, 5, config.hidden_dim)

        # Forward pass
        target_encoded = encoder(target_features, entity_ids=torch.tensor([0]))
        cross_attn_output = cross_attn(target_encoded.hidden_states, context_encoded)
        positions = model(target_features, cross_attn_output.output, entity_ids=torch.tensor([0]))

        # Loss and backward
        loss = sharpe_loss(positions, target_returns, warmup_steps=20)
        loss.backward()

        # Check gradients
        assert target_features.grad is not None
        assert not torch.isnan(target_features.grad).any()
        print("✓ Gradients flow end-to-end")

    def test_zero_shot_mode(self, config, realistic_data):
        """Models work in zero-shot mode (no entity embeddings)."""
        encoder = LSTMEncoder(config, use_entity=False, entity_embedding=None)
        cross_attn = XTrendCrossAttention(config)
        model = XTrend(config, entity_embedding=None)

        # Prepare data
        target_features = realistic_data['features']['ASSET0'][:63]
        target_returns = realistic_data['returns']['ASSET0'][:63]

        # Encode target (no entity)
        target_encoded = encoder(target_features.unsqueeze(0), entity_ids=None)

        # Mock context
        context_encoded = torch.randn(1, 3, config.hidden_dim)

        # Cross-attention
        cross_attn_output = cross_attn(target_encoded.hidden_states, context_encoded)

        # Forward pass
        positions = model(target_features.unsqueeze(0), cross_attn_output.output, entity_ids=None)

        assert positions.shape == (1, 63)
        print("✓ Zero-shot mode working")
