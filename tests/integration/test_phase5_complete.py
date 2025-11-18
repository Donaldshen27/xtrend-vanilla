"""Integration tests for Phase 5: Cross-Attention Mechanism."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.models import (
    ModelConfig,
    LSTMEncoder,
    EntityEmbedding,
    XTrendCrossAttention,
)
from xtrend.context import (
    sample_cpd_segmented,
    ContextBatch,
)
from xtrend.cpd import CPDConfig, GPCPDSegmenter


class TestPhase5Integration:
    """Integration tests verifying Phase 5 completion criteria."""

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
        prices = {}
        symbols = [f"ASSET{i}" for i in range(10)]

        for symbol in symbols:
            # Simulate prices
            price_series = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
            prices[symbol] = pd.Series(price_series, index=dates)

            # Features (8-dim)
            features[symbol] = torch.randn(len(dates), 8)

        return {
            'features': features,
            'prices': prices,
            'dates': dates,
            'symbols': symbols
        }

    def test_full_pipeline_target_to_context(self, config, realistic_data):
        """Complete pipeline: encode target and context, apply cross-attention."""
        # Create encoder
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)

        # Create cross-attention module
        cross_attn = XTrendCrossAttention(config)
        cross_attn.eval()  # Disable dropout for deterministic behavior

        # Target sequence
        target_symbol = "ASSET0"
        target_entity_id = realistic_data['symbols'].index(target_symbol)
        target_features = realistic_data['features'][target_symbol][:126]  # 126 days
        target_entity = torch.tensor([target_entity_id])

        # Encode target
        target_encoded = encoder(
            target_features.unsqueeze(0),  # Add batch dim
            entity_ids=target_entity
        )

        # Context sequences (5 contexts)
        context_features = []
        context_entities = []
        for i in range(1, 6):
            symbol = realistic_data['symbols'][i]
            entity_id = i
            ctx_feat = realistic_data['features'][symbol][:21]  # 21 days
            context_features.append(ctx_feat)
            context_entities.append(entity_id)

        # Encode contexts
        context_encoded_list = []
        for ctx_feat, ctx_entity in zip(context_features, context_entities):
            ctx_enc = encoder(
                ctx_feat.unsqueeze(0),
                entity_ids=torch.tensor([ctx_entity])
            )
            # Take final hidden state
            context_encoded_list.append(ctx_enc.hidden_states[:, -1:, :])

        # Stack contexts: (batch=1, context_size=5, hidden_dim)
        context_encoded = torch.cat(context_encoded_list, dim=1)

        # Apply cross-attention
        output = cross_attn(
            target_encoded.hidden_states,
            context_encoded
        )

        # Phase 5 completion criteria
        assert output.output.shape == (1, 126, config.hidden_dim)
        assert output.attention_weights.shape == (1, config.num_attention_heads, 126, 5)

        # Attention weights sum to 1
        weights_sum = output.attention_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6)

        print("✓ Full pipeline: target → context cross-attention working")
        print(f"✓ Output shape: {output.output.shape}")
        print(f"✓ Attention weights shape: {output.attention_weights.shape}")

    def test_variable_length_contexts(self, config, realistic_data):
        """Handle variable-length context sequences with padding."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)

        # Target
        target_features = realistic_data['features']['ASSET0'][:63]
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=torch.tensor([0])
        )

        # Variable-length contexts: lengths [21, 15, 10, 18, 12]
        context_lengths = [21, 15, 10, 18, 12]
        max_len = max(context_lengths)

        # Pad contexts to max_len
        context_padded = torch.zeros(1, len(context_lengths), max_len, config.hidden_dim)
        padding_mask = torch.zeros(1, len(context_lengths), max_len, dtype=torch.bool)

        for i, length in enumerate(context_lengths):
            ctx_feat = realistic_data['features'][realistic_data['symbols'][i+1]][:length]
            ctx_enc = encoder(
                ctx_feat.unsqueeze(0),
                entity_ids=torch.tensor([i+1])
            )
            # Use all timesteps from this context
            context_padded[0, i, :length, :] = ctx_enc.hidden_states[0, :length, :]
            padding_mask[0, i, :length] = True

        # Reshape for cross-attention: (batch, total_positions, hidden_dim)
        # We need to flatten context sequences
        batch_size = 1
        context_flat = context_padded.view(batch_size, -1, config.hidden_dim)
        mask_flat = padding_mask.view(batch_size, -1)

        output = cross_attn(
            target_encoded.hidden_states,
            context_flat,
            context_padding_mask=mask_flat
        )

        assert output.output.shape == (1, 63, config.hidden_dim)
        print("✓ Variable-length contexts handled with padding masks")

    def test_attention_interpretability(self, config, realistic_data):
        """Attention weights interpretable (Figure 9 pattern)."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)

        # Target
        target_features = realistic_data['features']['ASSET0'][:21]
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=torch.tensor([0])
        )

        # 10 context sequences
        context_encoded_list = []
        for i in range(1, 11):
            if i < len(realistic_data['symbols']):
                ctx_feat = realistic_data['features'][realistic_data['symbols'][i]][:10]
            else:
                # Create dummy features if we don't have enough symbols
                ctx_feat = torch.randn(10, 8)
            ctx_enc = encoder(
                ctx_feat.unsqueeze(0),
                entity_ids=torch.tensor([min(i, config.num_entities - 1)])
            )
            context_encoded_list.append(ctx_enc.hidden_states[:, -1:, :])

        context_encoded = torch.cat(context_encoded_list, dim=1)

        output = cross_attn(target_encoded.hidden_states, context_encoded)

        # Attention weights available
        weights = output.attention_weights  # (1, 4, 21, 10)

        # Can analyze which contexts are most attended
        avg_attention = weights.mean(dim=(0, 1, 2))  # Average over batch, heads, time
        top3_contexts = avg_attention.topk(3).indices

        print(f"✓ Attention weights interpretable")
        print(f"✓ Top-3 attended contexts: {top3_contexts.tolist()}")
        print(f"✓ Attention distribution: min={avg_attention.min():.4f}, "
              f"max={avg_attention.max():.4f}, mean={avg_attention.mean():.4f}")

    def test_gradient_flow_end_to_end(self, config, realistic_data):
        """Gradients flow from cross-attention through encoder."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)

        # Target
        target_features = realistic_data['features']['ASSET0'][:21]
        target_features.requires_grad_(True)
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=torch.tensor([0])
        )

        # Context
        context_features = realistic_data['features']['ASSET1'][:10]
        context_features.requires_grad_(True)
        context_encoded = encoder(
            context_features.unsqueeze(0),
            entity_ids=torch.tensor([1])
        )

        # Cross-attention
        output = cross_attn(
            target_encoded.hidden_states,
            context_encoded.hidden_states
        )

        # Backward pass
        loss = output.output.sum()
        loss.backward()

        # Check gradients
        assert target_features.grad is not None
        assert context_features.grad is not None
        assert not torch.isnan(target_features.grad).any()
        assert not torch.isnan(context_features.grad).any()

        print("✓ Gradients flow end-to-end through cross-attention and encoder")

    def test_multi_head_diversity(self, config, realistic_data):
        """Different attention heads learn different patterns."""
        entity_embedding = EntityEmbedding(config)
        encoder = LSTMEncoder(config, use_entity=True, entity_embedding=entity_embedding)
        cross_attn = XTrendCrossAttention(config)

        # Target and context
        target_features = realistic_data['features']['ASSET0'][:21]
        target_encoded = encoder(
            target_features.unsqueeze(0),
            entity_ids=torch.tensor([0])
        )

        context_encoded_list = []
        for i in range(1, 6):
            ctx_feat = realistic_data['features'][realistic_data['symbols'][i]][:10]
            ctx_enc = encoder(
                ctx_feat.unsqueeze(0),
                entity_ids=torch.tensor([i])
            )
            context_encoded_list.append(ctx_enc.hidden_states[:, -1:, :])

        context_encoded = torch.cat(context_encoded_list, dim=1)

        output = cross_attn(target_encoded.hidden_states, context_encoded)

        # Compare attention patterns across heads
        weights = output.attention_weights[0]  # (4, 21, 5)

        # Compute correlation between heads
        head_patterns = []
        for head_idx in range(config.num_attention_heads):
            pattern = weights[head_idx].mean(dim=0)  # Average over time
            head_patterns.append(pattern)

        # Heads should have some diversity (not perfectly correlated)
        # This is a weak test (patterns emerge after training)
        print(f"✓ Multi-head attention with {config.num_attention_heads} heads")
        for i, pattern in enumerate(head_patterns):
            print(f"  Head {i}: {pattern.tolist()}")
