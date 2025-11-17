"""Tests for entity embeddings."""
import pytest
import torch
from xtrend.models.embeddings import EntityEmbedding, ConditionalFFN
from xtrend.models.types import ModelConfig


class TestEntityEmbedding:
    def test_embedding_initialization(self, model_config):
        """EntityEmbedding initializes with correct dimensions."""
        emb = EntityEmbedding(model_config)

        # Should have embedding table for num_entities
        assert emb.embedding.num_embeddings == model_config.num_entities
        assert emb.embedding.embedding_dim == model_config.hidden_dim

    def test_embedding_forward_shape(self, model_config, sample_entity_ids):
        """EntityEmbedding produces correct output shape."""
        emb = EntityEmbedding(model_config)

        output = emb(sample_entity_ids)

        batch_size = sample_entity_ids.shape[0]
        assert output.shape == (batch_size, model_config.hidden_dim)

    def test_embedding_unique_per_entity(self, model_config):
        """Different entities get different embeddings."""
        emb = EntityEmbedding(model_config)

        entity_0 = torch.tensor([0])
        entity_1 = torch.tensor([1])

        emb_0 = emb(entity_0)
        emb_1 = emb(entity_1)

        # Different entities should have different embeddings
        assert not torch.allclose(emb_0, emb_1)

    def test_embedding_same_entity_consistent(self, model_config):
        """Same entity gets same embedding."""
        emb = EntityEmbedding(model_config)

        entity_0 = torch.tensor([0, 0, 0])

        output = emb(entity_0)

        # All three outputs should be identical
        assert torch.allclose(output[0], output[1])
        assert torch.allclose(output[1], output[2])


class TestConditionalFFN:
    def test_conditional_ffn_initialization(self, model_config):
        """ConditionalFFN initializes correctly."""
        entity_embedding = EntityEmbedding(model_config)
        cffn = ConditionalFFN(model_config, use_entity=True, entity_embedding=entity_embedding)

        assert cffn.linear1 is not None
        assert cffn.linear2 is not None
        assert cffn.linear3 is not None
        assert cffn.entity_embedding is entity_embedding  # Verify shared instance

    def test_conditional_ffn_equation_12(self, model_config):
        """ConditionalFFN implements Equation 12."""
        # Equation 12: FFN(h_t, s) = Linear3 ∘ ELU(Linear1(h_t) + Linear2(Embedding(s)))

        entity_embedding = EntityEmbedding(model_config)
        cffn = ConditionalFFN(model_config, use_entity=True, entity_embedding=entity_embedding)

        batch_size = 32
        h_t = torch.randn(batch_size, model_config.hidden_dim)
        entity_ids = torch.randint(0, model_config.num_entities, (batch_size,))

        output = cffn(h_t, entity_ids)

        assert output.shape == (batch_size, model_config.hidden_dim)

    def test_conditional_ffn_with_no_entity(self, model_config):
        """ConditionalFFN works without entity info (zero-shot)."""
        cffn = ConditionalFFN(model_config, use_entity=False, entity_embedding=None)

        batch_size = 32
        h_t = torch.randn(batch_size, model_config.hidden_dim)

        # Should work with entity_ids=None for zero-shot
        output = cffn(h_t, entity_ids=None)

        assert output.shape == (batch_size, model_config.hidden_dim)
