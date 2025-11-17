"""Entity embeddings for futures contracts (Equation 12)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig


class EntityEmbedding(nn.Module):
    """Learnable embeddings for futures contract types.

    Maps each contract ticker to a learned embedding vector.
    This allows the model to learn similarities between contracts
    (e.g., crude oil and heating oil should have similar embeddings).

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding table: num_entities x hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=config.num_entities,
            embedding_dim=config.hidden_dim
        )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for entity IDs.

        Args:
            entity_ids: Entity indices (batch,)
                Note: While nn.Embedding supports (batch, seq_len), ConditionalFFN
                expects (batch,) as each sequence corresponds to one entity.

        Returns:
            Embeddings (batch, hidden_dim)
        """
        return self.embedding(entity_ids)


class ConditionalFFN(nn.Module):
    """Feed-forward network conditioned on entity embeddings (Equation 12).

    Implements: FFN(h_t, s) = Linear3 ∘ ELU(Linear1(h_t) + Linear2(Embedding(s)))

    This fuses time-series representations with entity-specific information.
    For zero-shot learning, entity info can be excluded (entity_ids=None).

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
        entity_embedding: Shared EntityEmbedding instance (required if use_entity=True)
    """

    def __init__(
        self,
        config: ModelConfig,
        use_entity: bool = True,
        entity_embedding: Optional[EntityEmbedding] = None
    ):
        super().__init__()
        self.config = config
        self.use_entity = use_entity

        # Linear transformations
        self.linear1 = nn.Linear(config.hidden_dim, config.hidden_dim)

        if use_entity:
            assert entity_embedding is not None, "entity_embedding required when use_entity=True"
            self.entity_embedding = entity_embedding
            self.linear2 = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(config.dropout)
        self.linear3 = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply conditional FFN.

        Args:
            h_t: Hidden states (batch, hidden_dim) or (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) - one entity per sequence. None for zero-shot.

        Returns:
            Output (same shape as h_t)
        """
        # Linear1(h_t)
        transformed = self.linear1(h_t)

        # Add entity information if available
        if self.use_entity and entity_ids is not None:
            # Entity IDs must be (batch,) - one entity per sequence
            assert entity_ids.dim() == 1, f"entity_ids must be (batch,), got shape {entity_ids.shape}"

            entity_emb = self.entity_embedding(entity_ids)  # (batch, hidden_dim)

            # Broadcast entity embedding to match h_t shape
            if h_t.dim() == 3:  # (batch, seq_len, hidden_dim)
                entity_emb = entity_emb.unsqueeze(1)  # (batch, 1, hidden_dim)

            # Linear2(Embedding(s))
            entity_transformed = self.linear2(entity_emb)
            transformed = transformed + entity_transformed

        # ELU activation
        activated = self.activation(transformed)
        activated = self.dropout(activated)

        # Linear3
        output = self.linear3(activated)

        return output
