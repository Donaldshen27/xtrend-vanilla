"""Variable Selection Network (Equation 14)."""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from xtrend.models.types import ModelConfig


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network from X-Trend paper (Equation 14).

    Implements feature-wise processing with learned attention weights
    conditioned on entity embeddings:
        VSN(x_t, s) = Σ_{j=1}^{|X|} w_{t,j} * FFN_j(x_{t,j})
        where w_t = Softmax(Linear3(ELU(Linear1(x_t) + Linear2(Embedding(s)))))

    This allows the model to learn which features are most important
    at each timestep for each specific entity/asset. The entity conditioning
    uses an additive pattern similar to ConditionalFFN (Equation 12).

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
        entity_embedding: Shared EntityEmbedding instance (required if use_entity=True)
    """

    def __init__(
        self,
        config: ModelConfig,
        use_entity: bool = False,
        entity_embedding: Optional['EntityEmbedding'] = None
    ):
        super().__init__()
        self.config = config
        self.use_entity = use_entity

        # Shared entity embedding
        if use_entity:
            assert entity_embedding is not None, "entity_embedding required when use_entity=True"
            self.entity_embedding = entity_embedding

        # Feature-wise FFNs (one per input feature)
        self.feature_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, config.hidden_dim),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
            for _ in range(config.input_dim)
        ])

        # Attention FFN - separate paths for features and entity embeddings
        # Following ConditionalFFN pattern from embeddings.py
        self.linear1 = nn.Linear(config.input_dim, config.hidden_dim)

        if use_entity:
            self.linear2 = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.attention_activation = nn.ELU()
        self.attention_dropout = nn.Dropout(config.dropout)
        self.linear3 = nn.Linear(config.hidden_dim, config.input_dim)

    def forward(
        self,
        x: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply variable selection to input features.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            entity_ids: Entity IDs (batch,) - None for zero-shot

        Returns:
            output: Weighted feature representations (batch, seq_len, hidden_dim)
            weights: Attention weights (batch, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # Compute attention weights using additive entity conditioning
        # Following the pattern from ConditionalFFN (Equation 12)
        # w_t = Softmax(Linear3(ELU(Linear1(x_t) + Linear2(Embedding(s)))))

        # Linear1(x_t)
        transformed = self.linear1(x)  # (batch, seq_len, hidden_dim)

        # Add entity information if available
        if self.use_entity and entity_ids is not None:
            # Get entity embeddings (batch, hidden_dim)
            entity_emb = self.entity_embedding(entity_ids)
            # Broadcast to (batch, seq_len, hidden_dim)
            entity_emb = entity_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
            # Linear2(Embedding(s))
            entity_transformed = self.linear2(entity_emb)
            # Add to transformed: Linear1(x_t) + Linear2(Embedding(s))
            transformed = transformed + entity_transformed

        # Activation and dropout
        activated = self.attention_activation(transformed)
        activated = self.attention_dropout(activated)

        # Linear3 to get attention logits
        attn_logits = self.linear3(activated)  # (batch, seq_len, input_dim)
        weights = torch.softmax(attn_logits, dim=-1)

        # Process each feature with its dedicated FFN
        # and weight by attention
        processed_features = []
        for j in range(input_dim):
            # Extract j-th feature: (batch, seq_len, 1)
            feature_j = x[:, :, j:j+1]
            # Process with j-th FFN: (batch, seq_len, hidden_dim)
            processed_j = self.feature_ffns[j](feature_j)
            # Weight by attention: (batch, seq_len, 1) * (batch, seq_len, hidden_dim)
            weighted_j = weights[:, :, j:j+1] * processed_j
            processed_features.append(weighted_j)

        # Sum weighted features: Σ w_{t,j} * FFN_j(x_{t,j})
        output = sum(processed_features)

        return output, weights
