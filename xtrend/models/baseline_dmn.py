"""Baseline Deep Momentum Network (DMN) without cross-attention (Equation 7)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig
from xtrend.models.embeddings import EntityEmbedding
from xtrend.models.encoder import LSTMEncoder


class BaselineDMN(nn.Module):
    """Baseline neural forecaster without context/cross-attention.

    Implements Equation 7: z_t = tanh(Linear(g(x_t)))
    where g(·) is the LSTM encoder.

    This serves as the baseline comparison for X-Trend. It uses the same
    encoder architecture but without the cross-attention mechanism that
    enables few-shot learning from context sets.

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
    """

    def __init__(self, config: ModelConfig, use_entity: bool = True):
        super().__init__()
        self.config = config
        self.use_entity = use_entity

        # Shared entity embedding (one instance for entire model)
        if use_entity:
            self.entity_embedding = EntityEmbedding(config)
        else:
            self.entity_embedding = None

        # Encoder: g(x_t) from Equation 7
        self.encoder = LSTMEncoder(
            config,
            use_entity=use_entity,
            entity_embedding=self.entity_embedding
        )

        # Position head: tanh(Linear(g(x_t))) - Equation 7
        # Paper-faithful: single linear projection before tanh
        self.position_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Tanh()
        )

    def forward(
        self,
        x: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict trading positions.

        Args:
            x: Input features (batch, seq_len, input_dim)
            entity_ids: Entity IDs (batch,)

        Returns:
            Positions (batch, seq_len) in range (-1, 1)
        """
        # Encode: g(x_t)
        encoder_output = self.encoder(x, entity_ids)
        hidden_states = encoder_output.hidden_states  # (batch, seq_len, hidden_dim)

        # Position head: tanh(Linear(g(x_t)))
        positions = self.position_head(hidden_states)  # (batch, seq_len, 1)
        positions = positions.squeeze(-1)  # (batch, seq_len)

        return positions
