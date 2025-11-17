"""LSTM encoder with skip connections (Equation 14)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import ConditionalFFN, EntityEmbedding


class LSTMEncoder(nn.Module):
    """LSTM encoder with Variable Selection and skip connections (Equation 14).

    Architecture:
        x'_t = VSN(x_t, s)
        (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        a_t = LayerNorm(h_t + x'_t)  # Skip connection 1
        Ξ = LayerNorm(FFN(a_t, s) + a_t)  # Skip connection 2

    The skip connections allow the model to suppress components
    when needed, enabling adaptive complexity.

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

        # Shared entity embedding
        if use_entity:
            assert entity_embedding is not None, "entity_embedding required when use_entity=True"
            self.entity_embedding = entity_embedding

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(config)

        # LSTM cell (single layer, so dropout parameter is ignored)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True
        )

        # Explicit dropout on LSTM outputs (since single-layer LSTM ignores dropout param)
        self.lstm_dropout = nn.Dropout(config.dropout)

        # Layer normalization after LSTM
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)

        # FFN with entity conditioning (shares entity_embedding)
        self.ffn = ConditionalFFN(
            config,
            use_entity=use_entity,
            entity_embedding=entity_embedding if use_entity else None
        )

        # Layer normalization after FFN
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)

        # Learnable initial LSTM state (per entity if using entity info)
        if use_entity:
            self.init_h = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
        else:
            self.init_h = nn.Parameter(torch.randn(1, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(1, config.hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> EncoderOutput:
        """Encode input sequences.

        Args:
            x: Input features (batch, seq_len, input_dim)
            entity_ids: Entity IDs (batch,) - None for zero-shot

        Returns:
            EncoderOutput with hidden_states (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Variable Selection Network
        # x'_t = VSN(x_t, s)
        x_prime, _ = self.vsn(x)  # (batch, seq_len, hidden_dim)

        # Step 2: Initialize LSTM state
        if self.use_entity and entity_ids is not None:
            # Entity-specific initialization
            h_0 = self.init_h[entity_ids]  # (batch, hidden_dim)
            c_0 = self.init_c[entity_ids]  # (batch, hidden_dim)
        else:
            # Generic initialization for zero-shot
            h_0 = self.init_h.expand(batch_size, -1)  # (batch, hidden_dim)
            c_0 = self.init_c.expand(batch_size, -1)  # (batch, hidden_dim)

        # LSTM expects (num_layers, batch, hidden_dim)
        h_0 = h_0.unsqueeze(0)  # (1, batch, hidden_dim)
        c_0 = c_0.unsqueeze(0)  # (1, batch, hidden_dim)

        # Step 3: LSTM forward pass
        # (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        lstm_out, _ = self.lstm(x_prime, (h_0, c_0))  # (batch, seq_len, hidden_dim)

        # Apply dropout to LSTM outputs
        lstm_out = self.lstm_dropout(lstm_out)

        # Step 4: First skip connection with layer norm
        # a_t = LayerNorm(h_t + x'_t)
        a_t = self.layer_norm1(lstm_out + x_prime)

        # Step 5: FFN with entity conditioning
        ffn_out = self.ffn(a_t, entity_ids)  # (batch, seq_len, hidden_dim)

        # Step 6: Second skip connection with layer norm
        # Ξ = LayerNorm(FFN(a_t, s) + a_t)
        output = self.layer_norm2(ffn_out + a_t)

        return EncoderOutput(
            hidden_states=output,
            sequence_length=seq_len
        )
