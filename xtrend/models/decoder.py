"""LSTM decoder with cross-attention fusion (Equations 19a-19d)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import ConditionalFFN, EntityEmbedding


class LSTMDecoder(nn.Module):
    """LSTM decoder that fuses target features with cross-attention output.

    Architecture (Equation 19a-19d, Page 9):
        x'_t = LayerNorm ∘ FFN_1 ∘ Concat(VSN(x_t, s), y_t)
        (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        a_t = LayerNorm(h_t + x'_t)  # Skip connection 1
        Ξ_Dec = LayerNorm(FFN_2(a_t, s) + a_t)  # Skip connection 2

    The decoder combines:
    - Target features x_t (passed through VSN)
    - Cross-attention output y_t (from Phase 5)

    This allows the model to integrate both target and context information.

    Args:
        config: Model configuration
        use_entity: Whether to use entity embeddings (False for zero-shot)
        entity_embedding: Shared entity embedding (if use_entity=True)
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
        self.entity_embedding = entity_embedding

        # ✅ FIXED: Variable selection with entity conditioning
        self.vsn = VariableSelectionNetwork(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # ✅ FIXED: Fusion layer using entity-conditioned FFN (Issue #6)
        # Note: ConditionalFFN doesn't support input_dim/output_dim parameters
        # We need to project first, then apply conditioning
        self.fusion_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fusion_ffn = ConditionalFFN(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)

        # LSTM with entity-conditioned initial state
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
            dropout=0.0  # We add explicit dropout layer
        )
        self.lstm_dropout = nn.Dropout(config.dropout)

        # Skip connection 1: after LSTM
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)

        # ✅ FIXED: Skip connection 2 with entity conditioning
        self.ffn = ConditionalFFN(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)

        # Learnable initial states (per entity + generic fallback just like encoder)
        if self.use_entity:
            assert entity_embedding is not None, "entity_embedding required when use_entity=True"
            self.init_h = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(config.num_entities, config.hidden_dim))
            self.init_h_generic = nn.Parameter(torch.randn(1, config.hidden_dim))
            self.init_c_generic = nn.Parameter(torch.randn(1, config.hidden_dim))
        else:
            self.init_h = nn.Parameter(torch.randn(1, config.hidden_dim))
            self.init_c = nn.Parameter(torch.randn(1, config.hidden_dim))

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> EncoderOutput:
        """Apply decoder to fuse target and cross-attention output.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs for conditioning (batch,) or None for zero-shot

        Returns:
            EncoderOutput with fused hidden states
        """
        batch_size = target_features.shape[0]

        # ✅ FIXED: VSN returns (output, weights) tuple - unpack it (Issue #1)
        x_selected, _ = self.vsn(target_features, entity_ids)

        # ✅ FIXED: Step 2: Concatenate and fuse with entity conditioning (Issue #6)
        concatenated = torch.cat([x_selected, cross_attn_output], dim=-1)
        x_projected = self.fusion_proj(concatenated)  # Project to hidden_dim
        x_fused_raw = self.fusion_ffn(x_projected, entity_ids)  # Entity conditioning
        x_fused = self.fusion_norm(x_fused_raw)  # Equation 19a: x'_t

        # Step 3: Initialize LSTM state (matches encoder behavior)
        if self.use_entity and entity_ids is not None:
            h0 = self.init_h[entity_ids]
            c0 = self.init_c[entity_ids]
        elif self.use_entity and entity_ids is None:
            h0 = self.init_h_generic.expand(batch_size, -1)
            c0 = self.init_c_generic.expand(batch_size, -1)
        else:
            h0 = self.init_h.expand(batch_size, -1)
            c0 = self.init_c.expand(batch_size, -1)

        h0 = h0.unsqueeze(0)
        c0 = c0.unsqueeze(0)
        initial_state = (h0, c0)

        # Step 4: LSTM processing
        # (h_t, c_t) = LSTM(x'_t, h_{t-1}, c_{t-1})
        lstm_out, _ = self.lstm(x_fused, initial_state)
        lstm_out = self.lstm_dropout(lstm_out)

        # Step 5: Skip connection 1
        # a_t = LayerNorm(h_t + x'_t)
        skip1 = self.layer_norm1(lstm_out + x_fused)

        # Step 6: Conditional FFN with skip connection 2
        # Ξ_Dec = LayerNorm(FFN(a_t, s) + a_t)
        ffn_out = self.ffn(skip1, entity_ids)
        output = self.layer_norm2(ffn_out + skip1)

        # ✅ FIXED: Return EncoderOutput with sequence_length (Issue #4)
        return EncoderOutput(
            hidden_states=output,
            sequence_length=output.shape[1]
        )
