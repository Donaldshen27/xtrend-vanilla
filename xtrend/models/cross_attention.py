"""Multi-head cross-attention between target and context (Equations 15-18)."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.cross_attention_types import AttentionConfig, AttentionOutput


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: target attends to context.

    Following Equations 15-18:
    - Query from target: q_t
    - Keys from context: K_t
    - Values from context: V_t (after self-attention)
    - Attention: y_t = LayerNorm ∘ FFN ∘ Att(q_t, K_t, V'_t)

    Attention weights stored for interpretability (Figure 9).

    Args:
        config: Attention configuration
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        # Multi-head attention mechanism
        self.mha = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # FFN after attention (Equation 18)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> AttentionOutput:
        """Apply cross-attention from target to context.

        Args:
            query: Target queries (batch, target_len, hidden_dim)
            key: Context keys (batch, context_size, hidden_dim)
            value: Context values (batch, context_size, hidden_dim)
            key_padding_mask: Context padding mask (batch, context_size)
                True = valid, False = padding

        Returns:
            AttentionOutput with output and attention weights
        """
        # Convert padding mask for PyTorch convention
        if key_padding_mask is not None:
            # True=valid,False=padding -> True=padding,False=valid
            pytorch_mask = ~key_padding_mask
        else:
            pytorch_mask = None

        # Cross-attention: query from target, key/value from context
        attn_output, attn_weights = self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=pytorch_mask,
            need_weights=True,
            average_attn_weights=False  # Keep per-head weights
        )

        # FFN with residual connection (Equation 18)
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm(ffn_output + attn_output)

        return AttentionOutput(
            output=output,
            attention_weights=attn_weights
        )
