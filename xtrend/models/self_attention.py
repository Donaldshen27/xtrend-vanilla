"""Multi-head self-attention over context set (Equation 17)."""
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple

from xtrend.models.cross_attention_types import AttentionConfig


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention: V'_t = FFN ∘ Att(V_t, V_t, V_t).

    This processes the context set with self-attention before
    cross-attention with the target (Equation 17).

    Args:
        config: Attention configuration
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        # Use PyTorch's efficient MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True  # (batch, seq, feature) format
        )

        # FFN after attention (Equation 17)
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
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply self-attention to context set.

        Args:
            x: Context values (batch, context_size, hidden_dim)
            key_padding_mask: Padding mask (batch, context_size)
                True = valid, False = padding
            return_attention_weights: Return attention weights for interpretability

        Returns:
            If return_attention_weights=False:
                output: (batch, context_size, hidden_dim)
            If return_attention_weights=True:
                (output, attention_weights)
        """
        # Self-attention: query=key=value=x
        # Note: PyTorch's key_padding_mask uses opposite convention (True = ignore)
        # So we need to invert our mask
        if key_padding_mask is not None:
            # Convert: True=valid,False=padding -> True=padding,False=valid
            pytorch_mask = ~key_padding_mask
        else:
            pytorch_mask = None

        attn_output, attn_weights = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=pytorch_mask,
            need_weights=return_attention_weights,
            average_attn_weights=False  # Keep per-head weights
        )

        # FFN with residual connection
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm(ffn_output + attn_output)

        if return_attention_weights:
            return output, attn_weights
        return output
