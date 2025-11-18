"""Complete X-Trend cross-attention module integrating all components."""
import torch
import torch.nn as nn
from typing import Optional

from xtrend.models.types import ModelConfig
from xtrend.models.cross_attention_types import AttentionOutput
from xtrend.models.self_attention import MultiHeadSelfAttention
from xtrend.models.cross_attention import MultiHeadCrossAttention
from xtrend.models.qkv_projections import QKVProjections


class XTrendCrossAttention(nn.Module):
    """Complete X-Trend cross-attention module.

    Pipeline (following Equations 15-18):
    1. Self-attention over context: V'_t = SelfAtt(V_t)
    2. Project target to queries: q_t = Q_proj(target)
    3. Project context to keys: K_t = K_proj(V'_t)
    4. Project context to values: V_t = V_proj(V'_t)
    5. Cross-attention: y_t = CrossAtt(q_t, K_t, V_t)

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Create attention config from model config
        from xtrend.models.cross_attention_types import AttentionConfig
        self.attn_config = AttentionConfig(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout
        )

        # Step 1: Self-attention over context set (Equation 17)
        self.context_self_attention = MultiHeadSelfAttention(self.attn_config)

        # Step 2-4: Q/K/V projections (Equations 15-16)
        self.qkv_projections = QKVProjections(config)

        # Step 5: Cross-attention (Equation 18)
        self.cross_attention = MultiHeadCrossAttention(self.attn_config)

    def forward(
        self,
        target_encoded: torch.Tensor,
        context_encoded: torch.Tensor,
        context_padding_mask: Optional[torch.Tensor] = None
    ) -> AttentionOutput:
        """Apply complete cross-attention pipeline.

        Args:
            target_encoded: Encoded target (batch, target_len, hidden_dim)
            context_encoded: Encoded context (batch, context_size, hidden_dim)
            context_padding_mask: Context padding (batch, context_size)
                True = valid, False = padding

        Returns:
            AttentionOutput with attended target and attention weights
        """
        # Step 1: Self-attention over context set
        # V'_t = SelfAtt(V_t)
        context_self_attended = self.context_self_attention(
            context_encoded,
            key_padding_mask=context_padding_mask
        )

        # Step 2: Project target to queries
        queries = self.qkv_projections.project_query(target_encoded)

        # Step 3-4: Project context to keys and values
        keys = self.qkv_projections.project_key(context_self_attended)
        values = self.qkv_projections.project_value(context_self_attended)

        # Step 5: Cross-attention
        # y_t = CrossAtt(q_t, K_t, V_t)
        output = self.cross_attention(
            query=queries,
            key=keys,
            value=values,
            key_padding_mask=context_padding_mask
        )

        return output
