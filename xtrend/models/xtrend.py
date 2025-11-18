"""Complete X-Trend model variants."""
import torch
import torch.nn as nn
from typing import Optional, Dict

from xtrend.models.types import ModelConfig
from xtrend.models.decoder import LSTMDecoder
from xtrend.models.heads import PositionHead, GaussianHead, QuantileHead, PTP_G, PTP_Q
from xtrend.models.embeddings import EntityEmbedding


class XTrend(nn.Module):
    """X-Trend model: Direct position prediction (Equation 7).

    Architecture:
        Encoder (Phase 3) -> Cross-Attention (Phase 5) -> Decoder -> PositionHead

    Loss:
        L = L_Sharpe(z, r)

    Args:
        config: Model configuration
        entity_embedding: Shared entity embedding (optional for zero-shot)
    """

    def __init__(
        self,
        config: ModelConfig,
        entity_embedding: Optional[EntityEmbedding] = None
    ):
        super().__init__()
        self.config = config
        self.entity_embedding = entity_embedding
        self.use_entity = entity_embedding is not None

        # Decoder
        self.decoder = LSTMDecoder(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # Position prediction head
        self.position_head = PositionHead(config)

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) or None for zero-shot

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # Decode
        decoder_output = self.decoder(target_features, cross_attn_output, entity_ids)

        # Predict positions
        positions = self.position_head(decoder_output.hidden_states)

        return positions


class XTrendG(nn.Module):
    """X-Trend-G model: Gaussian prediction with PTP (Equations 20-21).

    Architecture:
        Encoder -> Cross-Attention -> Decoder -> GaussianHead -> PTP_G

    Loss:
        L = α * L_MLE(μ, σ, r) + L_Sharpe(PTP_G(μ, σ), r)

    Args:
        config: Model configuration
        entity_embedding: Shared entity embedding (optional for zero-shot)
    """

    def __init__(
        self,
        config: ModelConfig,
        entity_embedding: Optional[EntityEmbedding] = None
    ):
        super().__init__()
        self.config = config
        self.entity_embedding = entity_embedding
        self.use_entity = entity_embedding is not None

        # Decoder
        self.decoder = LSTMDecoder(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # Gaussian prediction head
        self.gaussian_head = GaussianHead(config)

        # PTP module
        self.ptp = PTP_G(config)

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) or None for zero-shot

        Returns:
            dict with keys:
                - 'mean': Predicted mean (batch, seq_len)
                - 'std': Predicted std dev (batch, seq_len)
                - 'positions': Trading positions from PTP (batch, seq_len)
        """
        # Decode
        decoder_output = self.decoder(target_features, cross_attn_output, entity_ids)

        # Predict Gaussian parameters
        mean, std = self.gaussian_head(decoder_output.hidden_states)

        # Map to positions via PTP
        positions = self.ptp(mean, std)

        return {
            'mean': mean,
            'std': std,
            'positions': positions
        }


class XTrendQ(nn.Module):
    """X-Trend-Q model: Quantile prediction with PTP (Equations 22-23).

    Architecture:
        Encoder -> Cross-Attention -> Decoder -> QuantileHead -> PTP_Q

    Loss:
        L = α * L_QRE(Q, r) + L_Sharpe(PTP_Q(Q), r)

    Best performing variant (Table 1, Page 12).

    Args:
        config: Model configuration
        entity_embedding: Shared entity embedding (optional for zero-shot)
        num_quantiles: Number of quantiles (default: 13)
    """

    def __init__(
        self,
        config: ModelConfig,
        entity_embedding: Optional[EntityEmbedding] = None,
        num_quantiles: int = 13
    ):
        super().__init__()
        self.config = config
        self.entity_embedding = entity_embedding
        self.use_entity = entity_embedding is not None
        self.num_quantiles = num_quantiles

        # Decoder
        self.decoder = LSTMDecoder(
            config,
            use_entity=self.use_entity,
            entity_embedding=entity_embedding
        )

        # Quantile prediction head
        self.quantile_head = QuantileHead(config, num_quantiles=num_quantiles)

        # PTP module
        self.ptp = PTP_Q(config, num_quantiles=num_quantiles)

    def forward(
        self,
        target_features: torch.Tensor,
        cross_attn_output: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            target_features: Raw target features (batch, seq_len, input_dim)
            cross_attn_output: Cross-attention output (batch, seq_len, hidden_dim)
            entity_ids: Entity IDs (batch,) or None for zero-shot

        Returns:
            dict with keys:
                - 'quantiles': Predicted quantiles (batch, seq_len, num_quantiles)
                - 'positions': Trading positions from PTP (batch, seq_len)
        """
        # Decode
        decoder_output = self.decoder(target_features, cross_attn_output, entity_ids)

        # Predict quantiles
        quantiles = self.quantile_head(decoder_output.hidden_states)

        # Map to positions via PTP
        positions = self.ptp(quantiles)

        return {
            'quantiles': quantiles,
            'positions': positions
        }
