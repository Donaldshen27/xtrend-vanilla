"""xtrend.models — Neural network architectures."""
from xtrend.models.types import ModelConfig, EncoderOutput
from xtrend.models.vsn import VariableSelectionNetwork
from xtrend.models.embeddings import EntityEmbedding, ConditionalFFN
from xtrend.models.encoder import LSTMEncoder
from xtrend.models.decoder import LSTMDecoder
from xtrend.models.baseline_dmn import BaselineDMN
from xtrend.models.cross_attention_types import AttentionConfig, AttentionOutput
from xtrend.models.xtrend_cross_attention import XTrendCrossAttention
from xtrend.models.heads import PositionHead, GaussianHead, QuantileHead, PTP_G, PTP_Q
from xtrend.models.losses import (
    sharpe_loss,
    gaussian_nll_loss,
    quantile_loss,
    joint_gaussian_loss,
    joint_quantile_loss
)
from xtrend.models.xtrend import XTrend, XTrendG, XTrendQ

__all__ = [
    "ModelConfig",
    "EncoderOutput",
    "VariableSelectionNetwork",
    "EntityEmbedding",
    "ConditionalFFN",
    "LSTMEncoder",
    "LSTMDecoder",
    "BaselineDMN",
    "AttentionConfig",
    "AttentionOutput",
    "XTrendCrossAttention",
    "PositionHead",
    "GaussianHead",
    "QuantileHead",
    "PTP_G",
    "PTP_Q",
    "sharpe_loss",
    "gaussian_nll_loss",
    "quantile_loss",
    "joint_gaussian_loss",
    "joint_quantile_loss",
    "XTrend",
    "XTrendG",
    "XTrendQ",
]
