"""Type definitions for neural network models."""
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


@dataclass
class ModelConfig:
    """Configuration for X-Trend neural architecture.

    Attributes:
        input_dim: Number of input features (5 returns + 3 MACD)
        hidden_dim: Hidden state dimension d_h (Paper: 64 or 128)
        dropout: Dropout rate (Paper: 0.3, 0.4, or 0.5)
        num_entities: Number of futures contracts for embeddings (Paper: 50)
        num_attention_heads: Number of parallel attention heads (Paper: 4)
    """
    input_dim: int = 8
    hidden_dim: int = 64
    dropout: float = 0.3
    num_entities: int = 50
    num_attention_heads: int = 4

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim ({self.input_dim}) must be positive")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout ({self.dropout}) must be in [0, 1)")
        if self.num_entities <= 0:
            raise ValueError(f"num_entities ({self.num_entities}) must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be positive")


class EncoderOutput(NamedTuple):
    """Output from encoder module.

    Attributes:
        hidden_states: Encoded representations (batch, seq_len, hidden_dim)
        sequence_length: Length of sequences in batch
    """
    hidden_states: torch.Tensor
    sequence_length: int
