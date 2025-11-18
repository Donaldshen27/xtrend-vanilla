"""Type definitions for cross-attention mechanism."""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class AttentionOutput:
    """Output from attention mechanism.

    Attributes:
        output: Attention output (batch, seq_len, hidden_dim)
        attention_weights: Attention weights (batch, num_heads, seq_len, context_size)
            for interpretability (Figure 9 in paper)
    """
    output: torch.Tensor
    attention_weights: torch.Tensor

    def __post_init__(self):
        """Validate shapes."""
        batch_size, seq_len, hidden_dim = self.output.shape
        batch_size_w, num_heads, seq_len_w, context_size = self.attention_weights.shape

        if batch_size != batch_size_w or seq_len != seq_len_w:
            raise ValueError(
                f"Shape mismatch: output {self.output.shape}, "
                f"weights {self.attention_weights.shape}"
            )


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism.

    Attributes:
        hidden_dim: Hidden dimension (must be divisible by num_heads)
        num_heads: Number of attention heads (paper uses 4)
        dropout: Dropout rate
    """
    hidden_dim: int
    num_heads: int = 4
    dropout: float = 0.3

    def __post_init__(self):
        """Validate configuration."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads
