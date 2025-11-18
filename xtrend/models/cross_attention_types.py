"""Type definitions for cross-attention mechanism."""
from dataclasses import dataclass
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
        # Check ranks first
        if len(self.output.shape) != 3:
            raise ValueError(
                f"AttentionOutput.output must be rank-3 (batch, seq_len, hidden_dim), "
                f"got shape {self.output.shape}"
            )
        if len(self.attention_weights.shape) != 4:
            raise ValueError(
                f"AttentionOutput.attention_weights must be rank-4 "
                f"(batch, num_heads, seq_len, context_size), "
                f"got shape {self.attention_weights.shape}"
            )

        # Now safe to destructure
        batch_size, seq_len, hidden_dim = self.output.shape
        batch_size_w, num_heads, seq_len_w, context_size = self.attention_weights.shape

        if batch_size != batch_size_w or seq_len != seq_len_w:
            raise ValueError(
                f"Shape mismatch: output {self.output.shape}, "
                f"weights {self.attention_weights.shape}"
            )

        # NEW: Multi-head constraint
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}). "
                f"Expected head_dim = {hidden_dim} / {num_heads} to be an integer."
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
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads
