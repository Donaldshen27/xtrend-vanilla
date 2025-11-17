"""Variable Selection Network (Equation 13)."""
import torch
import torch.nn as nn
from typing import Tuple

from xtrend.models.types import ModelConfig


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network from X-Trend paper (Equation 13).

    Implements feature-wise processing with learned attention weights:
        VSN(x_t) = Σ_{j=1}^{|X|} w_{t,j} * FFN_j(x_{t,j})
        where w_t = Softmax(FFN(x_t))

    This allows the model to learn which features are most important
    at each timestep.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Feature-wise FFNs (one per input feature)
        self.feature_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, config.hidden_dim),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
            for _ in range(config.input_dim)
        ])

        # Attention FFN for computing feature weights
        self.feature_attention = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply variable selection to input features.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            output: Weighted feature representations (batch, seq_len, hidden_dim)
            weights: Attention weights (batch, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # Compute attention weights: w_t = Softmax(FFN(x_t))
        # Shape: (batch, seq_len, input_dim) -> (batch, seq_len, input_dim)
        attn_logits = self.feature_attention(x)
        weights = torch.softmax(attn_logits, dim=-1)

        # Process each feature with its dedicated FFN
        # and weight by attention
        processed_features = []
        for j in range(input_dim):
            # Extract j-th feature: (batch, seq_len, 1)
            feature_j = x[:, :, j:j+1]
            # Process with j-th FFN: (batch, seq_len, hidden_dim)
            processed_j = self.feature_ffns[j](feature_j)
            # Weight by attention: (batch, seq_len, 1) * (batch, seq_len, hidden_dim)
            weighted_j = weights[:, :, j:j+1] * processed_j
            processed_features.append(weighted_j)

        # Sum weighted features: Σ w_{t,j} * FFN_j(x_{t,j})
        output = torch.stack(processed_features, dim=-1).sum(dim=-1)

        return output, weights
