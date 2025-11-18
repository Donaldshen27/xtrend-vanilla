"""Prediction heads for X-Trend model variants."""
import torch
import torch.nn as nn
from typing import Tuple

from xtrend.models.types import ModelConfig


class PositionHead(nn.Module):
    """Direct position prediction head for X-Trend (Equation 7).

    Maps decoder output directly to trading position z ∈ (-1, 1).

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Single linear layer + tanh
        # Equation 7: z = tanh(Linear(g(x)))
        self.linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Predict trading positions.

        Args:
            decoder_output: Decoder hidden states (batch, seq_len, hidden_dim)

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # Linear projection + tanh
        logits = self.linear(decoder_output).squeeze(-1)  # (batch, seq_len)
        positions = torch.tanh(logits)
        return positions


class GaussianHead(nn.Module):
    """Gaussian prediction head for X-Trend-G (Equation 20).

    Predicts mean μ and standard deviation σ of return distribution.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Separate networks for mean and std
        self.mean_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        self.std_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Gaussian parameters.

        Args:
            decoder_output: Decoder hidden states (batch, seq_len, hidden_dim)

        Returns:
            mean: Predicted mean (batch, seq_len)
            std: Predicted std dev (batch, seq_len), positive
        """
        mean = self.mean_net(decoder_output).squeeze(-1)  # (batch, seq_len)
        std_logits = self.std_net(decoder_output).squeeze(-1)

        # ✅ FIXED: Ensure std is positive and safe after squaring (Issue #5)
        # σ = softplus(logits) + 1e-3 (safe after squaring to variance)
        std = torch.nn.functional.softplus(std_logits) + 1e-3

        return mean, std


class QuantileHead(nn.Module):
    """Quantile prediction head for X-Trend-Q (Equation 22).

    Predicts multiple quantiles of return distribution.

    Paper uses 13 quantiles:
    [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    Args:
        config: Model configuration
        num_quantiles: Number of quantiles to predict (default: 13)
    """

    def __init__(self, config: ModelConfig, num_quantiles: int = 13):
        super().__init__()
        self.config = config
        self.num_quantiles = num_quantiles

        # Network to predict all quantiles
        self.quantile_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_quantiles)
        )

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Predict quantiles.

        Args:
            decoder_output: Decoder hidden states (batch, seq_len, hidden_dim)

        Returns:
            quantiles: Predicted quantiles (batch, seq_len, num_quantiles)
                Monotonically increasing along last dimension
        """
        # Predict quantile differences (ensure ordering)
        raw_quantiles = self.quantile_net(decoder_output)  # (batch, seq_len, num_quantiles)

        # ✅ FIXED: Vectorized monotonicity enforcement (Issue #9)
        # First quantile (no constraint)
        first_quantile = raw_quantiles[:, :, :1]

        # Remaining quantiles as positive increments
        increments = torch.nn.functional.softplus(raw_quantiles[:, :, 1:])

        # Cumulative sum ensures monotonicity
        # quantiles = [q_0, q_0 + inc_1, q_0 + inc_1 + inc_2, ...]
        quantiles = torch.cat([
            first_quantile,
            first_quantile + torch.cumsum(increments, dim=-1)
        ], dim=-1)  # Vectorized, single backward pass

        return quantiles


class PTP_G(nn.Module):
    """Predictive To Position module for Gaussian head (Page 9).

    Maps Gaussian parameters (mean, std) to trading position z ∈ (-1, 1).

    Used for joint training: L = α * L_MLE + L_Sharpe^PTP_G

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ✅ FIXED: No dropout - paper specifies deterministic mapping (Issue #10)
        self.ffn = nn.Sequential(
            nn.Linear(2, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Map Gaussian parameters to positions.

        Args:
            mean: Predicted mean (batch, seq_len)
            std: Predicted std dev (batch, seq_len)

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # Stack mean and std
        params = torch.stack([mean, std], dim=-1)  # (batch, seq_len, 2)

        # FFN + tanh
        logits = self.ffn(params).squeeze(-1)  # (batch, seq_len)
        positions = torch.tanh(logits)

        return positions


class PTP_Q(nn.Module):
    """Predictive To Position module for Quantile head (Page 9).

    Maps quantile predictions to trading position z ∈ (-1, 1).

    Used for joint training: L = α * L_QRE + L_Sharpe^PTP_Q

    Args:
        config: Model configuration
        num_quantiles: Number of quantiles (default: 13)
    """

    def __init__(self, config: ModelConfig, num_quantiles: int = 13):
        super().__init__()
        self.config = config
        self.num_quantiles = num_quantiles

        # ✅ FIXED: No dropout - paper specifies deterministic mapping (Issue #10)
        self.ffn = nn.Sequential(
            nn.Linear(num_quantiles, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Map quantiles to positions.

        Args:
            quantiles: Predicted quantiles (batch, seq_len, num_quantiles)

        Returns:
            positions: Trading positions (batch, seq_len) in (-1, 1)
        """
        # FFN + tanh
        logits = self.ffn(quantiles).squeeze(-1)  # (batch, seq_len)
        positions = torch.tanh(logits)

        return positions
