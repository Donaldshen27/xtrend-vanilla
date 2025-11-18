"""Query/Key/Value projection networks (Equations 15-16)."""
import torch
import torch.nn as nn

from xtrend.models.types import ModelConfig


class QKVProjections(nn.Module):
    """Separate projection networks for Query, Key, Value.

    Following paper Equations 15-16:
    - Query from target: q_t = Ξ_query(x_t, s)
    - Keys from context: K_t = {Ξ_key(x^c, s^c)}_C
    - Values from context: V_t = {Ξ_value(ξ^c, s^c)}_C

    Each projection is a separate FFN to allow different transformations.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Separate projection networks
        # Each is a 2-layer FFN with GELU activation
        self.query_proj = self._make_projection_network(config)
        self.key_proj = self._make_projection_network(config)
        self.value_proj = self._make_projection_network(config)

    def _make_projection_network(self, config: ModelConfig) -> nn.Module:
        """Create a projection network (2-layer FFN).

        Args:
            config: Model configuration

        Returns:
            Sequential FFN module
        """
        return nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

    def project_query(self, target_states: torch.Tensor) -> torch.Tensor:
        """Project target encoded states to queries.

        Args:
            target_states: Encoded target (batch, seq_len, hidden_dim)

        Returns:
            Queries (batch, seq_len, hidden_dim)
        """
        return self.query_proj(target_states)

    def project_key(self, context_states: torch.Tensor) -> torch.Tensor:
        """Project context states to keys.

        Args:
            context_states: Encoded context (batch, context_size, hidden_dim)

        Returns:
            Keys (batch, context_size, hidden_dim)
        """
        return self.key_proj(context_states)

    def project_value(self, context_states: torch.Tensor) -> torch.Tensor:
        """Project context states to values.

        Args:
            context_states: Encoded context (batch, context_size, hidden_dim)

        Returns:
            Values (batch, context_size, hidden_dim)
        """
        return self.value_proj(context_states)
