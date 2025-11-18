"""Type definitions for context set construction."""
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import pandas as pd


ContextMethod = Literal["final_hidden_state", "time_equivalent", "cpd_segmented"]


@dataclass
class ContextSequence:
    """A single context sequence for cross-attention.

    Attributes:
        features: Input features (seq_len, input_dim)
        entity_id: Entity/asset ID for this sequence
        start_date: Start date of sequence (timezone-aware)
        end_date: End date of sequence (timezone-aware)
        method: Construction method used
        padding_mask: Optional mask for padded positions (True = valid, False = padding)
    """
    features: torch.Tensor
    entity_id: torch.Tensor
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    method: ContextMethod
    padding_mask: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate and create default padding mask."""
        # Ensure timestamps are timezone-aware (UTC)
        if self.start_date.tz is None:
            self.start_date = self.start_date.tz_localize('UTC')
        if self.end_date.tz is None:
            self.end_date = self.end_date.tz_localize('UTC')

        # Create default padding mask if not provided (all valid)
        if self.padding_mask is None:
            self.padding_mask = torch.ones(self.features.shape[0], dtype=torch.bool)

        # Validate padding mask shape
        if self.padding_mask.shape[0] != self.features.shape[0]:
            raise ValueError(
                f"Padding mask length ({self.padding_mask.shape[0]}) "
                f"must match features length ({self.features.shape[0]})"
            )

    @property
    def length(self) -> int:
        """Sequence length."""
        return self.features.shape[0]

    @property
    def input_dim(self) -> int:
        """Feature dimension."""
        return self.features.shape[1]

    def is_causal(self, target_start_date: pd.Timestamp, buffer_days: int = 1) -> bool:
        """Check if context is before target with buffer (causality).

        Args:
            target_start_date: Target sequence START date (not end)
            buffer_days: Buffer between context end and target start (default 1 day)

        Returns:
            True if context ends before target starts with buffer
        """
        # Ensure target is timezone-aware
        if target_start_date.tz is None:
            target_start_date = target_start_date.tz_localize('UTC')

        # Context must end at least buffer_days before target starts
        buffer = pd.Timedelta(days=buffer_days)
        return self.end_date + buffer <= target_start_date


@dataclass
class ContextBatch:
    """Batch of context sequences for a single target.

    Attributes:
        sequences: List of C context sequences
        C: Context set size
        max_length: Maximum sequence length in batch
    """
    sequences: list[ContextSequence]

    def __post_init__(self):
        """Validate batch homogeneity (Codex feedback)."""
        if len(self.sequences) == 0:
            raise ValueError("Context batch cannot be empty")

        # All sequences must use same construction method
        methods = {seq.method for seq in self.sequences}
        if len(methods) > 1:
            raise ValueError(f"Mixed methods in batch: {methods}")

        # All sequences must have same feature dimension (schema validation)
        input_dims = {seq.input_dim for seq in self.sequences}
        if len(input_dims) > 1:
            raise ValueError(
                f"Mixed feature dimensions in batch: {input_dims}. "
                "All sequences must have same input_dim."
            )

    @property
    def C(self) -> int:
        """Context set size."""
        return len(self.sequences)

    @property
    def max_length(self) -> int:
        """Maximum sequence length in batch."""
        return max(seq.length for seq in self.sequences)

    @property
    def input_dim(self) -> int:
        """Feature dimension (consistent across batch)."""
        return self.sequences[0].input_dim

    def verify_causality(
        self,
        target_start_date: pd.Timestamp,
        buffer_days: int = 1
    ) -> bool:
        """Verify all context sequences are before target with buffer.

        Args:
            target_start_date: Target sequence START date
            buffer_days: Buffer between context and target

        Returns:
            True if all contexts are causal
        """
        return all(
            seq.is_causal(target_start_date, buffer_days)
            for seq in self.sequences
        )

    def to_padded_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert batch to padded tensor with mask for attention.

        Returns:
            features: (C, max_length, input_dim) padded tensor
            mask: (C, max_length) padding mask (True = valid, False = padding)
        """
        C = self.C
        max_len = self.max_length
        input_dim = self.input_dim

        # Preallocate tensors
        features = torch.zeros(C, max_len, input_dim)
        mask = torch.zeros(C, max_len, dtype=torch.bool)

        # Fill with sequences
        for i, seq in enumerate(self.sequences):
            seq_len = seq.length
            features[i, :seq_len] = seq.features
            mask[i, :seq_len] = seq.padding_mask

        return features, mask
