"""Tests for context set types."""
import pytest
import torch
import pandas as pd
from xtrend.context.types import ContextSequence, ContextBatch


class TestContextSequence:
    """Test ContextSequence type."""

    def test_context_sequence_creation(self):
        """Create a single context sequence."""
        features = torch.randn(21, 8)  # 21 days, 8 features
        entity_id = torch.tensor(5)
        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2020-01-25')

        seq = ContextSequence(
            features=features,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            method="cpd_segmented"
        )

        assert seq.features.shape == (21, 8)
        assert seq.entity_id == 5
        assert seq.length == 21
        assert seq.method == "cpd_segmented"

    def test_causality_check(self):
        """Verify causality: context must be before target."""
        features = torch.randn(21, 8)
        entity_id = torch.tensor(5)

        seq = ContextSequence(
            features=features,
            entity_id=entity_id,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="final_hidden_state"
        )

        # Context before target = OK
        assert seq.is_causal(target_start_date=pd.Timestamp('2020-02-01'))

        # Context after target = NOT OK
        assert not seq.is_causal(target_start_date=pd.Timestamp('2019-12-01'))


class TestContextBatch:
    """Test ContextBatch type."""

    def test_context_batch_creation(self):
        """Create a batch of context sequences."""
        # Create 3 context sequences
        sequences = []
        for i in range(3):
            features = torch.randn(21, 8)
            entity_id = torch.tensor(i)
            seq = ContextSequence(
                features=features,
                entity_id=entity_id,
                start_date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i*30),
                end_date=pd.Timestamp('2020-01-25') + pd.Timedelta(days=i*30),
                method="cpd_segmented"
            )
            sequences.append(seq)

        batch = ContextBatch(sequences=sequences)

        assert batch.C == 3
        assert batch.max_length == 21

    def test_causality_verification(self):
        """Verify batch-level causality check."""
        sequences = []
        for i in range(3):
            features = torch.randn(21, 8)
            entity_id = torch.tensor(i)
            seq = ContextSequence(
                features=features,
                entity_id=entity_id,
                start_date=pd.Timestamp('2020-01-01'),
                end_date=pd.Timestamp('2020-01-25'),
                method="final_hidden_state"
            )
            sequences.append(seq)

        batch = ContextBatch(sequences=sequences)

        # All contexts before target = OK
        assert batch.verify_causality(target_start_date=pd.Timestamp('2020-03-01'))

        # Some contexts after target = NOT OK
        assert not batch.verify_causality(target_start_date=pd.Timestamp('2020-01-15'))

    def test_mixed_methods_rejected(self):
        """Reject batch with mixed construction methods."""
        seq1 = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(0),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="final_hidden_state"
        )
        seq2 = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(1),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="cpd_segmented"
        )

        with pytest.raises(ValueError, match="Mixed methods"):
            ContextBatch(sequences=[seq1, seq2])
