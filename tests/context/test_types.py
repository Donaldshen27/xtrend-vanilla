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

    def test_custom_padding_mask(self):
        """Test custom padding mask support."""
        features = torch.randn(21, 8)
        # Custom mask: first 15 valid, last 6 padding
        custom_mask = torch.cat([
            torch.ones(15, dtype=torch.bool),
            torch.zeros(6, dtype=torch.bool)
        ])

        seq = ContextSequence(
            features=features,
            entity_id=torch.tensor(5),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="cpd_segmented",
            padding_mask=custom_mask
        )

        assert seq.padding_mask is not None
        assert seq.padding_mask.shape == (21,)
        assert seq.padding_mask[:15].all()
        assert not seq.padding_mask[15:].any()

    def test_timezone_conversion(self):
        """Test automatic timezone conversion to UTC."""
        # Create sequence with naive (no timezone) timestamps
        seq = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(5),
            start_date=pd.Timestamp('2020-01-01'),  # naive
            end_date=pd.Timestamp('2020-01-25'),    # naive
            method="final_hidden_state"
        )

        # Verify timestamps are now timezone-aware (UTC)
        assert seq.start_date.tz is not None
        assert seq.end_date.tz is not None
        assert str(seq.start_date.tz) == 'UTC'
        assert str(seq.end_date.tz) == 'UTC'

    def test_buffer_days_parameter(self):
        """Test buffer_days parameter in causality check."""
        seq = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(5),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),  # ends on Jan 25
            method="final_hidden_state"
        )

        # With 1-day buffer (default): context must end before Jan 26
        # Target on Jan 26 is OK (25 + 1 day buffer <= 26)
        assert seq.is_causal(target_start_date=pd.Timestamp('2020-01-26'), buffer_days=1)

        # Target on Jan 25 is NOT OK (need buffer)
        assert not seq.is_causal(target_start_date=pd.Timestamp('2020-01-25'), buffer_days=1)

        # With 5-day buffer: context must end before Jan 30
        # Target on Jan 30 is OK (25 + 5 days = 30)
        assert seq.is_causal(target_start_date=pd.Timestamp('2020-01-30'), buffer_days=5)

        # Target on Jan 29 is NOT OK
        assert not seq.is_causal(target_start_date=pd.Timestamp('2020-01-29'), buffer_days=5)

    def test_to_padded_tensor(self):
        """Test conversion to padded tensor for attention."""
        # Create sequences of different lengths
        seq1 = ContextSequence(
            features=torch.randn(10, 8),
            entity_id=torch.tensor(0),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-10'),
            method="final_hidden_state"
        )
        seq2 = ContextSequence(
            features=torch.randn(15, 8),
            entity_id=torch.tensor(1),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-15'),
            method="final_hidden_state"
        )
        seq3 = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(2),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-21'),
            method="final_hidden_state"
        )

        batch = ContextBatch(sequences=[seq1, seq2, seq3])
        features, mask = batch.to_padded_tensor()

        # Check shapes
        assert features.shape == (3, 21, 8)  # (C=3, max_len=21, input_dim=8)
        assert mask.shape == (3, 21)

        # Check padding mask
        assert mask[0, :10].all()   # seq1: first 10 valid
        assert not mask[0, 10:].any()  # seq1: last 11 padding
        assert mask[1, :15].all()   # seq2: first 15 valid
        assert not mask[1, 15:].any()  # seq2: last 6 padding
        assert mask[2, :21].all()   # seq3: all 21 valid

        # Check feature values preserved
        assert torch.allclose(features[0, :10], seq1.features)
        assert torch.allclose(features[1, :15], seq2.features)
        assert torch.allclose(features[2, :21], seq3.features)

        # Check padding is zeros
        assert torch.allclose(features[0, 10:], torch.zeros(11, 8))

    def test_mixed_input_dims_rejected(self):
        """Reject batch with mixed feature dimensions."""
        seq1 = ContextSequence(
            features=torch.randn(21, 8),  # input_dim=8
            entity_id=torch.tensor(0),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="final_hidden_state"
        )
        seq2 = ContextSequence(
            features=torch.randn(21, 10),  # input_dim=10 (different!)
            entity_id=torch.tensor(1),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="final_hidden_state"
        )

        with pytest.raises(ValueError, match="Mixed feature dimensions"):
            ContextBatch(sequences=[seq1, seq2])

    def test_empty_batch_rejected(self):
        """Reject empty context batch."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ContextBatch(sequences=[])

    def test_padding_mask_shape_validation(self):
        """Test validation of padding mask shape."""
        features = torch.randn(21, 8)
        invalid_mask = torch.ones(15, dtype=torch.bool)  # Wrong length!

        with pytest.raises(ValueError, match="Padding mask length"):
            ContextSequence(
                features=features,
                entity_id=torch.tensor(5),
                start_date=pd.Timestamp('2020-01-01'),
                end_date=pd.Timestamp('2020-01-25'),
                method="final_hidden_state",
                padding_mask=invalid_mask
            )

    def test_input_dim_property(self):
        """Test input_dim property for ContextSequence and ContextBatch."""
        # Test ContextSequence.input_dim
        seq = ContextSequence(
            features=torch.randn(21, 8),
            entity_id=torch.tensor(5),
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-01-25'),
            method="final_hidden_state"
        )
        assert seq.input_dim == 8

        # Test ContextBatch.input_dim
        sequences = [
            ContextSequence(
                features=torch.randn(21, 8),
                entity_id=torch.tensor(i),
                start_date=pd.Timestamp('2020-01-01'),
                end_date=pd.Timestamp('2020-01-25'),
                method="final_hidden_state"
            )
            for i in range(3)
        ]
        batch = ContextBatch(sequences=sequences)
        assert batch.input_dim == 8

    def test_batch_verify_causality_with_buffer(self):
        """Test batch-level causality verification with custom buffer."""
        sequences = []
        for i in range(3):
            seq = ContextSequence(
                features=torch.randn(21, 8),
                entity_id=torch.tensor(i),
                start_date=pd.Timestamp('2020-01-01'),
                end_date=pd.Timestamp('2020-01-25'),  # All end on Jan 25
                method="final_hidden_state"
            )
            sequences.append(seq)

        batch = ContextBatch(sequences=sequences)

        # With 1-day buffer, target on Jan 26 is OK
        assert batch.verify_causality(
            target_start_date=pd.Timestamp('2020-01-26'),
            buffer_days=1
        )

        # With 5-day buffer, target must be on Jan 30 or later
        assert batch.verify_causality(
            target_start_date=pd.Timestamp('2020-01-30'),
            buffer_days=5
        )
        assert not batch.verify_causality(
            target_start_date=pd.Timestamp('2020-01-29'),
            buffer_days=5
        )
