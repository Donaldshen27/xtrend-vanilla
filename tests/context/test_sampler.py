"""Tests for context sampling methods."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.context.sampler import sample_final_hidden_state
from xtrend.context.types import ContextBatch


class TestFinalHiddenStateMethod:
    """Test Final Hidden State sampling method."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature panel."""
        # 100 days of data for 5 assets
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')
        features = {}
        np.random.seed(42)
        for asset_id in range(5):
            # Shape: (100, 8) features
            features[f"ASSET{asset_id}"] = torch.randn(len(dates), 8)

        return {
            'features': features,
            'dates': dates,
            'symbols': [f"ASSET{i}" for i in range(5)]
        }

    def test_sample_final_hidden_state_basic(self, sample_features):
        """Sample C context sequences with fixed length."""
        target_date = pd.Timestamp('2020-04-01')

        batch = sample_final_hidden_state(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=10,  # Sample 10 context sequences
            l_c=21,  # Each 21 days long
            seed=42
        )

        assert isinstance(batch, ContextBatch)
        assert batch.C == 10
        assert batch.max_length == 21

        # Verify causality
        assert batch.verify_causality(target_date)

        # Verify all sequences have correct shape
        for seq in batch.sequences:
            assert seq.features.shape == (21, 8)
            assert seq.method == "final_hidden_state"

    def test_respects_causality(self, sample_features):
        """All sampled contexts must end before target."""
        # Use tz-naive to match the fixture dates
        target_date = pd.Timestamp('2020-02-01')

        batch = sample_final_hidden_state(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_c=21,
            seed=42
        )

        # Every context sequence must end before target
        for seq in batch.sequences:
            assert seq.end_date < target_date

    def test_timezone_naive_compatibility(self, sample_features):
        """Test that tz-naive dates work correctly and can be compared."""
        # Use tz-naive dates (most common case for users)
        dates = sample_features['dates']  # Already tz-naive from fixture
        target_date = dates[50]  # tz-naive

        batch = sample_final_hidden_state(
            features=sample_features['features'],
            dates=dates,
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_c=10,
            seed=42
        )

        # Verify all returned dates are tz-naive
        for seq in batch.sequences:
            assert seq.start_date.tz is None, "start_date should be tz-naive"
            assert seq.end_date.tz is None, "end_date should be tz-naive"

            # Critical: This should NOT raise TypeError
            gap = target_date - seq.end_date
            assert gap > pd.Timedelta(days=0), "Context must end before target"

    def test_exclude_symbols_functionality(self, sample_features):
        """Test that exclude_symbols filters out specified symbols."""
        target_date = pd.Timestamp('2020-04-01')
        excluded = ['ASSET0', 'ASSET1']

        batch = sample_final_hidden_state(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_c=10,
            exclude_symbols=excluded,
            seed=42
        )

        # Verify no sequences use excluded symbols
        symbols_list = sample_features['symbols']
        for seq in batch.sequences:
            entity_id = seq.entity_id.item()
            symbol = symbols_list[entity_id]
            assert symbol not in excluded, f"Excluded symbol {symbol} found in batch"

        # All sequences should come from ASSET2, ASSET3, ASSET4 only
        used_entity_ids = {seq.entity_id.item() for seq in batch.sequences}
        expected_entity_ids = {2, 3, 4}  # ASSET2, ASSET3, ASSET4
        assert used_entity_ids.issubset(expected_entity_ids)

    def test_seed_reproducibility(self, sample_features):
        """Test that same seed produces identical samples."""
        target_date = pd.Timestamp('2020-04-01')

        # Sample twice with same seed
        batch1 = sample_final_hidden_state(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_c=10,
            seed=123
        )

        batch2 = sample_final_hidden_state(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_c=10,
            seed=123
        )

        # Should produce identical sequences
        assert batch1.C == batch2.C
        for seq1, seq2 in zip(batch1.sequences, batch2.sequences):
            assert seq1.entity_id == seq2.entity_id
            assert seq1.start_date == seq2.start_date
            assert seq1.end_date == seq2.end_date
            assert torch.allclose(seq1.features, seq2.features)

    def test_insufficient_candidates_error(self, sample_features):
        """Test error when not enough candidates available."""
        # Use very early target date with long context length
        # This leaves few valid candidates
        target_date = pd.Timestamp('2020-01-10')

        with pytest.raises(ValueError, match="Not enough causal candidates"):
            sample_final_hidden_state(
                features=sample_features['features'],
                dates=sample_features['dates'],
                symbols=sample_features['symbols'],
                target_date=target_date,
                C=100,  # Request more than possible
                l_c=10,
                seed=42
            )
