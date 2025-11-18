"""Tests for context sampling methods."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.context.sampler import sample_final_hidden_state, sample_time_equivalent, sample_cpd_segmented
from xtrend.context.types import ContextBatch
from xtrend.cpd import RegimeSegment, RegimeSegments, CPDConfig


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


class TestTimeEquivalentMethod:
    """Test Time-Equivalent sampling method."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature panel."""
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')
        features = {}
        np.random.seed(42)
        for asset_id in range(5):
            features[f"ASSET{asset_id}"] = torch.randn(len(dates), 8)

        return {
            'features': features,
            'dates': dates,
            'symbols': [f"ASSET{i}" for i in range(5)]
        }

    def test_sample_time_equivalent_matches_target_length(self, sample_features):
        """Time-equivalent sequences match target length."""
        target_date = pd.Timestamp('2020-04-01')
        l_t = 63  # Target sequence length

        batch = sample_time_equivalent(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_t=l_t,
            seed=42
        )

        assert batch.C == 5
        assert batch.max_length == l_t  # Same length as target

        # All sequences should be l_t days
        for seq in batch.sequences:
            assert seq.length == l_t
            assert seq.method == "time_equivalent"

    def test_time_alignment(self, sample_features):
        """k-th timestep aligned across contexts."""
        target_date = pd.Timestamp('2020-03-01')
        l_t = 21

        batch = sample_time_equivalent(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=5,
            l_t=l_t,
            seed=42
        )

        # All sequences must be same length (time-aligned)
        lengths = [seq.length for seq in batch.sequences]
        assert len(set(lengths)) == 1  # All same length
        assert lengths[0] == l_t

        # Verify identical calendar span as the target window
        dates = sample_features['dates']
        target_idx = dates.get_loc(target_date)
        expected_start = dates[target_idx - l_t]
        expected_end = dates[target_idx - 1]

        for seq in batch.sequences:
            assert seq.start_date == expected_start
            assert seq.end_date == expected_end

    def test_raises_when_c_exceeds_aligned_symbols(self, sample_features):
        """Requesting more contexts than aligned symbols should fail."""
        target_date = pd.Timestamp('2020-04-01')

        with pytest.raises(ValueError, match="Not enough causal candidates"):
            sample_time_equivalent(
                features=sample_features['features'],
                dates=sample_features['dates'],
                symbols=sample_features['symbols'],
                target_date=target_date,
                C=6,
                l_t=63,
                seed=1
            )

    def test_excludes_symbols_without_aligned_window(self, sample_features):
        """Symbols missing the target window must be ignored."""
        target_date = pd.Timestamp('2020-04-01')
        l_t = 30

        # Remove the most recent 11 days from ASSET0 so it lacks the aligned window
        sample_features['features']['ASSET0'] = sample_features['features']['ASSET0'][:-11]

        batch = sample_time_equivalent(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            target_date=target_date,
            C=4,
            l_t=l_t,
            seed=0
        )

        symbols = sample_features['symbols']
        for seq in batch.sequences:
            assert symbols[seq.entity_id.item()] != 'ASSET0'

    def test_raises_when_target_window_not_available(self, sample_features):
        """Raise when the target window is longer than available history."""
        target_date = pd.Timestamp('2020-01-15')

        with pytest.raises(ValueError, match="sufficient history"):
            sample_time_equivalent(
                features=sample_features['features'],
                dates=sample_features['dates'],
                symbols=sample_features['symbols'],
                target_date=target_date,
                C=2,
                l_t=40,
                seed=1
            )


class TestCPDSegmentedMethod:
    """Test CPD-Segmented sampling method (primary method from paper)."""

    @pytest.fixture
    def sample_regimes(self):
        """Create sample regime segmentations."""
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')

        # Create regimes for 3 assets
        regimes = {}
        config = CPDConfig(min_length=5, max_length=21)

        for asset_id in range(3):
            symbol = f"ASSET{asset_id}"
            segments = []

            # Create 5 regimes per asset
            start_idx = 0
            for i in range(5):
                length = np.random.randint(5, 22)
                end_idx = min(start_idx + length - 1, len(dates) - 1)

                seg = RegimeSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    severity=0.95,
                    start_date=dates[start_idx],
                    end_date=dates[end_idx]
                )
                segments.append(seg)

                start_idx = end_idx + 1
                if start_idx >= len(dates):
                    break

            regimes[symbol] = RegimeSegments(segments=segments, config=config)

        return regimes

    @pytest.fixture
    def sample_features(self):
        """Create sample feature panel."""
        dates = pd.date_range('2020-01-01', '2020-04-10', freq='D')
        features = {}
        np.random.seed(42)
        for asset_id in range(3):
            features[f"ASSET{asset_id}"] = torch.randn(len(dates), 8)

        return {
            'features': features,
            'dates': dates,
            'symbols': [f"ASSET{i}" for i in range(3)]
        }

    def test_sample_cpd_segmented_basic(self, sample_features, sample_regimes):
        """Sample C regime sequences from CPD segmentation."""
        target_date = pd.Timestamp('2020-04-01')

        batch = sample_cpd_segmented(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            regimes=sample_regimes,
            target_date=target_date,
            C=10,
            max_length=21,
            seed=42
        )

        assert isinstance(batch, ContextBatch)
        assert batch.C == 10
        assert batch.max_length <= 21

        # Verify causality
        assert batch.verify_causality(target_date)

        # Verify method
        for seq in batch.sequences:
            assert seq.method == "cpd_segmented"
            assert seq.length <= 21  # Respect max_length

    def test_respects_max_length(self, sample_features, sample_regimes):
        """Long regimes truncated to max_length."""
        target_date = pd.Timestamp('2020-03-15')

        batch = sample_cpd_segmented(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            regimes=sample_regimes,
            target_date=target_date,
            C=5,
            max_length=10,  # Short max
            seed=42
        )

        # No sequence should exceed max_length
        for seq in batch.sequences:
            assert seq.length <= 10

    def test_regime_diversity(self, sample_features, sample_regimes):
        """Context set includes regimes from multiple assets."""
        target_date = pd.Timestamp('2020-04-01')

        batch = sample_cpd_segmented(
            features=sample_features['features'],
            dates=sample_features['dates'],
            symbols=sample_features['symbols'],
            regimes=sample_regimes,
            target_date=target_date,
            C=10,
            max_length=21,
            seed=42
        )

        # Should have regimes from multiple entities
        entity_ids = {seq.entity_id.item() for seq in batch.sequences}
        assert len(entity_ids) > 1  # Multiple assets represented
