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
        target_date = pd.Timestamp('2020-02-01', tz='UTC')

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
