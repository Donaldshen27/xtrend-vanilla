"""Integration tests for Phase 4: Context Set Construction."""
import pytest
import torch
import pandas as pd
import numpy as np

from xtrend.context import (
    sample_final_hidden_state,
    sample_time_equivalent,
    sample_cpd_segmented,
)
from xtrend.cpd import CPDConfig, GPCPDSegmenter
from xtrend.data.returns_vol import simple_returns


class TestPhase4Integration:
    """Integration tests verifying Phase 4 completion criteria."""

    @pytest.fixture
    def realistic_data(self):
        """Create realistic multi-asset data."""
        # 2 years of daily data
        dates = pd.date_range('2019-01-01', '2020-12-31', freq='D')
        np.random.seed(42)

        # Create 10 assets with realistic price movements
        features = {}
        prices = {}
        symbols = [f"ASSET{i}" for i in range(10)]

        for symbol in symbols:
            # Simulate prices with drift and volatility
            price_series = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
            prices[symbol] = pd.Series(price_series, index=dates)

            # Calculate features (returns)
            returns = simple_returns(prices[symbol])
            # Simple feature: just returns (repeat 8 times for 8 features)
            features[symbol] = torch.tensor(
                np.tile(returns.values[:, None], (1, 8)),
                dtype=torch.float32
            )

        return {
            'features': features,
            'prices': prices,
            'dates': dates,
            'symbols': symbols
        }

    def test_final_hidden_state_full_pipeline(self, realistic_data):
        """Complete pipeline: Final Hidden State method."""
        target_date = pd.Timestamp('2020-03-15')  # COVID period

        batch = sample_final_hidden_state(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=20,
            l_c=21,
            seed=42
        )

        # Phase 4 completion criteria
        assert batch.C == 20
        assert batch.max_length == 21
        assert batch.verify_causality(target_date)

        # Asset diversity
        entity_ids = {seq.entity_id.item() for seq in batch.sequences}
        assert len(entity_ids) >= 5  # Multiple assets

        print(f"✓ Final Hidden State: C={batch.C}, max_length={batch.max_length}")
        print(f"✓ Asset diversity: {len(entity_ids)} unique entities")

    def test_time_equivalent_full_pipeline(self, realistic_data):
        """Complete pipeline: Time-Equivalent method."""
        target_date = pd.Timestamp('2020-06-01')
        l_t = 63  # Target length
        max_contexts = len(realistic_data['symbols'])
        C = min(20, max_contexts)

        batch = sample_time_equivalent(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=C,
            l_t=l_t,
            seed=42
        )

        # All sequences same length (time-aligned)
        assert all(seq.length == l_t for seq in batch.sequences)
        assert batch.verify_causality(target_date)
        assert batch.C == C

        print(f"✓ Time-Equivalent: all sequences length={l_t}")

    def test_cpd_segmented_full_pipeline(self, realistic_data):
        """Complete pipeline: CPD-Segmented method (primary)."""
        # First run CPD on subset of assets (for performance)
        # Use only first 3 assets to speed up test
        config = CPDConfig(
            lookback=21,
            threshold=0.9,
            min_length=5,
            max_length=21
        )
        segmenter = GPCPDSegmenter(config)

        regimes = {}
        # Only segment first 3 assets to reduce test time
        for symbol in list(realistic_data['prices'].keys())[:3]:
            price_series = realistic_data['prices'][symbol]
            regimes[symbol] = segmenter.fit_segment(price_series)

        # Sample context set
        target_date = pd.Timestamp('2020-09-01')

        batch = sample_cpd_segmented(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            regimes=regimes,
            target_date=target_date,
            C=10,  # Reduced from 20 since we only have 3 assets
            max_length=21,
            seed=42
        )

        # Phase 4 completion criteria
        assert batch.C == 10
        assert batch.verify_causality(target_date)

        # All sequences respect max_length
        assert all(seq.length <= 21 for seq in batch.sequences)

        # Regime diversity
        entity_ids = {seq.entity_id.item() for seq in batch.sequences}
        assert len(entity_ids) >= 2  # At least 2 assets (reduced from 3)

        print(f"✓ CPD-Segmented: C={batch.C}, regime diversity={len(entity_ids)}")
        print(f"✓ Length distribution: min={min(s.length for s in batch.sequences)}, "
              f"max={max(s.length for s in batch.sequences)}")

    def test_zero_shot_exclusion(self, realistic_data):
        """Zero-shot: exclude target asset from context."""
        target_symbol = "ASSET0"
        target_date = pd.Timestamp('2020-08-01')

        batch = sample_final_hidden_state(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=15,
            l_c=21,
            seed=42,
            exclude_symbols=[target_symbol]  # Zero-shot: exclude target
        )

        # No context sequence should be from target symbol
        target_entity_id = realistic_data['symbols'].index(target_symbol)
        context_entity_ids = [seq.entity_id.item() for seq in batch.sequences]

        assert target_entity_id not in context_entity_ids
        print(f"✓ Zero-shot: target asset {target_symbol} excluded from context")

    def test_causality_enforcement(self, realistic_data):
        """Strict causality: all contexts before target."""
        target_date = pd.Timestamp('2020-01-15')  # Early date

        batch = sample_final_hidden_state(
            features=realistic_data['features'],
            dates=realistic_data['dates'],
            symbols=realistic_data['symbols'],
            target_date=target_date,
            C=10,
            l_c=10,  # Short sequences
            seed=42
        )

        # Every context must end before target
        for seq in batch.sequences:
            assert seq.end_date < target_date

        print(f"✓ Causality enforced: all {batch.C} contexts before target")
