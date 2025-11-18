"""Context set sampling methods with strict causality enforcement."""
from typing import Dict, List, Optional
import torch
import pandas as pd
import numpy as np

from xtrend.context.types import ContextSequence, ContextBatch


def sample_final_hidden_state(
    features: Dict[str, torch.Tensor],
    dates: pd.DatetimeIndex,
    symbols: List[str],
    target_date: pd.Timestamp,
    C: int,
    l_c: int,
    seed: Optional[int] = None,
    exclude_symbols: Optional[List[str]] = None
) -> ContextBatch:
    """Sample C random sequences of fixed length l_c (Final Hidden State method).

    This method samples random historical sequences and uses their final
    hidden state as context for cross-attention.

    Args:
        features: Dict mapping symbol -> feature tensor (T, input_dim)
        dates: DatetimeIndex of length T
        symbols: List of available symbols
        target_date: Target sequence date (for causality)
        C: Number of context sequences to sample
        l_c: Fixed context length (days)
        seed: Random seed for reproducibility
        exclude_symbols: Symbols to exclude (for zero-shot)

    Returns:
        ContextBatch with C sequences of length l_c
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter symbols
    available_symbols = [s for s in symbols if exclude_symbols is None or s not in exclude_symbols]
    if len(available_symbols) == 0:
        raise ValueError("No symbols available after exclusions")

    # Normalize target_date to match dates timezone
    if dates.tz is None and target_date.tz is not None:
        # dates is tz-naive, convert target_date to tz-naive
        target_date_normalized = target_date.tz_localize(None)
    elif dates.tz is not None and target_date.tz is None:
        # dates is tz-aware, convert target_date to tz-aware
        target_date_normalized = target_date.tz_localize(dates.tz)
    else:
        target_date_normalized = target_date

    # Find target date index
    target_idx = dates.get_loc(target_date_normalized)

    # Build candidate sequences (all possible sequences before target)
    candidates = []
    for symbol in available_symbols:
        entity_id = symbols.index(symbol)  # Map symbol to entity ID
        feature_tensor = features[symbol]

        # Sample all valid windows ending before target
        # Valid window: [start_idx, end_idx] where end_idx < target_idx
        max_end_idx = target_idx - 1
        min_start_idx = l_c - 1  # Need at least l_c days

        for end_idx in range(min_start_idx, max_end_idx + 1):
            start_idx = end_idx - l_c + 1
            candidates.append({
                'symbol': symbol,
                'entity_id': entity_id,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    if len(candidates) < C:
        raise ValueError(f"Not enough causal candidates ({len(candidates)}) for C={C}")

    # Random sample C sequences
    selected = np.random.choice(len(candidates), size=C, replace=False)

    sequences = []
    for idx in selected:
        cand = candidates[idx]

        # Extract feature window
        feature_window = features[cand['symbol']][cand['start_idx']:cand['end_idx']+1]

        seq = ContextSequence(
            features=feature_window,
            entity_id=torch.tensor(cand['entity_id']),
            start_date=dates[cand['start_idx']],
            end_date=dates[cand['end_idx']],
            method="final_hidden_state"
        )
        sequences.append(seq)

    return ContextBatch(sequences=sequences)
