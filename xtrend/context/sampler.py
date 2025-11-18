"""Context set sampling methods with strict causality enforcement."""
from typing import Dict, List, Optional
import torch
import pandas as pd
import numpy as np

from xtrend.context.types import ContextSequence, ContextBatch
from xtrend.cpd import RegimeSegments


def _normalize_target_timestamp(
    dates: pd.DatetimeIndex, target_date: pd.Timestamp
) -> pd.Timestamp:
    """Align target_date timezone to match dates index for lookups."""
    if dates.tz is None:
        # DatetimeIndex is tz-naive, so remove timezone info from target
        return target_date if target_date.tz is None else target_date.tz_localize(None)

    # DatetimeIndex is tz-aware
    if target_date.tz is None:
        return target_date.tz_localize(dates.tz)

    # Both are tz-aware; convert to index timezone if needed
    return (
        target_date
        if target_date.tz == dates.tz
        else target_date.tz_convert(dates.tz)
    )


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
    target_timestamp = _normalize_target_timestamp(dates, target_date)

    # Find target date index
    target_idx = dates.get_loc(target_timestamp)

    # Build candidate sequences (all possible sequences before target)
    candidates = []
    for symbol in available_symbols:
        entity_id = symbols.index(symbol)  # Map symbol to entity ID
        feature_tensor = features[symbol]
        available_len = feature_tensor.shape[0]

        if available_len < l_c:
            continue  # Not enough history for even one window

        # Sample all valid windows ending before target
        # Valid window: [start_idx, end_idx] where end_idx < target_idx
        symbol_max_end_idx = min(target_idx - 1, available_len - 1)
        min_start_idx = l_c - 1  # Need at least l_c days

        if symbol_max_end_idx < min_start_idx:
            continue  # Target date is before symbol accrued enough history

        for end_idx in range(min_start_idx, symbol_max_end_idx + 1):
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


def sample_time_equivalent(
    features: Dict[str, torch.Tensor],
    dates: pd.DatetimeIndex,
    symbols: List[str],
    target_date: pd.Timestamp,
    C: int,
    l_t: int,
    seed: Optional[int] = None,
    exclude_symbols: Optional[List[str]] = None
) -> ContextBatch:
    """Sample C sequences with same length as target (Time-Equivalent method).

    This method ensures temporal alignment: the k-th target timestep
    attends to the k-th timestep of each context sequence.

    Args:
        features: Dict mapping symbol -> feature tensor (T, input_dim)
        dates: DatetimeIndex of length T
        symbols: List of available symbols
        target_date: Target sequence date (for causality)
        C: Number of context sequences to sample
        l_t: Target sequence length (context length = l_t)
        seed: Random seed for reproducibility
        exclude_symbols: Symbols to exclude (for zero-shot)

    Returns:
        ContextBatch with C sequences of length l_t
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter symbols
    available_symbols = [s for s in symbols if exclude_symbols is None or s not in exclude_symbols]
    if len(available_symbols) == 0:
        raise ValueError("No symbols available after exclusions")

    # Normalize target_date to match dates timezone
    target_timestamp = _normalize_target_timestamp(dates, target_date)

    # Find target date index
    target_idx = dates.get_loc(target_timestamp)

    # Build candidate sequences
    candidates = []
    for symbol in available_symbols:
        entity_id = symbols.index(symbol)
        feature_tensor = features[symbol]
        available_len = feature_tensor.shape[0]

        if available_len < l_t:
            continue

        # Valid windows of length l_t ending before target
        symbol_max_end_idx = min(target_idx - 1, available_len - 1)
        min_start_idx = l_t - 1

        if symbol_max_end_idx < min_start_idx:
            continue

        for end_idx in range(min_start_idx, symbol_max_end_idx + 1):
            start_idx = end_idx - l_t + 1
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

        # Extract feature window (exactly l_t days)
        feature_window = features[cand['symbol']][cand['start_idx']:cand['end_idx']+1]

        seq = ContextSequence(
            features=feature_window,
            entity_id=torch.tensor(cand['entity_id']),
            start_date=dates[cand['start_idx']],
            end_date=dates[cand['end_idx']],
            method="time_equivalent"
        )
        sequences.append(seq)

    return ContextBatch(sequences=sequences)


def sample_cpd_segmented(
    features: Dict[str, torch.Tensor],
    dates: pd.DatetimeIndex,
    symbols: List[str],
    regimes: Dict[str, RegimeSegments],
    target_date: pd.Timestamp,
    C: int,
    max_length: int,
    seed: Optional[int] = None,
    exclude_symbols: Optional[List[str]] = None
) -> ContextBatch:
    """Sample C regime sequences from CPD segmentation (Primary method).

    This method uses GP-CPD to identify market regimes and samples
    complete regime sequences as context. This is the primary method
    from the X-Trend paper.

    Args:
        features: Dict mapping symbol -> feature tensor (T, input_dim)
        dates: DatetimeIndex of length T
        symbols: List of available symbols
        regimes: Dict mapping symbol -> RegimeSegments from CPD
        target_date: Target sequence date (for causality)
        C: Number of context sequences to sample
        max_length: Maximum regime length (truncate if longer)
        seed: Random seed for reproducibility
        exclude_symbols: Symbols to exclude (for zero-shot)

    Returns:
        ContextBatch with C regime sequences
    """
    if seed is not None:
        np.random.seed(seed)

    # Filter symbols
    available_symbols = [s for s in symbols if exclude_symbols is None or s not in exclude_symbols]
    if len(available_symbols) == 0:
        raise ValueError("No symbols available after exclusions")

    # Normalize target_date to match dates timezone
    target_timestamp = _normalize_target_timestamp(dates, target_date)

    # Find target date index
    target_idx = dates.get_loc(target_timestamp)

    # Build candidate regimes (all regimes ending before target)
    candidates = []
    for symbol in available_symbols:
        if symbol not in regimes:
            continue

        entity_id = symbols.index(symbol)
        regime_segs = regimes[symbol]

        for regime in regime_segs.segments:
            # Check causality: regime must end before target
            if regime.end_date >= target_timestamp:
                continue

            # Regime length
            regime_length = regime.end_idx - regime.start_idx + 1

            # Truncate if exceeds max_length
            # CRITICAL (Codex feedback): Preserve from CPD boundary START, not end
            # This preserves early regime characteristics that define the regime
            if regime_length > max_length:
                start_idx = regime.start_idx  # Keep from boundary
                end_idx = regime.start_idx + max_length - 1  # Truncate at max
            else:
                start_idx = regime.start_idx
                end_idx = regime.end_idx

            candidates.append({
                'symbol': symbol,
                'entity_id': entity_id,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    if len(candidates) < C:
        raise ValueError(f"Not enough causal regime candidates ({len(candidates)}) for C={C}")

    # Random sample C regimes
    selected = np.random.choice(len(candidates), size=C, replace=False)

    sequences = []
    for idx in selected:
        cand = candidates[idx]

        # Extract feature window for regime
        feature_window = features[cand['symbol']][cand['start_idx']:cand['end_idx']+1]

        seq = ContextSequence(
            features=feature_window,
            entity_id=torch.tensor(cand['entity_id']),
            start_date=dates[cand['start_idx']],
            end_date=dates[cand['end_idx']],
            method="cpd_segmented"
        )
        sequences.append(seq)

    return ContextBatch(sequences=sequences)
