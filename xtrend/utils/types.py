"""
Typed containers used across modules.

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
class FeatureMatrix(NamedTuple):
    """(dates, X, feature_names, metadata). See features.builder for construction."""
    dates: Sequence[Any]
    X: Any
    feature_names: Sequence[str]
    metadata: Mapping[str, Any]

class RegimeSegments(NamedTuple):
    """(dates, segments, params). segments: list of (start_idx, end_idx)."""
    dates: Sequence[Any]
    segments: Sequence[Tuple[int, int]]
    params: Mapping[str, Any]

class ContextBatch(NamedTuple):
    """(keys, values, key_padding_mask, metas) for cross-attention."""
    keys: Any
    values: Any
    key_padding_mask: Optional[Any]
    metas: Sequence[Mapping[str, Any]]

class TargetBatch(NamedTuple):
    """(X, y_next, sigma_t, meta) for model forward."""
    X: Any
    y_next: Optional[Any]
    sigma_t: Optional[Any]
    meta: Sequence[Mapping[str, Any]]

class ModelOutputs(NamedTuple):
    """Unified outputs: (mu, sigma) or quantiles, plus position and aux diagnostics."""
    mu: Optional[Any]
    sigma: Optional[Any]
    quantiles: Optional[Any]
    position: Any
    aux: Optional[Mapping[str, Any]]

class Predictions(NamedTuple):
    """(dates, positions, mu?, sigma?, quantiles?, meta) aligned to daily index."""
    dates: Sequence[Any]
    positions: Any
    mu: Optional[Any]
    sigma: Optional[Any]
    quantiles: Optional[Any]
    meta: Mapping[str, Any]
