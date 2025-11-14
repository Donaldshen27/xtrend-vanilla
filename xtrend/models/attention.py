"""
Cross-attention wrapper delegating to selected backend.

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def cross_attention(query: "Any",
                    key: "Any",
                    value: "Any",
                    key_padding_mask: Optional["Any"] = None,
                    backend: Literal["sdpa","torch_mha","flash","xformers"] = "sdpa") -> "Tuple[Any, Any]":
    """
    Apply cross-attention from query (target) to key/value (contexts).

    Args:
        query: (B, L_t, H) encoded target states.
        key: (B or C, L_c, H) context keys (batched to match query strategy).
        value: Values aligned with keys.
        key_padding_mask: Optional mask for variable-length contexts.
        backend: Attention backend to use (SDPA by default).

    Returns:
        attn_output: (B, L_t, H) contextualized target representation.
        attn_weights: Raw weights for interpretability.

    Notes:
        - Internally, this should call into third_party.adapters.attention.make_attention(...).
        - This is a declaration only.
    """
    pass
