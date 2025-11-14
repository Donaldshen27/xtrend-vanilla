"""
Hydra/OmegaConf config helpers (declarations only).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def compose_config(config_name: str, overrides: Sequence[str] | None = None) -> "Any":
    """
    Compose a Hydra configuration from conf/ directory.

    Args:
        config_name: Name of the root config (e.g., "experiment").
        overrides: Optional list of CLI-style overrides, e.g., ["model.head=quantile", "cpd.backend=ruptures"].

    Returns:
        A frozen, hierarchical config (OmegaConf-like).

    Notes:
        - Intended to wrap hydra.initialize/config_path + hydra.compose.
        - This skeleton intentionally contains no Hydra logic.
    """
    pass

def save_config(cfg: "Any", path: str) -> None:
    """
    Persist a configuration object to YAML for reproducibility.

    Args:
        cfg: Hydra/OmegaConf-like object.
        path: Destination file path.

    Notes:
        This function is a declaration only (no I/O here).
    """
    pass
