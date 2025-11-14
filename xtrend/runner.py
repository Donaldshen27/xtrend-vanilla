"""
Hydra-driven experiment runner (no orchestration here).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def main(config_name: str = "experiment") -> None:
    """
    Orchestrate an experiment using Hydra configs.

    Workflow:
        1) Compose config from conf/ (Hydra).
        2) Load data via DataSource adapter; align sessions.
        3) Build features; run CPD with chosen backend (ruptures/gpflow).
        4) Sample contexts; train models with Lightning; ensemble predictions.
        5) Evaluate using vectorbt; report with empyrical/quantstats.

    Notes:
        This is a declaration only and performs no work.
    """
    pass
