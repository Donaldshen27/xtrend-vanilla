"""
Training entrypoints using Lightning or Accelerate (no loops here).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def train_with_lightning(cfg: "Any") -> "str":
    """
    Train models with PyTorch Lightning Trainer using Hydra configs.

    Args:
        cfg: Composed configuration with model/data/optimizer/schedule sections.

    Returns:
        Path to checkpoint directory (string).

    Notes:
        - Early stopping and checkpointing should be configured via Trainer callbacks.
    """
    pass

def ensemble_predictions(checkpoints: "Sequence[str]", dataset: "Any") -> "Predictions":
    """
    Produce ensemble predictions by aggregating multiple checkpoints.

    Args:
        checkpoints: Paths to saved model checkpoints.
        dataset: Iterable of target/context batches.

    Returns:
        Predictions aligned to dataset dates.
    """
    pass
