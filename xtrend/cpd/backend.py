"""
CPD backends: ruptures (default), GPflow ChangePoints, optional GPyTorch.

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
class CPDBackend(Protocol):
    """
    Protocol for change-point detection backends.

    Methods:
        segments(series: Sequence[float], **kwargs) -> List[(start_idx, end_idx)]
        name() -> str
    """
    def segments(self, series: "Sequence[float]", **kwargs: Any) -> "List[Tuple[int, int]]": ...
    def name(self) -> str: ...

class RupturesCPD:
    """
    Adapter for the 'ruptures' library (PELT/Binseg/Kernel).

    Notes:
        - Use this as the default CPD method for speed and robustness.
        - Configure model ('rbf', 'l2', etc.) and penalty via kwargs.
    """
    def __init__(self, model: str = "rbf", method: str = "pelt"):
        """Initialize backend selection (declaration only)."""
        pass

    def segments(self, series: "Sequence[float]", **kwargs: Any) -> "List[Tuple[int, int]]":
        """Return change-point segments for a 1D series (declaration only)."""
        pass

    def name(self) -> str:
        """Return human-readable backend name ('ruptures')."""
        pass

class GPflowChangePointsCPD:
    """
    Adapter for GPflow's ChangePoints kernel to emulate the paper's GP-based CPD.

    Notes:
        - Implement as an *offline* preprocessor; do not embed TF into the training graph.
        - Fit stationary vs. change-point GP models locally and compare marginal likelihoods.
    """
    def __init__(self, kernel: str = "matern32"):
        """Initialize kernel choice and hyperparameters (declaration only)."""
        pass

    def segments(self, series: "Sequence[float]", **kwargs: Any) -> "List[Tuple[int, int]]":
        """Return change-point segments using GPflow models (declaration only)."""
        pass

    def name(self) -> str:
        """Return 'gpflow-changepoints'."""
        pass

class GPyTorchChangePointCPD:
    """
    Optional adapter using GPyTorch with a custom ChangePoint kernel.

    Notes:
        - Useful if you prefer a pure-PyTorch stack; still avoid hand-rolled GP math by using GPyTorch primitives.
    """
    def __init__(self, kernel: str = "matern32"):
        """Initialize kernel choice (declaration only)."""
        pass

    def segments(self, series: "Sequence[float]", **kwargs: Any) -> "List[Tuple[int, int]]":
        """Return change-point segments using GPyTorch (declaration only)."""
        pass

    def name(self) -> str:
        """Return 'gpytorch-changepoint'."""
        pass

def cpd_segments(prices: "Sequence[float]",
                 lmax: int,
                 nu: float,
                 lbw: int = 21,
                 lmin: int = 5,
                 backend: Literal["ruptures", "gpflow", "gpytorch"] = "ruptures") -> "RegimeSegments":
    """
    Segment a price series into regimes using the selected backend.

    Args:
        prices: Close prices ordered oldest -> newest.
        lmax: Maximum regime length (e.g., 21 or 63).
        nu: Threshold/severity parameter (interpretation backend-specific).
        lbw: Local lookback window for local fits (if applicable).
        lmin: Minimum regime length.
        backend: Which CPD backend to use.

    Returns:
        RegimeSegments object with (start_idx, end_idx) segments and params.

    Notes:
        - The paper tunes the threshold so average segment length ≈ lmax/2 for lmax=63.
        - This function declares the interface only.
    """
    pass
