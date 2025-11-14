"""
Losses: Sharpe proxy + distributional losses (reuse libraries).

IMPORTANT: This is a *skeleton* that avoids reinventing the wheel.
It exposes interfaces that are thin wrappers/adapters around mature libraries.
All functions/classes contain only docstrings and 'pass' bodies.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Protocol, NamedTuple
def sharpe_loss(positions: "Any", returns: "Any", eps: float = 1e-8) -> "Any":
    """
    Differentiable proxy for negative Sharpe (maximize Sharpe).

    Notes:
        - Keep this minimal and local to training; for reporting use empyrical/pyfolio.
    """
    pass

def gaussian_nll(mu: "Any", sigma: "Any", y: "Any") -> "Any":
    """Negative log-likelihood for Normal(y | mu, sigma) (declaration only)."""
    pass

def quantile_loss(q: "Any", y: "Any", quantiles: "Sequence[float]") -> "Any":
    """
    Pinball/quantile loss.
    Notes:
        Prefer TorchMetrics' MeanPinballLoss or PyTorch Forecasting's QuantileLoss in real training.
    """
    pass

def joint_loss_gauss(positions: "Any", returns: "Any", mu: "Any", sigma: "Any", alpha: float = 1.0) -> "Any":
    """Sharpe loss + alpha * Gaussian NLL (declaration only)."""
    pass

def joint_loss_quantile(positions: "Any", returns: "Any", q: "Any", quantiles: "Sequence[float]", alpha: float = 5.0) -> "Any":
    """Sharpe loss + alpha * quantile error (declaration only)."""
    pass
