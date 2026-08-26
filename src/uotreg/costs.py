"""Ground costs c(x, y) for the transport problem (the "phi" knob).

The paper uses the quadratic cost.  We expose a small registry so later
experiments can swap the ground cost without touching the trainer.  All costs
return a *per-sample* tensor (mean over the feature dimension), matching the
``mse_loss`` scaling used in the original code so that previously tuned ``tau``
values transfer unchanged.

A cost takes two tensors of shape ``(B, NUM, d)`` (or ``(B, d)``) and returns
``(B, NUM)`` (or ``(B,)``): the transport cost of moving each source point to
its mapped target.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch

CostFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def squared_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared difference over the last (feature) axis: ``||x-y||^2 / d``.

    This matches ``F.mse_loss`` used in the original UOTReg code.
    """
    return ((x - y) ** 2).mean(dim=-1)


def euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Root-mean-squared (scaled L2) distance over the feature axis."""
    return torch.sqrt(((x - y) ** 2).mean(dim=-1) + 1e-12)


_COSTS: Dict[str, CostFn] = {
    "sqeuclidean": squared_euclidean,
    "l2": euclidean,
}


def get_cost(name_or_fn) -> CostFn:
    """Resolve a cost by name (``"sqeuclidean"``, ``"l2"``) or pass a callable."""
    if callable(name_or_fn):
        return name_or_fn
    key = str(name_or_fn).lower()
    if key not in _COSTS:
        raise ValueError(f"Unknown cost '{name_or_fn}'. Available: {list(_COSTS)} or a callable.")
    return _COSTS[key]


def register_cost(name: str, fn: CostFn) -> None:
    _COSTS[name.lower()] = fn
