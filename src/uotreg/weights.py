r"""Local-Frechet regression weights with the revised nonnegative refinement.

Local-linear Frechet weights ``a_i(t,h)`` reproduce constants and linear trends
but can be **negative** near the boundary of the observed time range.  For a
Euclidean response that is harmless, but for an (unbalanced) Wasserstein
barycenter negative weights break convexity and the monotone-descent guarantee
(reviewer / editor major comment).  The revision therefore uses
**positive-part normalized** weights

    w_i(t,h) = [a_i(t,h)]_+ / sum_j [a_j(t,h)]_+ .

This module exposes both behaviours through one function:

  * ``scheme="positive"`` (default) -- continuous positive-part weights; the
    estimand varies continuously in t and equals the local-linear weights
    wherever those are already nonnegative.
  * ``scheme="raw"``               -- the old signed weights (for diagnostics).
  * ``threshold=eta``              -- optional *hard pruning*: drop references
    with weight < eta and renormalize, keeping the few nearest snapshots.  This
    is the behaviour the notebooks already relied on (weight-0 references never
    entered training); here it is an explicit, documented option, separate from
    the continuous estimand.

The returned :class:`WeightResult` reports the diagnostics the revision asks for
(total negative raw mass, dropped mass, active set, effective sample size).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #

def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u ** 2)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)


_KERNELS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
}


def get_kernel(name) -> Callable[[np.ndarray], np.ndarray]:
    if callable(name):
        return name
    key = str(name).lower()
    if key not in _KERNELS:
        raise ValueError(f"Unknown kernel '{name}'. Available: {list(_KERNELS)}.")
    return _KERNELS[key]


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #

@dataclass
class WeightResult:
    """Weights at one query time, plus the revision's diagnostics."""
    weights: np.ndarray          # length-N nonnegative weights summing to 1 (0 for pruned refs)
    raw_weights: np.ndarray      # signed local-linear weights a_i(t,h)
    active_index: List[int]      # references actually used in training (weight > 0)
    active_weights: np.ndarray   # weights restricted & renormalized to active_index (sums to 1)
    query_time: float
    bandwidth: float
    negative_mass: float         # P^-(t,h) = sum_{a_i<0} |a_i|  (boundary diagnostic)
    dropped_mass: float          # mass removed by hard threshold (q_eta)
    ess: float                   # effective sample size 1 / sum_i w_i^2 over active refs
    scheme: str = "positive"
    meta: Dict = field(default_factory=dict)

    @property
    def num_active(self) -> int:
        return len(self.active_index)


def _moments(times: np.ndarray, t: float, h: float, kernel) -> tuple:
    n = len(times)
    k = kernel((times - t) / h) / h
    mu0 = np.sum(k) / n
    mu1 = np.sum(k * (times - t)) / n
    mu2 = np.sum(k * (times - t) ** 2) / n
    sigma2 = mu0 * mu2 - mu1 ** 2
    return mu0, mu1, mu2, sigma2, k


def frechet_weights(
    timepoints: Sequence[float],
    query_time: float,
    bandwidth: float,
    *,
    kernel: str = "gaussian",
    scheme: str = "positive",
    threshold: Optional[float] = None,
) -> WeightResult:
    """Compute local-Frechet regression weights at ``query_time``.

    Parameters
    ----------
    timepoints : observed time points t_1 < ... < t_N.
    query_time : the time t at which to estimate the distribution.
    bandwidth  : kernel bandwidth h (controls temporal smoothing / borrowing).
    kernel     : ``"gaussian"`` (default) or ``"epanechnikov"``.
    scheme     : ``"positive"`` (continuous positive-part, default) or ``"raw"``.
    threshold  : optional hard-prune cutoff eta in [0, 1); references with weight
                 below eta are dropped and the rest renormalized.  Use e.g.
                 ``0.01`` to reproduce the notebooks' "drop ~zero-weight
                 snapshots" behaviour while keeping the 5--10 nearest.

    Returns
    -------
    :class:`WeightResult` with the final ``weights`` (length N), the ``active_index``
    / ``active_weights`` to feed the trainer, and diagnostics.
    """
    times = np.asarray(timepoints, dtype=float)
    n = len(times)
    K = get_kernel(kernel)
    mu0, mu1, mu2, sigma2, k = _moments(times, float(query_time), float(bandwidth), K)
    if abs(sigma2) < 1e-15:
        # Degenerate design (e.g. h far too large); fall back to kernel weights.
        raw = k / max(np.sum(k), 1e-15)
    else:
        raw = (1.0 / sigma2) * k * (mu2 - mu1 * (times - float(query_time)))
        raw = raw / n
    raw = np.asarray(raw, dtype=float)

    negative_mass = float(np.sum(np.abs(raw[raw < 0])))

    if scheme == "raw":
        # Old behaviour: normalize signed weights by their sum (can be negative).
        denom = np.sum(raw)
        w = raw / denom if abs(denom) > 1e-15 else raw
    elif scheme == "positive":
        pos = np.clip(raw, 0.0, None)
        s = pos.sum()
        w = pos / s if s > 0 else np.ones(n) / n
    else:
        raise ValueError("scheme must be 'positive' or 'raw'.")

    dropped_mass = 0.0
    if threshold is not None and threshold > 0:
        keep = w >= threshold
        if not keep.any():
            keep = np.zeros(n, dtype=bool)
            keep[int(np.argmax(w))] = True  # never drop everything
        dropped_mass = float(w[~keep].sum())
        w = np.where(keep, w, 0.0)
        s = w.sum()
        w = w / s if s > 0 else w

    active_index = [int(i) for i in np.nonzero(w > 0)[0]]
    active_weights = w[active_index]
    if active_weights.sum() > 0:
        active_weights = active_weights / active_weights.sum()
    ess = float(1.0 / np.sum(active_weights ** 2)) if len(active_weights) else 0.0

    return WeightResult(
        weights=w,
        raw_weights=raw,
        active_index=active_index,
        active_weights=active_weights,
        query_time=float(query_time),
        bandwidth=float(bandwidth),
        negative_mass=negative_mass,
        dropped_mass=dropped_mass,
        ess=ess,
        scheme=scheme,
        meta={"mu0": mu0, "mu1": mu1, "mu2": mu2, "sigma2": sigma2, "kernel": kernel},
    )


def effective_sample_size(weights: Sequence[float]) -> float:
    """ESS(t,h) = 1 / sum_i w_i^2 -- the effective number of contributing snapshots."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    return float(1.0 / np.sum(w ** 2)) if len(w) else 0.0
