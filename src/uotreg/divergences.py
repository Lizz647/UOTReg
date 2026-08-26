r"""Csiszar divergences for the unbalanced marginal penalty (the "psi" knob).

The unbalanced OT loss penalises a relaxed marginal by
``tau * D_psi(rho || nu)`` with ``D_psi(rho||nu) = \int psi(d rho/d nu) d nu``.
What actually enters the *semi-dual* training objective is the (tau-scaled)
convex conjugate ``psi^*_tau(s) = tau * psi^*(s / tau)``; for KL this is the
smooth ``tau (e^{s/tau} - 1)`` used in the original code.

Reviewers asked whether divergences other than KL are possible.  We provide a
small registry so the choice is a one-line config change.  KL is the default
because its conjugate is smooth and finite on all of R, which is what makes the
neural min--max training well behaved; the others are offered for comparison.

Each divergence exposes:
  * ``psi(r)``           -- the entropy function (for reference / diagnostics)
  * ``psi_star_tau(s)``  -- the tau-scaled conjugate used in the loss
  * ``smooth``           -- whether the conjugate is differentiable everywhere
  * ``finite_everywhere``-- whether the conjugate is finite for all real s
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch

# Clamp exponent arguments to keep exp() from overflowing in fp16/fp32.
_EXP_CLAMP = 30.0


@dataclass
class Divergence:
    name: str
    tau: float
    psi: Callable[[torch.Tensor], torch.Tensor]
    psi_star_tau: Callable[[torch.Tensor], torch.Tensor]
    smooth: bool = True
    finite_everywhere: bool = True
    note: str = ""

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        return self.psi_star_tau(s)


def _kl(tau: float) -> Divergence:
    # psi(r) = r log r - r + 1 ; psi^*(s) = e^s - 1 ; psi^*_tau(s) = tau (e^{s/tau}-1)
    def psi(r):
        return r * torch.log(r.clamp_min(1e-12)) - r + 1.0

    def psi_star_tau(s):
        return tau * (torch.exp((s / tau).clamp(max=_EXP_CLAMP)) - 1.0)

    return Divergence("kl", tau, psi, psi_star_tau, smooth=True, finite_everywhere=True,
                      note="Default. Smooth, finite everywhere; recommended for neural UOT.")


def _chi2(tau: float) -> Divergence:
    # Pearson chi^2: psi(r) = (r-1)^2 ; psi^*(s) = s + s^2/4 for s >= -2, else -1.
    def psi(r):
        return (r - 1.0) ** 2

    def psi_star_tau(s):
        u = s / tau
        smooth_branch = u + u ** 2 / 4.0
        return tau * torch.where(u >= -2.0, smooth_branch, torch.full_like(u, -1.0))

    return Divergence("chi2", tau, psi, psi_star_tau, smooth=True, finite_everywhere=True,
                      note="Pearson chi-square. Smooth, finite; heavier penalty on large mass ratios than KL.")


def _hellinger(tau: float) -> Divergence:
    # psi(r) = (sqrt(r) - 1)^2 ; psi^*(s) = s/(1-s) for s < 1, +inf otherwise.
    def psi(r):
        return (torch.sqrt(r.clamp_min(0.0)) - 1.0) ** 2

    def psi_star_tau(s):
        u = (s / tau).clamp(max=1.0 - 1e-4)  # enforce domain s < tau
        return tau * (u / (1.0 - u))

    return Divergence("hellinger", tau, psi, psi_star_tau, smooth=True, finite_everywhere=False,
                      note="Squared Hellinger. Conjugate finite only for s < tau (clamped); use moderate tau.")


def _tv(tau: float) -> Divergence:
    # Total variation: psi(r) = |r-1| ; psi^*(s) = s on [-1,1], -1 for s<-1, +inf for s>1.
    def psi(r):
        return torch.abs(r - 1.0)

    def psi_star_tau(s):
        u = (s / tau).clamp(max=1.0)  # +inf region clamped to the boundary
        u = torch.where(u < -1.0, torch.full_like(u, -1.0), u)
        return tau * u

    return Divergence("tv", tau, psi, psi_star_tau, smooth=False, finite_everywhere=False,
                      note="Total variation. Non-smooth / constrained conjugate; offered for comparison only.")


_BUILDERS: Dict[str, Callable[[float], Divergence]] = {
    "kl": _kl,
    "chi2": _chi2,
    "pearson": _chi2,
    "hellinger": _hellinger,
    "tv": _tv,
    "totalvariation": _tv,
}


def get_divergence(name: str, tau: float) -> Divergence:
    """Build a :class:`Divergence` for ``name`` with tolerance ``tau``.

    Available: ``"kl"`` (default), ``"chi2"`` / ``"pearson"``, ``"hellinger"``,
    ``"tv"``.  Pass ``tau`` (the unbalanced tolerance); as ``tau -> inf`` every
    relaxed penalty tends to the hard marginal constraint (balanced OT).
    """
    key = str(name).lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown divergence '{name}'. Available: {sorted(set(_BUILDERS))}.")
    return _BUILDERS[key](float(tau))


def available_divergences():
    return sorted(set(_BUILDERS))
