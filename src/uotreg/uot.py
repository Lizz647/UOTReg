r"""Semi-dual loss primitives for (unbalanced) optimal transport.

We solve each transport sub-problem in its semi-dual (min--max) form, exactly as
in the original code, but unify the three relaxation modes and any divergence
behind two functions.

Notation (all tensors broadcast over an optional reference axis):
  * ``cost``  : c(x, T(x))   -- the realized ground cost of the map  (>= 0)
  * ``d_Tx``  : v(T(x))      -- potential at mapped source points
  * ``d_y``   : v(y)         -- potential at target samples y ~ nu

The c-transform ``v^c(x) = min_y [c(x,y) - v(y)]`` is realized by the map T,
so the **map** is always trained to minimize ``c(x,T(x)) - v(T(x))`` regardless
of the marginal relaxation (the c-transform depends only on the cost).

The **potential** maximizes the semi-dual value
  J(v) = S_src(v^c; mu) + S_tgt(v; nu),
where a *hard* (balanced) marginal contributes the linear term ``E[.]`` and a
*relaxed* marginal contributes ``-E[psi^*_tau(-.)]``.  Minimizing ``-J`` gives:

  balanced   :  E[v(Tx)]            - E[v(y)]
  one-sided  :  E[v(Tx)]            + E[psi^*_tau(-v(y))]          (relax target)
  two-sided  :  E[psi^*_tau(-v^c)]  + E[psi^*_tau(-v(y))]          (relax both)

with ``v^c ≈ cost - v(Tx)``.  The constant ``cost`` is dropped where it does not
depend on the potential (balanced / one-sided) and kept where it does (two-sided).
"""
from __future__ import annotations

import torch

from .divergences import Divergence

VALID_RELAXATIONS = ("balanced", "one-sided", "two-sided")


def map_loss(cost: torch.Tensor, d_Tx: torch.Tensor) -> torch.Tensor:
    """Loss for the transport map(s): realize the c-transform ``v^c = inf(c - v)``."""
    return (cost - d_Tx).mean()


def potential_loss(
    cost: torch.Tensor,
    d_Tx: torch.Tensor,
    d_y: torch.Tensor,
    *,
    relaxation: str,
    divergence: Divergence,
) -> torch.Tensor:
    """Loss for the potential network(s) under the chosen relaxation/divergence.

    Returns a scalar to **minimize** (it is ``-J`` of the semi-dual value).
    """
    if relaxation == "balanced":
        return d_Tx.mean() - d_y.mean()
    if relaxation == "one-sided":
        return d_Tx.mean() + divergence(-d_y).mean()
    if relaxation == "two-sided":
        vc = cost - d_Tx                     # realized c-transform value v^c(x)
        return divergence(-vc).mean() + divergence(-d_y).mean()
    raise ValueError(f"relaxation must be one of {VALID_RELAXATIONS}, got '{relaxation}'.")


def check_relaxation(relaxation: str) -> str:
    if relaxation not in VALID_RELAXATIONS:
        raise ValueError(f"relaxation must be one of {VALID_RELAXATIONS}, got '{relaxation}'.")
    return relaxation
