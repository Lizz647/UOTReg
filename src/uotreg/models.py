"""Neural network building blocks for UOTReg.

Consolidates the many overlapping classes in the old ``models.py`` into a few
clear ones:

  * :class:`MLP`            -- the single feed-forward backbone everyone reused.
  * :class:`Generator`      -- parameterizes a distribution mu = G_# (latent).
  * :class:`TransportMaps`  -- N independent maps T_i sharing an input (the old
                               ``Seperate_T``); used inside the barycenter solver.
  * :class:`Potentials`     -- N independent potentials v_i (the old ``Seperate_D``).
  * :class:`MapNet` / :class:`PotentialNet` -- single-head versions for pairwise
                               (trajectory) UOT.
  * :class:`PlanarFlow` / :class:`NormalizingFlow` -- for mode-capturing init.
  * :class:`VelocityField`  -- time-conditioned field for flow-matching trajectories.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    """Feed-forward net: input -> [hidden]*n_hidden -> output, ReLU + dropout."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_hidden: int = 4,
                 dropout: float = 0.05, batchnorm: bool = False):
        super().__init__()
        sizes = [in_dim] + [hidden] * n_hidden + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))
        self.bn = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(n_hidden)) if batchnorm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.bn is not None and i < len(self.bn):
                x = self.bn[i](x)
            x = self.dropout(F.relu(x))
        return self.layers[-1](x)


class Generator(nn.Module):
    """Generative network mu = G_#(latent).  Width grows with the data dim.

    This is the old ``Gnet`` renamed; the layer construction (and therefore the
    ``state_dict`` keys ``network.0/3/6/9/11.*``) is byte-identical, so generators
    saved by the original code load directly.  ``latent_dim`` defaults to ``dim``
    and ``width = max(hidden, 2*dim)`` exactly as before.
    """

    def __init__(self, dim: int, latent_dim: int = None, hidden: int = 256,
                 n_layers: int = 4, dropout: float = 0.01):
        super().__init__()
        latent_dim = latent_dim or dim
        width = max(hidden, 2 * dim)
        layers: List[nn.Module] = [nn.Linear(latent_dim, width), nn.ReLU(True), nn.Dropout(dropout)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(width, width), nn.ReLU(True), nn.Dropout(dropout)]
        layers += [nn.Linear(width, width), nn.ReLU(True), nn.Linear(width, dim)]
        # attribute name kept as `network` to match old Gnet state_dict keys
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class TransportMaps(nn.Module):
    """``num`` independent transport maps applied to a shared input x.

    ``forward(x)`` returns a tensor of shape ``(B, num * dim)`` -- head i occupies
    columns ``[i*dim:(i+1)*dim]``.  (Equivalent to the old ``Seperate_T``.)
    """

    def __init__(self, dim: int, num: int, hidden: int = 256, n_hidden: int = 5,
                 dropout: float = 0.05, batchnorm: bool = False):
        super().__init__()
        self.dim, self.num = dim, num
        self.heads = nn.ModuleList(
            MLP(dim, dim, hidden, n_hidden, dropout, batchnorm) for _ in range(num)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=1)


class Potentials(nn.Module):
    """``num`` independent scalar potentials v_i.

    ``forward(y)`` expects ``y`` of shape ``(B, num*dim)`` and applies head i to
    chunk i, returning ``(B, num)``.  (Equivalent to the old ``Seperate_D``.)
    """

    def __init__(self, dim: int, num: int, hidden: int = 256, n_hidden: int = 5,
                 dropout: float = 0.05, batchnorm: bool = False):
        super().__init__()
        self.dim, self.num = dim, num
        self.heads = nn.ModuleList(
            MLP(dim, 1, hidden, n_hidden, dropout, batchnorm) for _ in range(num)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(y, self.num, dim=1)
        return torch.cat([head(chunks[i]) for i, head in enumerate(self.heads)], dim=1)


class MapNet(nn.Module):
    """Single transport map T: R^d -> R^d (pairwise / trajectory UOT)."""

    def __init__(self, dim: int, hidden: int = 256, n_hidden: int = 5,
                 dropout: float = 0.05, batchnorm: bool = False):
        super().__init__()
        self.net = MLP(dim, dim, hidden, n_hidden, dropout, batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PotentialNet(nn.Module):
    """Single scalar potential v: R^d -> R (pairwise / trajectory UOT)."""

    def __init__(self, dim: int, hidden: int = 256, n_hidden: int = 5,
                 dropout: float = 0.05, batchnorm: bool = False):
        super().__init__()
        self.net = MLP(dim, 1, hidden, n_hidden, dropout, batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Normalizing flow (mode-capturing initialization)
# --------------------------------------------------------------------------- #

class PlanarFlow(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, dim).uniform_(-0.01, 0.01))
        self.scale = nn.Parameter(torch.empty(1, dim).uniform_(-0.01, 0.01))
        self.bias = nn.Parameter(torch.empty(1).uniform_(-0.01, 0.01))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.scale * torch.tanh(F.linear(z, self.weight, self.bias))

    def log_abs_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        f = F.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f) ** 2) * self.weight
        det = 1 + psi @ self.scale.t()
        return torch.log(det.abs() + 1e-9)


class NormalizingFlow(nn.Module):
    """A stack of planar flows mapping a base Gaussian to the data.

    Trained by maximum likelihood for a few epochs to *capture the modes* of the
    pooled cells, then used to initialize the generator (see ``initialization``).
    """

    def __init__(self, dim: int, flow_length: int = 16):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList(PlanarFlow(dim) for _ in range(flow_length))

    def forward(self, z: torch.Tensor):
        log_det = torch.zeros(z.shape[0], 1, device=z.device)
        for flow in self.flows:
            log_det = log_det + flow.log_abs_det_jacobian(z)
            z = flow(z)
        return z, log_det


class Encoder(nn.Module):
    """Encoder for the VAE/NF generator initialization (matches the old ``Encoder``)."""

    def __init__(self, nin: int, n_latent: int, size: int = 256, num_layers: int = 4,
                 dropout: float = 0.05):
        super().__init__()
        layers = [nn.Linear(nin, size), nn.ReLU(True), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(size, size), nn.ReLU(True), nn.Dropout(dropout)]
        layers += [nn.Linear(size, n_latent)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# --------------------------------------------------------------------------- #
# Flow-matching velocity field
# --------------------------------------------------------------------------- #

class VelocityField(nn.Module):
    """Time-conditioned velocity v_theta(x, t) for dynamic-OT / flow-matching.

    Time is appended as an extra input feature, so a single network represents
    the whole trajectory and is integrated with an ODE solver.
    """

    def __init__(self, dim: int, hidden: int = 256, n_hidden: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = MLP(dim + 1, dim, hidden, n_hidden, dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))
