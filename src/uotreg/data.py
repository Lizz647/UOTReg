"""Lightweight, device-aware samplers.

Consolidates the several ``Sampler`` / ``tensorSampler`` / ``DatasetSampler``
variants from the old ``distributions.py`` and ``tools.py`` into one small class.
A sampler just holds a tensor of points (cells x features) and draws random
mini-batches on the requested device.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from .device import Device, resolve_device

ArrayLike = Union[np.ndarray, torch.Tensor, "pd.DataFrame"]  # noqa: F821


def _to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    if hasattr(x, "values"):  # pandas DataFrame / Series
        x = x.values
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class TensorSampler:
    """Random mini-batch sampler over a fixed point cloud.

    Parameters
    ----------
    data   : array / tensor / DataFrame of shape (n_cells, d).
    device : where ``sample`` returns batches ("auto", "cpu", "cuda", ...).
    pin    : keep the master copy on CPU (pinned) and move batches on demand;
             useful for very large clouds on the GPU.  If False the whole cloud
             lives on ``device``.
    """

    def __init__(self, data: ArrayLike, device: Optional[Device] = "auto", pin: bool = False):
        self.device = resolve_device(device)
        self._pin = pin
        cloud = _to_tensor(data)
        self._cloud = cloud if pin else cloud.to(self.device)
        self.n, self.dim = self._cloud.shape

    def __len__(self) -> int:
        return self.n

    def sample(self, batch_size: int = 64) -> torch.Tensor:
        idx = torch.randint(0, self.n, (batch_size,), device=self._cloud.device)
        batch = self._cloud[idx]
        return batch.to(self.device, non_blocking=True) if self._pin else batch

    def all(self) -> torch.Tensor:
        return self._cloud.to(self.device)

    def to(self, device: Device) -> "TensorSampler":
        self.device = resolve_device(device)
        if not self._pin:
            self._cloud = self._cloud.to(self.device)
        return self


class GaussianLatentSampler:
    """Standard-normal latent sampler for the generator (replaces StandardNormalSampler)."""

    def __init__(self, dim: int, device: Optional[Device] = "auto", scale: float = 1.0):
        self.dim = dim
        self.device = resolve_device(device)
        self.scale = scale

    def sample(self, batch_size: int = 64) -> torch.Tensor:
        return self.scale * torch.randn(batch_size, self.dim, device=self.device)

    def to(self, device: Device) -> "GaussianLatentSampler":
        self.device = resolve_device(device)
        return self


class GaussianMixtureLatentSampler:
    """Multi-modal latent: a mixture of ``k`` Gaussians in latent space.

    Giving the generator a *disconnected* latent lets its image be disconnected,
    so it can form separate modes instead of a connecting "bridge".  ``separation``
    sets how far apart the components are along the first latent axis.
    """

    def __init__(self, dim: int, k: int = 2, separation: float = 4.0, std: float = 1.0,
                 device: Optional[Device] = "auto"):
        self.dim, self.k, self.sep, self.std = dim, k, separation, std
        self.device = resolve_device(device)
        centers = torch.zeros(k, dim)
        centers[:, 0] = torch.linspace(-(k - 1) / 2.0, (k - 1) / 2.0, k) * separation
        self.centers = centers.to(self.device)

    def sample(self, batch_size: int = 64) -> torch.Tensor:
        comp = torch.randint(0, self.k, (batch_size,), device=self.device)
        return self.centers[comp] + self.std * torch.randn(batch_size, self.dim, device=self.device)

    def to(self, device: Device) -> "GaussianMixtureLatentSampler":
        self.device = resolve_device(device)
        self.centers = self.centers.to(self.device)
        return self


def samplers_from_arrays(
    arrays: Sequence[ArrayLike],
    device: Optional[Device] = "auto",
    pin: bool = False,
) -> List[TensorSampler]:
    """Build one :class:`TensorSampler` per time point from a list of point clouds."""
    return [TensorSampler(a, device=device, pin=pin) for a in arrays]


def samplers_from_labeled(
    features: ArrayLike,
    labels: Sequence,
    order: Optional[Sequence] = None,
    device: Optional[Device] = "auto",
    pin: bool = False,
):
    """Split a single feature matrix into per-time-point samplers.

    Parameters
    ----------
    features : (n_cells, d) matrix (e.g. ``adata.obsm['X_pca'][:, :d]``).
    labels   : length-n_cells time labels (one per cell).
    order    : optional explicit ordering of the unique labels; otherwise sorted.

    Returns ``(samplers, ordered_labels)``.
    """
    feats = _to_tensor(features)
    labels = np.asarray(labels)
    uniq = list(order) if order is not None else sorted(np.unique(labels).tolist())
    samplers = []
    for lab in uniq:
        mask = labels == lab
        samplers.append(TensorSampler(feats[mask], device=device, pin=pin))
    return samplers, uniq
