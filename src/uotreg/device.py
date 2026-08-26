"""Device, precision, and reproducibility helpers.

The original notebooks hard-coded ``device="cpu"`` everywhere.  Here we resolve
the device once (``"auto"`` picks CUDA / Apple-MPS when available) and reuse it.
On a large GPU (e.g. H200) you can simply pass ``device="cuda"`` and bump the
batch sizes; nothing else in the API changes.
"""
from __future__ import annotations

import contextlib
import random
from typing import Optional, Union

import numpy as np
import torch

Device = Union[str, torch.device]


def resolve_device(device: Optional[Device] = "auto") -> torch.device:
    """Turn a user-facing device spec into a concrete ``torch.device``.

    ``"auto"`` (or ``None``) -> CUDA if available, else Apple MPS, else CPU.
    """
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def seed_everything(seed: Optional[int]) -> None:
    """Seed python / numpy / torch for reproducible runs (no-op if ``seed`` is None)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool):
    """Return a mixed-precision autocast context (only meaningful on CUDA).

    Use as ``with autocast_context(dev, cfg.amp): ...`` to speed up training on
    the GPU.  On CPU / MPS it is a no-op so the same code path runs everywhere.
    """
    if enabled and device.type == "cuda":
        # bfloat16 has fp32-range exponent, so the min--max UOT objective trains
        # stably without a GradScaler (recommended on H100/H200).
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").float().numpy()
