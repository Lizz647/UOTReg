"""Loaders for the two real datasets (embryoid, statefate).

Reads the precomputed principal-component matrices shipped under ``data/``, so no scanpy/anndata
is needed for ``d <= 20`` (the regime used in the paper). For ``d > 20`` we fall back to the
``.h5ad`` and read (or recompute) PCs, which needs ``h5py`` (or ``anndata``).

Returns a small :class:`Dataset` bundle: per-time-point arrays (time order), the numeric time
points, the pooled matrix, and the integer time label per cell. ``data_dir`` is the folder holding
``embryoid/`` and ``scrna-statefate/`` (in this repository: ``<repo>/data``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Dataset:
    arrays: List[np.ndarray]   # one (n_i, d) array per time point, in time order
    timepoints: np.ndarray     # numeric time of each snapshot
    pooled: np.ndarray         # (sum n_i, d) all cells
    labels: np.ndarray         # integer time label per pooled cell
    name: str = ""
    dim: int = 0

    @property
    def n_times(self) -> int:
        return len(self.arrays)


def _pcs_from_h5ad(h5ad_path: str, d: int) -> np.ndarray:
    """Extract / compute ``d`` PCs from an .h5ad.

    Fast path (no scanpy): both shipped .h5ad files already store ``obsm/X_pca`` with 50
    precomputed components (verified to match the shipped 10-/20-PC .npy files exactly in
    their leading columns), so we read them directly with ``h5py``. Only if that is missing
    do we fall back to anndata + a fresh PCA of the expression matrix.
    """
    try:                                            # fast path: read obsm/X_pca via h5py
        import h5py
        with h5py.File(h5ad_path, "r") as f:
            if "obsm" in f and "X_pca" in f["obsm"] and f["obsm"]["X_pca"].shape[1] >= d:
                return np.asarray(f["obsm"]["X_pca"][:, :d], dtype=np.float32)
    except Exception:
        pass                                        # fall through to the anndata path
    try:
        import anndata as ad
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f"d={d} needs >20 PCs from {os.path.basename(h5ad_path)}, but it has no readable "
            "obsm/X_pca and anndata is not installed (`pip install anndata`).") from e
    A = ad.read_h5ad(h5ad_path)
    if "X_pca" in A.obsm and A.obsm["X_pca"].shape[1] >= d:
        return np.asarray(A.obsm["X_pca"][:, :d], dtype=np.float32)
    # recompute PCA from the (log-normalized) expression matrix
    from sklearn.decomposition import PCA
    X = A.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return PCA(n_components=d, random_state=0).fit_transform(X).astype(np.float32)


def _load(data_dir: str, name: str, pc10: str, pc20: str, h5ad: str, labels_file: str,
          label_to_time: dict, d: int) -> Dataset:
    folder = os.path.join(data_dir, name)
    if d <= 10:
        X = np.load(os.path.join(folder, pc10))[:, :d]
    elif d <= 20:
        X = np.load(os.path.join(folder, pc20))[:, :d]
    else:
        X = _pcs_from_h5ad(os.path.join(folder, h5ad), d)
    X = np.asarray(X, dtype=np.float32)
    labels = np.load(os.path.join(folder, labels_file))
    uniq = sorted(np.unique(labels).tolist())
    timepoints = np.array([label_to_time[u] for u in uniq], dtype=float)
    arrays = [X[labels == u] for u in uniq]
    return Dataset(arrays=arrays, timepoints=timepoints, pooled=X, labels=labels,
                   name=name, dim=d)


def load_embryoid(data_dir: str, d: int = 20) -> Dataset:
    """Embryoid body differentiation: 5 time points (Day 00-03 ... Day 24-27)."""
    label_to_time = {0: 1.5, 6: 7.5, 12: 13.5, 18: 19.5, 24: 25.5}
    return _load(data_dir, "embryoid", "embryoid_pc.npy", "embryoid_pc_20.npy",
                 "embryoid_data.h5ad", "time_labels.npy", label_to_time, d)


def load_statefate(data_dir: str, d: int = 20) -> Dataset:
    """Hematopoiesis state-fate: 3 time points (Day 2, Day 4, Day 6)."""
    label_to_time = {1: 2.0, 2: 4.0, 3: 6.0}
    return _load(data_dir, "scrna-statefate", "statefate_pc.npy", "statefate_pc20.npy",
                 "invitro-hvg.h5ad", "time_labels.npy", label_to_time, d)


def subsample(ds: Dataset, n_per_time: Optional[int] = None, seed: int = 0) -> Dataset:
    """Return a cell-subsampled copy (handy for fast CPU demos)."""
    if n_per_time is None:
        return ds
    rng = np.random.default_rng(seed)
    arrays, labels_parts = [], []
    uniq = sorted(np.unique(ds.labels).tolist())
    for arr, u in zip(ds.arrays, uniq):
        k = min(n_per_time, len(arr))
        idx = rng.choice(len(arr), k, replace=False)
        arrays.append(arr[idx])
        labels_parts.append(np.full(k, u))
    pooled = np.concatenate(arrays, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    return Dataset(arrays, ds.timepoints, pooled, labels, ds.name, ds.dim)
