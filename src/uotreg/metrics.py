"""Distributional distances for evaluation (benchmarks, batch-effect alignment).

All functions take two point clouds ``X`` (n, d) and ``Y`` (m, d) as numpy
arrays or tensors and return a python float.  POT (``pip install pot``) is used
for exact Wasserstein distances when available; otherwise we fall back to the
sliced-Wasserstein approximation (always available) and say so once.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

try:  # optional exact-OT backend
    import ot as _pot
    _HAS_POT = True
except Exception:  # pragma: no cover
    _HAS_POT = False

_WARNED = {"pot": False}


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Maximum mean discrepancy (multi-bandwidth RBF)
# --------------------------------------------------------------------------- #

def mmd(X, Y, sigmas: Sequence[float] = (1, 2, 4, 8, 16), subsample: int = 1000) -> float:
    """Squared MMD with a sum-of-RBF kernel (matches the benchmark's MMD)."""
    X, Y = _np(X), _np(Y)
    rng = np.random.default_rng(0)
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]

    def k(A, B):
        d2 = np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T
        out = np.zeros_like(d2)
        for s in sigmas:
            out += np.exp(-d2 / (2 * s ** 2))
        return out / len(sigmas)

    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


# --------------------------------------------------------------------------- #
# Sliced-Wasserstein (always available)
# --------------------------------------------------------------------------- #

def sliced_wasserstein(X, Y, p: int = 2, n_proj: int = 200, seed: int = 0) -> float:
    """Sliced p-Wasserstein distance via random 1-D projections."""
    X, Y = _np(X), _np(Y)
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    total = 0.0
    for u in dirs:
        a = np.sort(X @ u)
        b = np.sort(Y @ u)
        m = min(len(a), len(b))
        qa = np.quantile(a, np.linspace(0, 1, m))
        qb = np.quantile(b, np.linspace(0, 1, m))
        total += np.mean(np.abs(qa - qb) ** p)
    return float((total / n_proj) ** (1.0 / p))


# --------------------------------------------------------------------------- #
# Wasserstein (exact via POT, else sliced fallback)
# --------------------------------------------------------------------------- #

def wasserstein(X, Y, p: int = 2, subsample: int = 1500) -> float:
    """Exact p-Wasserstein distance (POT) or sliced fallback if POT is absent."""
    X, Y = _np(X), _np(Y)
    if not _HAS_POT:
        if not _WARNED["pot"]:
            warnings.warn("POT not installed; using sliced-Wasserstein as a proxy for "
                          "W%d. `pip install pot` for exact values." % p)
            _WARNED["pot"] = True
        return sliced_wasserstein(X, Y, p=p)
    rng = np.random.default_rng(0)
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    M = _pot.dist(X, Y, metric="euclidean") ** p
    val = _pot.emd2(a, b, M)
    return float(val ** (1.0 / p))


def emd(X, Y, **kw) -> float:
    """Earth mover's distance = 1-Wasserstein."""
    return wasserstein(X, Y, p=1, **kw)


def w2(X, Y, **kw) -> float:
    """2-Wasserstein distance."""
    return wasserstein(X, Y, p=2, **kw)


def energy_distance(X, Y) -> float:
    """Energy distance (kernel-free; cheap, robust)."""
    X, Y = _np(X), _np(Y)

    def md(A, B):
        return np.sqrt(np.maximum(
            np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T, 0)).mean()

    return float(2 * md(X, Y) - md(X, X) - md(Y, Y))


def mmd_rbf(X, Y, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma=None) -> float:
    """Adaptive multi-scale RBF MMD -- reproduces the old benchmark's ``mmd_loss``.

    Bandwidth defaults to the mean pairwise squared distance (median-heuristic
    style), then a geometric ladder of ``kernel_num`` bandwidths is summed.
    """
    X, Y = _np(X), _np(Y)
    n = X.shape[0] + Y.shape[0]
    total = np.concatenate([X, Y], axis=0)
    L2 = np.sum((total[None, :, :] - total[:, None, :]) ** 2, axis=2)
    bandwidth = fix_sigma if fix_sigma else np.sum(L2) / (n ** 2 - n)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bws = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    K = sum(np.exp(-L2 / bw) for bw in bws)
    b = X.shape[0]
    XX, YY = K[:b, :b], K[b:, b:]
    XY, YX = K[:b, b:], K[b:, :b]
    return float(np.mean(XX + YY - XY - YX))


def _kde_metrics(X, Y, bandwidth: float = 0.5):
    from sklearn.neighbors import KernelDensity
    X, Y = _np(X), _np(Y)
    kp = KernelDensity(bandwidth=bandwidth).fit(X)
    kq = KernelDensity(bandwidth=bandwidth).fit(Y)
    pX, qX = np.exp(kp.score_samples(X)), np.exp(kq.score_samples(X))
    pY, qY = np.exp(kp.score_samples(Y)), np.exp(kq.score_samples(Y))
    kl = float(np.mean(kp.score_samples(X) - kq.score_samples(X)))
    tvd = float(0.5 * (np.mean(np.abs(pX - qX)) + np.mean(np.abs(pY - qY))))
    return kl, tvd


def kl_divergence_kde(X, Y, bandwidth: float = 0.5) -> float:
    return _kde_metrics(X, Y, bandwidth)[0]


def tvd_kde(X, Y, bandwidth: float = 0.5) -> float:
    return _kde_metrics(X, Y, bandwidth)[1]


def all_metrics(X, Y) -> dict:
    """Convenience: MMD, EMD (W1), W2, sliced-W in one dict."""
    return {
        "mmd": mmd(X, Y),
        "emd": emd(X, Y),
        "w2": w2(X, Y),
        "sliced_w2": sliced_wasserstein(X, Y, p=2),
    }
