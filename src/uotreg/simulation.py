r"""Synthetic batch-effect simulations for the paper's numerical study.

Consolidates the previously-scattered `batcheffect_data.py` (2-D `bd` bifurcation, faithful to the
original ``Simu_noise``) and the `hd_batcheffect_data.py` high-dimensional lift into one package module.

* The 2-D bifurcation / linear / reverse geometries are the true signal.
* `GeomHD` lifts them to `dim` dimensions (block-replicate the 2 signal coords + optional pure-noise
  dims + rotation) and keeps a 2-D `Geom`-style interface: `observed`/`clean` are d-dim clouds (what
  the estimator sees), `curves`/`branch_means` are 2-D, `project2d` maps back to the signal plane.
* Shape metrics (`adherence`, branch-aware `adherence_labeled`, `branch_balance`) live on the 2-D
  projection; distribution fidelity (W2) is measured in full d.

Set `dim=2` to recover the original 2-D experiment. `make_bifurcation`, `make_reverse`, `make_linear`
are the entry points; `roughness` lives in `uotreg.trajectory_metrics`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .trajectory_metrics import roughness

# --------------------------------------------------------------------------- #
# 2-D bifurcation (faithful to the original Simu_noise): 10 times t=0..9.
# Trunk for t<=SPLIT at y=2 (+/-DELTA), then A -> y=t-2, B -> y=-t+6.
# --------------------------------------------------------------------------- #
N_TIMES = 10
N_PER_TIME = 500
SPLIT = 4.0
DELTA_PRE = 0.25
Y_JITTER_PRE = 0.03
COV_A = np.diag([0.32 ** 2, 0.30 ** 2])
COV_B = np.diag([0.22 ** 2, 0.20 ** 2])
STRENGTH = 0.5


def branch_means(t):
    """True (noise-free) means of the two branches at time t: (mA, mB), each (2,)."""
    t = float(t)
    if t <= SPLIT:
        yA, yB = 2.0 + DELTA_PRE, 2.0 - DELTA_PRE
    else:
        yA, yB = t - 2.0, -t + 6.0
    return np.array([t, yA]), np.array([t, yB])


def _affine(rng, strength=STRENGTH):
    theta = np.deg2rad(rng.uniform(-10, 10) * strength)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale = 1.0 + rng.uniform(-0.20, 0.20, size=2) * strength
    A = R * scale
    b = rng.normal(0, 0.30, size=2) * np.sqrt(strength)
    return A, b


def _snapshot(t, n, rng):
    mA, mB = branch_means(t)
    nA = n // 2
    nB = n - nA
    LA, LB = np.linalg.cholesky(COV_A), np.linalg.cholesky(COV_B)
    pA = mA + rng.standard_normal((nA, 2)) @ LA.T
    pB = mB + rng.standard_normal((nB, 2)) @ LB.T
    if t <= SPLIT:
        pA[:, 1] += rng.normal(0, Y_JITTER_PRE, nA)
        pB[:, 1] += rng.normal(0, Y_JITTER_PRE, nB)
    pts = np.concatenate([pA, pB], axis=0).astype(np.float32)
    lab = np.concatenate([np.zeros(nA, int), np.ones(nB, int)])
    perm = rng.permutation(n)
    return pts[perm], lab[perm]


@dataclass
class BatchData:
    tlist: np.ndarray
    observed: List[np.ndarray]
    clean: List[np.ndarray]
    labels: List[np.ndarray]
    curves: np.ndarray


def generate(seed: int = 0, n_per_time: int = N_PER_TIME, strength: float = STRENGTH) -> BatchData:
    """The faithful 2-D bifurcation: per-time clean snapshot + an independent affine batch effect."""
    rng = np.random.default_rng(seed)
    tlist = np.linspace(0, N_TIMES - 1, N_TIMES)
    observed, clean, labels = [], [], []
    for t in tlist:
        pts, lab = _snapshot(float(t), n_per_time, rng)
        A, b = _affine(rng, strength)
        observed.append((pts @ A.T + b).astype(np.float32))
        clean.append(pts)
        labels.append(lab)
    curves = np.stack([np.stack(branch_means(t), axis=0) for t in tlist], axis=0)
    return BatchData(tlist, observed, clean, labels, curves)


# --------------------------------------------------------------------------- #
# high-dimensional lift helpers
# --------------------------------------------------------------------------- #
def lift_spec(dim, method):
    """How to fill `dim` dims. Columns = [block_x (block), block_y (block), noise (n_noise)]."""
    assert dim >= 2, "dim must be >= 2"
    if method == "replicate":
        block = dim // 2; n_noise = dim - 2 * block
    elif method == "noise":
        block = 1; n_noise = dim - 2
    elif method == "combo":
        block = max(1, dim // 4); n_noise = dim - 2 * block
    else:
        raise ValueError(f"unknown lift method '{method}' (use replicate|noise|combo)")
    return block, n_noise


def _rotation(dim, seed, rotate):
    if not rotate or dim == 2:
        return None
    rng = np.random.default_rng(seed + 9999)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return Q.astype(np.float32)


def _lift(X2d, block, n_noise, lift_noise, noise_scale, Q, rng):
    """Embed 2-D points into (2*block + n_noise) dims (block-replicate signal + noise dims)."""
    X2d = np.asarray(X2d, np.float32)
    if block == 1 and n_noise == 0:
        X = X2d
    else:
        cols = []
        for c in range(2):
            rep = np.repeat(X2d[:, [c]], block, axis=1) + lift_noise * rng.standard_normal((len(X2d), block))
            cols.append(rep)
        if n_noise > 0:
            cols.append(noise_scale * rng.standard_normal((len(X2d), n_noise)))
        X = np.concatenate(cols, axis=1).astype(np.float32)
    return (X @ Q.T).astype(np.float32) if Q is not None else X


def _reverse_2d(observed, clean, curves, labels=None):
    """Flip the bifurcation horizontally (x->xmax-x) AND reverse time -> starts APART and CONVERGES."""
    n = len(observed); xmax = float(n - 1)

    def flipx(P):
        P = np.asarray(P, np.float32).copy(); P[..., 0] = xmax - P[..., 0]; return P

    obs = [flipx(observed[n - 1 - i]) for i in range(n)]
    cln = [flipx(clean[n - 1 - i]) for i in range(n)]
    crv = np.stack([flipx(curves[n - 1 - i]) for i in range(n)], 0)
    lab = [np.asarray(labels[n - 1 - i]) for i in range(n)] if labels is not None else None

    def bmeans(t):
        mA, mB = branch_means(xmax - float(t))
        return np.array([xmax - mA[0], mA[1]]), np.array([xmax - mB[0], mB[1]])

    return obs, cln, crv, bmeans, lab


# --------------------------------------------------------------------------- #
# GeomHD: original 2-D `Geom` interface, clouds lifted to `dim`
# --------------------------------------------------------------------------- #
class GeomHD:
    """A lifted geometry: d-dim `observed`/`clean` clouds with a 2-D signal-plane interface."""

    def __init__(self, kind, tlist, observed_2d, clean_2d, curves, branch_means_fn,
                 dim=2, method="replicate", lift_noise=0.05, noise_scale=1.0, rotate=False, seed=0,
                 labels=None):
        self.kind = kind; self.dim = int(dim); self.method = method
        self.tlist = [float(t) for t in tlist]
        self.curves = np.asarray(curves, np.float32)
        self.branch_means_fn = branch_means_fn
        self.block, self.n_noise = lift_spec(self.dim, method)
        self._Q = _rotation(self.dim, seed, rotate)
        rng = np.random.default_rng(seed + 777)
        self.observed = [_lift(o, self.block, self.n_noise, lift_noise, noise_scale, self._Q, rng)
                         for o in observed_2d]
        self.clean = [_lift(c, self.block, self.n_noise, lift_noise, noise_scale, self._Q, rng)
                      for c in clean_2d]
        self.observed_2d = [np.asarray(o, np.float32) for o in observed_2d]
        self.labels = [np.asarray(l) for l in labels] if labels is not None else None

    # -- geometry / projection --
    def project2d(self, X):
        """Map d-dim points back to the interpretable 2-D signal plane (block-average, undo rotation)."""
        X = np.asarray(X, float)
        if self.dim == 2:
            return X[:, :2]
        if self._Q is not None:
            X = X @ self._Q
        return np.stack([X[:, :self.block].mean(1), X[:, self.block:2 * self.block].mean(1)], 1)

    def project_traj(self, traj):
        return np.stack([self.project2d(frame) for frame in np.asarray(traj)], 0)

    def branch_means(self, t):
        mA, mB = self.branch_means_fn(float(t)); return np.asarray(mA, float), np.asarray(mB, float)

    def truth(self, t):
        return self.clean[int(round(float(t)))]

    def without_time(self, j):
        """A shallow view with time-index j removed (no re-lifting) -- for leave-one-out."""
        g = GeomHD.__new__(GeomHD)
        g.kind = self.kind; g.dim = self.dim; g.method = self.method
        g.tlist = [t for i, t in enumerate(self.tlist) if i != j]
        g.curves = self.curves; g.branch_means_fn = self.branch_means_fn
        g.block = self.block; g.n_noise = self.n_noise; g._Q = self._Q
        g.observed = [o for i, o in enumerate(self.observed) if i != j]
        g.clean = [c for i, c in enumerate(self.clean) if i != j]
        g.observed_2d = [o for i, o in enumerate(self.observed_2d) if i != j]
        g.labels = [l for i, l in enumerate(self.labels) if i != j] if self.labels is not None else None
        return g

    # -- shape metrics (on the 2-D projection) --
    def adherence(self, traj2d, times):
        """Mean dist to the NEAREST branch. Rewards being on-manifold; blind to imbalance -- use
        `adherence_labeled` for a branch-aware version."""
        traj2d = np.asarray(traj2d); tot, cnt = 0.0, 0
        for k, t in enumerate(times):
            mA, mB = self.branch_means(t)
            d = np.minimum(np.linalg.norm(traj2d[k] - mA, axis=1), np.linalg.norm(traj2d[k] - mB, axis=1))
            tot += d.sum(); cnt += traj2d[k].shape[0]
        return float(tot / max(cnt, 1))

    def adherence_labeled(self, traj2d, times, labels0):
        """Mean dist to each cell's ASSIGNED branch (labels0: 0=A upper, 1=B lower); penalizes the
        'all mass to one branch' collapse the nearest-branch metric misses."""
        traj2d = np.asarray(traj2d); labels0 = np.asarray(labels0).ravel()
        tot, cnt = 0.0, 0
        for k, t in enumerate(times):
            mA, mB = self.branch_means(t)
            assigned = np.where(labels0[:, None] == 0, np.asarray(mA)[None, :], np.asarray(mB)[None, :])
            tot += np.linalg.norm(traj2d[k] - assigned, axis=1).sum(); cnt += traj2d[k].shape[0]
        return float(tot / max(cnt, 1))

    def branch_balance(self, traj2d, times, labels0=None):
        """Fraction on branch A at the most-separated time (~0.5 ideal); if labels0 given also returns
        branch-recovery accuracy. nan when the branches coincide (linear geometry)."""
        traj2d = np.asarray(traj2d)
        seps = [float(np.linalg.norm(np.subtract(*self.branch_means(t)))) for t in times]
        k = int(np.argmax(seps))
        if seps[k] < 1e-6:
            return float("nan")
        mA, mB = self.branch_means(times[k]); P = traj2d[k]
        near_A = np.linalg.norm(P - np.asarray(mA), axis=1) < np.linalg.norm(P - np.asarray(mB), axis=1)
        if labels0 is None:
            return float(near_A.mean())
        labels0 = np.asarray(labels0).ravel()
        recovery = float(((near_A & (labels0 == 0)) | (~near_A & (labels0 == 1))).mean())
        return float(near_A.mean()), recovery

    def truth_roughness(self, times):
        cA = np.stack([self.branch_means(t)[0] for t in times])
        cB = np.stack([self.branch_means(t)[1] for t in times])
        return 0.5 * (roughness(cA[:, None, :]) + roughness(cB[:, None, :]))


# --------------------------------------------------------------------------- #
# make_* : the three geometries, lifted to `dim`
# --------------------------------------------------------------------------- #
def make_bifurcation(seed=0, strength=0.5, n_per_time=500, dim=2,
                     method="replicate", lift_noise=0.05, noise_scale=1.0, rotate=False):
    """Faithful bd bifurcation lifted to `dim`. `dim=2` reproduces the original 2-D data exactly."""
    d = generate(seed=seed, n_per_time=n_per_time, strength=strength)
    obs2d = [np.asarray(o, np.float32) for o in d.observed]
    cln2d = [np.asarray(c, np.float32) for c in d.clean]
    return GeomHD("bifurcation", d.tlist, obs2d, cln2d, d.curves, lambda t: branch_means(float(t)),
                  dim=dim, seed=seed, labels=d.labels, method=method, lift_noise=lift_noise,
                  noise_scale=noise_scale, rotate=rotate)


def make_reverse(seed=0, strength=0.5, n_per_time=500, dim=2,
                 method="replicate", lift_noise=0.05, noise_scale=1.0, rotate=False):
    """Bifurcation flipped (starts apart, converges = a merge), lifted to `dim`."""
    d = generate(seed=seed, n_per_time=n_per_time, strength=strength)
    obs, cln, crv, bmeans, lab = _reverse_2d(d.observed, d.clean, d.curves, d.labels)
    return GeomHD("reverse", d.tlist, obs, cln, crv, bmeans, dim=dim, seed=seed, labels=lab,
                  method=method, lift_noise=lift_noise, noise_scale=noise_scale, rotate=rotate)


def make_linear(seed=0, strength=0.5, n_per_time=500, dim=2, n_times=10, trunk_y=2.0, slope=0.5,
                method="replicate", lift_noise=0.05, noise_scale=1.0, rotate=False):
    """Single moving Gaussian on y=trunk_y+slope*t with centroid-local batch noise, lifted to `dim`."""
    cov = np.diag([0.30 ** 2, 0.28 ** 2]); L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed); tlist = np.linspace(0, n_times - 1, n_times)
    mean = lambda t: np.array([float(t), trunk_y + slope * float(t)])
    observed, clean = [], []
    for t in tlist:
        pts = (mean(t) + rng.standard_normal((n_per_time, 2)) @ L.T).astype(np.float32); clean.append(pts)
        th = np.deg2rad(rng.uniform(-10, 10) * strength)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        A = R * (1.0 + rng.uniform(-0.20, 0.20, 2) * strength); b = rng.normal(0, 0.30, 2) * np.sqrt(strength)
        c = pts.mean(0); observed.append(((pts - c) @ A.T + c + b).astype(np.float32))
    curves = np.stack([np.stack([mean(t), mean(t)], 0) for t in tlist], 0)
    return GeomHD("linear", tlist, observed, clean, curves, lambda t: (mean(t), mean(t)),
                  dim=dim, seed=seed, labels=None, method=method, lift_noise=lift_noise,
                  noise_scale=noise_scale, rotate=rotate)
