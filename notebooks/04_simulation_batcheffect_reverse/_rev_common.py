"""Shared, env-portable helpers for the REVERSE-NOISE batch-effect benchmark.

Same contract as the bifurcation benchmark's `_be_common.py` (numpy-only, so it imports in any
env), pointed at this experiment's own results tree:

    results/batcheffectreverse/
        baselines/data/reverse_d{dim}_seed{k}_data.npz     <- the shared data every method fits on
        <METHOD>/reverse_d{dim}_seed{k}_{METHOD}_traj.npy  <- one folder per baseline method
        reverse_d{dim}_new_seed{k}_est.npz / _trajs.npz / _metrics.json

The main file writes the data npz; each baseline loads it, fits on `raw_series`, transports the
shared `X0`, and saves a full-dim `(T, N, dim)` trajectory.
"""
import json
import os
import numpy as np

# Snapshot indices used as the trajectory grid. NOTE this variant starts at t=0, one time point MORE
# in the first stage than the bifurcation benchmark (which uses 1..8): the first stage is where the
# batch effect lives here, so it gets 5 snapshots (t=0..4) rather than 4.
TRAJ_T = list(range(0, 9))

# Times whose distribution is ESTIMATED. t=0 is deliberately NOT estimated: it is the trajectory's
# starting cloud, taken OBSERVED, exactly as the bifurcation benchmark does. So the estimator runs on
# t=1..8 and the series handed to the trajectory fitter is  [observed t=0] + [estimated t=1..8],
# which is len(TRAJ_T) long. `est_series` in the saved npz therefore has len(EST_T) = 8 entries, and
# `est_t` is stored alongside it so a loader never has to guess the alignment.
EST_T = TRAJ_T[1:]

# The canonical seed set for this benchmark. The reverse-noise runs use 10-19, distinct from the
# bifurcation benchmark's 0-9, so the two never collide in a shared results tree.
SEEDS = list(range(10, 20))

# the reverse-noise data knobs -- kept here so every file (and env) agrees on the data.
# EDIT THESE TWO to retune the simulation; everything downstream reads them (and both are echoed
# into every saved npz, so an older run still rebuilds its own data).
DELTA_PRE = 0.6           # trunk margin (bifurcation: 0.25)

STRENGTH_SCHEDULE = {     # per-time affine batch strength (bifurcation: flat 0.5)
    0: 0.5, 1: 1.5, 2: 1.5,        # first stage: the noisy one
    3: 1.0,                        # tapering into the split
    4: 0.65,                       # the split time itself (geometrically already stage 2)
    "default": 0.25,               # t >= 5: the clean second stage
}


def schedule_to_json(sch):
    """npz cannot hold a dict -> store the schedule as a JSON string."""
    return json.dumps({str(k): float(v) for k, v in sch.items()})


def schedule_from_json(txt):
    """Parse back, restoring the int keys that JSON stringified and keeping "default"."""
    raw = json.loads(str(txt))
    return {(int(k) if str(k).lstrip("-").isdigit() else k): float(v) for k, v in raw.items()}

RESULTS_DIRNAME = "batcheffectreverse"

import sys
_here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
# this folder (its helper modules) + tools/ (the shared `_repo.py` path resolver)
for _d in (_here, os.path.abspath(os.path.join(_here, os.pardir, os.pardir, "tools"))):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _repo


def find_root():
    return _repo.REPO


def _bdir(write=False):
    d = _repo.results(RESULTS_DIRNAME, "baselines", write=write)
    if write:
        os.makedirs(os.path.join(d, "data"), exist_ok=True)
    return d


def data_path(dim, seed, quick=False, write=False, root=None):
    """Path to the shared per-seed data npz. `write=True` -> under `new_results/`; `root=` pins
    the read to one results tree (the figures notebook passes the shipped tree)."""
    q = "_quick" if quick else ""
    base = os.path.join(root, "baselines") if root else _bdir(write)
    return os.path.join(base, "data", f"reverse_d{dim}_seed{seed}{q}_data.npz")


def traj_path(dim, seed, method, quick=False, write=False, root=None):
    """`batcheffectreverse/<method>/reverse_d{dim}_seed{seed}_{method}_traj.npy`."""
    q = "_quick" if quick else ""
    d = os.path.join(root, method) if root else _repo.results(RESULTS_DIRNAME, method,
                                                              write=write, make=write)
    return os.path.join(d, f"reverse_d{dim}_seed{seed}{q}_{method}_traj.npy")


def load_data(dim, seed, quick=False, root=None):
    """dict(raw_series [list of (n,dim)], times, X0 (n_start,dim), labels0|None, dim, + data knobs).

    `quick` MUST match the setting the exporter ran with: `QUICK=True` writes `..._quick_data.npz`
    and `QUICK=False` writes `..._data.npz`, so a mismatch looks like missing data. The error below
    says which files actually exist rather than just naming the one that does not."""
    p = data_path(dim, seed, quick, root=root)
    if not os.path.exists(p):
        d = os.path.dirname(p)
        have = sorted(f for f in os.listdir(d) if f.endswith("_data.npz")) if os.path.isdir(d) else []
        q = [f for f in have if "_quick_" in f]
        r = [f for f in have if "_quick_" not in f]
        raise FileNotFoundError(
            f"no exported data at {p}\n"
            f"  QUICK={quick} -> looking for {'_quick_data.npz' if quick else '_data.npz'} files.\n"
            f"  present in {d}:\n"
            f"    {len(r)} full : {', '.join(f.split('_seed')[1].split('_')[0] for f in r) or '(none)'}\n"
            f"    {len(q)} quick: {', '.join(f.split('_seed')[1].split('_')[0] for f in q) or '(none)'}\n"
            f"  Fix: set QUICK to match the run that exported the data "
            f"(Section 1 of reverse_estimate, run with SAVE = 1), or re-export this seed there.")
    z = np.load(p)
    rs = z["raw_series"]
    out = dict(raw_series=[np.asarray(rs[i], np.float32) for i in range(rs.shape[0])],
               times=[float(t) for t in z["times"]], X0=np.asarray(z["X0"], np.float32),
               labels0=(np.asarray(z["labels0"]) if z["labels0"].size else None), dim=int(z["dim"]))
    for k in ("delta_pre", "n_per", "n_start"):
        if k in z.files:
            out[k] = float(z[k])
    if "schedule" in z.files:
        out["schedule"] = schedule_from_json(z["schedule"])
    return out


def save_traj(dim, seed, method, traj, quick=False):
    p = traj_path(dim, seed, method, quick, write=True)
    np.save(p, np.asarray(traj, np.float32))
    return p


def project2d(X, dim=None):
    """d-dim points/trajectories -> the 2-D signal plane, numpy-only (== GeomHD.project2d for the
    default `method="replicate", rotate=False` lift). Raw dims 0,1 are BOTH copies of signal-x, so
    always plot through this."""
    X = np.asarray(X, float)
    d = X.shape[-1] if dim is None else int(dim)
    if d == 2:
        return X[..., :2]
    block = d // 2
    return np.stack([X[..., :block].mean(-1), X[..., block:2 * block].mean(-1)], -1)
