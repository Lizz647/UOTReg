"""Shared, env-portable helpers for the batch-effect BASELINE files (TIGON / MioFlow / raw WOT+MMFM).

The trajectory metrics need the `GeomHD` object from `uotreg.simulation`, which only imports in the
`mmfm_clean` env -- NOT in the `tigon` / `mioflow` envs. So the split is:

  * `export_data_bifurcation.py` (mmfm_clean) generates the data DETERMINISTICALLY from `sim` and saves
    one npz per (dim, seed): `raw_series` (the snapshots the baselines fit on), `X0`/`labels0` (the shared
    start cells), `times`. This is the single source of truth -> every method sees IDENTICAL data.
  * the baseline files LOAD that npz, fit on `raw_series` (RAW snapshots, not our estimates), transport
    `X0` -> a FULL-DIM trajectory `(T, N, dim)`, and save it with `save_traj`. No metrics/plots there.
  * `raw_wot_mmfm_bifurcation.py` (mmfm_clean) reconstructs `G` from the seed (deterministic == the export)
    and SCORES + PLOTS every saved trajectory with the same `trajectory_metrics` our method uses.

This module is numpy-only so it imports in all three envs.
"""
import os
import numpy as np

TRAJ_T = list(range(1, 9))     # snapshot indices used as the trajectory grid (matches batcheffect_bifurcation)
STRENGTH = 0.5


import sys
_here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
# this folder (its helper modules) + tools/ (the shared `_repo.py` path resolver)
for _d in (_here, os.path.abspath(os.path.join(_here, os.pardir, os.pardir, "tools"))):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _repo


def find_root():
    return _repo.REPO


# Where trajectories are WRITTEN. Edit to redirect, so a final production run can land in its
# own folder without touching the exploratory tree.
RESULTS_DIRNAME = "batcheffectnew"
# Where the shared per-seed data npz is READ from -- deliberately NOT redirected with the output, so
# a redirected run still fits the identical data every other method saw.
DATA_DIRNAME = "batcheffectnew"


def _bdir(write=False):
    d = _repo.results(DATA_DIRNAME, "baselines", write=write)
    if write:
        os.makedirs(os.path.join(d, "data"), exist_ok=True)
    return d


def data_path(dim, seed, quick=False, write=False, root=None):
    """Path to the shared per-seed data npz. `write=True` -> under `new_results/`; `root=` pins
    the read to one results tree (the figures notebook passes the shipped tree)."""
    q = "_quick" if quick else ""
    base = os.path.join(root, "baselines") if root else _bdir(write)
    return os.path.join(base, "data", f"bifurcation_d{dim}_seed{seed}{q}_data.npz")


# Baseline methods live under `baselines/<method>/`, ours under `ours/`, so the tree says at a
# glance which rows are competitors and which are the paper's own. `METHOD_DIR` maps a method name
# to its folder; anything unlisted keeps the flat `<method>/` layout (the TIGONs_* sweep variants).
BASELINE_METHODS = {"WOT": "baselines/WOT", "MMFM": "baselines/MMFM",
                    "MioFlow": "baselines/mioflow",          # currently the v2 (new-API) rows
                    "MioFlowOld": "baselines/mioflow_old",   # old-API refits, kept SEPARATE so the
                    #                                          two can be compared before deciding
                    "TIGON": "baselines/tigon",
                    "TIGONfwd": "baselines/tigon"}


def method_dir(method, write=False):
    """Absolute directory a method's trajectories belong in."""
    sub = BASELINE_METHODS.get(method, method)
    return _repo.results(RESULTS_DIRNAME, *sub.split("/"), write=write, make=write)


def traj_path(dim, seed, method, quick=False, write=False, root=None):
    """One method's trajectory: `<root>/baselines/<method>/bifurcation_d{dim}_seed{seed}_{method}_traj.npy`."""
    q = "_quick" if quick else ""
    d = os.path.join(root, *BASELINE_METHODS.get(method, method).split("/")) if root \
        else method_dir(method, write)
    return os.path.join(d, f"bifurcation_d{dim}_seed{seed}{q}_{method}_traj.npy")


def ours_dir(write=False):
    """`batcheffectnew/ours/` -- the paper's own rows: per-seed estimated distributions,
    fitted trajectories, metrics, and the trained generators."""
    return _repo.results(RESULTS_DIRNAME, "ours", write=write, make=write)


def load_data(dim, seed, quick=False, root=None):
    """Return dict(raw_series [list of (n,dim)], times [list], X0 (n_start,dim), labels0 (n_start,)|None, dim)."""
    z = np.load(data_path(dim, seed, quick, root=root))
    rs = z["raw_series"]
    return dict(raw_series=[np.asarray(rs[i], np.float32) for i in range(rs.shape[0])],
                times=[float(t) for t in z["times"]], X0=np.asarray(z["X0"], np.float32),
                labels0=(np.asarray(z["labels0"]) if z["labels0"].size else None), dim=int(z["dim"]))


def save_traj(dim, seed, method, traj, quick=False):
    p = traj_path(dim, seed, method, quick, write=True)
    np.save(p, np.asarray(traj, np.float32))
    return p


def project2d(X, dim=None):
    """d-dim points/trajectories -> the 2-D signal plane, numpy-only (== GeomHD.project2d for the
    default `method="replicate", rotate=False` lift: dims [0:block] are copies of signal-x and
    [block:2*block] copies of signal-y, block = dim//2). Works on (..., d) arrays.

    NOTE for viz: raw dims 0,1 are BOTH copies of x -> plotting them gives a diagonal line, not the
    bifurcation. Always plot through this projection."""
    X = np.asarray(X, float)
    d = X.shape[-1] if dim is None else int(dim)
    if d == 2:
        return X[..., :2]
    block = d // 2
    return np.stack([X[..., :block].mean(-1), X[..., block:2 * block].mean(-1)], -1)
