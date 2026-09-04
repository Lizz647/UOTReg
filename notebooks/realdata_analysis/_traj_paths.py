"""Where the real-data ANALYSIS trajectories live -- one folder per method.

The same reorganization `_loo_paths.py` did for the LOO predictions, applied to the trajectories.
Everything used to sit flat in `results/realdata_analysis/<ds>/d<dim>/` with the method encoded
in the filename, which makes two different fits of "MMFM" indistinguishable once they are on disk --
and here that is not hypothetical:

    the flat `traj_<ds><dim>_MMFM.npy` files were written by `trajectory_analysis.py`'s in-file MMFM
    (`_traj_K`: hidden 128 / iters 3000 / tuples 800), NOT by the LOO settings of record
    (hidden 256 / layers 4 / iters 8000 / tuples 400 / LINEAR spline). They are a different fit of a
    different configuration under the same name.

So each method owns a folder inside its `(dataset, dim)` cell:

    <ds>/d<dim>/ours_final/      ours: UOT maps, ours: OT-CFM   <- trajectory_analysis.py
    <ds>/d<dim>/wot_final/       WOT                            <- trajectory_analysis.py
    <ds>/d<dim>/mmfm_final/      MMFM                           <- mmfm_traj.py     (`torch26`)
    <ds>/d<dim>/mioflow_final/   MioFlow                        <- mioflow_traj.py  (`mioflow2`)
    <ds>/d<dim>/tigon_final/     TIGON, TIGON (bwd)             <- tigon_traj.py, on the cluster

Resolution is per-method folder FIRST, flat directory SECOND, so every existing reader
(`trajectory_analysis`, `traj_comparison`, `cross_dimension`, `visualization`, `seed_analysis`) keeps
working against the flat files while the folders fill.

**STRICT methods never fall back.** MMFM / MioFlow / TIGON resolve ONLY inside their own folder:
* MMFM, because the flat file is the wrong configuration (above) and would silently become the
  paper's "MMFM" row;
* MioFlow and TIGON, because no flat file exists yet and a fallback could only find something stale.
Until those folders fill, those methods read as MISSING -- which is what the figure is meant to show.
Set `_strict`'s `allow` to "MMFM" (or "all") to restore the fallback for a one-off look.

**Staleness.** `organize()` COPIES the settled ours/WOT trajectories into their folders and records
the source mtime in `_provenance.json`. If `trajectory_analysis` is later re-run, the flat file moves
ahead of the copy -- `describe()` prints a STALE warning rather than letting the figure quietly draw
the older paths. Re-run `organize()` to refresh.
"""
import os
import sys
import json
import shutil
import time

_here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
# this folder (its helper modules) + tools/ (the shared `_repo.py` path resolver)
for _d in (_here, os.path.abspath(os.path.join(_here, os.pardir, os.pardir, "tools"))):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _repo

# file tag (as it appears in `traj_<ds><dim>_<tag>.npy`) -> the folder that owns it
METHOD_DIR = {
    "oursUOTmaps": "ours_final",
    "oursOT-CFM":  "ours_final",
    "WOT":         "wot_final",
    "MMFM":        "mmfm_final",
    "MioFlow":     "mioflow_final",
    "TIGON":       "tigon_final",
    "TIGONbwd":    "tigon_final",
}

# display name <-> file tag (the display names the figures and `traj_comparison` use)
PRETTY = {"oursUOTmaps": "ours: UOT maps", "oursOT-CFM": "ours: OT-CFM", "WOT": "WOT",
          "MMFM": "MMFM", "MioFlow": "MioFlow", "TIGON": "TIGON", "TIGONbwd": "TIGON (bwd)"}
TAG = {v: k for k, v in PRETTY.items()}

# the five methods the paper's trajectory figure compares, in panel order
FIGURE_METHODS = ["ours: UOT maps", "WOT", "MMFM", "MioFlow", "TIGON"]

STRICT_TAGS = {"MMFM", "MioFlow", "TIGON", "TIGONbwd"}

DATASETS = ["embryoid", "statefate"]
DIMS = [10, 20, 50]
PROV = "_provenance.json"


def _strict(tag):
    allow = ""
    if allow.strip().lower() == "all":
        return False
    return tag in STRICT_TAGS and tag not in {a.strip() for a in allow.split(",")}


def results_root(repo=None, write=False, shipped=False):
    """Root of the analysis results tree. `write=True` resolves into `new_results/`;
    `shipped=True` pins the read to the shipped paper tree."""
    if shipped:
        return _repo.shipped("realdata_analysis")
    return _repo.results("realdata_analysis", write=write)


def cell_dir(root, ds, dim):
    return os.path.join(root, ds, f"d{dim}")


def fname(ds, dim, tag):
    return f"traj_{ds}{dim}_{tag}.npy"


def traj_path(root, ds, dim, tag):
    """Per-method folder if the file is there, else the flat cell directory.

    For a STRICT tag the per-method path is returned even when nothing is there, so a caller
    reporting it as missing points at the folder that is still empty rather than at a flat file that
    must not be used."""
    cell = cell_dir(root, ds, dim)
    sub = METHOD_DIR.get(tag)
    if sub:
        p = os.path.join(cell, sub, fname(ds, dim, tag))
        if os.path.exists(p) or _strict(tag):
            return p
    return os.path.join(cell, fname(ds, dim, tag))


def out_dir(root, ds, dim, tag):
    """Where a NEW run must write. Creates it. Use this in the fitting scripts."""
    d = os.path.join(cell_dir(root, ds, dim), METHOD_DIR.get(tag, ""))
    os.makedirs(d, exist_ok=True)
    return d


def load(root, ds, dim, tag):
    """(array, path) or (None, path) -- np.load of the resolved trajectory if it exists."""
    import numpy as np
    p = traj_path(root, ds, dim, tag)
    return (np.load(p) if os.path.exists(p) else None), p


# ----------------------------------------------------------------------------- one-shot organizer
def organize(root, tags=("oursUOTmaps", "oursOT-CFM", "WOT"), datasets=None, dims=None, dry=False):
    """Create every method folder, and COPY the settled flat trajectories into theirs.

    Only non-STRICT tags are copied: MMFM's flat file is a different configuration and MioFlow/TIGON
    have none, so their folders are created EMPTY on purpose. Idempotent -- a copy that is already
    current is skipped; a flat file that has moved ahead is re-copied."""
    datasets = datasets or DATASETS
    dims = dims or DIMS
    n_new = n_fresh = n_skip = 0
    for ds in datasets:
        for dim in dims:
            cell = cell_dir(root, ds, dim)
            if not os.path.isdir(cell):
                continue
            for sub in sorted(set(METHOD_DIR.values())):          # every folder, even ones left empty
                os.makedirs(os.path.join(cell, sub), exist_ok=True)
            for tag in tags:
                src = os.path.join(cell, fname(ds, dim, tag))
                if not os.path.exists(src):
                    n_skip += 1
                    continue
                dst_dir = os.path.join(cell, METHOD_DIR[tag])
                dst = os.path.join(dst_dir, fname(ds, dim, tag))
                prov_fp = os.path.join(dst_dir, PROV)
                prov = json.load(open(prov_fp)) if os.path.exists(prov_fp) else {}
                rec = prov.get(fname(ds, dim, tag))
                src_mt = os.path.getmtime(src)
                if os.path.exists(dst) and rec and abs(rec.get("src_mtime", -1) - src_mt) < 1:
                    n_fresh += 1
                    continue
                print(f"  {'[dry] ' if dry else ''}{ds} d={dim}: {tag} -> {METHOD_DIR[tag]}/"
                      + ("  (refresh: flat moved ahead)" if os.path.exists(dst) else ""))
                if not dry:
                    shutil.copy2(src, dst)
                    prov[fname(ds, dim, tag)] = {
                        "copied_from": os.path.relpath(src, root), "src_mtime": src_mt,
                        "src_mtime_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(src_mt)),
                        "bytes": os.path.getsize(src),
                        "producer": "trajectory_analysis.py (FIT section)"}
                    json.dump(prov, open(prov_fp, "w"), indent=1)
                n_new += 1
    print(f"  organize: {n_new} copied, {n_fresh} already current, {n_skip} flat files absent")
    return n_new


# ----------------------------------------------------------------------------- status report
def describe(root, methods=None, datasets=None, dims=None, verbose=True):
    """Per method: how many of the 6 (dataset, dim) cells are on disk, and where they resolved.

    This is the "what is usable right now" table. `stale` counts folder copies whose flat source has
    since been rewritten -- those must be refreshed with `organize()` before they are trusted."""
    methods = methods or list(PRETTY.values())
    datasets = datasets or DATASETS
    dims = dims or DIMS
    rows = {}
    if verbose:
        print(f"  root = {root}")
        print(f"  {'method':16s}{'folder':15s}{'found':>6}{'flat':>6}{'stale':>6}{'missing':>8}   cells")
    for m in methods:
        tag = TAG.get(m, m)
        sub = METHOD_DIR.get(tag, "(flat)")
        infolder = flat = stale = 0
        gaps, have = [], []
        for ds in datasets:
            for dim in dims:
                cell = cell_dir(root, ds, dim)
                p = traj_path(root, ds, dim, tag)
                if not os.path.exists(p):
                    gaps.append(f"{ds}{dim}")
                    continue
                have.append(f"{ds}{dim}")
                if os.path.dirname(p) == cell:
                    flat += 1
                else:
                    infolder += 1
                    prov_fp = os.path.join(os.path.dirname(p), PROV)
                    rec = (json.load(open(prov_fp)).get(os.path.basename(p))
                           if os.path.exists(prov_fp) else None)
                    src = os.path.join(cell, os.path.basename(p))
                    if rec and os.path.exists(src) and os.path.getmtime(src) - rec.get("src_mtime", 0) > 1:
                        stale += 1
        rows[m] = dict(folder=sub, found=infolder + flat, flat=flat, stale=stale,
                       missing=len(gaps), have=have, gaps=gaps, strict=_strict(tag))
        if verbose:
            note = (", ".join(have) if have else "-- none --")
            if _strict(tag) and gaps:
                note += "   [strict: no flat fallback]"
            print(f"  {m:16s}{sub:15s}{infolder + flat:>6}{flat:>6}{stale:>6}{len(gaps):>8}   {note}")
    if verbose:
        print("\n  found = resolvable now | flat = still read from the legacy cell directory"
              "\n  stale = a folder copy whose flat source was rewritten later -- re-run organize()"
              "\n  MMFM/MioFlow/TIGON are STRICT: they never read flat, so an empty folder reads as"
              "\n  missing and the figure draws a 'run pending' panel instead of the wrong fit.")
    return rows
