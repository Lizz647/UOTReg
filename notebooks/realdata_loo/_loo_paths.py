"""Where the real-data LOO artifacts live. Used by `loo_figs` (metrics + figures).

Predictions used to sit in ONE flat directory, `results/realdata/`, with the method encoded in
the filename. That made a re-run of one method able to silently overwrite another's rows (it already
happened: `mmfm_*.py` saves "MMFM fwd" and "MMFM bwd" to the same `pred_..._MMFM.npy`), and it made
provenance invisible -- two of the twelve `pred_*_MioFlow.npy` files were written by the NEW-API
script and ten by the old one, distinguishable only by mtime.

So each settled method now owns a folder:

    results/realdata/ours_final/      UOTReg, OT       <- loo_{embryoid,statefate}.py
    results/realdata/naive_final/     Naive1, Naive2   <- same driver
    results/realdata/MMFM_final/      MMFM             <- baselines/mmfm_*.py
    results/realdata/mioflow_final/   MioFlow          <- baselines/mioflow_*_old.py

    results/realdata/tigon_final/     TIGON + read-out variants  <- baselines/tigon_loo_final.py

**TIGON's folder is mapped but may still be empty** -- its production run is on the cluster. Until
those files land the resolver falls through to the flat directory, so the old rows keep working and
the new ones take over automatically the moment they are pulled back.

Resolution order is per-method folder FIRST, flat directory SECOND. So a half-migrated tree reads
correctly, and a method that has not been reorganized behaves exactly as before.

TIGON is the ONE exception: it resolves STRICTLY inside `tigon_final/` (see `STRICT_METHODS`).
"""
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
# this folder (its helper modules) + tools/ (the shared `_repo.py` path resolver)
for _d in (_here, os.path.abspath(os.path.join(_here, os.pardir, os.pardir, "tools"))):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _repo

# method key -> subdirectory of results/realdata/. Methods absent from this map resolve flat.
METHOD_DIR = {
    "UOTReg":  "ours_final",
    "OT":      "ours_final",
    "Naive1":  "naive_final",
    "Naive2":  "naive_final",
    "MMFM":    "MMFM_final",
    "MioFlow": "mioflow_final",
    # TIGON's production run (`baselines/tigon_loo_final.py`) writes here.
    "TIGON":       "tigon_final",
    "TIGONfwd":    "tigon_final",
    "TIGONnbr":    "tigon_final",
    "TIGONnbrfwd": "tigon_final",
}

# Methods that resolve ONLY inside their own folder -- never the flat fallback.
#
# The flat directory holds ~70 exploratory TIGON predictions from the diagnosis and calibration work:
# the unconditioned runs that returned their own input cloud, plus every sigma-sweep variant. They are
# named `pred_<ds><dim>_Day<h>_TIGON.npy` -- exactly the name the production run will use -- so a flat
# fallback would quietly put a stale, differently-configured number in the paper's table under the
# name "TIGON". Until `tigon_final/` fills from the cluster, TIGON reads as MISSING, which is what the
# table and figures are meant to show.
#
# Set `_strict`'s `allow` to "TIGON,TIGONfwd" (or "all") to restore the old fallback for a
# one-off look at the exploratory rows.
STRICT_METHODS = {"TIGON", "TIGONfwd", "TIGONnbr", "TIGONnbrfwd"}


def _strict(method):
    allow = ""
    if allow.strip().lower() == "all":
        return False
    return method in STRICT_METHODS and method not in {a.strip() for a in allow.split(",")}

# the grid every reader in this folder iterates over
DATASETS = ["embryoid", "statefate"]
DIMS     = [10, 20, 50]
HELD     = {"embryoid": [7.5, 13.5, 19.5], "statefate": [4.0]}


def results_root(repo=None, write=False, shipped=False):
    """The ROOT of the realdata results tree (the flat dir that holds the per-method folders).

    `write=True` resolves into `new_results/`; `shipped=True` pins the read to the shipped paper
    tree; the default read prefers your own run and falls back to shipped (see `_repo.results`)."""
    if shipped:
        return _repo.shipped("realdata")
    return _repo.results("realdata", write=write)


def stem(dataset, dim, held, method):
    return f"{dataset}{dim}_Day{held}_{method}"


def _resolve(root, method, fname):
    """Per-method folder if the file is there, else the flat legacy directory.

    The flat path is also what comes back when the file exists in NEITHER place, so callers get a
    sensible name to report as missing -- except for a STRICT method, where the returned name is the
    per-method one, so "missing" reads as "its folder is still empty" rather than pointing at a flat
    file that must not be used."""
    sub = METHOD_DIR.get(method)
    if sub:
        p = os.path.join(root, sub, fname)
        if os.path.exists(p) or _strict(method):
            return p
    return os.path.join(root, fname)


def pred_path(root, dataset, dim, held, method):
    return _resolve(root, method, f"pred_{stem(dataset, dim, held, method)}.npy")


def metrics_path(root, dataset, dim, held, method):
    return _resolve(root, method, f"metrics_{stem(dataset, dim, held, method)}.json")


def out_dir(root, method):
    """Where a NEW run of `method` must write. Creates it. Use this in the baseline scripts."""
    d = os.path.join(root, METHOD_DIR.get(method, ""))
    os.makedirs(d, exist_ok=True)
    return d


def describe(root, methods=("UOTReg", "OT", "Naive1", "Naive2", "MMFM", "MioFlow", "TIGON"),
             datasets=None, dims=None, held=None):
    """Print where each method actually resolves, and how many of its grid cells are on disk.

    Run this after reorganizing or after a re-run -- it is the check that the folders are right."""
    datasets = datasets or DATASETS
    dims = dims or DIMS
    held = held or HELD
    print(f"  root = {root}")
    print(f"  {'method':10s}{'folder':16s}{'found':>7}{'flat':>7}{'missing':>9}   cells missing")
    for m in methods:
        sub = METHOD_DIR.get(m, "(flat)")
        infolder = flat = 0
        gaps = []
        for ds in datasets:
            for dim in dims:
                for h in held[ds]:
                    p = pred_path(root, ds, dim, h, m)
                    if not os.path.exists(p):
                        gaps.append(f"{ds}{dim}/Day{h}")
                    elif os.path.dirname(p) == root:
                        flat += 1
                    else:
                        infolder += 1
        note = ", ".join(gaps[:4]) + (f" (+{len(gaps)-4})" if len(gaps) > 4 else "")
        if _strict(m):
            note = (note + "   [strict: flat fallback OFF]") if gaps else note
        print(f"  {m:10s}{sub:16s}{infolder:>7}{flat:>7}{len(gaps):>9}   {note}")
    print("\n  'flat' = the per-method folder has no file for that cell, so the legacy directory "
          "was used.\n  Anywhere it appears now, a re-run wrote to the old place -- every settled "
          "method owns a folder.\n  TIGON is strict (see STRICT_METHODS): it never reads flat, so "
          "until the cluster run is\n  pulled back into tigon_final/ it reports its cells as "
          "missing, and the table/figures\n  leave it blank rather than showing the exploratory "
          "calibration runs.")
