"""Repository root / data / results resolution, shared by every notebook folder.

This is the single copy. Every notebook puts both its own folder and this ``tools/`` directory on
``sys.path`` in its first cell, then imports ``_repo`` by bare name.

Two layouts are supported, checked in this order while walking up from this file's folder:

1. **RELEASE** -- a checkout of this repository, recognized by ``<root>/src/uotreg`` next to
   ``<root>/notebooks``. Input data lives in ``<root>/data/``.
2. **IN-PLACE** -- the author's working tree, where this folder sits beside ``newcode/`` and
   ``oldcode/``: recognized by ``<root>/newcode/src/uotreg``.

**Two results trees, and the rule that separates them.**

* ``results/`` ships with the repository and holds the paper's cached artifacts. Notebooks READ
  from it. Nothing here ever writes to it.
* ``new_results/`` is yours. Every notebook that saves anything writes THERE, and only when you
  set ``SAVE = 1`` -- so a default run leaves no files behind and can never overwrite the shipped
  results.

:func:`shipped` points into the shipped tree -- the figure/metric notebooks read it by default,
so they always redraw the paper's artifacts. :func:`results` resolves a READ that prefers
``new_results/`` when that exact path exists (used by the pipeline notebooks, whose stages hand
off through disk). :func:`out` resolves a WRITE and always lands in ``new_results/``.
"""
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))


def _find():
    p = _HERE
    for _ in range(10):
        if os.path.isdir(os.path.join(p, "src", "uotreg")) and os.path.isdir(os.path.join(p, "notebooks")):
            return p, True
        if os.path.isdir(os.path.join(p, "newcode", "src", "uotreg")):
            return p, False
        p = os.path.dirname(p)
    raise RuntimeError(
        f"cannot locate the repository root above {_HERE}: expected src/uotreg + notebooks/ "
        "(release checkout) or newcode/src/uotreg (in-place working tree)")


REPO, IS_RELEASE = _find()

# where the `uotreg` package sources live (put on sys.path by `add_paths`)
SRC_DIR = os.path.join(REPO, *(["src"] if IS_RELEASE else ["newcode", "src"]))
# the shipped results tree -- READ ONLY by convention
RESULTS_ROOT = os.path.join(REPO, *(["results"] if IS_RELEASE else ["newcode", "results_final"]))
# where anything this repository writes goes. Point it somewhere else if you like; nothing else
# needs changing, and the shipped tree stays untouched either way.
OUT_ROOT = os.path.join(REPO, "new_results")
# the input data tree (embryoid/, scrna-statefate/)
DATA_DIR = os.path.join(REPO, *(["data"] if IS_RELEASE else ["oldcode", "UOTReg", "data", "timedata"]))

# Fixed auxiliary artifacts that live outside data/timedata in the working tree (pretrained
# generator inits, the published statefate start cells, the outlier run of record). Keyed by
# their RELEASE-relative location; the value is the in-place location.
_AUX = {
    "data/ini/G_embryoid20_256_Day4_ini.pth":
        "oldcode/UOTReg/results/dynamics/embryoid/G_embryoid20_256_Day4_ini.pth",
    "data/ini/G_statefate20_256_Dayall_ini.pth":
        "oldcode/UOTReg/results/dynamics/statefate/G_statefate20_256_Dayall_ini.pth",
    "data/scrna-statefate/starting3000.npy":
        "oldcode/UOTReg/results/dynamics/statefate/starting3000.npy",
    "results/outliers_run":
        "oldcode/UOTReg/results/simu/outliernew",
}


def aux_path(rel):
    """Resolve one of the fixed auxiliary artifacts by its release-relative path (see `_AUX`)."""
    if rel not in _AUX:
        raise KeyError(f"unknown aux artifact {rel!r}; known: {sorted(_AUX)}")
    return os.path.join(REPO, *(rel if IS_RELEASE else _AUX[rel]).split("/"))


def shipped(*parts):
    """A path in the SHIPPED results tree (read-only). The figure notebooks default to this;
    point them at :func:`results` instead to prefer your own ``new_results/`` re-run."""
    return os.path.join(RESULTS_ROOT, *parts)


def results(*parts, write=False, make=False):
    """Resolve a path under the results tree.

    Read (the default): `new_results/<parts>` when that exists -- your own run wins -- otherwise
    the shipped `results/<parts>`. Write (`write=True`): always `new_results/<parts>`.
    `make=True` creates the directory (pass the parts of a DIRECTORY, not of a file).
    """
    rel = os.path.join(*parts) if parts else ""
    if write:
        p = os.path.join(OUT_ROOT, rel)
        if make:
            os.makedirs(p, exist_ok=True)
        return p
    p_new = os.path.join(OUT_ROOT, rel)
    p = p_new if os.path.exists(p_new) else os.path.join(RESULTS_ROOT, rel)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def find(*parts):
    """Resolve one FILE for reading: `new_results/<parts>` when that file is there, else the
    shipped `results/<parts>`.

    Use this instead of joining a directory from :func:`results` and appending a name. The
    directory-level check cannot tell a folder that holds YOUR figures from one that also holds the
    shipped labels, so a partially-populated `new_results/` directory would otherwise hide the file
    you actually want.
    """
    rel = os.path.join(*parts)
    p_new = os.path.join(OUT_ROOT, rel)
    return p_new if os.path.exists(p_new) else os.path.join(RESULTS_ROOT, rel)


def out(*parts, make=True):
    """Resolve a WRITE path under `new_results/`; the parent directory is created."""
    p = os.path.join(OUT_ROOT, *parts)
    if make:
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def data(*parts):
    """Join under the data tree."""
    return os.path.join(DATA_DIR, *parts)


def add_paths(*extra):
    """Put this folder (the notebook's helpers) and `src/` on sys.path, plus any extras."""
    for d in (_HERE, SRC_DIR) + tuple(extra):
        if d not in sys.path:
            sys.path.insert(0, d)
