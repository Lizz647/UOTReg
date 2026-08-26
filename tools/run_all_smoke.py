#!/usr/bin/env python3
"""Run every notebook once at its shipped SMOKE=1 defaults -- the release plumbing check.

    python tools/run_all_smoke.py                 # run everything, keep going on failure
    python tools/run_all_smoke.py --only 03 04    # run only entries whose path matches
    python tools/run_all_smoke.py --list          # print the plan and exit

Each notebook's code cells are executed in order in one namespace with the notebook's folder as
cwd, headless (MPLBACKEND=Agg), with stdout+stderr captured to `_smoke_runs/<name>.log`. Notebooks
run in dependency order: the batch-effect stages are pipelines (`*_estimate` writes the `_quick`
files `*_trajectories` and `*_figs` then read). Dataset-parameterized notebooks get a second run
with `DATASET="statefate"` pre-set in the namespace.

The batch-effect `*_estimate` -> `*_trajectories` stages hand off through disk, so those four run
with `SAVE = 1`; their output goes to `new_results/`, never to the shipped tree. The figure/metric
notebooks read the shipped results and write nothing; every fitter runs at its shipped `SAVE = 0`.

Everything runs with the interpreter this script is launched with. Statefate gene-space runs are
skipped automatically when the large statefate .h5ad is absent.

Delete `new_results/` afterwards to get back to a pristine checkout.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
RELEASE = os.path.dirname(HERE)
LOGDIR = os.path.join(RELEASE, "_smoke_runs")

# The notebooks run with the SAME interpreter this script is launched with -- activate the
# environment you want tested (e.g. `conda activate uotreg`) and run `python tools/run_all_smoke.py`.
ENV_PY = {
    "torch": sys.executable,
}

SF_H5AD = os.path.join(RELEASE, "data", "scrna-statefate", "invitro-hvg.h5ad")
NO_H5AD = None if os.path.exists(SF_H5AD) else "statefate .h5ad not present (see README)"

# (notebook path relative to the release root, env key, params pre-set in the namespace or None,
#  skip-reason or None)
PLAN = [
    ("tutorials/embryoid_tutorial.ipynb", "torch", None, None),
    ("notebooks/01_simulation_outliers/run_outliers.ipynb", "torch", None, None),
    ("notebooks/01_simulation_outliers/outlier_figs.ipynb", "torch", None, None),
    ("notebooks/02_simulation_divergence/divergence_outliers.ipynb", "torch", None, None),
    ("notebooks/02_simulation_divergence/divergence_realdata.ipynb", "torch", None, None),
    ("notebooks/02_simulation_divergence/divergence_figs.ipynb", "torch", None, None),
    ("notebooks/03_simulation_batcheffect/bifurcation_estimate.ipynb", "torch", {"SAVE": 1}, None),
    ("notebooks/03_simulation_batcheffect/bifurcation_trajectories.ipynb", "torch", {"SAVE": 1}, None),
    ("notebooks/03_simulation_batcheffect/bifurcation_figs.ipynb", "torch", None, None),
    ("notebooks/04_simulation_batcheffect_reverse/reverse_estimate.ipynb", "torch", {"SAVE": 1}, None),
    ("notebooks/04_simulation_batcheffect_reverse/reverse_trajectories.ipynb", "torch", {"SAVE": 1}, None),
    ("notebooks/04_simulation_batcheffect_reverse/reverse_figs.ipynb", "torch", None, None),
    ("notebooks/05_realdata_loo/loo_embryoid.ipynb", "torch", None, None),
    ("notebooks/05_realdata_loo/loo_statefate.ipynb", "torch", None, None),
    ("notebooks/05_realdata_loo/loo_figs.ipynb", "torch", None, None),
    ("notebooks/06_realdata_analysis/estimate_and_trajectories.ipynb", "torch", None, None),
    ("notebooks/06_realdata_analysis/estimate_and_trajectories.ipynb", "torch",
     {"DATASET": "statefate"}, None),
    ("notebooks/06_realdata_analysis/fates_and_markers.ipynb", "torch", None, None),
    ("notebooks/06_realdata_analysis/fates_and_markers.ipynb", "torch", {"DATASET": "statefate"}, None),
    ("notebooks/06_realdata_analysis/figures.ipynb", "torch", None, None),
    ("notebooks/06_realdata_analysis/figures.ipynb", "torch", {"DATASET": "statefate"}, NO_H5AD),
    ("notebooks/06_realdata_analysis/cross_dimension.ipynb", "torch", None, None),
]

DRIVER = r"""
import json, sys
ns = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
cells = [ "".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]
          if c["cell_type"] == "code" and "".join(c["source"]).strip() ]
for i, src in enumerate(cells):
    print(f"### cell {i+1}/{len(cells)}", flush=True)
    exec(compile(src, f"<cell {i+1}>", "exec"), ns)
print("### notebook completed", flush=True)
"""


def cleanup_smoke_output():
    """Delete the `_quick` files the SAVE=1 pipeline stages wrote into `new_results/`.

    Only `*_quick*` is removed -- that tag is on everything a SMOKE run produces and on nothing
    else -- so a real `SMOKE = 0, SAVE = 1` run of your own is never touched. Empty directories
    (and `new_results/` itself, when nothing else is in it) are removed afterwards.
    """
    out = os.path.join(RELEASE, "new_results")
    if not os.path.isdir(out):
        return
    n = 0
    for dirpath, _dirnames, filenames in os.walk(out):
        for fn in filenames:
            if "_quick" in fn:
                os.remove(os.path.join(dirpath, fn))
                n += 1
    kept = 0
    for dirpath, _dirnames, filenames in os.walk(out, topdown=False):
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
        else:
            kept += len(filenames)
    if n:
        print(f"cleaned {n} smoke file(s) from new_results/"
              + (f" ({kept} of your own file(s) kept)" if kept else " -- now empty")
              + "   [--keep-smoke-output to keep them]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only entries whose path contains one of these substrings")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--keep-smoke-output", action="store_true",
                    help="keep the `_quick` files the SAVE=1 pipeline stages write into "
                         "new_results/ (they are deleted at the end of the run by default)")
    args = ap.parse_args()

    plan = [(nb, env, params, skip) for nb, env, params, skip in PLAN
            if not args.only or any(s in nb for s in args.only)]
    if args.list:
        for nb, env, params, skip in plan:
            tag = f" {params}" if params else ""
            print(f"{env:7s} {nb}{tag}" + (f"   [SKIP: {skip}]" if skip else ""))
        return

    for env, py in ENV_PY.items():
        if not os.path.exists(py):
            sys.exit(f"python for env {env!r} not found at {py} -- edit ENV_PY in this script")
    os.makedirs(LOGDIR, exist_ok=True)

    results = []
    for nb, env, params, skip in plan:
        name = os.path.basename(nb)[:-6] + ("_" + params["DATASET"] if params and "DATASET" in params else "")
        if skip:
            print(f"SKIP  {name:42s} {skip}")
            results.append((name, "SKIP", 0.0))
            continue
        log = os.path.join(LOGDIR, name + ".log")
        cmd = [ENV_PY[env], "-c", DRIVER, os.path.basename(nb)]
        if params:
            cmd.append(json.dumps(params))
        t0 = time.time()
        with open(log, "w") as lf:
            r = subprocess.run(cmd, cwd=os.path.join(RELEASE, os.path.dirname(nb)),
                               stdout=lf, stderr=subprocess.STDOUT,
                               env={**os.environ, "MPLBACKEND": "Agg"})
        dt = time.time() - t0
        ok = r.returncode == 0
        results.append((name, "ok" if ok else "FAIL", dt))
        print(f"{'ok   ' if ok else 'FAIL '} {name:42s} {dt:7.0f}s   ({log if not ok else ''})")
        if not ok and args.stop_on_fail:
            break

    print("\n---- summary ----")
    for name, status, dt in results:
        print(f"{status:4s}  {name:42s} {dt:7.0f}s")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n{sum(1 for _, s, _ in results if s == 'ok')} ok, {n_fail} failed, "
          f"{sum(1 for _, s, _ in results if s == 'SKIP')} skipped "
          f"({sum(dt for _, _, dt in results)/60:.0f} min total). Logs in _smoke_runs/.")

    if not args.keep_smoke_output:
        cleanup_smoke_output()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
