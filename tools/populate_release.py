#!/usr/bin/env python3
"""Populate (or re-sync) this release folder's `data/` and `results/` from the working tree.

Run from anywhere inside the author's working tree (the release folder must sit beside
`newcode/` and `oldcode/`):

    python tools/populate_release.py            # copy data + aux + the results manifest
    python tools/populate_release.py --dry-run  # print what would be copied and the sizes
    python tools/populate_release.py --checksum # md5-verify every shipped file against its source
    python tools/populate_release.py --prune    # delete release results/ files the manifest does not ship
    python tools/populate_release.py --with-statefate-h5ad   # also copy the ~860 MB statefate .h5ad

Copies are one-way (working tree -> release) and idempotent: a file is re-copied only when the
source is newer or the sizes differ. That heuristic cannot see a same-size overwrite with a NEWER
mtime (it happened: a SMOKE run's output was moved over shipped artifacts) -- `--checksum` compares
content and restores any mismatch; `--prune` removes files under `results/` that the manifest does
not produce. `--checksum --prune --dry-run` is the full integrity report.

This script is a RELEASE-BUILD tool for the author's machine; it is not needed (and will refuse
to run) in a standalone checkout, where `data/` and `results/` ship pre-populated.
"""
import argparse
import fnmatch
import os
import hashlib
import shutil
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
RELEASE = os.path.dirname(HERE)
TREE = os.path.dirname(RELEASE)              # the folder holding newcode/ + oldcode/
NEWRES = os.path.join(TREE, "newcode", "results_final")
OLDUOT = os.path.join(TREE, "oldcode", "UOTReg")

# ---------------------------------------------------------------- data + aux (fixed, small)
# release-relative destination <- working-tree source (relative to TREE)
DATA = {
    "data/embryoid/embryoid_data.h5ad":      "oldcode/UOTReg/data/timedata/embryoid/embryoid_data.h5ad",
    "data/embryoid/embryoid_pc.npy":         "oldcode/UOTReg/data/timedata/embryoid/embryoid_pc.npy",
    "data/embryoid/embryoid_pc_20.npy":      "oldcode/UOTReg/data/timedata/embryoid/embryoid_pc_20.npy",
    "data/embryoid/time_labels.npy":         "oldcode/UOTReg/data/timedata/embryoid/time_labels.npy",
    "data/scrna-statefate/statefate_pc.npy": "oldcode/UOTReg/data/timedata/scrna-statefate/statefate_pc.npy",
    "data/scrna-statefate/statefate_pc20.npy": "oldcode/UOTReg/data/timedata/scrna-statefate/statefate_pc20.npy",
    "data/scrna-statefate/time_labels.npy":  "oldcode/UOTReg/data/timedata/scrna-statefate/time_labels.npy",
    # aux artifacts (see notebooks' _repo.py `_AUX`)
    "data/ini/G_embryoid20_256_Day4_ini.pth":    "oldcode/UOTReg/results/dynamics/embryoid/G_embryoid20_256_Day4_ini.pth",
    "data/ini/G_statefate20_256_Dayall_ini.pth": "oldcode/UOTReg/results/dynamics/statefate/G_statefate20_256_Dayall_ini.pth",
    "data/scrna-statefate/starting3000.npy":     "oldcode/UOTReg/results/dynamics/statefate/starting3000.npy",
}
STATEFATE_H5AD = ("data/scrna-statefate/invitro-hvg.h5ad",
                  "oldcode/UOTReg/data/timedata/scrna-statefate/invitro-hvg.h5ad")

# the outlier run of record: the paper's trained generators, read (never written) by outlier_viz
DIRS = {
    "results/outliers_run": "oldcode/UOTReg/results/simu/outliernew",
}

# ---------------------------------------------------------------- cached results (manifest)
# Each entry: (release-relative destination dir, source dir under newcode/results_final,
#              [include glob patterns, matched against the path relative to the source dir]).
# The manifest is READ-DRIVEN: it ships exactly what the notebooks load (audited 2026-08-25),
# not what the working tree holds. Deliberately excluded, with the reason:
#   * dm_*.npy DM caches (~470 MB)      -- cross_dimension recomputes them from the trajectory
#   * model_*.pth / est_G_*.pth / ckpt* -- retraining artifacts nothing in the release reads
#     (exception: the d20 WOT models, which spare the tune notebooks a full WOT refit)
#   * hist_*.json, logs, *.out/.err, .prev, TIGONs_* / TIGONnbr* sweep variants -- exploratory
#   * figures/** except the two labels_published_*.npy + four de_*_full.csv the notebooks READ
#   * anything *_quick* -- smoke output never ships
RESULTS = [
    # 02: divergence_figs reads metrics + samples + preds (not the G_*.pth generators)
    ("results/divergence", "divergence",
     ["*_metrics.json", "*_preds.npz", "outliers_samples.npz"]),
    # 03: the bifurcation pipeline's own est/trajs live at the tree ROOT; per-seed shared data
    #     and the four baseline trajectory sets live under baselines/
    ("results/batcheffectnew", "batcheffectnew",
     ["bifurcation_d10_new_seed*_est.npz", "bifurcation_d10_new_seed*_trajs.npz",
      "bifurcation_d10_paper_metrics.json"]),
    ("results/batcheffectnew/baselines", "batcheffectnew/baselines",
     ["data/bifurcation_d10_seed*_data.npz", "WOT/*_WOT_traj.npy", "MMFM/*_MMFM_traj.npy",
      "mioflow_old/*_MioFlowOld_traj.npy", "tigon/*_TIGON_traj.npy", "tigon/*_TIGONfwd_traj.npy"]),
    # 04: same layout, reverse-noise benchmark (seeds 10-19; methods are FLAT folders here)
    ("results/batcheffectreverse", "batcheffectreverse",
     ["reverse_d10_new_seed*_est.npz", "reverse_d10_new_seed*_trajs.npz",
      "reverse_d10_paper_metrics.json",
      "baselines/data/reverse_d10_seed*_data.npz", "WOT/*_traj.npy", "MMFM/*_traj.npy",
      "MioFlow/*_traj.npy", "TIGON/*_traj.npy", "TIGONfwd/*_traj.npy"]),
    # 05: the LOO benchmark -- per-method prediction folders + the scored csv/summary
    ("results/realdata", "realdata",
     ["ours_final/pred_*.npy", "naive_final/pred_*.npy",
      "MMFM_final/pred_*.npy", "MMFM_final/metrics_*.json",
      "mioflow_final/pred_*.npy", "mioflow_final/metrics_*.json",
      "tigon_final/pred_*.npy", "tigon_final/metrics_*.json",
      "benchmark_detailed.csv", "benchmark_summary.json"]),
    # 06: estimated clouds (all dims -- small), d20 baseline trajectory folders, the tune-seed
    #     runs cross_dimension + the figures read (traj/clusters/startidx/meta; no dm, no model),
    #     the seed_selection labels + DE caches, and the d20 WOT models (spare a refit)
    ("results/realdata_analysis", "realdata_analysis",
     ["*/d10/est_cloud_Day*.npy", "*/d20/est_cloud_Day*.npy", "*/d50/est_cloud_Day*.npy",
      "*/d20/*_final/traj_*.npy", "*/d20/*_final/meta_*.json", "*/d20/*_final/startidx_*.npy",
      "*/d20/*_final/_provenance.json",
      "*/d20/traj_meta_*.json", "*/d20/metrics_bars_*.json", "*/d20/model_*_WOT.pth",
      "*/d*_tune/seed*/traj_*.npy", "*/d*_tune/seed*/clusters_*.npy",
      "*/d*_tune/seed*/startidx_*.npy", "*/d*_tune/seed*/meta_*.json",
      "*/seed_selection/*"]),
    # chain artifacts: written by traj_figs / de_tables, read by gene_dynamics / marker_tables --
    # shipped so the notebooks run in any order
    ("results/figures/realdata", "figures/realdata",
     ["trajs_comparisons/labels_published_embryoid20_seed5.npy",
      "trajs_comparisons/labels_published_statefate20_seed2.npy",
      "tables/de_embryoid20_seed5_final_full.csv", "tables/de_embryoid20_seed5_initial_full.csv",
      "tables/de_statefate20_seed2_final_full.csv", "tables/de_statefate20_seed2_initial_full.csv"]),
]

# never ship: smoke/quick output, previous-save backups
EXCLUDE_SUBSTRINGS = ("_quick", "_smoke", ".prev")


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src, dst, dry, stats, checksum=False):
    if not os.path.exists(src):
        stats["missing"].append(src)
        return
    if os.path.exists(dst):
        if checksum:
            if _md5(dst) == _md5(src):
                stats["kept"] += 1
                return
            stats["mismatch"].append(os.path.relpath(dst, RELEASE))
        elif os.path.getsize(dst) == os.path.getsize(src) \
                and os.path.getmtime(dst) >= os.path.getmtime(src):
            stats["kept"] += 1
            return
    stats["copied"] += 1
    stats["bytes"] += os.path.getsize(src)
    if not dry:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _copy_dir(srcd, dstd, patterns, dry, stats, checksum=False):
    for dirpath, _dirnames, filenames in os.walk(srcd):
        rel = os.path.relpath(dirpath, srcd)
        for fn in filenames:
            if fn.startswith(".") or any(s in fn for s in EXCLUDE_SUBSTRINGS):
                continue
            relf = os.path.normpath(os.path.join(rel, fn))
            if patterns and not any(fnmatch.fnmatch(relf, p) for p in patterns):
                continue
            stats["expected"].add(os.path.relpath(os.path.join(dstd, relf), RELEASE))
            _copy(os.path.join(dirpath, fn), os.path.join(dstd, relf), dry, stats, checksum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-statefate-h5ad", action="store_true",
                    help="also copy the ~860 MB statefate .h5ad (needed only for statefate at "
                         "d=50 and for de_tables/gene_dynamics on statefate)")
    ap.add_argument("--checksum", action="store_true",
                    help="compare CONTENT (md5) instead of size+mtime; restores any shipped file "
                         "that no longer matches its working-tree source")
    ap.add_argument("--prune", action="store_true",
                    help="delete files under the release's results/ that the manifest does not "
                         "produce (SMOKE deposits, figures, stray models)")
    args = ap.parse_args()

    if not os.path.isdir(NEWRES) or not os.path.isdir(OLDUOT):
        sys.exit("this release folder does not sit beside newcode/ and oldcode/ -- nothing to "
                 "populate from (a standalone checkout ships data/ and results/ already).")

    stats = {"copied": 0, "kept": 0, "bytes": 0, "missing": [], "mismatch": [], "expected": set()}
    todo = dict(DATA)
    if args.with_statefate_h5ad:
        todo[STATEFATE_H5AD[0]] = STATEFATE_H5AD[1]
    for dst_rel, src_rel in sorted(todo.items()):
        stats["expected"].add(dst_rel)
        _copy(os.path.join(TREE, src_rel), os.path.join(RELEASE, dst_rel), args.dry_run, stats,
              args.checksum)
    for dst_rel, src_rel in sorted(DIRS.items()):
        _copy_dir(os.path.join(TREE, src_rel), os.path.join(RELEASE, dst_rel), None,
                  args.dry_run, stats, args.checksum)
    for dst_rel, src_rel, patterns in RESULTS:
        _copy_dir(os.path.join(NEWRES, src_rel), os.path.join(RELEASE, dst_rel), patterns,
                  args.dry_run, stats, args.checksum)

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {stats['copied']} files ({stats['bytes']/1e6:.1f} MB), "
          f"{stats['kept']} already up to date")
    if stats["mismatch"]:
        print(f"content mismatches {'found' if args.dry_run else 'RESTORED'} "
              f"({len(stats['mismatch'])}):")
        for m in stats["mismatch"]:
            print("   ", m)

    if args.prune:
        n_pruned = 0
        for dirpath, _dirnames, filenames in os.walk(os.path.join(RELEASE, "results")):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), RELEASE)
                if fn.startswith(".") or rel in stats["expected"]:
                    continue
                n_pruned += 1
                print(f"  prune  {rel}")
                if not args.dry_run:
                    os.remove(os.path.join(RELEASE, rel))
        print(f"{'would prune' if args.dry_run else 'pruned'} {n_pruned} files")
        if not args.dry_run:
            for dirpath, dirnames, filenames in os.walk(os.path.join(RELEASE, "results"),
                                                        topdown=False):
                if not dirnames and not filenames:
                    os.rmdir(dirpath)
    if stats["missing"]:
        print(f"MISSING {len(stats['missing'])} sources:")
        for m in stats["missing"]:
            print("   ", m)
    if not args.with_statefate_h5ad:
        print("note: the ~860 MB statefate .h5ad was NOT copied (pass --with-statefate-h5ad); "
              "without it, statefate d=50 and the statefate DE/gene notebooks cannot run.")


if __name__ == "__main__":
    main()
