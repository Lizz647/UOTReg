r"""Trajectory-quality metrics for the batch-effect study.

All operate on a projected trajectory ``traj2d`` of shape ``(T, N, 2)`` (cells tracked over the
trajectory times), and a geometry ``G`` (a :class:`uotreg.simulation.GeomHD`) that supplies the true
``branch_means`` / ``truth`` / ``labels``. They are duck-typed on ``G`` (no import of ``simulation``),
so this module has no package cycle and ``simulation`` imports :func:`roughness` from here.

* :func:`roughness` -- zig-zag penalty (compare to ``G.truth_roughness``).
* branch-shape metrics live on ``GeomHD`` itself: ``adherence`` (nearest), ``adherence_labeled``
  (branch-aware), ``branch_balance``.
* here we add the *spread-aware* metrics that see thinness/diffuseness that the prototype-adherence
  misses: :func:`adher_indiv`, :func:`w2_path`, :func:`spread_ratio`, :func:`spread_post`.
"""
from __future__ import annotations

import numpy as np

from .metrics import w2


def roughness(traj):
    """Zig-zag penalty = mean squared 2nd time-difference on a (T, n, d) trajectory."""
    tr = np.asarray(traj, float); acc = tr[2:] - 2 * tr[1:-1] + tr[:-2]
    return float((acc ** 2).sum(-1).mean())


def _rms_spread(P):
    P = np.asarray(P, float)
    return float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())) if len(P) else 0.0


def adher_indiv(G, traj2d, traj_t, labels0):
    """Per-cell individual-path adherence: target_i(t) = branch_{label_i}(t) + (p_i(t0) - branch_{label_i}(t0)).
    Each cell's true path is its branch-mean path SHIFTED by its initial offset, so a thin/collapsed
    trajectory that loses the offset is penalized (unlike the prototype-mean `adherence_labeled`)."""
    traj2d = np.asarray(traj2d); labels0 = np.asarray(labels0).ravel()
    m0 = [np.asarray(m) for m in G.branch_means(traj_t[0])]
    base0 = np.where(labels0[:, None] == 0, m0[0][None], m0[1][None])
    shift = traj2d[0] - base0
    tot, cnt = 0.0, 0
    for k, t in enumerate(traj_t):
        mA, mB = G.branch_means(t)
        base = np.where(labels0[:, None] == 0, np.asarray(mA)[None], np.asarray(mB)[None])
        tot += np.linalg.norm(traj2d[k] - (base + shift), axis=1).sum(); cnt += traj2d[k].shape[0]
    return float(tot / max(cnt, 1))


def w2_path(G, traj, traj_t, n_sub=200, seed=0, project=True):
    """Mean over t of W2(trajectory cloud, truth cloud) -- the honest distribution match (penalizes
    wrong position + spread + imbalance at once). Clouds subsampled to a common size.

    `project=True` (default) measures on the 2-D signal plane: both clouds go through
    ``G.project2d``, so `traj` may be given either projected `(T,N,2)` or in ambient dim `(T,N,d)`.
    `project=False` measures in the FULL ambient dimension -- `traj` must then be `(T,N,G.dim)` and is
    compared to the unprojected truth. The full-d number is larger and on a different scale (the
    block-replicated signal dims each contribute, and the projection averages their lift noise away),
    so compare full-d to full-d only.
    """
    traj = np.asarray(traj); rng = np.random.default_rng(seed); vals = []
    if not project and traj.shape[-1] != G.dim:
        raise ValueError(f"project=False needs an ambient-dim trajectory (T,N,{G.dim}), "
                         f"got {traj.shape}")
    for k, t in enumerate(traj_t):
        Tp, P = np.asarray(G.truth(t)), traj[k]
        if project:
            Tp = G.project2d(Tp)
            if P.shape[-1] != Tp.shape[-1]:
                P = G.project2d(P)
        m = min(len(P), len(Tp), n_sub)
        a = P[rng.integers(0, len(P), m)]; b = Tp[rng.integers(0, len(Tp), m)]
        vals.append(w2(a, b))
    return float(np.mean(vals))


def spread_ratio(G, traj2d, traj_t, labels0, split=None):
    """Mean over t,branch of (traj within-branch RMS spread)/(truth within-branch RMS spread).
    ~1 = matches truth spread, <1 = too thin, >1 = too diffuse. If `split` is given, only times with
    branch separation > 0 beyond `split` are counted (see :func:`spread_post`)."""
    traj2d = np.asarray(traj2d); labels0 = np.asarray(labels0).ravel(); ratios = []
    for k, t in enumerate(traj_t):
        if split is not None and float(t) <= float(split):
            continue
        Tcl = G.project2d(G.truth(t)); Tlab = G.labels[int(round(float(t)))]
        for b in (0, 1):
            tr = traj2d[k][labels0 == b]; tt = Tcl[Tlab == b]
            if len(tr) >= 2 and len(tt) >= 2:
                s_tt = _rms_spread(tt)
                if s_tt > 1e-6:
                    ratios.append(_rms_spread(tr) / s_tt)
    return float(np.mean(ratios)) if ratios else float("nan")


def spread_post(G, traj2d, traj_t, labels0, split=4.0):
    """`spread_ratio` restricted to SEPARATED times (t > split) -- avoids the trunk inflation where a
    single trajectory blob is naturally wider than one clean branch. Use this for the honest read."""
    return spread_ratio(G, traj2d, traj_t, labels0, split=split)


def metric_row(G, traj2d, traj_t, labels0=None, split=4.0, traj_full=None):
    """Convenience: all metrics for one trajectory as a dict (nan-safe where labels are missing).
    Pass the SAME trajectory in ambient dim as `traj_full` to also get `w2_path_full` (W2 measured in
    all `G.dim` dimensions instead of on the 2-D projection)."""
    t = np.asarray(traj2d); have = labels0 is not None
    bb = G.branch_balance(t, traj_t, labels0 if have else None)
    bA, rec = (bb if isinstance(bb, tuple) else (bb, float("nan")))
    return dict(
        adherence=G.adherence(t, traj_t),
        adher_labeled=(G.adherence_labeled(t, traj_t, labels0) if have else float("nan")),
        adher_indiv=(adher_indiv(G, t, traj_t, labels0) if have else float("nan")),
        w2_path=w2_path(G, t, traj_t),
        w2_path_full=(w2_path(G, np.asarray(traj_full), traj_t, project=False)
                      if traj_full is not None else float("nan")),
        spread=(spread_ratio(G, t, traj_t, labels0) if have else float("nan")),
        spread_post=(spread_post(G, t, traj_t, labels0, split) if have else float("nan")),
        roughness=roughness(t),
        balanceA=bA, recovery=rec,
    )


def report(G, trajs, traj_t, labels0=None, split=4.0, verbose=True, trajs_full=None):
    """Compute `metric_row` for every method in `trajs` and (optionally) print a table.
    `spread` shown is `spread_post` (post-split, avoids trunk inflation). Returns {name: row}.
    `trajs_full` = {name: ambient-dim trajectory} adds a `w2_{G.dim}d` column (full-dimensional W2)."""
    rows = {nm: metric_row(G, t, traj_t, labels0, split,
                           traj_full=(trajs_full or {}).get(nm)) for nm, t in trajs.items()}
    if verbose:
        cols = [("adher_labeled", "adher*"), ("adher_indiv", "indiv"), ("w2_path", "w2_path"),
                ("spread_post", "spread"), ("roughness", "rough"), ("balanceA", "balA")]
        if trajs_full:
            cols.insert(3, ("w2_path_full", f"w2_{G.dim}d"))
        print(f"[{G.kind}] trajectory metrics (truth roughness ref = {G.truth_roughness(traj_t):.3f}; "
              "good = low adher*/w2_path, spread ~ 1, balA ~ 0.5):")
        print("   " + f"{'method':32s} " + " ".join(f"{h:>8}" for _, h in cols))
        for nm, r in rows.items():
            def _f(x):
                return "     n/a" if (isinstance(x, float) and np.isnan(x)) else f"{x:>8.3f}"
            print("   " + f"{nm:32s} " + " ".join(_f(r[k]) for k, _ in cols))
    return rows
