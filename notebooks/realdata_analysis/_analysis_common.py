"""Shared engine for the real-data ANALYSIS notebooks (dist-est -> trajectories -> clustering ->
cross-dimension), run in the `torch26` env (has `uotreg`). Keeps `dist_est`, `trajectory_analysis`
and `cross_dimension` in sync.

Pipeline (mirrors the old `oldcode/.../realdata_analysis` notebooks with the NEW `uotreg` API):
  1. dist-est: estimate the UOTReg barycenter at a DENSE time grid using ALL observed snapshots
     (no leave-out) -> save a cloud + generator per grid time.
  2. trajectories: fit "ours" (UOT maps + balanced-OT + flow global-coupling) on the anchored estimated
     series, plus MMFM (in-file); official TIGON / MioFlow trajectories come from own-env sibling files
     and are LOADED if present. Start cells transported -> (T, N, d) per method.
  3. clustering: aligned-Euclidean trajectory distance matrix -> MDS -> KMeans (old `kmeans_from_dist`).
  4. cross-dimension: run d=10/20/50 and compare (cluster ARI/AMI, DM correlation, branch purity, PCA var).

Results tree: `results/realdata_analysis/<dataset>/d<DIM>/` (clouds, generators, trajs, clusters, figs).
"""
import os
import sys
import json
import numpy as np

# ----------------------------------------------------------------------------- paths / bootstrap
_here = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
# this folder (its helper modules) + tools/ (the shared `_repo.py` path resolver)
for _d in (_here, os.path.abspath(os.path.join(_here, os.pardir, os.pardir, "tools"))):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import _repo

ROOT = _repo.REPO
_repo.add_paths()


def results_dir(ds, dim, write=False, shipped=False):
    """The `(dataset, dim)` cell of the analysis tree. `shipped=True` pins the read to the shipped
    paper tree; the default read prefers your own `new_results/` run and falls back to shipped;
    `write=True` always resolves into `new_results/`."""
    if shipped and not write:
        return os.path.join(_repo.shipped("realdata_analysis"), ds, f"d{dim}")
    out = _repo.results("realdata_analysis", ds, f"d{dim}", write=write, make=write)
    return out


# ----------------------------------------------------------------------------- per-dataset config
# TRAJ_T = the full trajectory time grid (endpoints observed, interior estimated). EST_T = the interior
# grid actually estimated. Recipe (h/tau/arch) mirrors the old dist-est notebooks + the loo EST config.
CFG = {
    "embryoid": dict(
        traj_t=[1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5],
        est_t=[4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5],   # interior; endpoints 1.5 / 25.5 use raw cells
        h=4.0, tau=5.0, budget=85, divergence="kl",
        gen_hidden=256, gen_layers=4, map_hidden=256, map_layers=5,
        ini=_repo.aux_path("data/ini/G_embryoid20_256_Day4_ini.pth"),
    ),
    "statefate": dict(
        traj_t=[2.0, 3.0, 4.0, 5.0, 6.0],
        est_t=[3.0, 4.0, 5.0],                            # interior; endpoints 2.0 / 6.0 use raw cells
        h=None, tau=1.0, budget=85, divergence="kl",      # h=None -> Silverman on the observed times
        gen_hidden=256, gen_layers=4, map_hidden=196, map_layers=5,
        ini=_repo.aux_path("data/ini/G_statefate20_256_Dayall_ini.pth"),
    ),
}


def silverman_h(times):
    t = np.asarray(times, float)
    return float(1.06 * np.std(t) * len(t) ** (-1 / 5))


def est_recipe(ds, dim, smoke=False):
    """The `U.estimate` kwargs for this dataset/dim. Matches the loo EST config exactly at d=10/20
    (budget/R=85, d/t/g=50/10/50, batch 64/128, gen/map 256(sf map 196)/4-5, gaussian init scale 10).

    **d=50 bump:** the estimate is harder in 50-D, so R (outer budget) and the T/D/G inner iters are
    raised (budget 85->120, d/g 50->65, t 10->12, init 10000->15000) for a good high-dim estimate.
    Returns (kwargs, use_ini, ini_path)."""
    c = CFG[ds]
    h = c["h"] if c["h"] is not None else silverman_h(load_data(ds, dim)[1])
    budget, d_it, t_it, g_it, init_it = 85, 50, 10, 50, 10000
    if dim == 50 and not smoke:                       # high-D: larger R + slightly larger T/D/G
        budget, d_it, t_it, g_it, init_it = 120, 65, 12, 65, 15000
    kw = dict(h=h, tau=c["tau"], divergence=c["divergence"], std_mode="none", relaxation="one-sided",
              gen_hidden=(64 if smoke else c["gen_hidden"]), gen_layers=c["gen_layers"],
              map_hidden=(64 if smoke else c["map_hidden"]), map_layers=c["map_layers"],
              budget=(6 if smoke else budget),
              d_iters=(15 if smoke else d_it),
              t_iters=(5 if smoke else t_it),
              g_iters=(15 if smoke else g_it),
              batch_size=(32 if smoke else 64), batch_size_g=(64 if smoke else 128),
              init="gaussian", init_iters=(500 if smoke else init_it), gaussian_scale=10.0)
    use_ini = (dim == 20) and (not smoke) and os.path.exists(c["ini"])
    return kw, use_ini, c["ini"]


# ----------------------------------------------------------------------------- data
_ds_cache = {}


def load_data(ds, dim):
    """Return (arrays[list per obs time], timepoints[list], labels[list or None]) via the numpy loader."""
    key = (ds, dim)
    if key not in _ds_cache:
        from uotreg import datasets
        d = (datasets.load_embryoid(_repo.DATA_DIR, d=dim) if ds == "embryoid"
             else datasets.load_statefate(_repo.DATA_DIR, d=dim))
        arrays = [np.asarray(a, np.float32) for a in d.arrays]
        labels = list(d.labels) if getattr(d, "labels", None) is not None else None
        _ds_cache[key] = (arrays, [float(t) for t in d.timepoints], labels)
    return _ds_cache[key]


# ----------------------------------------------------------------------------- dist-est save/load
# `smoke=True` suffixes the filename, so a SMOKE run's estimates never shadow the full-quality
# ones a later read would want; `write=True` puts them under `new_results/`.
def est_cloud_path(ds, dim, t, smoke=False, write=False):
    return os.path.join(results_dir(ds, dim, write),
                        f"est_cloud_Day{t}{'_smoke' if smoke else ''}.npy")


def est_gen_path(ds, dim, t, smoke=False, write=False):
    return os.path.join(results_dir(ds, dim, write),
                        f"est_G_Day{t}{'_smoke' if smoke else ''}.pth")


def load_series(ds, dim, traj_t=None):
    """The anchored trajectory series over `traj_t`: raw observed cells at the endpoints (times present
    in the data) and the saved estimated cloud at every interior grid time. Missing estimates -> error."""
    arrays, tp, _ = load_data(ds, dim)
    traj_t = traj_t or CFG[ds]["traj_t"]
    series = []
    for t in traj_t:
        if t in tp:                                   # observed time -> raw cells (endpoints, and any coinciding)
            series.append(arrays[tp.index(t)])
        else:
            fp = est_cloud_path(ds, dim, t)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"missing estimate {fp} -- run dist_est for {ds} d={dim} first")
            series.append(np.load(fp))
    return series, [float(t) for t in traj_t]


# For the SERIES the trajectory sees, interior grid points ALWAYS use the estimate (even where a grid
# time coincides with an observed day), matching the old notebook. Only the two endpoints are raw.
def analysis_series(ds, dim):
    arrays, tp, _ = load_data(ds, dim)
    c = CFG[ds]; traj_t = c["traj_t"]
    series = []
    for i, t in enumerate(traj_t):
        if i == 0 or i == len(traj_t) - 1:            # endpoints -> raw observed cloud
            series.append(arrays[tp.index(t)])
        else:                                         # interior -> estimated barycenter cloud
            fp = est_cloud_path(ds, dim, t)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"missing estimate {fp} -- run dist_est for {ds} d={dim} first")
            series.append(np.load(fp))
    return series, [float(t) for t in traj_t]


# ----------------------------------------------------------------------------- trajectory fitting (new API)
def _traj_K(dim, smoke=False, dataset=None):
    """Trajectory-fitting budgets. Non-smoke reproduces the OLD `*_newanalysis` UOT-map training EXACTLY:
    `UOT_relax_on_2` did `for d in range(D_ITERS): [for t in range(T_ITERS): T-update]; one D-update`, with
    **embryoid D/T_ITERS=300/100, statefate 200/100** -- the SAME loop structure as the new
    `TrajectoryFitter._fit_pair` (d_iters x t_iters T-updates + d_iters D-updates per transition). Hidden
    256 / 5 layers (old `task_specific_hidden_size_T=256, n_hidden=5`). flow/mmfm = `pipelines._DEFAULT_K`.
    smoke == the batcheffect SMOKE config."""
    if smoke:
        return dict(map_hidden=64, map_layers=5, d_iters=20, t_iters=5, tau=5.0,
                    flow_hidden=64, flow_iters=300, global_tuples=120,
                    mmfm_hidden=64, mmfm_iters=300, mmfm_tuples=120)
    d_default = {"embryoid": 300, "statefate": 200}.get(dataset, 250)   # old per-dataset D_ITERS
    # tau = UOT relaxation for the one-sided maps ONLY (balanced OT ignores it). The OLD notebooks
    # trained the trajectory maps with `UOT_relax_on_2(..., tau=50)` (hence "reg50"/"direct50"), so
    # tau=50 is the FAITHFUL default. KL conjugate tau*(e^{s/tau}-1): LARGE tau -> balanced-like /
    # robust; SMALL tau -> more unbalanced (drops far mass).
    return dict(map_hidden=256, map_layers=5,
                d_iters= d_default,
                t_iters= 100,
                tau= 50.0,
                # flow = OT-CFM UNSHARED (one field per interval), the recipe settled by the
                # batcheffect flow screen: hidden 128, batch 256, minibatch-OT coupling.
                flow_hidden= 128,
                flow_iters= 3000,
                flow_batch= 256,
                global_tuples= 1000,   # flow-global only (not reported)
                # MMFM baseline: LINEAR interpolant + LIGHTER net + more tuples (cubic Runge-overshoots
                # catastrophically through the 5 sparse observed knots; linear can't overshoot between knots).
                mmfm_hidden=128, mmfm_iters=3000, mmfm_tuples=800)


def model_path(ds, dim, tag, write=False):
    """Where trajectory_analysis saves the trained trajectory MODEL for method `tag` (UOT/OT map chains
    via `TrajectoryFitter.save`, OT-CFM fields via `FlowMatchingTrajectory.save`) -- reload to transport
    ANY start cells later without retraining."""
    return os.path.join(results_dir(ds, dim, write), f"model_{ds}{dim}_{tag}.pth")


def fit_ours_uot(series, traj_t, X0, relaxation="one-sided", dim=None, smoke=False, device="cpu",
                 dataset=None, tau=None, return_model=False, d_iters=None, t_iters=None):
    """UOT-maps trajectory (relaxation='one-sided'=UOTReg, 'balanced'=OT). Returns (T,N,d), or
    (traj, fitter) with `return_model=True` (fitter.save(path) persists the map chain; fitter.transport(x)
    re-applies it to new cells).
    Budgets = the validated `_DEFAULT_K` (d_iters 250-300, t_iters 100) -> minutes on CPU; pass device='cuda'.
    `tau` overrides the relaxation strength for the one-sided maps (None -> `_traj_K` default;
    LOWER tau = less conservative / more unbalanced). Ignored when relaxation='balanced'."""
    from uotreg.trajectory import TrajectoryFitter
    from uotreg.config import TrajectoryConfig, UOTConfig, ModelConfig
    K = _traj_K(dim, smoke, dataset)
    if d_iters is not None: K["d_iters"] = int(d_iters)
    if t_iters is not None: K["t_iters"] = int(t_iters)
    tau = K["tau"] if tau is None else float(tau)
    tf = TrajectoryFitter(
        model=ModelConfig(dim=dim, map_hidden=K["map_hidden"], map_layers=K["map_layers"],
                          pot_hidden=K["map_hidden"], pot_layers=K["map_layers"], dropout=0.05),
        config=TrajectoryConfig(uot=UOTConfig(relaxation=relaxation, tau=tau),
                                d_iters=K["d_iters"], t_iters=K["t_iters"], batch_size=64,
                                warm_start=True, device=device, verbose=False),
    ).fit(series)
    traj = tf.transport(X0)                            # (len(series), N, d)
    return (traj, tf) if return_model else traj


def load_uot_model(ds, dim, tag, d, device="cpu"):
    """Reload a saved UOT/OT map chain -> a TrajectoryFitter with `.transport(x0)` (no retraining)."""
    from uotreg.trajectory import TrajectoryFitter
    from uotreg.config import ModelConfig, TrajectoryConfig
    K = _traj_K(d, False, ds)
    tf = TrajectoryFitter(model=ModelConfig(dim=d, map_hidden=K["map_hidden"], map_layers=K["map_layers"],
                                            pot_hidden=K["map_hidden"], pot_layers=K["map_layers"], dropout=0.05),
                          config=TrajectoryConfig(device=device, verbose=False))
    return tf.load(model_path(ds, dim, tag), map_location=device)


def _trim_clouds(clouds, pct=99.5):
    """Drop each cloud's tail points beyond the `pct`-percentile radius (generator sampling artifacts).
    Flow-matching regresses velocity to the coupling, so a single far tail point teaches a huge velocity
    and makes cells overshoot off-manifold; trimming fixes it. `pct=None` -> no trim. Start cells (X0)
    are observed data, untouched."""
    if pct is None:
        return clouds
    out = []
    for c in clouds:
        c = np.asarray(c); r = np.linalg.norm(c, axis=1)
        out.append(c[r <= np.percentile(r, pct)])
    return out


def fit_ours_flow_global(series, traj_t, X0, dim=None, smoke=False, device="cpu", dataset=None, trim_pct=99.5):
    """Flow global-coupling (one OT-CFM field on a global chained-OT coupling). Returns (T,N,d).
    NOTE: trains on a FIXED `global_tuples` chained-OT coupling -> under-covers big real clouds and can
    extrapolate/overshoot off-manifold at real time scales. Prefer `fit_ours_flow_matching` (OT-CFM,
    fresh-sampled) for real data; kept for parity with the batcheffect sim.
    `trim_pct` drops estimated-cloud tail points before fitting (see `_trim_clouds`)."""
    from uotreg import baselines as B
    K = _traj_K(dim, smoke, dataset)
    ser = _trim_clouds(series, trim_pct)
    field = B.global_coupling_flow(ser, [float(t) for t in traj_t], X0, hidden=K["flow_hidden"],
                                   iters=K["flow_iters"], n_tuples=K["global_tuples"],
                                   device=device, return_field=True)
    return B.rk4_sample(field, X0, np.asarray(traj_t, float), n_per=10, device=device)


def fit_ours_flow_matching(series, traj_t, X0, dim=None, smoke=False, device="cpu", dataset=None,
                           trim_pct=99.5, shared=False, return_model=False,
                           flow_iters=None, flow_batch=None, flow_hidden=None):
    """OT-CFM flow matching, **UNSHARED** field by default (one velocity field per interval) -- the
    recipe settled by the batcheffect flow screen (unshared > shared; a single shared field averages
    conflicting targets where the population branches). Samples FRESH minibatch-OT-coupled batches from
    the full consecutive clouds each iteration; `trim_pct` tail-clips the estimated clouds (a single far
    generator-artifact point teaches a huge velocity). RK4-integrated interval-by-interval exactly at
    the snapshot times (`simulate_at_times`) -> (T,N,d)."""
    from uotreg.trajectory import FlowMatchingTrajectory
    K = _traj_K(dim, smoke, dataset)
    if flow_iters is not None: K["flow_iters"] = int(flow_iters)
    if flow_batch is not None: K["flow_batch"] = int(flow_batch)
    if flow_hidden is not None: K["flow_hidden"] = int(flow_hidden)
    times = [float(t) for t in traj_t]
    ser = _trim_clouds(series, trim_pct)
    fm = FlowMatchingTrajectory(dim=dim, coupling="ot", hidden=K["flow_hidden"], n_layers=4,
                                device=device, seed=0, shared=shared, field="mlp")
    fm.fit(ser, times=times, iters=K["flow_iters"], batch_size=K.get("flow_batch", 128),
           lr=1e-3, verbose=False)
    traj = np.asarray(fm.simulate_at_times(X0, n_per=20))               # (T, N, d) at the grid times
    return (traj, fm) if return_model else traj


def load_flow_model(ds, dim, tag, d, device="cpu"):
    """Reload a saved OT-CFM flow -> a FlowMatchingTrajectory with `.simulate_at_times(x0)`."""
    from uotreg.trajectory import FlowMatchingTrajectory
    fm = FlowMatchingTrajectory(dim=d, device=device)
    return fm.load(model_path(ds, dim, tag), map_location=device)


def fit_mmfm_traj(obs_arrays, obs_times, traj_t, X0, dim=None, spline="linear", smoke=False, device="cpu", dataset=None):
    """MMFM baseline trajectory: fit on the RAW observed snapshots, integrate X0 FORWARD across traj_t."""
    from uotreg import baselines as B
    K = _traj_K(dim, smoke, dataset)
    field = B.mmfm_fit(obs_arrays, obs_times, dim=dim, hidden=K["mmfm_hidden"], iters=K["mmfm_iters"],
                       n_tuples=K["mmfm_tuples"], sigma=0.05, coupling="ot", spline=spline,
                       device=device, seed=0)
    return B.rk4_sample(field, X0, np.asarray(traj_t, float), n_per=20, device=device)


def load_external_traj(ds, dim, method):
    """Load an own-env baseline trajectory `traj_<ds><dim>_<method>.npy` (T,N,d) if present, else None.
    (official TIGON / MioFlow are produced by sibling files in their own conda envs.)"""
    fp = os.path.join(results_dir(ds, dim), f"traj_{ds}{dim}_{method}.npy")
    return np.load(fp) if os.path.exists(fp) else None


# ----------------------------------------------------------------------------- trajectory distance + clustering
#   (faithful copies of the old `embryoid_newanalysis` helpers)
def pairwise_aligned_euclid(traj):
    """Average Euclidean distance at corresponding time steps -> (N,N) distance matrix. traj (T,N,D).
    (Vectorized per time step -- identical values to the old per-pair loop, orders of magnitude faster.)"""
    from scipy.spatial.distance import cdist
    traj = np.asarray(traj, float)
    T, N, D = traj.shape
    DM = np.zeros((N, N), float)
    for t in range(T):
        DM += cdist(traj[t], traj[t])
    DM /= T
    np.fill_diagonal(DM, 0.0)
    return DM


def dm_path(ds, dim, tag, write=False):
    """Where the aligned-Euclid DM for method `tag` is cached."""
    return os.path.join(results_dir(ds, dim, write), f"dm_{ds}{dim}_{tag}.npy")


def _check_dm(DM):
    DM = np.asarray(DM, float)
    assert DM.ndim == 2 and DM.shape[0] == DM.shape[1], "DM must be square"
    DM = 0.5 * (DM + DM.T); np.fill_diagonal(DM, 0.0)
    return DM


def kmeans_from_dist(DM, n_clusters=6, n_components=10, random_state=0):
    """Metric-MDS the trajectory distance matrix to Euclidean space, then KMeans (old behaviour)."""
    from sklearn.manifold import MDS
    from sklearn.cluster import KMeans
    DM = _check_dm(DM)
    X_emb = MDS(n_components=n_components, dissimilarity="precomputed",
                random_state=random_state, n_init=4).fit_transform(DM)
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(X_emb)
    return labels, X_emb


def obs_time_view(tr, grid, timepoints):
    """Restrict a (T,N,d) trajectory to the OBSERVED times present in its grid -> (T', N, d), times.
    Grid-matching for fair cross-method metrics: a dense-grid trajectory has more time steps, which
    mechanically changes time-averaged quantities (purity/DM); evaluate everyone at the same times."""
    idx = [i for i, t in enumerate(grid) if float(t) in [float(x) for x in timepoints]]
    return np.asarray(tr)[idx], [float(grid[i]) for i in idx]


def _knn_idx(X, k):
    D = np.sum(X ** 2, 1)[:, None] + np.sum(X ** 2, 1)[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, 1)[:, :k]


def traj_metrics(tr, grid, arrays, timepoints, n_sub=1000, k=10, seed=0):
    """The comparison metrics for ONE trajectory (T,N,d) on grid `grid`:
      fidelity_t  : {t: W2(traj cloud, observed snapshot)} at every OBSERVED time in the grid
      fidelity    : their mean
      retention   : mean Jaccard of each cell's k-NN set at t0 vs at the final time (endpoint-only,
                    grid-independent)
      purity_own  : neighborhood purity on the method's OWN grid (old definition, grid-DEPENDENT)
      purity_obs  : the same purity computed on the OBSERVED-times view (grid-MATCHED -- the fair one)
    """
    from uotreg.metrics import w2
    tr = np.asarray(tr)
    rng = np.random.default_rng(seed)
    n = min(n_sub, tr.shape[1])
    # RANDOM subsample, not the first `n` rows. The prefix was a real bias: on statefate ours carries
    # the 3000 published cells at scattered ids while the baselines carry all 28249 in global order,
    # and that day-0 array is ORDERED -- its first 1000 cells differ from a random 1000 by up to 1.03
    # SD on PC4. The prefix therefore inflated WOT/MMFM/MioFlow by 1.8-2.1 in W2 while leaving ours
    # untouched, i.e. it manufactured most of the reported margin. Embryoid was unaffected (every
    # method carries the same 2381 cells; prefix vs random differ by <=0.03).
    sel = (np.sort(rng.choice(tr.shape[1], n, replace=False)) if n < tr.shape[1]
           else np.arange(tr.shape[1]))
    fid = {}
    for i, t in enumerate(grid):
        if float(t) in [float(x) for x in timepoints]:
            obs = arrays[[float(x) for x in timepoints].index(float(t))]
            fid[float(t)] = float(w2(tr[i, sel, :], obs[rng.integers(0, len(obs), n)]))
    nn0, nnT = _knn_idx(tr[0, sel], k), _knn_idx(tr[-1, sel], k)
    jac = [len(set(nn0[i]) & set(nnT[i])) / len(set(nn0[i]) | set(nnT[i])) for i in range(n)]
    p_own, _, _ = neighborhood_purity_euclid(tr[:, sel, :], k=k, start_t=0)
    trv, _ = obs_time_view(tr, grid, timepoints)
    p_obs, _, _ = neighborhood_purity_euclid(trv[:, sel, :], k=k, start_t=0)
    return dict(fidelity_t=fid, fidelity=float(np.mean(list(fid.values()))) if fid else float("nan"),
                retention=float(np.mean(jac)), purity_own=float(p_own), purity_obs=float(p_obs))


def neighborhood_purity_euclid(traj, k=10, start_t=0, DM=None):
    """Neighborhood spread: mean aligned-Euclid distance from each trajectory to its k nearest neighbors
    (defined at time `start_t`). Lower = tighter neighborhoods / better-separated branches. (T,N,D)."""
    T, N, D = traj.shape
    X0 = traj[start_t]
    dist0 = np.sum(X0 ** 2, 1, keepdims=True) + np.sum(X0 ** 2, 1)[None, :] - 2.0 * (X0 @ X0.T)
    np.fill_diagonal(dist0, np.inf)
    nn_idx = np.argsort(dist0, axis=1)[:, :k]
    if DM is None:
        DM = pairwise_aligned_euclid(traj)
    pur_all = DM[np.arange(N)[:, None], nn_idx].mean(axis=1)
    return float(pur_all.mean()), pur_all, nn_idx


# ----------------------------------------------------------------------------- plotting helpers
def metric_barh(ax, labels, values, xlabel, highlight=None, fs=1.0, fmt="{:.2f}",
                higher_better=False, missing="pending", ref=None, ref_label="", arrow=True):
    """A compact horizontal bar chart of one metric, in the paper figures' shared style.

    Lives here rather than in either figure file so `traj_figs` (methods) and `cross_dimension`
    (dimension pairs) cannot drift apart -- they are printed side by side in the same paper.

    Greyscale with `highlight` in black: the surrounding panels already carry a time colour ramp, and
    a second colour scheme competing with it is what makes a small multiple hard to read. A
    non-finite value is drawn as no bar plus the word `missing` -- a bar of height zero and a number
    that does not exist mean very different things, and only one of them is "this method lost".
    """
    import numpy as _np
    y = _np.arange(len(labels))
    vals = [float(v) if v is not None and _np.isfinite(v) else _np.nan for v in values]
    ok = [v for v in vals if _np.isfinite(v)]
    cols = ["0.15" if (highlight is not None and l == highlight) else "0.62" for l in labels]
    ax.barh(y, [0 if not _np.isfinite(v) else v for v in vals], height=0.68, color=cols,
            edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9 * fs)
    ax.invert_yaxis()
    # `arrow=None` drops the "(lower better)" suffix. In a narrow column beside a second bar chart
    # it is the same clause twice, and it reads better once in the caption -- the CALLER decides.
    suffix = ("higher better" if higher_better else "lower better") if arrow else None
    ax.set_xlabel(f"{xlabel}  ({suffix})" if suffix else xlabel, fontsize=9.5 * fs, labelpad=2)
    ax.tick_params(axis="x", labelsize=8.5 * fs, pad=1)
    hi = max(ok + ([ref] if ref is not None and np.isfinite(ref) else [])) if ok else 1
    ax.set_xlim(0, hi * 1.35)
    if ref is not None and np.isfinite(ref):
        # the reference line is what turns a bare number into a judgement: without it a reader has
        # no way to know whether 50% is close to the best achievable or close to meaningless
        ax.axvline(ref, color="0.25", lw=0.9, ls="--", zorder=4)
        # inside the axes, not above them: anchored to the axes top it overlapped the legend sitting
        # in the row above in the cross-dimension figure
        ax.annotate(ref_label, xy=(ref, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -2), textcoords="offset points", fontsize=7 * fs, color="0.25",
                    va="top", ha="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="0.88", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for yi, v in zip(y, vals):
        if _np.isfinite(v):
            ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", fontsize=8.5 * fs)
        else:
            ax.text(0, yi, " " + missing, va="center", ha="left", fontsize=8.5 * fs,
                    color="0.45", style="italic")
    return ax


def project2d(X):
    """PC1/PC2 (the data is already in PC space)."""
    return np.asarray(X)[..., :2]


def pc_extent(clouds, q=1.0, margin=0.12):
    """Robust PC1/PC2 view limits from `clouds` (list of (N,d) or one array): the [q, 100-q] percentile
    box + `margin`. Used to CLIP figure axes to the data so a few baseline strays (MMFM / flow-global
    integrate one field over the long real span and can overshoot) go off-frame instead of blowing up the plot."""
    P = project2d(np.concatenate([np.asarray(c) for c in clouds], 0) if isinstance(clouds, (list, tuple)) else clouds)
    lo = np.percentile(P, q, axis=0); hi = np.percentile(P, 100 - q, axis=0)
    pad = margin * (hi - lo)
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (float(lo[1] - pad[1]), float(hi[1] + pad[1]))


def plot_estimated_distributions(series, traj_t, ax=None, title=None):
    """Grey-scale scatter of every cloud in the series (PC1/PC2), light->dark over time (b&w-safe)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=140)
    shades = np.linspace(0.75, 0.0, len(series))
    for t, X, s in zip(traj_t, series, shades):
        P = project2d(X)
        ax.scatter(P[:, 0], P[:, 1], s=6, color=str(float(s)), alpha=0.5, label=f"t={t}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(title or "Estimated distributions over time")
    return ax


_TRAJ_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]   # b&w-safe: distinguish methods by SHAPE


def plot_trajectories(traj_dict, ax=None, max_cells=60, title=None, seed=0, xlim=None, ylim=None):
    """Overlay a subsample of per-cell paths (PC1/PC2) for each method. Methods distinguished by marker
    SHAPE at the endpoint (grayscale-safe); paths are thin grey lines. Pass `xlim`/`ylim` (e.g. from
    `pc_extent(series)`) to CLIP the view to the data so a few baseline strays go off-frame."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    rng = np.random.default_rng(seed)
    for mi, (name, traj) in enumerate(traj_dict.items()):
        if traj is None:
            continue
        T, N, D = traj.shape
        sel = rng.choice(N, min(max_cells, N), replace=False)
        P = project2d(traj[:, sel, :])                 # (T, n, 2)
        shade = str(0.15 + 0.6 * mi / max(1, len(traj_dict) - 1))
        for c in range(P.shape[1]):
            ax.plot(P[:, c, 0], P[:, c, 1], "-", color=shade, lw=0.5, alpha=0.5)
        ax.scatter(P[-1, :, 0], P[-1, :, 1], s=26, marker=_TRAJ_MARKERS[mi % len(_TRAJ_MARKERS)],
                   facecolors="none", edgecolors="black", linewidths=0.9, label=name)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(title or "Reconstructed trajectories"); ax.legend(fontsize=9, loc="best")
    return ax


# ----------------------------------------------------------------------------- dist-est QUALITY metrics
def distest_metrics(ds, dim, n=1000, repeats=10, seed=0):
    """Sanity-check the estimate: at every grid time that COINCIDES with an OBSERVED day, compare the
    estimated cloud vs the true observed cells (MMD / EMD / W2, POT-backed). Lower = the estimated
    barycenter reproduces the real marginal. Returns {time: {MMD,EMD,W2}}."""
    from uotreg.metrics import mmd_rbf, emd, w2
    arrays, tp, _ = load_data(ds, dim)
    rng = np.random.default_rng(seed)
    out = {}
    for t in CFG[ds]["est_t"]:
        if t not in tp:                       # only where we can compare to a real snapshot
            continue
        fp = est_cloud_path(ds, dim, t)
        if not os.path.exists(fp):
            continue
        est = np.load(fp); truth = arrays[tp.index(t)]
        acc = {"MMD": [], "EMD": [], "W2": []}
        for _ in range(repeats):
            p = est[rng.integers(0, len(est), n)]; r = truth[rng.integers(0, len(truth), n)]
            acc["MMD"].append(mmd_rbf(p, r)); acc["EMD"].append(emd(p, r)); acc["W2"].append(w2(p, r))
        out[float(t)] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in acc.items()}
    return out


# ----------------------------------------------------------------------------- COLOR plotting (old style)
def _time_palette(n):
    import seaborn as sns
    return sns.color_palette("RdBu_r", n_colors=n)


def _cap(X, max_cells, seed=0):
    """A reproducible subsample of at most `max_cells` rows (all of them when `max_cells` is None).

    Scatter plots of these clouds are dominated by overplotting long before every cell is drawn, and
    statefate's day-0 snapshot alone is 28k cells. The PUBLISHED statefate notebook does exactly this
    (`if idx.shape[0] > 1000: idx = np.random.choice(idx, 1000, replace=False)`)."""
    X = np.asarray(X)
    if max_cells is None or max_cells >= len(X):
        return X
    return X[np.random.default_rng(seed).choice(len(X), int(max_cells), replace=False)]


def plot_estimated_distributions_color(series, traj_t, ax=None, title=None, endpoints=(0, -1), s=12,
                                       max_cells=None, sub_seed=0, ends_on_top=False,
                                       mark_ends=None, legend=True):
    """Old-style: each cloud colored by time (RdBu_r), endpoints opaque, interior translucent. PC1/PC2.

    `max_cells` caps how many cells EACH cloud contributes (None = all, the previous behaviour).

    `ends_on_top` draws the two endpoint clouds LAST. The default paints in time order, so the final
    cloud sits on top of everything and the FIRST one ends up buried -- fine on Embryoid, where the
    clouds separate, and unreadable on Statefate, where they overlap. Off by default so no existing
    figure changes.

    `mark_ends` is a string appended to the two endpoint labels in the legend (e.g. "observed"). In
    the anchored series only the endpoints are measured cells; every interior cloud is an estimate,
    and a reader cannot tell which is which from the colours alone."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6), dpi=140)
    pal = _time_palette(len(series)); ends = {endpoints[0] % len(series), endpoints[1] % len(series)}
    order = ([i for i in range(len(series)) if i not in ends] + sorted(ends)) if ends_on_top \
        else list(range(len(series)))
    for z, i in enumerate(order):
        P = project2d(_cap(series[i], max_cells, sub_seed)); a = 0.85 if i in ends else 0.35
        ax.scatter(P[:, 0], P[:, 1], color=[pal[i]], s=s, alpha=a, edgecolors="none", zorder=z + 1)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title(title or "Estimated distributions over time")
    if legend:
        lab = [f"t={t}" + (f" ({mark_ends})" if (mark_ends and i in ends) else "")
               for i, t in enumerate(traj_t)]
        ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=pal[i], mec="none", label=lab[i])
                           for i, _t in enumerate(traj_t)], fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.01, 1))
    return ax


def plot_trajectory_color(series, traj_t, traj, n_paths=12, ax=None, title=None, seed=0,
                          our_label="UOTReg trajectory", clip=True, max_cells=None, sub_seed=0):
    """Old visualization_trajs style: background clouds colored by time (RdBu_r) + selected cell paths as
    dashed black arrows with time-colored, black-edged waypoints. `traj` (T,N,d). `clip` sets the axes to
    the data extent (`pc_extent(series)`) so a few off-manifold baseline strays don't blow up the view;
    prefers drawing paths that STAY in-frame (up to `n_paths`)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6), dpi=140)
    T = len(series); pal = _time_palette(T)
    xlim, ylim = pc_extent(series)              # extent from the FULL clouds, so capping cannot zoom
    for i, (X, c) in enumerate(zip(series, pal)):
        P = project2d(_cap(X, max_cells, sub_seed)); a = 0.75 if i in (0, T - 1) else 0.3
        ax.scatter(P[:, 0], P[:, 1], color=[c], s=13, alpha=a, edgecolors="none", zorder=i + 1)
    rng = np.random.default_rng(seed)
    # prefer cells whose whole path stays within the data box (so drawn paths are representative)
    Pall = project2d(traj)                                       # (T, N, 2)
    inbox = ((Pall[..., 0] >= xlim[0]) & (Pall[..., 0] <= xlim[1]) &
             (Pall[..., 1] >= ylim[0]) & (Pall[..., 1] <= ylim[1])).all(0)
    pool = np.where(inbox)[0] if inbox.sum() >= n_paths else np.arange(traj.shape[1])
    sel = rng.choice(pool, min(n_paths, len(pool)), replace=False)
    for i in sel:
        P = project2d(traj[:, i, :])
        for j in range(len(P) - 1):
            ax.plot([P[j, 0], P[j + 1, 0]], [P[j, 1], P[j + 1, 1]], ls="--", color="black", lw=1, alpha=0.8, zorder=10)
            ax.annotate("", xy=(P[j + 1, 0], P[j + 1, 1]), xytext=(P[j, 0], P[j, 1]),
                        arrowprops=dict(arrowstyle="->", color="black", alpha=0.8, lw=0.8), zorder=11)
        for j in range(len(P)):
            ax.scatter(P[j, 0], P[j, 1], color=[pal[j]], s=38, edgecolors="black", linewidths=1.2, zorder=12)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title(title or "Reconstructed trajectory")
    if clip:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=pal[i], mec="none", label=f"t={t}") for i, t in enumerate(traj_t)]
    handles += [Line2D([0], [0], color="black", lw=2, label=our_label)]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    return ax


def plot_clusters_medoids(traj, labels, DM, title=None, per_cluster_max=None, seed=0, names=None):
    """Old `plot_clusters_1x3_from_pcs_with_medoids`: trajectories colored by cluster (tab10) at low alpha,
    per-cluster MEDOID (min row-sum in the cluster sub-DM) overplotted thick. 1x3 PC panels if d>=4 else 1x1.

    `names` = {cluster id: legend label}, for labelling clusters by the fate they were matched to
    instead of by their arbitrary number."""
    import matplotlib.pyplot as plt
    T, N, D = traj.shape
    pairs = [(0, 1), (0, 2), (0, 3)] if D >= 4 else [(0, 1)]
    uniq = np.unique(labels); cmap = plt.get_cmap("tab10")
    cmap_c = {c: cmap(i % 10) for i, c in enumerate(uniq)}
    medoid = {}
    rng = np.random.default_rng(seed); plot_ids = {}
    for c in uniq:
        ids = np.where(labels == c)[0]
        sub = DM[np.ix_(ids, ids)]; medoid[c] = int(ids[int(np.argmin(sub.sum(1)))])
        if per_cluster_max and len(ids) > per_cluster_max:
            keep = rng.choice(ids[ids != medoid[c]], per_cluster_max - 1, replace=False)
            plot_ids[c] = np.append(keep, medoid[c])
        else:
            plot_ids[c] = ids
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.2), dpi=150, squeeze=False)
    for ax, (p, q) in zip(axes.ravel(), pairs):
        for c in uniq:
            col = cmap_c[c]
            for i in plot_ids[c]:
                ax.plot(traj[:, i, p], traj[:, i, q], color=col, lw=1.0, alpha=0.12)
            mi = medoid[c]
            ax.plot(traj[:, mi, p], traj[:, mi, q], color=col, lw=3.0, alpha=1.0, zorder=4)
            ax.scatter(traj[0, mi, p], traj[0, mi, q], color=[col], s=26, marker="o", edgecolors="k", lw=0.6, zorder=5)
            ax.scatter(traj[-1, mi, p], traj[-1, mi, q], color=[col], s=26, marker="D", edgecolors="k", lw=0.6, zorder=5)
        ax.set_xlabel(f"PC{p+1}"); ax.set_ylabel(f"PC{q+1}")
        ax.set_title(f"PC{p+1} vs PC{q+1}", pad=5)
    handles = [plt.Line2D([0], [0], color=cmap_c[c], lw=2,
                          label=(names or {}).get(int(c), f"Cluster {c}")) for c in uniq]
    handles += [plt.Line2D([0], [0], marker="o", color="w", mfc="gray", mec="k", label="Start", ms=6, lw=0),
                plt.Line2D([0], [0], marker="D", color="w", mfc="gray", mec="k", label="End", ms=6, lw=0),
                plt.Line2D([0], [0], color="k", lw=3, label="Rep. traj")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False,
           fontsize=plt.rcParams.get("legend.fontsize", 9))
    # `y=0.985` + a taller rect closes the gap between the suptitle and the per-panel "PC1 vs PC2"
    # subtitles; at the default the two sat a full line apart and read as unrelated headings.
    fig.suptitle(title or "Trajectories by cluster (+ medoids)", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.99, 0.99])
    fig.subplots_adjust(top=0.855)
    return fig


# ----------------------------------------------------------------------------- misc
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
