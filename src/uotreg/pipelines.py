r"""High-level two-stage driver for the batch-effect / distribution-estimation experiments.

Stage 1 -- estimate the UOTReg (denoised unbalanced-OT barycenter) distribution at a query time.
Stage 2 -- fit a configurable set of trajectory methods on the estimated series.

`estimate` works on plain arrays (a list of per-time clouds + the time grid), so it serves both the
synthetic `GeomHD` experiments and the real-data leave-one-out benchmarks. `run_stage1` /
`fit_trajectories` are the convenience wrappers used by the batch-effect notebooks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import baselines as B
from .barycenter import DistributionEstimator
from .config import ModelConfig, TrainConfig, TrajectoryConfig, UOTConfig
from .data import GaussianMixtureLatentSampler, TensorSampler, samplers_from_arrays
from .trajectory import FlowMatchingTrajectory, TrajectoryFitter


# --------------------------------------------------------------------------- #
# per-time standardization spec
# --------------------------------------------------------------------------- #
def resolve_std(std_mode, t, default="none"):
    """Resolve a possibly PER-TIME `std_mode` at query time `t`. Accepts a string
    ("none"|"std"|"iso"), a dict {int_time: mode} (missing -> its "default" key, else `default`), or a
    callable(t)->mode. e.g. `{6:"std",7:"std",8:"std","default":"none"}` = no-std trunk, std branches."""
    if callable(std_mode):
        return std_mode(t)
    if isinstance(std_mode, dict):
        return std_mode.get(int(round(float(t))), std_mode.get("default", default))
    return std_mode


# --------------------------------------------------------------------------- #
# Stage 1 -- the UOTReg distribution estimate at one query time
# --------------------------------------------------------------------------- #
def estimate(observed: Sequence[np.ndarray], tlist, query_time, dim, *, h=1.0, tau=5.0, budget=80,
             std_mode="none", relaxation="one-sided", divergence="kl", weight_scheme="positive",
             threshold=0.01, gen_hidden=100, gen_layers=4, map_hidden=None, map_layers=None,
             batch_size=32, batch_size_g=64, d_iters=25, t_iters=10, g_iters=30,
             seed=0, n_gen=500, latent_mode="gaussian", init="gaussian", init_iters=5000,
             gaussian_scale=None, vae_epochs=5, vae_flow_length=16, vae_coef=5.0, flow_epochs=250,
             init_kwargs=None, pretrained_generator=None, device="cpu", return_estimator=False):
    """UOTReg estimate of the cell distribution at `query_time` from the observed clouds `observed`
    (list, one per time in `tlist`), in ambient dimension `dim`. `std_mode` standardizes the pooled
    data (fit + samples mapped BACK to original coords). `map_hidden`/`map_layers` default to
    `gen_hidden`/`gen_layers`; `gaussian_scale` defaults to 2 (standardized) / 6 (raw).

    `init` selects the generator initialization, matching `reproduce/realdata/realdata_dist_est_dims`:
    `"gaussian"` (broad blob, uses `init_iters`/`gaussian_scale`), `"vae_nf"` (data-aware flow-VAE on
    the pooled cells, uses `vae_epochs`/`vae_flow_length`/`vae_coef`), `"data_gaussian"`, or `"flow"`.
    `init_kwargs` (dict) overrides the auto-built kwargs. Returns the sampled cloud `(n_gen, dim)`
    (and the fitted estimator if `return_estimator`)."""
    mu = np.zeros(dim, np.float32); sd = np.ones(dim, np.float32)
    if std_mode == "std":
        pool = np.concatenate(observed, 0); mu = pool.mean(0); sd = pool.std(0) + 1e-6
    elif std_mode == "iso":
        pool = np.concatenate(observed, 0); mu = pool.mean(0); sd = (pool.std(0).mean() + 1e-6) * np.ones(dim, np.float32)
    obs_s = [(np.asarray(o, np.float32) - mu) / sd for o in observed]
    sm = samplers_from_arrays(obs_s, device=device)
    pl = TensorSampler(np.concatenate(obs_s, 0), device=device)
    mh = gen_hidden if map_hidden is None else map_hidden
    ml = gen_layers if map_layers is None else map_layers
    model = ModelConfig(dim=dim, latent_dim=dim, gen_hidden=gen_hidden, gen_layers=gen_layers, gen_dropout=0.05,
                        map_hidden=mh, map_layers=ml, pot_hidden=mh, pot_layers=ml, dropout=0.05)
    tr = TrainConfig(outer_iters=budget, d_iters=d_iters, t_iters=t_iters, g_iters=g_iters,
                     batch_size=batch_size, batch_size_g=batch_size_g, lr_map=3e-4, lr_pot=3e-4,
                     lr_gen=1e-4, weight_decay_td=1e-10, weight_decay_gen=1e-8, device=device,
                     verbose=False, seed=seed)
    e = DistributionEstimator(model=model, train=tr,
                              uot=UOTConfig(relaxation=relaxation, tau=tau, divergence=divergence))
    latent = GaussianMixtureLatentSampler(dim=dim, k=2, separation=4.0, std=1.0, device=device) \
        if latent_mode == "mixture" else None
    tlist_f = [float(t) for t in tlist]
    if pretrained_generator is not None:
        # start from a saved generator that already mimics the data (the reproduce/realdata d=20 recipe).
        # The ini is in the ORIGINAL PC space, so use std_mode="none" (mu=0, sd=1) with it.
        e.fit(sm, tlist_f, query_time=float(query_time), h=h, weight_scheme=weight_scheme,
              threshold=threshold, pretrained_generator=pretrained_generator)
    else:
        if init_kwargs is None:
            if init == "vae_nf":
                # data-aware flow-VAE init on the pooled cells (faithful to the old VAEini / the
                # reproduce/realdata `init_fit_kwargs` vae_nf branch).
                ik = {"vae_epochs": vae_epochs, "flow_length": vae_flow_length, "vae_coef": vae_coef}
            elif init == "flow":
                ik = {"iters": init_iters, "flow_epochs": flow_epochs, "flow_length": vae_flow_length}
            elif init == "data_gaussian":
                ik = {"iters": init_iters}
            else:  # "gaussian"
                gscale = (2.0 if std_mode != "none" else 6.0) if gaussian_scale is None else gaussian_scale
                ik = {"iters": init_iters, "gaussian_scale": gscale}
        else:
            ik = init_kwargs
        e.fit(sm, tlist_f, query_time=float(query_time), h=h, weight_scheme=weight_scheme,
              threshold=threshold, init=init, init_data_sampler=pl, latent_sampler=latent,
              init_kwargs=ik)
    samples = e.sample(n_gen) * sd + mu
    return (samples, e) if return_estimator else samples


def run_stage1(G, traj_t, *, h=1.0, tau=5.0, budget=80, std_mode="none", gen_hidden=100, gen_layers=4,
               n_start=120, n_gen=500, latent_mode="gaussian", seed=0, x0_seed=None, device="cpu"):
    """Estimate the denoised series at every trajectory time (Stage 1). Returns a dict with
    `est_series` (denoised), `ours_series` (anchored: raw start + denoised path), `raw_series`, the
    start cells `X0`, and their true `labels0` -- everything Stage 2 needs. `std_mode` may be per-time.

    `x0_seed` selects the `n_start` starting cells at `traj_t[0]` (whose trajectories are transported
    and drawn): `None` = the first `n_start` cells (deterministic, the original behaviour); an int =
    a random subset via `rng(x0_seed)`. `labels0` is kept aligned to the same selection."""
    raw_series = [G.observed[t] for t in traj_t]
    est_series = [estimate(G.observed, G.tlist, t, G.dim, h=h, tau=tau, budget=budget,
                           std_mode=resolve_std(std_mode, t), gen_hidden=gen_hidden, gen_layers=gen_layers,
                           n_gen=n_gen, latent_mode=latent_mode, seed=seed, device=device) for t in traj_t]
    ours_series = [raw_series[0]] + est_series[1:]
    first = np.asarray(G.observed[traj_t[0]])
    k = min(n_start, len(first))
    sel = np.arange(k) if x0_seed is None else \
        np.random.default_rng(x0_seed).choice(len(first), k, replace=False)
    X0 = first[sel]
    labels0 = np.asarray(G.labels[traj_t[0]])[sel] if G.labels is not None else None
    return dict(est_series=est_series, ours_series=ours_series, raw_series=raw_series,
                X0=X0, labels0=labels0)


# --------------------------------------------------------------------------- #
# Stage 2 -- fit a configurable set of trajectory methods on the estimated series
# --------------------------------------------------------------------------- #
DEFAULT_METHODS = ("ours: UOT maps", "ours: flow-matching", "ours: flow global-coupling",
                   "raw MMFM", "raw WOT-like")

_DEFAULT_K = dict(traj_hidden=256, map_layers=5, d_iters=250, t_iters=100,
                  flow_hidden=256, flow_iters=3000, global_tuples=200,
                  mmfm_iters=3000, mmfm_tuples=200, tigon_iters=1500, diff_sigma=0.3, tau=5.0,
                  # knobs of the explicit flow variants ("ours: flow <coupling> <shared|unshared>")
                  flow_layers=4, flow_batch=128, flow_lr=1e-3, flow_sigma=0.0, flow_n_per=20,
                  flow_field="mlp", flow_reg=0.05, flow_reg_m=1.0, flow_seed=0,
                  uot_seed=None,      # set an int to make the UOT map chain reproducible run-to-run
                  map_batch=64)       # cells per UOT map/potential step


def fit_trajectories(G, ours_series, raw_series, X0, traj_t, K=None, methods=DEFAULT_METHODS,
                     device="cpu", return_models=False, project=True):
    """Fit each method in `methods` on the CACHED distributions. Known names: 'ours: UOT maps',
    'ours: flow-matching', 'ours: flow global-coupling', 'ours: diffusion', 'raw WOT-like',
    'raw MMFM', 'raw TIGON'. (MioFlow / official TIGON are loaded separately in their own env.)
    `K` overrides `_DEFAULT_K` (capacity/iters/tau).

    Plus the EXPLICIT flow variants `'ours: flow {OT-CFM|UOT-map|minibatch-UOT|independent}
    {shared|unshared}'` -- the training coupling x one field for all intervals / one field per
    interval, integrated with RK4 at the snapshot times (knobs: `flow_hidden/layers/iters/batch/lr/
    sigma/n_per/field/reg/reg_m/seed`). `'UOT-map'` trains the flow against OUR OWN transport maps
    and reuses the chain fitted for 'ours: UOT maps' (fitted once, whichever is requested first).

    `project=False` returns the AMBIENT-dim `(T,N,G.dim)` trajectory instead of the 2-D projection --
    project it yourself with `G.project_traj` for the shape metrics, and feed the full-d array to
    `trajectory_metrics.report(..., trajs_full=...)` for a full-dimensional W2.

    Each method is TRAINED ONCE. With `return_models=False` (default) returns {name: (T,N,2) projected
    trajectory} for the given `X0` -- unchanged. With `return_models=True` returns {name: applier} where
    `applier(x0)` -> (T,N,2) projected trajectory: the trained model re-applied to NEW start cells `x0`
    with NO re-training (cheap re-visualization from different starting points). ('ours: diffusion' is
    trained+sampled together, so under `return_models` its applier retrains; the other 6 do not.)"""
    K = {**_DEFAULT_K, **(K or {})}
    times = [float(t) for t in traj_t]; TH = K["traj_hidden"]; dim = G.dim; tvec = np.asarray(times, float)
    tcfg = TrajectoryConfig(uot=UOTConfig(relaxation="one-sided", tau=K["tau"]),
                            d_iters=K["d_iters"], t_iters=K["t_iters"], batch_size=K["map_batch"],
                            warm_start=True, device=device, verbose=False, seed=K["uot_seed"])
    _sub = lambda full: full[np.linspace(0, full.shape[0] - 1, len(traj_t)).astype(int)]

    _fitters = {}          # id(series) -> fitted TrajectoryFitter (shared by 'UOT maps' + 'uot-map' flows)

    def _uot_fitter(series):
        """Fit (once per series) the UOT map chain. `TrajectoryFitter.fit` re-seeds, so the chain is
        the same whichever method asks for it first."""
        key = id(series)
        if key not in _fitters:
            _fitters[key] = TrajectoryFitter(
                model=ModelConfig(dim=dim, map_hidden=TH, map_layers=K["map_layers"],
                                  pot_hidden=TH, pot_layers=K["map_layers"], dropout=0.05),
                config=tcfg).fit(series)
        return _fitters[key]

    def _uot(series):
        tf = _uot_fitter(series)
        return lambda x0: tf.transport(x0)

    def _flow(series):
        fm = FlowMatchingTrajectory(dim=dim, coupling="ot", hidden=K["flow_hidden"], n_layers=4,
                                    device=device, seed=0)
        fm.fit(series, times=times, iters=K["flow_iters"], batch_size=128, lr=1e-3, verbose=False)
        return lambda x0: _sub(fm.simulate(x0, n_steps=100))

    def _flow_var(series, coupling, shared):
        """Explicit flow variant: any coupling x shared/unshared field, RK4 at the snapshot times.
        `coupling='uot-map'` REUSES the UOT maps of 'ours: UOT maps' (fit once, see `_uot_fitter`)."""
        fm = FlowMatchingTrajectory(dim=dim, coupling=coupling, hidden=K["flow_hidden"],
                                    n_layers=K["flow_layers"], device=device, seed=K["flow_seed"],
                                    shared=shared, field=K["flow_field"])
        fm.fit(series, times=times, iters=K["flow_iters"], batch_size=K["flow_batch"], lr=K["flow_lr"],
               sigma=K["flow_sigma"], reg=K["flow_reg"], reg_m=K["flow_reg_m"], verbose=False,
               uot_maps=(_uot_fitter(series).map_chain if coupling == "uot-map" else None))
        return lambda x0: fm.simulate_at_times(x0, n_per=K["flow_n_per"])

    def _global(series):
        field = B.global_coupling_flow(series, times, X0, hidden=K["flow_hidden"], iters=K["flow_iters"],
                                       n_tuples=K["global_tuples"], device=device, return_field=True)
        return lambda x0: B.rk4_sample(field, x0, tvec, n_per=10, device=device)

    def _mmfm(series):
        field = B.mmfm_fit(series, times, dim=dim, hidden=TH, iters=K["mmfm_iters"],
                           n_tuples=K["mmfm_tuples"], sigma=0.05, device=device, seed=0)
        return lambda x0: B.ode_sample(field, x0, times, n_per=10, device=device)

    def _tigon(series):
        mdl = B.tigon_fit(series, times, dim=dim, hidden=32, iters=K["tigon_iters"], n_samples=128,
                          n_per=5, device=device, seed=0)
        return lambda x0: B.tigon_sample(mdl, x0, times, n_per=8, device=device)

    def _diffusion(series):    # trained + sampled together -> applier retrains (unused by the 6-method runs)
        return lambda x0: B.bridge_flow(series, times, x0, hidden=K["flow_hidden"], iters=K["flow_iters"],
                                        sigma=K["diff_sigma"], n_tuples=K["global_tuples"], device=device)

    reg = {
        "ours: UOT maps": lambda: _uot(ours_series),
        "ours: flow-matching": lambda: _flow(ours_series),
        "ours: flow global-coupling": lambda: _global(ours_series),
        "ours: diffusion": lambda: _diffusion(ours_series),
        "raw WOT-like": lambda: _uot(raw_series),
        "raw MMFM": lambda: _mmfm(raw_series),
        "raw TIGON": lambda: _tigon(raw_series),
    }
    # explicit flow variants: coupling x (shared | unshared) -- "ours: flow OT-CFM unshared", etc.
    for _lab, _cpl in [("OT-CFM", "ot"), ("UOT-map", "uot-map"),
                       ("minibatch-UOT", "uot"), ("independent", "independent")]:
        for _sh in (True, False):
            reg[f"ours: flow {_lab} {'shared' if _sh else 'unshared'}"] = \
                (lambda c, s: (lambda: _flow_var(ours_series, c, s)))(_cpl, _sh)
    models = {}
    for m in methods:
        if m not in reg:
            raise KeyError(f"unknown method '{m}'; known: {list(reg)}")
        models[m] = reg[m]()          # TRAIN once
    if return_models:
        if not project:               # ambient-dim output (project yourself, e.g. for a full-d W2)
            return {m: models[m] for m in methods}
        return {m: (lambda ap: (lambda x0: G.project_traj(ap(x0))))(models[m]) for m in methods}
    return {m: (G.project_traj(models[m](X0)) if project else models[m](X0)) for m in methods}
