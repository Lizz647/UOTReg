"""Plotting helpers (matplotlib imported lazily so the core has no hard dep).

The headline is :func:`live_dashboard`, a training callback that draws the 1x3
panel during ``DistributionEstimator.fit``:

    (1) G regression loss (log10)   (2) stationarity residual R_n
    (3) current generated samples over the data in PC space.

Panel (2) is the new "stationarity check": R_n = E_mu||x - Tbar(x)||^2, which
should decrease toward 0 (the revision's vanishing-residual stopping rule).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _lazy_plt():
    import matplotlib.pyplot as plt
    return plt


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def plot_training_summary(estimator, background=None, highlight=None, dims=(0, 1),
                          n_samples=2000, ax=None, title=""):
    """Static 1x3 summary after fitting (G loss / residual / PCA scatter)."""
    plt = _lazy_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), dpi=120)
    g = np.asarray(estimator.history["g_loss"])
    r = np.asarray(estimator.history["residual"])
    axes[0].plot(np.log10(np.clip(g, 1e-12, None)))
    axes[0].set_title("G loss (regression), log10"); axes[0].set_xlabel("outer iter")
    axes[1].plot(r, color="C3")
    axes[1].set_title(r"stationarity residual $R_n=E\|x-\bar T x\|^2$")
    axes[1].set_xlabel("outer iter"); axes[1].set_yscale("log")

    ax = axes[2]
    i, j = dims
    bg, hl = _to_np(background), _to_np(highlight)
    if bg is not None:
        ax.scatter(bg[:, i], bg[:, j], s=6, c="lightgrey", alpha=0.5, label="all cells")
    if hl is not None:
        ax.scatter(hl[:, i], hl[:, j], s=8, c="lightblue", alpha=0.7, label="target")
    gen = estimator.sample(n_samples)
    ax.scatter(gen[:, i], gen[:, j], s=8, c="lightcoral", alpha=0.7, label="generated")
    ax.set_xlabel(f"PC{i+1}"); ax.set_ylabel(f"PC{j+1}")
    ax.set_title("generated vs data"); ax.legend(fontsize=7, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def live_dashboard(background=None, highlight=None, dims=(0, 1), every: int = 2,
                   n_samples: int = 2000):
    """Return a ``fit(callback=...)`` closure that live-updates the 1x3 dashboard.

    Example
    -------
    >>> from uotreg.plotting import live_dashboard
    >>> est.fit(..., callback=live_dashboard(background=pooled_pc, highlight=target_pc))
    """
    bg, hl = _to_np(background), _to_np(highlight)

    def _cb(estimator, outer):
        if outer % every != 0 and outer != estimator.train_cfg.outer_iters - 1:
            return
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except Exception:
            pass
        fig = plot_training_summary(estimator, background=bg, highlight=hl,
                                    dims=dims, n_samples=n_samples,
                                    title=f"outer iter {outer+1}")
        plt = _lazy_plt()
        plt.show(); plt.close(fig)

    return _cb


def scatter_2d(points_by_label: dict, dims=(0, 1), ax=None, s=10, alpha=0.7,
               title="", xlabel=None, ylabel=None):
    """Scatter several labelled point clouds on shared axes."""
    plt = _lazy_plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=120)
    i, j = dims
    for label, pts in points_by_label.items():
        pts = _to_np(pts)
        ax.scatter(pts[:, i], pts[:, j], s=s, alpha=alpha, label=str(label))
    ax.set_xlabel(xlabel or f"dim {i}"); ax.set_ylabel(ylabel or f"dim {j}")
    ax.set_title(title); ax.legend(fontsize=7, loc="best")
    return ax


def plot_trajectories_2d(traj, dims=(0, 1), ax=None, max_cells=40, color="k",
                         alpha=0.5, lw=0.8, title="trajectories"):
    """Plot composed trajectories of shape (n_steps+1, n_cells, d) as polylines."""
    plt = _lazy_plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=120)
    traj = _to_np(traj)
    i, j = dims
    n_cells = traj.shape[1]
    idx = np.arange(n_cells)
    if n_cells > max_cells:
        idx = np.random.default_rng(0).choice(n_cells, max_cells, replace=False)
    for c in idx:
        ax.plot(traj[:, c, i], traj[:, c, j], color=color, alpha=alpha, lw=lw)
    ax.scatter(traj[0, idx, i], traj[0, idx, j], s=15, c="tab:green", label="start", zorder=3)
    ax.scatter(traj[-1, idx, i], traj[-1, idx, j], s=15, c="tab:red", label="end", zorder=3)
    ax.set_xlabel(f"dim {i}"); ax.set_ylabel(f"dim {j}")
    ax.set_title(title); ax.legend(fontsize=7)
    return ax


# --------------------------------------------------------------------------- #
# batch-effect simulation panels (duck-typed on a `GeomHD`-like G)
# --------------------------------------------------------------------------- #
def show_noisy_data(G, times, clean=True, s=6, figsize_per=2.4):
    """Scatter the OBSERVED (batch-perturbed) clouds at each time on the 2-D signal plane, with the
    clean truth underneath. Shows the noisy dataset the estimator has to denoise. `times` = indices."""
    plt = _lazy_plt()
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per * n, 3.0), dpi=110, sharex=True, sharey=True)
    for ax, t in zip(np.atleast_1d(axes), times):
        Op = G.project2d(G.observed[t])
        if clean:
            Cp = G.project2d(G.truth(t)); ax.scatter(Cp[:, 0], Cp[:, 1], s=s, c="0.75", alpha=0.35)
        ax.scatter(Op[:, 0], Op[:, 1], s=s, c="tab:orange", alpha=0.5)
        ax.plot(G.curves[:, 0, 0], G.curves[:, 0, 1], "b--", lw=0.5, alpha=0.6)
        ax.plot(G.curves[:, 1, 0], G.curves[:, 1, 1], "b--", lw=0.5, alpha=0.6)
        ax.set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"[{G.kind}] observed (orange) vs clean truth (grey) at each time")
    fig.tight_layout()
    return fig


def show_estimates(G, est_series, times, w2_fn=None, s=6, figsize_per=2.4):
    """Scatter the DENOISED estimate (purple) vs truth (grey) at each predicted time. If `w2_fn` is
    given (e.g. uotreg.metrics.w2) the panel titles show W2(est,truth) vs W2(raw,truth)."""
    plt = _lazy_plt()
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per * n, 3.0), dpi=110, sharex=True, sharey=True)
    for ax, t, g in zip(np.atleast_1d(axes), times, est_series):
        Tp, gp = G.project2d(G.truth(t)), G.project2d(g)
        ax.scatter(Tp[:, 0], Tp[:, 1], s=s - 1, c="0.7", alpha=0.35)
        ax.scatter(gp[:, 0], gp[:, 1], s=s, c="tab:purple", alpha=0.5)
        ax.plot(G.curves[:, 0, 0], G.curves[:, 0, 1], "b--", lw=0.5, alpha=0.6)
        ax.plot(G.curves[:, 1, 0], G.curves[:, 1, 1], "b--", lw=0.5, alpha=0.6)
        ttl = f"t={t}"
        if w2_fn is not None:
            ttl += f"\nW2 {w2_fn(g, G.truth(t)):.2f} (raw {w2_fn(G.observed[t], G.truth(t)):.2f})"
        ax.set_title(ttl, fontsize=7)
    fig.suptitle(f"[{G.kind}] denoised estimate (purple) vs truth (grey)")
    fig.tight_layout()
    return fig


def plot_trajectory_panel(G, trajs, title=None, max_cells=60, curves=True):
    """Side-by-side trajectory panels (one per method in `trajs`, each a (T,N,2) projected trajectory),
    with the true branch curves dashed. Returns the figure."""
    plt = _lazy_plt()
    n = len(trajs)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.9), dpi=110, sharex=True, sharey=True)
    for ax, (nm, t) in zip(np.atleast_1d(axes), trajs.items()):
        plot_trajectories_2d(np.asarray(t), ax=ax, max_cells=max_cells, title=nm.replace(" (", "\n("))
        if curves:
            ax.plot(G.curves[:, 0, 0], G.curves[:, 0, 1], "b--", lw=1, alpha=0.6)
            ax.plot(G.curves[:, 1, 0], G.curves[:, 1, 1], "b--", lw=1, alpha=0.6)
    fig.suptitle(title or f"[{G.kind}] trajectories")
    fig.tight_layout()
    return fig
