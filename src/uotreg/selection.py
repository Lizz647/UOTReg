"""Reproducible selection rules for the bandwidth ``h`` and the unbalanced tolerance ``tau``.

This module is the concrete, runnable form of the three rules the supplement describes
(Section ``supp:sensitivity``). A referee asked for "a diagnostic target, optimization criterion, or
algorithmic rule" rather than qualitative guidance; each function below is one, stated so that the
value it returns can be recomputed from the data alone.

============  =========================================================================  ==========
rule          criterion                                                                  applies to
============  =========================================================================  ==========
Silverman     ``h = 1.06 * sd(t_1..t_N) * N^(-1/5)``                                      h
ESS           smallest ``h`` on a grid with ``min_t ESS(t, h) >= target``, where           h
              ``ESS(t,h) = (sum_i w_i(t,h)^2)^{-1}`` for the local-linear weights
leave-one-    hold out each INTERIOR observed time in turn, estimate it from the           h and tau
time-out      remaining snapshots, minimise the average held-out W2 over a grid
============  =========================================================================  ==========

**Why interior times only.** The weights in ``frechet_weights`` are local-*linear*: holding out an
endpoint turns the estimate at that time into an extrapolation, which measures something the method
never does in use. With N observed times the LOO criterion therefore averages over N-2 held-out
fits — three for Embryoid, one for Statefate, seven for the bifurcation simulation.

**Why h and tau are selected separately.** A joint grid costs ``|H| x |T|`` fits per held-out time.
Coordinate-wise selection — scan ``h`` at a fixed ``tau_0``, then scan ``tau`` at the selected ``h``
— costs ``|H| + |T|`` and is what these functions do. It is a genuine restriction: it finds a
coordinate-wise minimum, not necessarily the joint one. Report it as such. `loo_select` will run the
joint grid if you ask for it (`joint=True`), which is the check to run once at low budget.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional, Sequence

import numpy as np

from .metrics import emd, mmd, w2
from .pipelines import estimate
from .weights import effective_sample_size, frechet_weights

__all__ = ["silverman_bandwidth", "ess_table", "ess_bandwidth", "interior_times", "loo_select"]


# --------------------------------------------------------------------------------- cheap rules
def silverman_bandwidth(times: Sequence[float]) -> float:
    """``h = 1.06 * sd(t) * N^(-1/5)`` on the OBSERVED time grid — the cheap default.

    Note this is a rule for the *time* axis: it scales the bandwidth to how spread out the
    observation times are, not to the data. It has no knowledge of the distributions, so it cannot
    react to a grid that is uneven in information (e.g. a dense early period and a sparse late one)."""
    t = np.asarray(times, dtype=float)
    return float(1.06 * np.std(t) * len(t) ** (-1 / 5))


def ess_table(times, query_times, bandwidths, *, kernel="gaussian", scheme="positive",
              threshold=0.01):
    """``{h: {t: ESS(t, h)}}`` for the local-linear weights actually used by the estimator.

    ESS(t,h) = 1 / sum_i w_i(t,h)^2, between 1 (all mass on one snapshot) and N (uniform). It is the
    number of snapshots that effectively contribute at t, so it reads directly as "how much am I
    borrowing"."""
    out = {}
    for h in bandwidths:
        row = {}
        for q in query_times:
            wr = frechet_weights(times, q, float(h), kernel=kernel, scheme=scheme,
                                 threshold=threshold)
            row[float(q)] = effective_sample_size(wr.weights)
        out[float(h)] = row
    return out


def ess_bandwidth(times, query_times=None, target=3.0, bandwidths=None, *, kernel="gaussian",
                  scheme="positive", threshold=0.01, verbose=True):
    """Smallest ``h`` on the grid whose ESS reaches ``target`` at EVERY query time.

    Returns ``(h, table)``; ``h`` is ``None`` when the target is unreachable, which is not a failure
    of the search but information about the design: **ESS is bounded above by N**, so a target of 3
    on a three-snapshot grid can only be met by weights that are essentially uniform, and a target
    at or above N is unreachable at any bandwidth. Read a `None` as "this grid is too short for that
    target" and lower it (or use the LOO rule, which has no such ceiling)."""
    times = [float(t) for t in times]
    q = list(query_times) if query_times is not None else interior_times(times)
    hs = list(bandwidths) if bandwidths is not None else \
        list(np.round(np.geomspace(0.25, 20.0, 25), 3))
    tab = ess_table(times, q, hs, kernel=kernel, scheme=scheme, threshold=threshold)
    pick = next((h for h in sorted(tab) if min(tab[h].values()) >= target), None)
    if verbose:
        n = len(times)
        print(f"  [ESS] target {target} over query times {q}: "
              + (f"h = {pick}" if pick is not None else
                 f"NOT REACHABLE (max over the grid = {max(min(r.values()) for r in tab.values()):.2f}; "
                 f"ESS <= N = {n}, so this target needs a longer time grid)"))
    return pick, tab


def interior_times(times):
    """The times that may be held out: everything but the two endpoints (see the module docstring)."""
    t = sorted(float(x) for x in times)
    return t[1:-1]


# --------------------------------------------------------------------------------- LOO
def _score(pred, truth, n_rep, estimator, n_gen, rng):
    """W2 / EMD / MMD of one fit. `n_rep` > 1 RESAMPLES the fitted generator rather than refitting:
    the estimate is stochastic through its sample, and averaging over draws is what the supplement's
    'averaged over repetitions' means. Refitting instead is `n_fit`, which is far more expensive."""
    rows = []
    for r in range(max(1, int(n_rep))):
        p = pred if r == 0 or estimator is None else np.asarray(estimator.sample(n_gen))
        rows.append(dict(w2=w2(p, truth), emd=emd(p, truth), mmd=mmd(p, truth)))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def loo_select(observed, tlist, dim, *, h_grid=(1.0, 2.0, 3.0, 5.0, 7.0, 15.0),
               tau_grid=(1.0, 5.0, 10.0, 20.0), h0=None, tau0=5.0, held_out=None,
               est_kwargs=None, n_rep=1, n_gen=500, seed=0, joint=False,
               cache_path=None, verbose=True):
    """Leave-one-time-point-out selection of ``h`` and ``tau``.

    For each held-out interior time ``t*``: refit the estimator on the remaining snapshots, predict
    the distribution at ``t*``, and score it against the observed cloud there. The selected value
    minimises the mean held-out W2.

    Parameters
    ----------
    observed, tlist : the snapshots and their times (the FULL grid; the held-out one is removed
                      inside, so nothing here has to be pre-split).
    est_kwargs      : passed through to `estimate` — use a SMALL net and a short budget here. This
                      is a selection run over ~30 fits, not a production fit; the ranking of h is
                      what has to be right, not the absolute error.
    n_rep           : draws from each fitted generator (cheap; see `_score`).
    joint           : scan the full ``h x tau`` grid instead of coordinate-wise. Costs
                      ``|H| x |T|`` fits per held-out time — run it once at low budget to check
                      the coordinate-wise answer, not routinely.
    cache_path      : a json that accumulates ``{key: metrics}``; re-running skips finished cells,
                      so a long selection survives an interrupted kernel.

    Returns a dict with ``h_table`` / ``tau_table`` (value -> per-time and mean metrics),
    ``h_selected`` / ``tau_selected``, the reference row, and everything needed to rebuild the
    supplement table.
    """
    tlist = [float(t) for t in tlist]
    est_kwargs = dict(est_kwargs or {})
    held = [float(t) for t in (held_out if held_out is not None else interior_times(tlist))]
    assert held, f"no interior times to hold out in {tlist} (need at least 3 observed times)"
    for t in held:
        assert t in tlist, f"held-out time {t} is not an observed time {tlist}"
    h0 = float(h0 if h0 is not None else silverman_bandwidth(tlist))

    # The cache is keyed by (h, tau, t*) -- which is NOT enough on its own: the same cell computed
    # under a different fitting budget is a different number. Without this signature, raising
    # `budget` and re-running silently MIXES old cheap cells with new expensive ones and the ranking
    # becomes meaningless. A mismatch discards the cache rather than half-using it.
    _sig = {"est": {k: est_kwargs[k] for k in sorted(est_kwargs)}, "n_rep": int(n_rep),
            "n_gen": int(n_gen), "seed": int(seed), "dim": int(dim),
            "tlist": [float(t) for t in tlist], "held": held}
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            _raw = json.load(open(cache_path))
            if _raw.get("_sig") == _sig:
                cache = {k: v for k, v in _raw.items() if k != "_sig"}
            else:
                diff = [k for k in _sig if _raw.get("_sig", {}).get(k) != _sig[k]]
                print(f"  [cache] {os.path.basename(cache_path)} was written under a DIFFERENT "
                      f"configuration (differs in: {', '.join(diff) or 'unknown'}) -- discarding it "
                      f"and refitting. Nothing is mixed.", flush=True)
        except Exception:
            cache = {}

    def _fit(h, tau, tstar):
        key = f"h={h:g}|tau={tau:g}|t={tstar:g}"
        if key in cache:
            return cache[key], True
        keep = [i for i, t in enumerate(tlist) if t != tstar]
        obs_k = [np.asarray(observed[i], np.float32) for i in keep]
        t_k = [tlist[i] for i in keep]
        truth = np.asarray(observed[tlist.index(tstar)], np.float32)
        t0 = time.time()
        pred, est = estimate(obs_k, t_k, query_time=tstar, dim=dim, h=float(h), tau=float(tau),
                             seed=seed, n_gen=n_gen, return_estimator=True, **est_kwargs)
        m = _score(np.asarray(pred), truth, n_rep, est, n_gen, np.random.default_rng(seed))
        m["minutes"] = (time.time() - t0) / 60
        cache[key] = m
        if cache_path:
            json.dump({"_sig": _sig, **cache}, open(cache_path, "w"), indent=1)
        return m, False

    # the reference every row is read against: the nearest REMAINING observed cloud, i.e. what you
    # get with no model at all. A grid point that does not beat this is not selecting anything.
    ref = {}
    for tstar in held:
        keep = [i for i, t in enumerate(tlist) if t != tstar]
        j = min(keep, key=lambda i: abs(tlist[i] - tstar))
        ref[tstar] = _score(np.asarray(observed[j], np.float32),
                            np.asarray(observed[tlist.index(tstar)], np.float32), 1, None, n_gen, None)
    ref_mean = {k: float(np.mean([ref[t][k] for t in held])) for k in ("w2", "emd", "mmd")}

    def _sweep(name, values, fixed):
        table = {}
        for v in values:
            h, tau = (float(v), fixed) if name == "h" else (fixed, float(v))
            per = {}
            for tstar in held:
                m, hit = _fit(h, tau, tstar)
                per[tstar] = m
                if verbose:
                    how = "cached" if hit else "%.1f min" % m["minutes"]
                    print(f"    {name}={v:<6g} t*={tstar:<6g} W2={m['w2']:.3f} ({how})", flush=True)
            table[float(v)] = dict(per_time=per,
                                   **{k: float(np.mean([per[t][k] for t in held]))
                                      for k in ("w2", "emd", "mmd")})
        return table

    out = dict(tlist=tlist, held_out=held, dim=dim, h_grid=list(map(float, h_grid)),
               tau_grid=list(map(float, tau_grid)), h0=h0, tau0=float(tau0),
               reference=dict(per_time=ref, **ref_mean), est_kwargs=est_kwargs,
               n_rep=n_rep, seed=seed, joint=bool(joint))

    if joint:
        grid = {}
        for h in h_grid:
            for tau in tau_grid:
                per = {t: _fit(float(h), float(tau), t)[0] for t in held}
                grid[f"h={float(h):g}|tau={float(tau):g}"] = dict(
                    h=float(h), tau=float(tau), per_time=per,
                    **{k: float(np.mean([per[t][k] for t in held])) for k in ("w2", "emd", "mmd")})
        best = min(grid, key=lambda k: grid[k]["w2"])
        out.update(joint_grid=grid, h_selected=grid[best]["h"], tau_selected=grid[best]["tau"])
    else:
        if verbose:
            print(f"  [LOO] stage 1: h over {list(h_grid)} at tau={tau0}")
        ht = _sweep("h", h_grid, float(tau0))
        h_sel = min(ht, key=lambda v: ht[v]["w2"])
        if verbose:
            print(f"  [LOO] h -> {h_sel}   (mean held-out W2 {ht[h_sel]['w2']:.3f})")
            print(f"  [LOO] stage 2: tau over {list(tau_grid)} at h={h_sel}")
        tt = _sweep("tau", tau_grid, h_sel)
        tau_sel = min(tt, key=lambda v: tt[v]["w2"])
        if verbose:
            print(f"  [LOO] tau -> {tau_sel}   (mean held-out W2 {tt[tau_sel]['w2']:.3f})")
        out.update(h_table=ht, tau_table=tt, h_selected=float(h_sel), tau_selected=float(tau_sel))
    return out


def loo_markdown(res, title="", used=None):
    """The result as a markdown table, ready for the supplement. `used` = {'h':..,'tau':..}, the
    value the paper reports, printed alongside so agreement (or not) is explicit rather than implied."""
    L = [f"### {title}" if title else "", ""]
    L.append(f"Held-out interior times: {res['held_out']}  |  "
             f"reference (nearest remaining snapshot): W2 {res['reference']['w2']:.3f}")
    L.append("")
    for key, name, sel in (("h_table", "h", "h_selected"), ("tau_table", "tau", "tau_selected")):
        if key not in res:
            continue
        tab = res[key]
        L.append(f"| {name} | " + " | ".join(f"W2(t*={t:g})" for t in res["held_out"])
                 + " | mean W2 | mean EMD | mean MMD |")
        L.append("|" + "---|" * (len(res["held_out"]) + 4))
        for v in sorted(tab):
            star = " **<-**" if v == res[sel] else ""
            L.append(f"| {v:g}{star} | "
                     + " | ".join(f"{tab[v]['per_time'][t]['w2']:.3f}" for t in res["held_out"])
                     + f" | {tab[v]['w2']:.3f} | {tab[v]['emd']:.3f} | {tab[v]['mmd']:.3f} |")
        L.append("")
    L.append(f"**Selected: h = {res.get('h_selected')}, tau = {res.get('tau_selected')}**"
             + (f"   (used in the paper: h = {used.get('h')}, tau = {used.get('tau')})" if used else ""))
    return "\n".join(L)
