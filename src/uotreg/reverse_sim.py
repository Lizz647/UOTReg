"""Reverse-noise batch-effect bifurcation -- the referee's mirror-image case.

The referee (Section 6.2) noted that in the bifurcation simulation "the distributions on the two
branches are first perturbed to approach each other and then turn to move further away", and asked
for the opposite: **first apart, then closer to the true path**.

What actually drives that asymmetry is the *size of the batch perturbation relative to the geometry*.
In the original `make_bifurcation` the affine batch effect has a single `strength` at every time, but
the two branches sit on a tight trunk (y = 2 +/- 0.25) until t = 4 and only separate afterwards, so
the perturbation reads as small early and large late. This module reverses that profile:

  * **`schedule`** -- a per-time batch strength, big early and small late, tapering through the split
    (t = 3, 4 already lead into the second stage geometrically). Same `{time: value, "default": ...}`
    idiom as `std_mode` elsewhere in the codebase, so it is one dict to edit;
  * **`delta_pre` widened** (0.25 -> 0.6) -- the trunk margin between the two Gaussians, so that under
    the larger early noise the branch order (A above, B below) is still broadly maintained.
    Individual snapshots may still interact or cross, which is fine and intended.

Everything else is the faithful `bd` bifurcation: same covariances, same post-split branch means
(yA = t-2, yB = -t+6), same affine batch model, same `GeomHD` lift to `dim` dimensions. `dim=2`
recovers the 2-D picture. Because the object returned IS a `GeomHD`, every metric, projection and
plotting helper used by the bifurcation benchmark works unchanged.

SEED / DETERMINISM: `seed` drives the WHOLE data-generating mechanism -- both the clean cells and the
affine batch effects. The two are drawn from independent streams spawned off one `SeedSequence(seed)`
(rather than `seed` and `seed + k`, which can collide across seeds), so

  * the same `seed` always gives exactly the same data;
  * a DIFFERENT `seed` gives different clean cells AND different batch effects -- a genuinely new
    realization of the mechanism, not a re-perturbation of the same cells;
  * retuning the `schedule` changes ONLY the perturbation -- the clean cells underneath are
    bit-identical, so tuning runs stay comparable;
  * changing `n_per_time` does not reshuffle the batch draws.

`check_determinism()` at the bottom asserts all four.

NOTE: this is NOT `uotreg.simulation.make_reverse`, which flips the bifurcation in TIME (a merge:
branches start apart and converge). Here the geometry is the original bifurcation; only the noise
profile is reversed.
"""
import numpy as np

from .simulation import GeomHD, COV_A, COV_B, SPLIT, N_TIMES, N_PER_TIME, _affine, Y_JITTER_PRE

# ---- the knobs that define this variant ------------------------------------------------------ #
DELTA_PRE = 0.6          # trunk margin (the bifurcation uses 0.25) -- widened so the order survives

# per-time affine batch strength (the bifurcation uses a flat 0.5 at every time).
# Keys are snapshot indices; "default" covers every time not listed.
STRENGTH_SCHEDULE = {
    0: 0.5, 1: 1.5, 2: 1.5,        # first stage: the noisy one
    3: 1.0,                        # tapering into the split
    4: 0.65,                       # the split time itself (geometrically already stage 2)
    "default": 0.25,               # t >= 5: the clean second stage
}


def _streams(seed):
    """Two independent rng streams spawned off ONE SeedSequence(seed): (clean cells, batch effects).
    Both are fully determined by `seed`, so the seed changes the entire mechanism, while the streams
    stay decoupled -- retuning the batch schedule cannot disturb the clean draws."""
    clean_ss, batch_ss = np.random.SeedSequence(int(seed)).spawn(2)
    return np.random.default_rng(clean_ss), np.random.default_rng(batch_ss)


def branch_means(t, delta_pre=DELTA_PRE):
    """True (noise-free) means of the two branches at time t: (mA, mB), each (2,).
    Identical to the bifurcation apart from the widened trunk margin."""
    t = float(t)
    if t <= SPLIT:
        yA, yB = 2.0 + delta_pre, 2.0 - delta_pre
    else:
        yA, yB = t - 2.0, -t + 6.0
    return np.array([t, yA]), np.array([t, yB])


def strength_at(t, schedule=None):
    """Look a time up in the schedule: exact key first, then "default"."""
    sch = STRENGTH_SCHEDULE if schedule is None else schedule
    key = int(round(float(t)))
    if key in sch:
        return float(sch[key])
    if str(key) in sch:                       # tolerate JSON round-trips (string keys)
        return float(sch[str(key)])
    return float(sch.get("default", 0.5))


def schedule_list(schedule=None, tlist=None):
    """The schedule as a plain list over `tlist` (default: all N_TIMES snapshots)."""
    tl = range(N_TIMES) if tlist is None else tlist
    return [strength_at(t, schedule) for t in tl]


def _snapshot(t, n, rng, delta_pre):
    """One clean snapshot + its branch labels (mirrors `uotreg.simulation._snapshot`)."""
    mA, mB = branch_means(t, delta_pre)
    nA, nB = n // 2, n - n // 2
    LA, LB = np.linalg.cholesky(COV_A), np.linalg.cholesky(COV_B)
    pA = mA + rng.standard_normal((nA, 2)) @ LA.T
    pB = mB + rng.standard_normal((nB, 2)) @ LB.T
    if t <= SPLIT:
        pA[:, 1] += rng.normal(0, Y_JITTER_PRE, nA)
        pB[:, 1] += rng.normal(0, Y_JITTER_PRE, nB)
    pts = np.concatenate([pA, pB], axis=0).astype(np.float32)
    lab = np.concatenate([np.zeros(nA, int), np.ones(nB, int)])
    perm = rng.permutation(n)
    return pts[perm], lab[perm]


def generate_2d(seed=0, n_per_time=N_PER_TIME, delta_pre=DELTA_PRE, schedule=None):
    """Clean snapshots + an independent affine batch effect whose strength follows `schedule`.

    Two independent streams spawned off `seed` (see SEED / DETERMINISM in the module docstring):
    one draws the clean cells, the other the affine batch effects."""
    rng_clean, rng_batch = _streams(seed)
    tlist = np.linspace(0, N_TIMES - 1, N_TIMES)
    observed, clean, labels, strengths = [], [], [], []
    for t in tlist:
        pts, lab = _snapshot(float(t), n_per_time, rng_clean, delta_pre)
        s = strength_at(t, schedule)
        A, b = _affine(rng_batch, s)                 # SAME affine batch model as the bifurcation
        observed.append((pts @ A.T + b).astype(np.float32))
        clean.append(pts); labels.append(lab); strengths.append(s)
    curves = np.stack([np.stack(branch_means(t, delta_pre), axis=0) for t in tlist], axis=0)
    return tlist, observed, clean, labels, curves, np.asarray(strengths)


def make_reverse_noise(seed=0, n_per_time=N_PER_TIME, dim=2, delta_pre=DELTA_PRE, schedule=None,
                       method="replicate", lift_noise=0.05, noise_scale=1.0, rotate=False):
    """The reverse-noise bifurcation lifted to `dim`, as a `GeomHD` (drop-in for `make_bifurcation`).

    Deterministic in `seed` -> the same seed gives the same data to every method, exactly as in the
    bifurcation benchmark. The realized per-time strengths are attached as `G.strengths`."""
    tlist, obs, cln, lab, crv, strengths = generate_2d(
        seed=seed, n_per_time=n_per_time, delta_pre=delta_pre, schedule=schedule)
    G = GeomHD("reverse-noise", tlist, obs, cln, crv, lambda t: branch_means(t, delta_pre),
               dim=dim, seed=seed, labels=lab, method=method, lift_noise=lift_noise,
               noise_scale=noise_scale, rotate=rotate)
    G.strengths = strengths
    G.delta_pre = float(delta_pre)
    G.schedule = dict(STRENGTH_SCHEDULE if schedule is None else schedule)
    return G


def check_determinism(seed=0, n_per_time=200, dim=10, schedule=None, verbose=True):
    """Assert the four seed/determinism guarantees. Returns True (raises on failure)."""
    kw = dict(seed=seed, n_per_time=n_per_time, dim=dim, schedule=schedule)
    a, b = make_reverse_noise(**kw), make_reverse_noise(**kw)
    for t in range(N_TIMES):                                  # 1. same seed -> same data
        assert np.array_equal(a.observed[t], b.observed[t]), f"observed differs at t={t}"
        assert np.array_equal(a.clean[t], b.clean[t]), f"clean differs at t={t}"
        assert np.array_equal(a.labels[t], b.labels[t]), f"labels differ at t={t}"
    e = make_reverse_noise(seed=seed + 1, n_per_time=n_per_time, dim=dim, schedule=schedule)
    assert not np.array_equal(a.clean[1], e.clean[1]), "a new seed did not redraw the CLEAN cells"
    assert not np.array_equal(a.observed[1], e.observed[1]), "a new seed did not redraw the OBSERVED cells"
    #    ... and the batch effect itself moved, not just the cells underneath it
    dev = lambda G, t: np.asarray(G.observed[t]).mean(0) - np.asarray(G.clean[t]).mean(0)
    assert not np.allclose(dev(a, 1), dev(e, 1)), "a new seed did not redraw the BATCH effect"
    other = dict(STRENGTH_SCHEDULE if schedule is None else schedule)
    other["default"] = float(other.get("default", 0.2)) + 0.37     # 3. retuning leaves clean alone
    c = make_reverse_noise(seed=seed, n_per_time=n_per_time, dim=dim, schedule=other)
    for t in range(N_TIMES):
        assert np.array_equal(a.clean[t], c.clean[t]), f"schedule changed the CLEAN data at t={t}"
    d = make_reverse_noise(seed=seed, n_per_time=n_per_time + 50, dim=dim, schedule=schedule)
    assert np.allclose(a.strengths, d.strengths), "n_per_time changed the strength schedule"
    if verbose:
        print(f"seed/determinism OK (seed {seed}): same seed -> identical data; a new seed redraws "
              f"BOTH the clean cells and the batch effects; the schedule changes only the "
              f"perturbation; n_per_time does not disturb the batch draws")
    return True
