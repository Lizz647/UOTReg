r"""Trajectory-inference baselines + our flow variants, as a package module.

Self-contained (torch + numpy + scipy; no POT/torchdiffeq/torchcfm) reimplementations of published
methods, plus the two OT-CFM variants used in the paper. Consolidates the previously-scattered
`traj_baselines.py` and the notebook `_global_coupling_flow` / `_bridge_flow` helpers.

* **MMFM** (Rohbeck et al., ICLR'25): `mmfm_fit` + `ode_sample` -- chained minibatch-OT tuples ->
  an interpolant through all K+1 marginals (`spline="cubic"` natural cubic | `"linear"`, both from the
  paper) -> velocity field regressed to the interpolant derivative. Generation is FORWARD-time.
* **TIGON** (Sha et al., NMI'23): `tigon_fit` (robust simulation-matched sliced-W variant, the one
  used) / `tigon_fit_cnf` (literal CNF port, slow) + `tigon_sample`.
* **ours flows**: `global_coupling_flow` (OT-CFM on ONE chained-OT coupling across all times -> keeps
  each cell's branch identity) and `bridge_flow` (stochastic-bridge / [SF]^2M: noisy Brownian-bridge
  interpolants + Euler-Maruyama SDE, `sigma` diffusion strength).

For 1:1 paper numbers run the authors' code (see `reproduce`/`additional/TIGON`); these are the
dependency-free ports used in the main comparison.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment


def _dev(device):
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------------------------------------------------------- #
# shared: time-conditioned velocity MLP + RK4 integrator
# --------------------------------------------------------------------------- #
class VelocityMLP(nn.Module):
    """v(x, t): concatenates a small sinusoidal time embedding with x."""
    def __init__(self, dim, hidden=128, layers=4, n_freq=8, act=nn.SELU):
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * np.pi)
        din = dim + 2 * n_freq
        net = [nn.Linear(din, hidden), act()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), act()]
        net += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*net)

    def temb(self, t):
        a = t * self.freqs
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def forward(self, x, t):
        if t.dim() == 1:
            t = t[:, None]
        return self.net(torch.cat([x, self.temb(t)], -1))


def rk4_sample(field, x0, times, n_per=10, device="cpu"):
    """Integrate dx/dt = field(x,t) with RK4; return the cloud at each time. (len(times), N, d)."""
    field.eval()
    x = torch.as_tensor(np.asarray(x0), dtype=torch.float32, device=device)
    out = [x.clone()]
    with torch.no_grad():
        for k in range(len(times) - 1):
            t0, t1 = float(times[k]), float(times[k + 1]); h = (t1 - t0) / n_per
            for j in range(n_per):
                t = t0 + j * h; tt = torch.full((x.shape[0], 1), t, device=device)
                k1 = field(x, tt); k2 = field(x + 0.5 * h * k1, tt + 0.5 * h)
                k3 = field(x + 0.5 * h * k2, tt + 0.5 * h); k4 = field(x + h * k3, tt + h)
                x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(x.clone())
    return np.stack([o.cpu().numpy() for o in out], 0)


# --------------------------------------------------------------------------- #
# MMFM
# --------------------------------------------------------------------------- #
def _chain_ot_tuples(clouds: Sequence[np.ndarray], n: int, seed: int = 0) -> np.ndarray:
    """(n, K+1, d) OT-coupled trajectory tuples by chained minibatch OT (exact assignment per pair)."""
    rng = np.random.default_rng(seed)
    S = [np.asarray(C, np.float32)[rng.choice(len(C), n, replace=len(C) < n)] for C in clouds]
    aligned = [S[0]]; prev = S[0]
    for k in range(1, len(S)):
        cost = ((prev[:, None, :] - S[k][None, :, :]) ** 2).sum(-1)
        _, col = linear_sum_assignment(cost)
        cur = S[k][col]; aligned.append(cur); prev = cur
    return np.stack(aligned, axis=1)


def _linear_diag(times, X, tq):
    """Per-tuple piecewise-LINEAR interpolant on the diagonal: for query time tq[i] and tuple i, return
    position P[i] and (segment-slope) velocity dP[i]. times (K,), X (n,K,d), tq (n,) -> P,dP (n,d).
    The linear-interpolation option from the MMFM paper (cubic's simpler, non-oscillating counterpart)."""
    ar = np.arange(len(tq))
    k = np.clip(np.searchsorted(times, tq, side="right") - 1, 0, len(times) - 2)   # segment per tuple
    t0, t1 = times[k], times[k + 1]
    x0, x1 = X[ar, k], X[ar, k + 1]
    slope = (x1 - x0) / (t1 - t0)[:, None]
    return x0 + slope * (tq - t0)[:, None], slope


def _ppoly_diag(pp, tq):
    """Evaluate a scipy PPoly built over axis=1 (coeffs (deg+1, K-1, n, d)) ON THE DIAGONAL:
    result[i] = pp_i(tq[i]) for each tuple i. O(n) via the coefficient array -- avoids the O(n^2)
    `pp(tq)[arange, arange]` full cross-product (which made cubic MMFM ~n_tuples times too slow)."""
    x = pp.x
    seg = np.clip(np.searchsorted(x, tq, side="right") - 1, 0, len(x) - 2)   # segment per tuple
    dx = (tq - x[seg])[:, None]                                              # (n, 1)
    c = pp.c[:, seg, np.arange(len(tq)), :]                                  # (deg+1, n, d)
    out = c[0].copy()
    for p in range(1, c.shape[0]):                                          # Horner
        out = out * dx + c[p]
    return out


def mmfm_fit(clouds, times, dim=2, hidden=128, layers=4, iters=3000, lr=1e-3,
             sigma=0.0, n_tuples=200, coupling="ot", spline="cubic", device="cpu", seed=0, verbose=False):
    """Fit the MMFM velocity field (interpolant through all marginals). Returns the trained VelocityMLP.
    `spline`: "cubic" (natural cubic spline, smooth velocity) | "linear" (piecewise-linear = chained
    OT-CFM between consecutive marginals, non-oscillating). Both are the interpolants offered in the paper."""
    device = _dev(device); torch.manual_seed(seed); times = np.asarray(times, float)
    n = min(n_tuples, min(len(C) for C in clouds))
    if coupling == "ot":
        X = _chain_ot_tuples(clouds, n, seed)
    else:
        rng = np.random.default_rng(seed)
        X = np.stack([np.asarray(C, np.float32)[rng.choice(len(C), n, replace=len(C) < n)]
                      for C in clouds], axis=1)
    if spline == "cubic":
        sp = CubicSpline(times, X, axis=1); dsp = sp.derivative()
    elif spline != "linear":
        raise ValueError(f"spline must be 'cubic' or 'linear', got {spline!r}")
    field = VelocityMLP(dim, hidden, layers).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    t_lo, t_hi = float(times[0]), float(times[-1]); field.train()
    rng_t = np.random.default_rng(seed)          # seed the time sampling too (reproducible given `seed`)
    for it in range(iters):
        tq = rng_t.uniform(t_lo, t_hi, size=n)
        if spline == "linear":
            P, dP = _linear_diag(times, X, tq)
        else:
            P = _ppoly_diag(sp, tq); dP = _ppoly_diag(dsp, tq)
        xt = torch.as_tensor(P, dtype=torch.float32, device=device)
        if sigma:
            xt = xt + sigma * torch.randn_like(xt)
        ut = torch.as_tensor(dP, dtype=torch.float32, device=device)
        tt = torch.as_tensor(tq, dtype=torch.float32, device=device)[:, None]
        loss = ((field(xt, tt) - ut) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it % max(1, iters // 8) == 0:
            print(f"[mmfm] {it}/{iters} loss={loss.item():.4e}")
    field.eval()
    return field


def ode_sample(field, X0, times, n_per=10, device="cpu"):
    """RK4-integrate a fitted velocity field through `times`."""
    return rk4_sample(field, X0, times, n_per=n_per, device=_dev(device))


# --------------------------------------------------------------------------- #
# TIGON (growth-augmented CNF; from-scratch)
# --------------------------------------------------------------------------- #
class TIGONField(nn.Module):
    """Velocity net v(x,t) and growth net g(x,t) (both Tanh MLPs on [t, x])."""
    def __init__(self, dim, hidden=32, v_layers=4, g_layers=3):
        super().__init__()
        def mlp(out, L):
            net = [nn.Linear(dim + 1, hidden), nn.Tanh()]
            for _ in range(L - 1):
                net += [nn.Linear(hidden, hidden), nn.Tanh()]
            net += [nn.Linear(hidden, out)]
            return nn.Sequential(*net)
        self.v = mlp(dim, v_layers); self.g = mlp(1, g_layers)

    def _in(self, x, t):
        tt = torch.full((x.shape[0], 1), float(t), device=x.device)
        return torch.cat([tt, x], -1)

    def vel(self, x, t):
        return self.v(self._in(x, t))

    def growth(self, x, t):
        return self.g(self._in(x, t))


def _sliced_w(x, y, n_proj=48):
    d = x.shape[1]; th = torch.randn(d, n_proj, device=x.device); th = th / (th.norm(dim=0, keepdim=True) + 1e-8)
    xs, _ = torch.sort(x @ th, dim=0); ys, _ = torch.sort(y @ th, dim=0)
    return ((xs - ys) ** 2).mean()


def _forward_flow(field, x0, times, n_per):
    x = x0; clouds = [x]
    for k in range(len(times) - 1):
        t0, t1 = float(times[k]), float(times[k + 1]); h = (t1 - t0) / n_per
        for j in range(n_per):
            x = x + h * field.vel(x, t0 + j * h)
        clouds.append(x)
    return clouds


def tigon_fit(clouds, times, dim=2, hidden=32, v_layers=4, g_layers=3, iters=1500, lr=3e-3,
              n_samples=128, n_per=5, kinetic=0.02, growth_pen=0.1, device="cpu", seed=0, verbose=False):
    """Fit TIGON's velocity + growth fields by simulation-based marginal matching (robust,
    dependency-free variant of TIGON's dynamic unbalanced OT). Returns TIGONField."""
    device = _dev(device); torch.manual_seed(seed); times = np.asarray(times, float)
    C = [torch.as_tensor(np.asarray(c, np.float32), device=device) for c in clouds]
    field = TIGONField(dim, hidden, v_layers, g_layers).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=lr)

    def draw(i, n):
        return C[i][torch.randint(0, C[i].shape[0], (n,), device=device)]

    field.train()
    for it in range(iters):
        x0 = draw(0, n_samples)
        clouds_t = _forward_flow(field, x0, times, n_per)
        loss = sum(_sliced_w(clouds_t[i], draw(i, n_samples)) for i in range(1, len(C)))
        xr = draw(np.random.randint(0, len(C)), n_samples)
        tr = float(np.random.uniform(times[0], times[-1]))
        vv, gg = field.vel(xr, tr), field.growth(xr, tr)
        loss = loss + kinetic * (vv ** 2).sum(-1).mean() + growth_pen * (gg ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it % max(1, iters // 8) == 0:
            print(f"[tigon] {it}/{iters} loss={float(loss):.4e}")
    field.eval()
    return field


def tigon_sample(field, X0, times, n_per=10, device="cpu"):
    """Integrate TIGON's velocity field forward through `times` (RK4). (len(times), N, d)."""
    device = _dev(device)

    class _V(nn.Module):
        def __init__(s): super().__init__(); s.f = field
        def forward(s, x, t): return s.f.vel(x, float(t.flatten()[0]))
    return rk4_sample(_V().to(device), X0, times, n_per=n_per, device=device)


# --------------------------------------------------------------------------- #
# ours: OT-CFM flow variants (global coupling + stochastic bridge)
# --------------------------------------------------------------------------- #
def global_coupling_flow(series, times, X0, hidden=256, iters=3000, n_tuples=200,
                         device="cpu", seed=0, n_per=10, return_field=False):
    """OT-CFM (single shared field) trained on ONE globally-consistent chained-OT coupling across all
    times, so each cell keeps a single branch identity (anchored by the well-separated late marginals)
    -> fixes the split imbalance of independent consecutive-pair OT-CFM. (len(times), N, d)."""
    device = _dev(device)
    d = int(np.asarray(series[0]).shape[1]); tvec = np.asarray(times, float)
    n = min(n_tuples, min(len(np.asarray(c)) for c in series))
    tup = torch.as_tensor(_chain_ot_tuples(series, n, seed=seed), dtype=torch.float32, device=device)
    field = VelocityMLP(d, hidden=hidden, layers=4).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=1e-3); field.train()
    for _ in range(iters):
        k = np.random.randint(0, len(tvec) - 1); t0, t1 = float(tvec[k]), float(tvec[k + 1])
        x0, x1 = tup[:, k], tup[:, k + 1]; s = torch.rand(n, 1, device=device)
        xt = (1 - s) * x0 + s * x1; tt = t0 + s * (t1 - t0); target = (x1 - x0) / (t1 - t0)
        loss = ((field(xt, tt) - target) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    field.eval()
    if return_field:                 # let callers reuse the trained field on new start cells
        return field
    return rk4_sample(field, X0, tvec, n_per=n_per, device=device)


def bridge_flow(series, times, X0, hidden=256, iters=3000, sigma=0.3, n_tuples=200,
                device="cpu", seed=0, n_steps=100):
    """Stochastic-bridge / [SF]^2M 'diffusion' trajectory: an OT-CFM drift trained on NOISY
    Brownian-bridge interpolants over the global chained-OT coupling, sampled by Euler-Maruyama as an
    SDE dx = v dt + sigma*dW. The noise lets the SDE split a connected blob and blurs the middle gap;
    sigma=0 recovers `global_coupling_flow`. (len(times), N, d)."""
    device = _dev(device)
    d = int(np.asarray(series[0]).shape[1]); tvec = np.asarray(times, float)
    n = min(n_tuples, min(len(np.asarray(c)) for c in series))
    tup = torch.as_tensor(_chain_ot_tuples(series, n, seed=seed), dtype=torch.float32, device=device)
    field = VelocityMLP(d, hidden=hidden, layers=4).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=1e-3); field.train()
    for _ in range(iters):
        k = np.random.randint(0, len(tvec) - 1); t0, t1 = float(tvec[k]), float(tvec[k + 1])
        x0, x1 = tup[:, k], tup[:, k + 1]; s = torch.rand(n, 1, device=device)
        xt = (1 - s) * x0 + s * x1 + sigma * torch.sqrt(s * (1 - s)) * torch.randn(n, d, device=device)
        tt = t0 + s * (t1 - t0); target = (x1 - x0) / (t1 - t0)
        loss = ((field(xt, tt) - target) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    field.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(X0), dtype=torch.float32, device=device)
        ts = torch.linspace(float(tvec[0]), float(tvec[-1]), n_steps + 1, device=device); out = [x.clone()]
        for i in range(n_steps):
            dt = (ts[i + 1] - ts[i])
            x = x + dt * field(x, ts[i].expand(x.shape[0], 1)) + sigma * torch.sqrt(dt.clamp_min(1e-12)) * torch.randn_like(x)
            out.append(x.clone())
        full = torch.stack(out, 0).cpu().numpy()
    return full[np.linspace(0, full.shape[0] - 1, len(times)).astype(int)]
