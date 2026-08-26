r"""Cell-trajectory reconstruction between estimated distributions.

Two complementary approaches, both behind small high-level classes:

1. :class:`TrajectoryFitter` -- the paper's approach.  Learn unbalanced-OT maps
   between consecutive distributions and compose them.  Crucially the maps are
   learned **iteratively with warm starting**: the *same* network learns
   t_i -> t_{i+1}, then continues from those weights to learn t_{i+1} -> t_{i+2},
   and so on (each transition is snapshotted so the full chain can be composed).
   The relaxation (balanced / one-sided / two-sided) and divergence are
   configurable, exactly as for the barycenter.

2. :class:`FlowMatchingTrajectory` -- a dynamic-OT / flow-matching alternative.
   Time-conditioned velocity field(s) trained by conditional flow-matching
   between consecutive snapshots and integrated as an ODE.  The training
   coupling is configurable (independent / minibatch-OT / minibatch-UOT / the
   UOT maps of 1.), and the field is either **shared** across all intervals
   (one v(x,t)) or **unshared** (one field per interval, more capacity where the
   population branches).  This yields smooth, continuous-time trajectories and
   is a useful comparator to the discrete UOT-map chain.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np
import torch

from .config import ModelConfig, TrajectoryConfig
from .costs import get_cost
from .device import resolve_device, seed_everything, to_numpy
from .divergences import get_divergence
from .models import MapNet, PotentialNet, VelocityField, weights_init
from .uot import check_relaxation, map_loss, potential_loss


def _set_train(module: torch.nn.Module, train: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(train)
    module.train(train)


def _as_sampler(obj, device):
    """Accept a sampler (has .sample) or a raw array; wrap arrays in a TensorSampler."""
    if hasattr(obj, "sample"):
        return obj.to(device) if hasattr(obj, "to") else obj
    from .data import TensorSampler
    return TensorSampler(obj, device=device)


# --------------------------------------------------------------------------- #
# 1. Iterative pairwise UOT maps
# --------------------------------------------------------------------------- #

class TrajectoryFitter:
    """Compose iteratively-learned unbalanced-OT maps into cell trajectories."""

    def __init__(
        self,
        dim: Optional[int] = None,
        model: Optional[ModelConfig] = None,
        config: Optional[TrajectoryConfig] = None,
    ):
        self.model_cfg = model or ModelConfig()
        if dim is not None:
            self.model_cfg.dim = dim
        self.cfg = config or TrajectoryConfig()
        self.device = resolve_device(self.cfg.device)
        self.map_chain: List[MapNet] = []     # one snapshot per transition
        self.map: Optional[MapNet] = None
        self.potential: Optional[PotentialNet] = None
        self.history: List[dict] = []

    def fit(
        self,
        distributions: Sequence,
        *,
        tau: Optional[float] = None,
        relaxation: Optional[str] = None,
        divergence: Optional[str] = None,
        cost: Optional[str] = None,
    ) -> "TrajectoryFitter":
        """Learn the chain of maps between consecutive ``distributions`` (time order)."""
        seed_everything(self.cfg.seed)
        uc = self.cfg.uot
        tau = uc.tau if tau is None else tau
        relaxation = check_relaxation(relaxation or uc.relaxation)
        div = get_divergence(divergence or uc.divergence, tau)
        cost_fn = get_cost(cost or uc.cost)
        mc = self.model_cfg

        dists = [_as_sampler(d, self.device) for d in distributions]
        self.map = MapNet(mc.dim, mc.map_hidden, mc.map_layers, mc.dropout, mc.batchnorm).to(self.device)
        self.potential = PotentialNet(mc.dim, mc.pot_hidden, mc.pot_layers, mc.dropout, mc.batchnorm).to(self.device)
        self.map.apply(weights_init); self.potential.apply(weights_init)
        t_opt = torch.optim.Adam(self.map.parameters(), lr=self.cfg.lr_map)
        d_opt = torch.optim.Adam(self.potential.parameters(), lr=self.cfg.lr_pot)

        self.map_chain = []
        for k in range(len(dists) - 1):
            src, tgt = dists[k], dists[k + 1]
            if not self.cfg.warm_start:
                # fresh networks per transition
                self.map.apply(weights_init); self.potential.apply(weights_init)
            losses = self._fit_pair(src, tgt, t_opt, d_opt, cost_fn, div, relaxation)
            snapshot = deepcopy(self.map).eval()
            for p in snapshot.parameters():
                p.requires_grad_(False)
            self.map_chain.append(snapshot)
            self.history.append({"transition": k, **losses})
            if self.cfg.verbose:
                print(f"[traj] transition {k}->{k+1}: loss_D={losses['loss_D']:.4e} "
                      f"loss_T={losses['loss_T']:.4e}")
        return self

    def _fit_pair(self, src, tgt, t_opt, d_opt, cost_fn, div, relaxation) -> dict:
        cfg = self.cfg
        T, D = self.map, self.potential
        last_T = last_D = 0.0
        for _ in range(cfg.d_iters):
            _set_train(T, True); _set_train(D, False)
            for _ in range(cfg.t_iters):
                X = src.sample(cfg.batch_size)
                T_X = T(X)
                cost = cost_fn(T_X, X)                 # (B,)
                d_Tx = D(T_X).squeeze(-1)              # (B,)
                loss_T = map_loss(cost, d_Tx)
                t_opt.zero_grad(); loss_T.backward(); t_opt.step()
                last_T = float(loss_T.item())

            _set_train(D, True); _set_train(T, False)
            X = src.sample(cfg.batch_size)
            with torch.no_grad():
                T_X = T(X)
                cost = cost_fn(T_X, X)
            Y = tgt.sample(cfg.batch_size)
            d_Tx = D(T_X).squeeze(-1)
            d_y = D(Y).squeeze(-1)
            loss_D = potential_loss(cost, d_Tx, d_y, relaxation=relaxation, divergence=div)
            d_opt.zero_grad(); loss_D.backward(); d_opt.step()
            last_D = float(loss_D.item())
        return {"loss_T": last_T, "loss_D": last_D}

    @torch.no_grad()
    def transport(self, initial_cells, return_intermediate: bool = True):
        """Push ``initial_cells`` through the learned map chain.

        Returns an array of shape ``(n_steps+1, n_cells, d)`` if
        ``return_intermediate`` else the final positions ``(n_cells, d)``.
        """
        if not self.map_chain:
            raise RuntimeError("Call fit() before transport().")
        x = torch.as_tensor(np.asarray(initial_cells), dtype=torch.float32, device=self.device)
        traj = [x.clone()]
        for mp in self.map_chain:
            x = mp(x)
            traj.append(x.clone())
        if return_intermediate:
            return np.stack([to_numpy(t) for t in traj], axis=0)
        return to_numpy(x)

    def save(self, path: str) -> None:
        torch.save({"chain": [m.state_dict() for m in self.map_chain],
                    "model_cfg": self.model_cfg.__dict__}, path)

    def load(self, path: str, map_location=None) -> "TrajectoryFitter":
        ckpt = torch.load(path, map_location=map_location or self.device)
        mc = self.model_cfg
        self.map_chain = []
        for sd in ckpt["chain"]:
            m = MapNet(mc.dim, mc.map_hidden, mc.map_layers, mc.dropout, mc.batchnorm).to(self.device)
            m.load_state_dict(sd); m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.map_chain.append(m)
        return self


# --------------------------------------------------------------------------- #
# 2. Flow-matching / dynamic-OT trajectories
# --------------------------------------------------------------------------- #

def _minibatch_ot_coupling(x0: torch.Tensor, x1: torch.Tensor, reg: float = 0.05):
    """Reorder x1 to (approximately) optimally couple it with x0 within the batch.

    Uses POT's exact EMD if available; otherwise falls back to a small numpy
    entropic Sinkhorn (no POT dependency) so the OT coupling still works.
    """
    with torch.no_grad():
        M = torch.cdist(x0, x1, p=2).pow(2).cpu().numpy()
        n = x0.shape[0]
        a = np.ones(n) / n
        try:
            import ot  # POT, optional
            plan = ot.emd(a, a, M)
        except Exception:
            Mn = M / (M.max() + 1e-12)
            K = np.exp(-Mn / reg)
            u, v = np.ones(n), np.ones(M.shape[1])
            b = np.ones(M.shape[1]) / M.shape[1]
            for _ in range(100):
                u = a / (K @ v + 1e-300)
                v = b / (K.T @ u + 1e-300)
            plan = u[:, None] * K * v[None, :]
        idx = plan.argmax(axis=1)
    return x1[torch.as_tensor(idx, device=x1.device)]


def _unbalanced_coupling(x0: torch.Tensor, x1: torch.Tensor, reg: float = 0.05, reg_m: float = 1.0):
    """Reorder x1 by the max-mass target of an UNBALANCED Sinkhorn plan (POT).

    `reg_m` is the marginal-relaxation strength: small = very unbalanced (mass may be created /
    destroyed), large -> the balanced plan.  Falls back to the balanced minibatch-OT coupling if
    POT is unavailable.
    """
    with torch.no_grad():
        try:
            import ot  # POT, optional
        except Exception:
            return _minibatch_ot_coupling(x0, x1, reg)
        M = torch.cdist(x0, x1, p=2).pow(2).cpu().numpy().astype(np.float64)
        M /= (M.max() + 1e-12)
        n, m = M.shape
        plan = ot.unbalanced.sinkhorn_unbalanced(np.ones(n) / n, np.ones(m) / m, M, reg, reg_m)
        idx = np.asarray(plan).argmax(axis=1)
    return x1[torch.as_tensor(idx, device=x1.device)]


def _as_map_fn(m, device):
    """Wrap a UOT map (a torch `MapNet` or a numpy callable) as a tensor -> tensor function."""
    def f(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                y = m(x)
            except Exception:                       # numpy-only callable
                y = m(to_numpy(x))
        if torch.is_tensor(y):
            return y.detach().to(device)
        return torch.as_tensor(np.asarray(y, np.float32), device=device)
    return f


# coupling spellings used by the diagnostics / notebooks -> canonical names
_COUPLING_ALIASES = {"minibatch-ot": "ot", "minibatch-uot": "uot", "uot_map": "uot-map",
                     "map": "uot-map", "rand": "independent", "random": "independent"}
_COUPLINGS = ("ot", "uot", "uot-map", "independent")


class FlowMatchingTrajectory:
    """Continuous-time trajectories via flow-matching velocity field(s) (dynamic OT).

    `coupling` selects how the training pairs ``(x0, x1)`` are drawn on each interval:

    * ``"ot"``          -- minibatch OT (standard OT-CFM);
    * ``"uot"``         -- minibatch UNBALANCED Sinkhorn (POT; see `reg` / `reg_m`);
    * ``"uot-map"``     -- ``x1 = T_k(x0)`` from our fitted UOTReg maps, passed to :meth:`fit` as
      `uot_maps` (e.g. a :attr:`TrajectoryFitter.map_chain`) -- i.e. the flow is trained to imitate
      the coupling our own method uses;
    * ``"independent"`` -- no pairing.

    `shared=True` trains ONE field v(x,t) for every interval (the classic single-field flow);
    `shared=False` trains one field PER interval.  At a bifurcation a shared single-valued field
    must average the bimodal targets of the splitting interval, so `shared=False` has strictly more
    capacity there (see the batch-effect study).

    `field` picks the architecture: ``"plain"`` = :class:`VelocityField` (time concatenated),
    ``"mlp"`` = :class:`~uotreg.baselines.VelocityMLP` (sinusoidal time embedding).
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        model: Optional[ModelConfig] = None,
        coupling: str = "ot",          # "ot" | "uot" | "uot-map" | "independent"
        hidden: int = 256,
        n_layers: int = 4,
        device: str = "auto",
        seed: Optional[int] = None,
        shared: bool = True,           # one v(x,t) for all intervals, or one field per interval
        field: str = "plain",          # "plain" (VelocityField) | "mlp" (VelocityMLP)
    ):
        self.model_cfg = model or ModelConfig()
        if dim is not None:
            self.model_cfg.dim = dim
        self.coupling = _COUPLING_ALIASES.get(coupling, coupling)
        if self.coupling not in _COUPLINGS:
            raise ValueError(f"unknown coupling '{coupling}'; known: {list(_COUPLINGS)}")
        self.device = resolve_device(device)
        self.seed = seed
        self.shared = shared
        self.field_type = field
        self.hidden, self.n_layers = hidden, n_layers
        self.fields: List[torch.nn.Module] = [self._new_field()]
        self.times: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- fields
    def _new_field(self) -> torch.nn.Module:
        if self.field_type == "mlp":
            from .baselines import VelocityMLP
            return VelocityMLP(self.model_cfg.dim, self.hidden, self.n_layers).to(self.device)
        return VelocityField(self.model_cfg.dim, self.hidden, self.n_layers).to(self.device)

    @property
    def field(self) -> torch.nn.Module:
        """The (first) velocity field -- backwards-compatible single-field handle."""
        return self.fields[0]

    @field.setter
    def field(self, f: torch.nn.Module) -> None:
        self.fields[0] = f

    def _field(self, k: int) -> torch.nn.Module:
        """The field governing interval k -> k+1."""
        return self.fields[0] if self.shared else self.fields[k]

    def _field_at(self, t: float) -> torch.nn.Module:
        """The field governing time `t` (for the free-running Euler integrator)."""
        if self.shared or self.times is None:
            return self.fields[0]
        k = int(np.searchsorted(np.asarray(self.times, float), t, side="right") - 1)
        return self.fields[min(max(k, 0), len(self.fields) - 1)]

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        distributions: Sequence,
        times: Optional[Sequence[float]] = None,
        *,
        iters: int = 5000,
        batch_size: int = 256,
        lr: float = 1e-3,
        sigma: float = 0.0,
        uot_maps: Optional[Sequence] = None,
        reg: float = 0.05,
        reg_m: float = 1.0,
        verbose: bool = True,
    ) -> "FlowMatchingTrajectory":
        """Train the velocity field(s) by flow-matching between consecutive snapshots.

        `sigma` > 0 replaces the straight interpolant by a Brownian bridge (the stochastic /
        [SF]^2M variant, which -- unlike a deterministic field -- can split a connected blob).
        `uot_maps` is required for ``coupling="uot-map"``: one map per interval, taking the cloud at
        time k to time k+1.  `reg` / `reg_m` configure the ``"uot"`` Sinkhorn coupling.
        """
        seed_everything(self.seed)
        dists = [_as_sampler(d, self.device) for d in distributions]
        n = len(dists)
        times = np.asarray(times if times is not None else np.linspace(0.0, 1.0, n), dtype=float)
        self.times = times
        maps = None
        if self.coupling == "uot-map":
            if uot_maps is None or len(uot_maps) < n - 1:
                raise ValueError(f"coupling='uot-map' needs {n - 1} maps in `uot_maps`, "
                                 f"got {0 if uot_maps is None else len(uot_maps)}")
            maps = [_as_map_fn(m, self.device) for m in uot_maps]
        # one field for everything, or one per interval
        self.fields = [self._new_field()] if self.shared else [self._new_field() for _ in range(n - 1)]
        params = [p for f in self.fields for p in f.parameters()]
        opt = torch.optim.Adam(params, lr=lr)
        for f in self.fields:
            f.train()
        for it in range(iters):
            k = np.random.randint(0, n - 1)             # random consecutive interval
            t0, t1 = float(times[k]), float(times[k + 1])
            x0 = dists[k].sample(batch_size)
            if self.coupling == "uot-map":
                x1 = maps[k](x0)
            else:
                x1 = dists[k + 1].sample(batch_size)
                if self.coupling == "ot":
                    x1 = _minibatch_ot_coupling(x0, x1)
                elif self.coupling == "uot":
                    x1 = _unbalanced_coupling(x0, x1, reg, reg_m)
            s = torch.rand(x0.shape[0], 1, device=self.device)
            xt = (1 - s) * x0 + s * x1                  # linear interpolant
            if sigma:
                xt = xt + sigma * torch.sqrt(s * (1 - s)) * torch.randn_like(xt)   # Brownian bridge
            t = t0 + s * (t1 - t0)
            target_v = (x1 - x0) / (t1 - t0)            # constant-speed velocity
            pred_v = self._field(k)(xt, t)
            loss = torch.nn.functional.mse_loss(pred_v, target_v)
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and it % max(1, iters // 10) == 0:
                print(f"[flow-match] it {it}/{iters}  loss={loss.item():.4e}")
        for f in self.fields:
            f.eval()
        return self

    # -------------------------------------------------------------- simulate
    @torch.no_grad()
    def simulate(self, initial_cells, n_steps: int = 100, return_intermediate: bool = True):
        """Integrate dx/dt = v(x,t) (Euler) from the first to the last time on a uniform grid.

        With `shared=False` each step uses the field of the interval it starts in.  Prefer
        :meth:`simulate_at_times` when you want the clouds exactly at the snapshot times.
        """
        if self.times is None:
            raise RuntimeError("Call fit() before simulate().")
        x = torch.as_tensor(np.asarray(initial_cells), dtype=torch.float32, device=self.device)
        t0, t1 = float(self.times[0]), float(self.times[-1])
        ts = torch.linspace(t0, t1, n_steps + 1, device=self.device)
        traj = [x.clone()]
        for i in range(n_steps):
            dt = (ts[i + 1] - ts[i])
            x = x + dt * self._field_at(float(ts[i]))(x, ts[i].expand(x.shape[0], 1))
            traj.append(x.clone())
        if return_intermediate:
            return np.stack([to_numpy(t) for t in traj], axis=0)
        return to_numpy(x)

    def save(self, path: str) -> None:
        """Save the trained field(s) + everything needed to rebuild and integrate them.
        (Plain-python metadata only, so torch>=2.6 `weights_only` loading works.)"""
        torch.save({"fields": [f.state_dict() for f in self.fields],
                    "times": (None if self.times is None else [float(t) for t in self.times]),
                    "cfg": dict(dim=self.model_cfg.dim, hidden=self.hidden, n_layers=self.n_layers,
                                shared=self.shared, field=self.field_type, coupling=self.coupling)}, path)

    def load(self, path: str, map_location=None) -> "FlowMatchingTrajectory":
        """Rebuild the field(s) from a `save` checkpoint (overrides this instance's architecture)."""
        ckpt = torch.load(path, map_location=map_location or self.device)
        c = ckpt["cfg"]
        self.model_cfg.dim = c["dim"]; self.hidden = c["hidden"]; self.n_layers = c["n_layers"]
        self.shared = c["shared"]; self.field_type = c["field"]; self.coupling = c["coupling"]
        self.fields = []
        for sd in ckpt["fields"]:
            f = self._new_field(); f.load_state_dict(sd); f.eval()
            for p in f.parameters():
                p.requires_grad_(False)
            self.fields.append(f)
        self.times = None if ckpt["times"] is None else np.asarray(ckpt["times"], float)
        return self

    @torch.no_grad()
    def simulate_at_times(self, initial_cells, n_per: int = 20):
        """RK4-integrate interval by interval (each with ITS field) -> `(len(times), N, d)`.

        The snapshot-aligned counterpart of :meth:`simulate`: no step ever straddles an interval
        boundary, so it is the correct integrator for `shared=False`.
        """
        if self.times is None:
            raise RuntimeError("Call fit() before simulate_at_times().")
        for f in self.fields:
            f.eval()
        x = torch.as_tensor(np.asarray(initial_cells), dtype=torch.float32, device=self.device)
        tv = [float(t) for t in self.times]
        out = [x.clone()]
        for k in range(len(tv) - 1):
            fk = self._field(k); h = (tv[k + 1] - tv[k]) / n_per
            for j in range(n_per):
                tt = torch.full((x.shape[0], 1), tv[k] + j * h, device=self.device)
                k1 = fk(x, tt); k2 = fk(x + 0.5 * h * k1, tt + 0.5 * h)
                k3 = fk(x + 0.5 * h * k2, tt + 0.5 * h); k4 = fk(x + h * k3, tt + h)
                x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(x.clone())
        return np.stack([to_numpy(o) for o in out], axis=0)
