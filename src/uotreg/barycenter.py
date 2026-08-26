r"""Distribution estimation: the robust local-Frechet UOT barycenter.

High-level API around the fixed-point algorithm.  Given per-time-point samplers
and a query time, :class:`DistributionEstimator` computes the nonnegative
local-Frechet weights, builds the generator / transport-map / potential networks
for the *active* references only, optionally initializes the generator, and runs
the descent iteration ``mu_{n+1} = (Tbar_{mu_n})_# mu_n``.

The training math is identical to the validated original code; what is new is
the clean API, the configurable relaxation / divergence / cost, GPU support, and
a monitored stationarity residual ``R_n = E_mu || x - Tbar(x) ||^2`` (which the
revision uses as a principled stopping rule -- Corollary on the vanishing
residual).
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np
import torch

from .config import ModelConfig, TrainConfig, UOTConfig, WeightConfig
from .costs import get_cost
from .data import GaussianLatentSampler, TensorSampler
from .device import autocast_context, resolve_device, seed_everything, to_numpy
from .divergences import get_divergence
from .initialization import initialize_generator, load_generator_state
from .models import Generator, Potentials, TransportMaps, weights_init
from .uot import check_relaxation, map_loss, potential_loss
from .weights import WeightResult, frechet_weights


def _set_train(module: torch.nn.Module, train: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(train)
    module.train(train)


class DistributionEstimator:
    """Estimate the cell-state distribution ``mu_bar(t)`` at a query time.

    Examples
    --------
    >>> est = DistributionEstimator(dim=20)
    >>> est.fit(samplers, timepoints=[1.5, 7.5, 13.5, 19.5, 25.5],
    ...         query_time=13.5, h=4, tau=5, relaxation="one-sided", init="flow")
    >>> X = est.sample(2000)          # numpy array of generated cells
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        model: Optional[ModelConfig] = None,
        uot: Optional[UOTConfig] = None,
        train: Optional[TrainConfig] = None,
        weights: Optional[WeightConfig] = None,
    ):
        self.model_cfg = model or ModelConfig()
        if dim is not None:
            self.model_cfg.dim = dim
        self.uot_cfg = uot or UOTConfig()
        self.train_cfg = train or TrainConfig()
        self.weight_cfg = weights or WeightConfig()

        self.device = resolve_device(self.train_cfg.device)
        self.generator: Optional[Generator] = None
        self.maps: Optional[TransportMaps] = None
        self.potentials: Optional[Potentials] = None
        self.latent: Optional[GaussianLatentSampler] = None
        self.weight_result: Optional[WeightResult] = None
        self.history = {"g_loss": [], "residual": []}

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(
        self,
        samplers: Sequence,
        timepoints: Optional[Sequence[float]] = None,
        query_time: Optional[float] = None,
        *,
        weights: Optional[Sequence[float]] = None,
        h: Optional[float] = None,
        tau: Optional[float] = None,
        relaxation: Optional[str] = None,
        divergence: Optional[str] = None,
        cost: Optional[str] = None,
        weight_scheme: Optional[str] = None,
        threshold: Optional[float] = "default",
        kernel: Optional[str] = None,
        init: str = "gaussian",
        init_data_sampler=None,
        init_kwargs: Optional[dict] = None,
        pretrained_generator: Optional[str] = None,
        latent_sampler=None,
        callback=None,
    ) -> "DistributionEstimator":
        """Estimate the (weighted) barycenter of ``samplers``.

        Two modes:
          * **time** (default): pass ``timepoints`` and ``query_time`` and the
            local-Frechet weights are computed (``h, weight_scheme, threshold, kernel``).
          * **explicit weights**: pass ``weights`` (length = #samplers) for a plain
            weighted Wasserstein barycenter with no time axis (e.g. the outlier
            simulation); ``timepoints``/``query_time`` are then ignored.

        ``init`` selects the generator initialization (``"gaussian"`` / ``"vae_nf"``
        / ``"flow"`` / ``"pretrained"``); for ``"vae_nf"``/``"flow"`` pass
        ``init_data_sampler`` (pooled cells).
        """
        # ---- resolve per-call overrides -------------------------------------
        wc, uc, tc = self.weight_cfg, self.uot_cfg, self.train_cfg
        h = wc.bandwidth if h is None else h
        tau = uc.tau if tau is None else tau
        relaxation = check_relaxation(relaxation or uc.relaxation)
        divergence = divergence or uc.divergence
        cost_name = cost or uc.cost
        scheme = weight_scheme or wc.scheme
        kernel = kernel or wc.kernel
        threshold = wc.threshold if threshold == "default" else threshold
        seed_everything(tc.seed)

        # ---- weights & active references ------------------------------------
        if weights is not None:
            ww = np.asarray(weights, dtype=float)
            if ww.sum() <= 0:
                raise ValueError("weights must have positive sum.")
            ww = ww / ww.sum()
            active_index = [int(i) for i in np.nonzero(ww > 0)[0]]
            active_weights = ww[active_index] / ww[active_index].sum()
            self.weight_result = None
            if tc.verbose:
                print(f"[fit] explicit-weight barycenter | tau={tau} relax={relaxation} "
                      f"psi={divergence} | {len(active_index)} references "
                      f"weights={np.round(active_weights, 3).tolist()}")
        else:
            if timepoints is None or query_time is None:
                raise ValueError("Provide either (timepoints, query_time) or explicit weights.")
            wr = frechet_weights(timepoints, query_time, h, kernel=kernel,
                                 scheme=scheme, threshold=threshold)
            self.weight_result = wr
            if wr.num_active < 1:
                raise RuntimeError("No active reference time points; check h / threshold.")
            active_index, active_weights = wr.active_index, wr.active_weights
            if tc.verbose:
                print(f"[fit] query={query_time} h={h} tau={tau} relax={relaxation} "
                      f"psi={divergence} | active refs={wr.active_index} "
                      f"weights={np.round(wr.active_weights, 3).tolist()} "
                      f"(neg-mass={wr.negative_mass:.3g}, dropped={wr.dropped_mass:.3g}, ESS={wr.ess:.2f})")

        active_samplers = [samplers[i] for i in active_index]
        active_samplers = [s.to(self.device) if hasattr(s, "to") else s for s in active_samplers]
        w = torch.tensor(active_weights, dtype=torch.float32, device=self.device)
        num = len(active_index)

        # ---- build networks --------------------------------------------------
        mc = self.model_cfg
        self.div = get_divergence(divergence, tau)
        self.cost_fn = get_cost(cost_name)
        self.relaxation = relaxation
        # latent sampler: default standard normal; override (e.g. a multi-modal
        # latent) lets the generator form disconnected modes -> less bridge.
        self.latent = (latent_sampler.to(self.device) if (latent_sampler is not None
                       and hasattr(latent_sampler, "to")) else
                       (latent_sampler if latent_sampler is not None
                        else GaussianLatentSampler(mc.latent_dim or mc.dim, device=self.device)))
        self.generator = Generator(mc.dim, mc.latent_dim, mc.gen_hidden, mc.gen_layers,
                                   mc.gen_dropout).to(self.device)
        self.maps = TransportMaps(mc.dim, num, mc.map_hidden, mc.map_layers,
                                  mc.dropout, mc.batchnorm).to(self.device)
        self.potentials = Potentials(mc.dim, num, mc.pot_hidden, mc.pot_layers,
                                     mc.dropout, mc.batchnorm).to(self.device)
        self.maps.apply(weights_init)
        self.potentials.apply(weights_init)

        # ---- initialize generator -------------------------------------------
        initialize_generator(
            self.generator,
            strategy=("pretrained" if pretrained_generator else init),
            latent_sampler=self.latent,
            data_sampler=init_data_sampler,
            device=self.device,
            checkpoint=pretrained_generator,
            verbose=tc.verbose,
            **(init_kwargs or {}),
        )

        # ---- optimizers ------------------------------------------------------
        self.t_opt = torch.optim.Adam(self.maps.parameters(), lr=tc.lr_map,
                                      weight_decay=tc.weight_decay_td)
        self.d_opt = torch.optim.Adam(self.potentials.parameters(), lr=tc.lr_pot,
                                      weight_decay=tc.weight_decay_td)
        self.g_opt = torch.optim.Adam(self.generator.parameters(), lr=tc.lr_gen,
                                      weight_decay=tc.weight_decay_gen)

        self._train_loop(active_samplers, w, callback=callback)
        return self

    # ------------------------------------------------------------------ #
    # Training loop (fixed-point barycenter iteration)
    # ------------------------------------------------------------------ #
    def _train_loop(self, samplers: List, w: torch.Tensor, callback=None) -> None:
        tc, mc = self.train_cfg, self.model_cfg
        dim, num = mc.dim, len(samplers)
        G, T, D = self.generator, self.maps, self.potentials

        for outer in range(tc.outer_iters):
            # ----- update potentials D and maps T -----
            for _ in range(tc.d_iters):
                # maps realize the c-transform
                _set_train(T, True); _set_train(D, False)
                for _ in range(tc.t_iters):
                    with torch.no_grad():
                        X = G(self.latent.sample(tc.batch_size))
                    with autocast_context(self.device, tc.amp):
                        T_X = T(X).view(-1, num, dim)
                        cost = self.cost_fn(T_X, X.unsqueeze(1))          # (B, num)
                        d_Tx = D(T_X.reshape(X.shape[0], -1))             # (B, num)
                        loss_T = map_loss(cost, d_Tx)
                    self.t_opt.zero_grad(); loss_T.backward(); self.t_opt.step()

                # potentials maximize the semi-dual
                _set_train(D, True); _set_train(T, False)
                with torch.no_grad():
                    X = G(self.latent.sample(tc.batch_size))
                    T_X = T(X).view(-1, num, dim)
                    Y = torch.cat([s.sample(tc.batch_size) for s in samplers], dim=1)
                with autocast_context(self.device, tc.amp):
                    cost = self.cost_fn(T_X, X.unsqueeze(1)).detach()
                    d_Tx = D(T_X.reshape(X.shape[0], -1))
                    d_y = D(Y)
                    loss_D = potential_loss(cost, d_Tx, d_y,
                                            relaxation=self.relaxation, divergence=self.div)
                self.d_opt.zero_grad(); loss_D.backward(); self.d_opt.step()

            # ----- update generator G (pushforward by the average map) -----
            g0 = self._optimize_generator(w, num, dim)
            self.history["g_loss"].append(g0)
            self.history["residual"].append(self._residual(w, num, dim))

            if tc.verbose and (outer % tc.log_every == 0 or outer == tc.outer_iters - 1):
                print(f"  [outer {outer+1}/{tc.outer_iters}] G_loss={g0:.4e} "
                      f"residual R_n={self.history['residual'][-1]:.4e}")

            if callback is not None:
                callback(self, outer)

    def _optimize_generator(self, w: torch.Tensor, num: int, dim: int) -> float:
        tc = self.train_cfg
        G, T = self.generator, self.maps
        G_old = deepcopy(G)
        _set_train(G_old, False); _set_train(G, True); _set_train(T, False)
        loss_start = 0.0
        for g in range(tc.g_iters):
            Z = self.latent.sample(tc.batch_size_g)
            with torch.no_grad():
                base = G_old(Z)
                T_base = T(base).view(-1, num, dim)
                target = (w.view(1, num, 1) * T_base).sum(dim=1)      # Tbar(base)
            with autocast_context(self.device, tc.amp):
                loss_G = 0.5 * torch.nn.functional.mse_loss(G(Z), target)
            self.g_opt.zero_grad(); loss_G.backward(); self.g_opt.step()
            if g == 0:
                loss_start = float(loss_G.item())
        return loss_start

    @torch.no_grad()
    def _residual(self, w: torch.Tensor, num: int, dim: int) -> float:
        """Stationarity residual R_n = E_mu || x - Tbar(x) ||^2 (vanishes at the fixed point)."""
        self.generator.eval(); self.maps.eval()
        X = self.generator(self.latent.sample(max(512, self.train_cfg.batch_size)))
        T_X = self.maps(X).view(-1, num, dim)
        Tbar = (w.view(1, num, 1) * T_X).sum(dim=1)
        return float(((X - Tbar) ** 2).mean().item())

    # ------------------------------------------------------------------ #
    # Sampling / IO
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_tensor(self, n: int = 2000) -> torch.Tensor:
        if self.generator is None:
            raise RuntimeError("Call fit() before sampling.")
        self.generator.eval()
        return self.generator(self.latent.sample(n))

    def sample(self, n: int = 2000) -> np.ndarray:
        return to_numpy(self.sample_tensor(n))

    def save(self, path: str) -> None:
        torch.save({
            "generator": self.generator.state_dict(),
            "model_cfg": self.model_cfg.__dict__,
            "weight_result": None if self.weight_result is None else self.weight_result.__dict__,
        }, path)

    def load(self, path: str, map_location=None) -> "DistributionEstimator":
        """Load a generator saved by this class *or* a raw old ``Gnet`` checkpoint."""
        mc = self.model_cfg
        self.latent = GaussianLatentSampler(mc.latent_dim or mc.dim, device=self.device)
        self.generator = Generator(mc.dim, mc.latent_dim, mc.gen_hidden, mc.gen_layers,
                                   mc.gen_dropout).to(self.device)
        load_generator_state(self.generator, path, map_location=map_location or self.device)
        self.generator.eval()
        return self
