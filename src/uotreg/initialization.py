"""Generator initialization strategies.

Before the fixed-point barycenter iteration we initialize the generator G so it
already produces a sensible spread of points.  Two strategies (selectable):

  * ``"gaussian"`` -- regress G towards the identity on a *broad* Gaussian, so
    G_#(latent) is a wide blob covering the data.  Cheap and what the original
    notebooks used (``mse(Z, G(Z))`` with an inflated Z).

  * ``"flow"`` -- first fit a small normalizing flow to the pooled cells for a
    few epochs to *capture the modes* of the data, then regress G onto the flow
    samples.  This gives a multi-modal, data-aware initialization, which helps
    the barycenter solver on real, clustered scRNA-seq data.

  * ``"pretrained"`` -- load generator weights from a checkpoint.

All routines are device-aware and leave G in train mode.
"""
from __future__ import annotations

from typing import Optional

import torch

from .data import GaussianLatentSampler
from .models import NormalizingFlow


# --------------------------------------------------------------------------- #
# Sample-based distribution distance (no bandwidth tuning): energy distance.
# --------------------------------------------------------------------------- #

def _pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased-ish energy distance between two sample sets (for mode-capture fitting)."""
    dxy = _pairwise_dist(x, y).mean()
    dxx = _pairwise_dist(x, x).mean()
    dyy = _pairwise_dist(y, y).mean()
    return 2 * dxy - dxx - dyy


def fit_normalizing_flow(
    flow: NormalizingFlow,
    data_sampler,
    latent_sampler,
    epochs: int = 800,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = False,
) -> NormalizingFlow:
    """Fit a normalizing flow to the pooled data by minimizing the energy distance."""
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    flow.train()
    for ep in range(epochs):
        z = latent_sampler.sample(batch_size)
        gen, _ = flow(z)
        real = data_sampler.sample(batch_size)
        loss = energy_distance(gen, real)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % max(1, epochs // 10) == 0:
            print(f"[flow-init] epoch {ep}/{epochs}  energy={loss.item():.4f}")
    flow.eval()
    return flow


def load_generator_state(generator: torch.nn.Module, checkpoint, map_location=None):
    """Load generator weights from a path or state-dict, tolerant to the format.

    Handles (a) raw ``state_dict`` saved by the old code (``torch.save(G.state_dict())``),
    (b) the new wrapped checkpoint ``{"generator": state_dict, ...}``, and
    (c) a ``net.``/``network.`` prefix mismatch.  Old ``Gnet`` checkpoints load
    directly because the new ``Generator`` keeps the ``network.*`` keys.
    """
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=False)
    sd = checkpoint.get("generator", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        generator.load_state_dict(sd)
        return generator
    except RuntimeError:
        target = set(generator.state_dict().keys())
        remap = {}
        for k, v in sd.items():
            nk = k
            if k not in target:
                if k.startswith("net.") and ("network." + k[4:]) in target:
                    nk = "network." + k[4:]
                elif k.startswith("network.") and ("net." + k[8:]) in target:
                    nk = "net." + k[8:]
            remap[nk] = v
        generator.load_state_dict(remap)
        return generator


def initialize_generator(
    generator: torch.nn.Module,
    strategy: str = "gaussian",
    *,
    latent_sampler,
    data_sampler=None,
    device=None,
    iters: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-4,
    gaussian_scale: float = 6.0,
    flow_length: int = 16,
    flow_epochs: int = 800,
    vae_epochs: int = 5,
    vae_coef: float = 5.0,
    vae_lr: float = 1e-3,
    checkpoint: Optional[str] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    """Initialize ``generator`` in place and return it.

    Strategies
    ----------
    ``"gaussian"``   : regress G toward identity on a broad Gaussian (cheap, wide blob).
    ``"vae_nf"``     : flow-VAE init -- train an encoder + (this generator as) decoder
                       + planar flows on the pooled cells; the decoder is the init.
                       The faithful reproduction of the original ``VAEini``.
    ``"flow"``       : lighter normalizing-flow-only init (energy-distance fit).
    ``"pretrained"`` : load generator weights from ``checkpoint`` (old or new format).
    """
    dev = device if device is not None else next(generator.parameters()).device
    generator.train()

    if strategy == "pretrained":
        if checkpoint is None:
            raise ValueError("strategy='pretrained' requires a checkpoint path.")
        load_generator_state(generator, checkpoint, map_location=dev)
        generator.eval()
        return generator

    if strategy == "vae_nf":
        if data_sampler is None:
            raise ValueError("strategy='vae_nf' requires data_sampler (pooled cells).")
        from .vae_init import vae_nf_initialize
        with torch.no_grad():
            dim = generator(latent_sampler.sample(2)).shape[1]
        return vae_nf_initialize(generator, data_sampler, dim, flow_length=flow_length,
                                 epochs=vae_epochs, coef=vae_coef, lr=vae_lr,
                                 batch_size=batch_size, device=dev, verbose=verbose)

    opt = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-8)

    if strategy == "gaussian":
        # Regress G towards identity on a broad Gaussian -> wide covering blob.
        # NOTE: this is centred at the ORIGIN and ignores the data, so per-time
        # generators whose target is far from the origin must learn a large
        # translation -> progressively worse fits along a drifting trajectory. Prefer
        # ``"data_gaussian"`` (below) for a time-LOCAL init when fitting a moving mode.
        for it in range(iters):
            z = latent_sampler.sample(batch_size).detach() * gaussian_scale
            loss = torch.nn.functional.mse_loss(z, generator(z))
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < 1e-2:
                break
            if verbose and it % 500 == 0:
                print(f"[gaussian-init] it {it}  mse={loss.item():.4e}")
        return generator

    if strategy == "data_gaussian":
        # Data-LOCAL init: moment-match G(Z) to the provided data as
        #   G(Z) ~= mean(data) + gaussian_scale * std(data) * Z.
        # The generator starts at the right location and (slightly inflated) spread,
        # so the barycenter iteration only has to DENOISE locally -- no large
        # translation -- which equalises fit quality across a drifting trajectory.
        if data_sampler is None:
            raise ValueError("strategy='data_gaussian' requires data_sampler (local cells).")
        if hasattr(data_sampler, "to"):
            data_sampler = data_sampler.to(dev)
        with torch.no_grad():
            X = data_sampler.all() if hasattr(data_sampler, "all") else \
                torch.cat([data_sampler.sample(batch_size) for _ in range(8)], dim=0)
            mean = X.mean(0, keepdim=True)
            std = X.std(0, keepdim=True).clamp_min(1e-3)
        # here gaussian_scale acts as a modest spread multiplier (start a bit wide);
        # a value ~1.5-2 covers the noisy cloud, then training contracts it.
        mult = gaussian_scale if gaussian_scale <= 3.0 else 1.5
        for it in range(iters):
            z = latent_sampler.sample(batch_size)
            target = mean + mult * std * z
            loss = torch.nn.functional.mse_loss(generator(z), target)
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and it % 500 == 0:
                print(f"[data_gaussian-init] it {it}  mse={loss.item():.4e}")
        return generator

    if strategy == "flow":
        if data_sampler is None:
            raise ValueError("strategy='flow' requires data_sampler (pooled cells).")
        if hasattr(data_sampler, "to"):
            data_sampler = data_sampler.to(dev)  # match the generator's device
        # Infer the data dimension from the generator's output.
        with torch.no_grad():
            dim = generator(latent_sampler.sample(2)).shape[1]
        flow = NormalizingFlow(dim, flow_length=flow_length).to(dev)
        fit_normalizing_flow(flow, data_sampler, latent_sampler,
                             epochs=flow_epochs, batch_size=batch_size, verbose=verbose)
        flow.eval()
        # Regress the generator onto the (mode-capturing) flow samples.
        for it in range(iters):
            z = latent_sampler.sample(batch_size)
            with torch.no_grad():
                target, _ = flow(z)
            loss = torch.nn.functional.mse_loss(generator(z), target)
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and it % 500 == 0:
                print(f"[flow-init] regress it {it}  mse={loss.item():.4e}")
        return generator

    raise ValueError(f"Unknown init strategy '{strategy}'. "
                     "Use 'gaussian', 'data_gaussian', 'flow', 'vae_nf', or 'pretrained'.")
