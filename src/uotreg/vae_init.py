r"""VAE + normalizing-flow generator initialization (faithful to the old code).

This reproduces the ``VAENormalizingFlow`` initialization used in the original
``UOT_embryoid_new2`` pipeline: an encoder + (this generator as) decoder + a
stack of planar flows on the latent, trained as a flow-VAE for a few epochs on
the pooled cells.  The trained **decoder is the generator**, so after this call
the generator already produces a multi-modal, data-aware cloud -- a much better
starting point than a broad Gaussian for clustered scRNA-seq data.

Architecture / training match the original:
  encoder  : Encoder(nin=d, n_latent=size, num_layers=4, dropout=0.05)
  decoder  : the Generator (Gnet) being initialized  (latent_dim = d)
  flow     : `flow_length` PlanarFlow blocks on the d-dim latent
  VAE bits : mu = Linear(size, d), sigma = Linear(size, d)->Softplus->Hardtanh(1e-4, 5)
  train    : recon MSE + beta * KL(flow-corrected), beta warmup, Adam(1e-3),
             ExponentialLR(0.99995), coef=5, epochs=5, batch_size=64.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import Encoder, PlanarFlow


class VAENormalizingFlow(nn.Module):
    """Flow-VAE whose decoder is the generator we want to initialize."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int,
                 encoder_dims: int, flow_length: int = 16):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mu = nn.Linear(encoder_dims, latent_dim)
        self.sigma = nn.Sequential(
            nn.Linear(encoder_dims, latent_dim),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.0),
        )
        self.flows = nn.ModuleList(PlanarFlow(latent_dim) for _ in range(flow_length))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.sigma(h)

    def forward(self, x):
        mu, sigma = self.encode(x)
        eps = torch.randn(x.shape[0], mu.shape[1], device=x.device)
        z0 = sigma * eps + mu
        z = z0
        ladj = []
        for flow in self.flows:
            ladj.append(flow.log_abs_det_jacobian(z))
            z = flow(z)
        # flow-corrected KL  (q(z0) - p(z_k) - sum log|det J|)
        log_p_zk = -0.5 * z * z
        log_q_z0 = -0.5 * (torch.log(sigma + 1e-12) + (z0 - mu) ** 2 / sigma)
        logs = (log_q_z0 - log_p_zk).sum() - torch.cat(ladj, dim=1).sum()
        kl = logs / float(x.shape[0])
        return self.decoder(z), kl


def vae_nf_initialize(
    generator: nn.Module,
    data,                         # pooled cells: TensorSampler (.all()) or array/tensor
    dim: int,
    *,
    encoder_size: int = 256,
    encoder_layers: int = 4,
    flow_length: int = 16,
    epochs: int = 5,
    coef: float = 5.0,
    lr: float = 1e-3,
    gamma: float = 0.99995,
    batch_size: int = 64,
    device=None,
    verbose: bool = False,
) -> nn.Module:
    """Initialize ``generator`` (the VAE decoder) in place by flow-VAE training."""
    dev = device if device is not None else next(generator.parameters()).device
    if hasattr(data, "all"):
        X = data.all()
    else:
        X = torch.as_tensor(data, dtype=torch.float32)
    X = X.to(dev).float()
    n = X.shape[0]

    encoder = Encoder(dim, encoder_size, size=encoder_size, num_layers=encoder_layers,
                      dropout=0.05).to(dev)
    model = VAENormalizingFlow(encoder, generator, latent_dim=dim,
                               encoder_dims=encoder_size, flow_length=flow_length).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    model.train()
    steps = max(1, n // batch_size)
    for ep in range(epochs):
        beta = ep / float(coef * epochs)
        perm = torch.randperm(n, device=dev)
        running = 0.0
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            x = X[idx]
            x_tilde, kl = model(x)
            loss = F.mse_loss(x_tilde, x) + beta * kl
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += float(loss.item())
        if verbose:
            print(f"[vae_nf-init] epoch {ep+1}/{epochs}  loss={running/steps:.4f}  beta={beta:.3f}")
    generator.eval()
    return generator
