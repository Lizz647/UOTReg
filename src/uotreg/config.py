"""Configuration dataclasses.

These group the knobs so that experiments over ``d`` (dimension), the ground
cost ``phi``, bandwidth ``h``, tolerance ``tau``, divergence ``psi``, and the
relaxation mode are one-line changes rather than scattered notebook edits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Network architecture (generator / transport maps / potentials)."""
    dim: int = 20                      # data dimension d (e.g. number of PCs)
    latent_dim: Optional[int] = None   # generator latent dim (defaults to dim)
    gen_hidden: int = 256
    gen_layers: int = 4
    gen_dropout: float = 0.01
    map_hidden: int = 256
    map_layers: int = 5                # n_hidden in each map/potential MLP
    pot_hidden: int = 256
    pot_layers: int = 5
    dropout: float = 0.05
    batchnorm: bool = False


@dataclass
class UOTConfig:
    """Unbalanced-OT loss settings (shared by barycenter and trajectory solvers)."""
    relaxation: str = "one-sided"      # "balanced" | "one-sided" | "two-sided"
    divergence: str = "kl"             # psi: "kl" | "chi2" | "hellinger" | "tv"
    tau: float = 5.0                   # unbalanced tolerance (tau -> inf => balanced)
    cost: str = "sqeuclidean"          # ground cost phi: "sqeuclidean" | "l2" | callable


@dataclass
class WeightConfig:
    """Local-Frechet weighting (only used by the distribution estimator)."""
    bandwidth: float = 4.0             # h
    kernel: str = "gaussian"
    scheme: str = "positive"           # "positive" (truncate-at-zero) | "raw"
    threshold: Optional[float] = 0.01  # hard-prune cutoff eta; None disables pruning


@dataclass
class TrainConfig:
    """Optimization schedule and hardware."""
    outer_iters: int = 42              # number of generator updates (the old G_time)
    d_iters: int = 50                  # potential updates per outer step
    t_iters: int = 10                  # map updates per potential update
    g_iters: int = 50                  # generator updates per outer step
    batch_size: int = 64
    batch_size_g: int = 128
    lr_map: float = 3e-4
    lr_pot: float = 3e-4
    lr_gen: float = 1e-4
    weight_decay_td: float = 1e-10
    weight_decay_gen: float = 1e-8
    device: str = "auto"
    amp: bool = False                  # mixed precision (CUDA only)
    seed: Optional[int] = None
    verbose: bool = True
    log_every: int = 5


@dataclass
class TrajectoryConfig:
    """Settings for iterative pairwise-UOT trajectory fitting."""
    uot: UOTConfig = field(default_factory=lambda: UOTConfig(tau=50.0))
    d_iters: int = 300                 # potential updates per pair
    t_iters: int = 100                 # map updates per potential update
    batch_size: int = 128
    lr_map: float = 3e-4
    lr_pot: float = 3e-4
    warm_start: bool = True            # continue the *same* net from pair to pair
    device: str = "auto"
    seed: Optional[int] = None
    verbose: bool = True
