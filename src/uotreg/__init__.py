"""UOTReg: robust local-Frechet regression with unbalanced neural optimal transport.

Reorganized, GPU-ready implementation for the JCGS revision.  Typical use::

    from uotreg import DistributionEstimator, TrajectoryFitter
    from uotreg.data import samplers_from_labeled

    samplers, times = samplers_from_labeled(adata.obsm['X_pca'][:, :20],
                                            adata.obs['time'], device='auto')
    est = DistributionEstimator(dim=20)
    est.fit(samplers, timepoints=[1.5, 7.5, 13.5, 19.5, 25.5], query_time=13.5,
            h=4, tau=5, relaxation='one-sided', divergence='kl', init='flow')
    cells_t = est.sample(2000)

See the repository README and ``tutorials/embryoid_tutorial.ipynb`` for the full tutorial.
"""
from .barycenter import DistributionEstimator
from .config import (
    ModelConfig,
    TrainConfig,
    TrajectoryConfig,
    UOTConfig,
    WeightConfig,
)
from .data import (
    GaussianLatentSampler,
    GaussianMixtureLatentSampler,
    TensorSampler,
    samplers_from_arrays,
    samplers_from_labeled,
)
from .divergences import available_divergences, get_divergence
from .trajectory import FlowMatchingTrajectory, TrajectoryFitter
from .weights import effective_sample_size, frechet_weights

# experiment layer (paper numerics): synthetic data, baselines, metrics, the two-stage driver
from . import baselines, outlier_sim, pipelines, reverse_sim, simulation, trajectory_metrics
from .pipelines import estimate, fit_trajectories, resolve_std, run_stage1

__version__ = "0.1.0"

__all__ = [
    "DistributionEstimator",
    "TrajectoryFitter",
    "FlowMatchingTrajectory",
    "ModelConfig",
    "UOTConfig",
    "WeightConfig",
    "TrainConfig",
    "TrajectoryConfig",
    "TensorSampler",
    "GaussianLatentSampler",
    "samplers_from_arrays",
    "samplers_from_labeled",
    "frechet_weights",
    "effective_sample_size",
    "get_divergence",
    "available_divergences",
    "simulation",
    "baselines",
    "trajectory_metrics",
    "pipelines",
    "estimate",
    "run_stage1",
    "fit_trajectories",
    "resolve_std",
    "__version__",
]
