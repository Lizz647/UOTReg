r"""The outlier-robustness simulation (10 GMM clouds in 10-D, 4 inlier + 4 outlier modes).

Two generators for the same setup:

* :func:`generate_outlier_gmms` -- the parametric form used by ``run_outliers`` and the
  divergence comparison (``numpy`` Generator, tunable perturbation).
* :func:`make_data` -- the verbatim legacy generator (``np.random.seed(25)``, SPD-squared
  covariances); reproduces exactly the data the paper's saved barycenters were trained on,
  so ``outlier_figs`` can overlay those checkpoints on the same cells.
"""
from __future__ import annotations

from typing import List

import numpy as np

DIM = 10

# 4 inlier means (mass 0.94) + 4 outlier means (mass 0.06)
INLIER_MEANS = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, 2, 2, 2, 2, 2],
    [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
], dtype=float)
OUTLIER_MEANS = np.array([
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, -10, -10, -10, -10, -10],
    [10, 10, 10, 10, 10, 0, 0, 0, 0, 0],
    [-10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
], dtype=float)
MEANS = np.vstack([INLIER_MEANS, OUTLIER_MEANS])
WEIGHTS = np.array([0.24, 0.24, 0.23, 0.23, 0.015, 0.015, 0.015, 0.015])
COV = np.eye(DIM) * 0.5


def _sample_gmm(means, weights, cov, n, rng):
    comp = rng.choice(len(weights), size=n, p=weights / weights.sum())
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, means.shape[1])) @ L.T
    return (means[comp] + z).astype(np.float32)


def generate_outlier_gmms(num: int = 10, n: int = 2000, perturb: float = 0.1,
                          seed: int = 0) -> List[np.ndarray]:
    """Return ``num`` point clouds (each ``n`` x 10) — perturbed copies of the GMM."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        means = MEANS + perturb * rng.standard_normal(MEANS.shape)   # slight per-dist perturbation
        out.append(_sample_gmm(means, WEIGHTS, COV, n, rng))
    return out


def outlier_contamination(samples: np.ndarray) -> float:
    """Fraction of points closer to an outlier mode than to any inlier mode.

    ~0 for a robust (UOT) barycenter; larger for the balanced barycenter that is
    pulled toward the outliers.
    """
    di = np.min(np.linalg.norm(samples[:, None, :] - INLIER_MEANS[None], axis=2), axis=1)
    do = np.min(np.linalg.norm(samples[:, None, :] - OUTLIER_MEANS[None], axis=2), axis=1)
    return float(np.mean(do < di))


def inlier_mode_distance(samples: np.ndarray) -> float:
    """Mean distance of each point to its nearest inlier mode (lower=better)."""
    di = np.min(np.linalg.norm(samples[:, None, :] - INLIER_MEANS[None], axis=2), axis=1)
    return float(di.mean())


# --- verbatim legacy generator (the paper's run of record) --------------------- #
def generate_perturbed_gmm_params(template_means, template_covs, n_mixtures,
                                  mean_perturb=0.5, cov_perturb=0.3,
                                  weights=None, weight_perturb=0.0):
    """Generate GMM parameters by perturbing template means/covs (verbatim)."""
    n_components = len(template_means)
    dim = template_means[0].shape[0]
    gmm_params = []
    if weights is None:
        base_weights = np.ones(n_components) / n_components
    else:
        base_weights = np.asarray(weights)
    for _ in range(n_mixtures):
        means, covariances = [], []
        if weight_perturb > 0:
            noise = np.random.randn(n_components) * weight_perturb
            perturbed_weights = base_weights + noise
            mn = perturbed_weights.min()
            if mn < 0:
                perturbed_weights = perturbed_weights - mn
            s = perturbed_weights.sum()
            perturbed_weights = perturbed_weights / s if s > 0 else np.ones(n_components) / n_components
        else:
            perturbed_weights = base_weights.copy()
        for k in range(n_components):
            shift = np.random.randn(dim) * mean_perturb
            perturbed_mean = template_means[k] + shift
            noise = np.random.randn(dim, dim)
            noise = 0.5 * (noise + noise.T)
            perturbed_cov = template_covs[k] + cov_perturb * noise
            perturbed_cov = perturbed_cov @ perturbed_cov.T + 1e-3 * np.eye(dim)
            means.append(perturbed_mean)
            covariances.append(perturbed_cov)
        gmm_params.append((means, covariances, perturbed_weights))
    return gmm_params


def sample_gmm(n_samples, means, covariances, weights):
    """Sample ``n_samples`` from a GMM (verbatim)."""
    n_components = len(means)
    n_per = np.random.multinomial(n_samples, weights)
    samples = [np.random.multivariate_normal(means[k], covariances[k], n_per[k]) for k in range(n_components)]
    return np.vstack(samples)


# --- exact template (4 inlier + 4 outlier modes in 10-D) -------------------- #
TEMPLATE_MEANS = [
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), np.array([2, 2, 2, 2, 2, -2, -2, -2, -2, -2]),
    np.array([-2, -2, -2, -2, -2, 2, 2, 2, 2, 2]), np.array([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2]),
    np.array([0, 0, 0, 0, 0, 10, 10, 10, 10, 10]), np.array([0, 0, 0, 0, 0, -10, -10, -10, -10, -10]),
    np.array([10, 10, 10, 10, 10, 0, 0, 0, 0, 0]), np.array([-10, -10, -10, -10, -10, 0, 0, 0, 0, 0]),
]
N_MIXTURES = 10
N_COMPONENTS = 8
SAMPLES_PER_MIXTURE = 2000
SEED = 25


def make_data(seed: int = SEED):
    """Reproduce the exact dataset; returns (X_all, all_data, labels_all)."""
    np.random.seed(seed)
    template_covs = [np.eye(DIM) * 0.5 for _ in range(N_COMPONENTS)]
    gmm_param_list = generate_perturbed_gmm_params(TEMPLATE_MEANS, template_covs,
                                                   n_mixtures=N_MIXTURES, weights=WEIGHTS)
    all_data, all_labels = [], []
    for i, (means, covs, w) in enumerate(gmm_param_list):
        X = sample_gmm(SAMPLES_PER_MIXTURE, means, covs, w)
        all_data.append(X.astype(np.float32))
        all_labels.extend([i] * X.shape[0])
    return np.vstack(all_data).astype(np.float32), all_data, np.array(all_labels)
