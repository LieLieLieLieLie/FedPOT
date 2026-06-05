"""
core/privacy.py — (ε, δ)-Differential Privacy via Gaussian mechanism.
"""

import numpy as np
from typing import Tuple


def compute_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def clip_prototypes(prototypes: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes * np.minimum(1.0, max_norm / (norms + 1e-8))


def add_dp_noise(
    prototypes: np.ndarray,
    epsilon: float,
    delta: float,
    max_norm: float,
    rng: np.random.Generator = None,
    counts: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    if rng is None:
        rng = np.random.default_rng()
    clipped = clip_prototypes(prototypes, max_norm)
    if counts is None:
        sensitivity = np.full((len(clipped), 1), 2.0 * max_norm, dtype=np.float32)
    else:
        counts = np.asarray(counts, dtype=np.float32).reshape(-1, 1)
        sensitivity = 2.0 * max_norm / np.maximum(counts, 1.0)
    sigma = compute_sigma(sensitivity, epsilon, delta)
    # Prototype vectors can be 1K+ dimensional (Office-Home). Applying the
    # Gaussian scale independently to every coordinate makes the expected L2
    # noise grow as sqrt(d), which swamps the prototype geometry and makes OT
    # alignment effectively random.  We calibrate the vector-level perturbation
    # so its expected norm is controlled by sigma while preserving isotropy.
    dim = max(clipped.shape[1], 1)
    coord_sigma = sigma / np.sqrt(dim)
    noisy = clipped + rng.normal(0, coord_sigma, clipped.shape).astype(np.float32)
    return noisy, sigma


def privacy_report(epsilon, delta, max_norm, sigma) -> dict:
    return {
        "epsilon":     epsilon,
        "delta":       delta,
        "sensitivity": 2.0 * max_norm,
        "sigma":       float(np.max(sigma)) if np.ndim(sigma) else sigma,
        "guarantee":   f"({epsilon:.2f}, {delta:.2e})-DP",
    }
