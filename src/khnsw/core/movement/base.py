"""Base utility functions for kinetic movement models and deterministic randomness."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def mean_nn_distance(
    X: NDArray[Any],
    sample: int = 1000,
    seed: int = 0,
) -> float:
    """Compute mean nearest-neighbor Euclidean distance over a deterministic sample of rows.

    Calculates exact brute-force Euclidean (L2, not squared) distances from each sampled
    row to its nearest non-self neighbor. If n < 2, returns 0.0.

    Parameters
    ----------
    X : NDArray[Any]
        2D array of spatial positions of shape (n, d), or 1D array of shape (n,).
    sample : int, default=1000
        Maximum number of rows to sample. If n <= sample, computes over all n rows.
    seed : int, default=0
        Deterministic seed for row sampling when n > sample.

    Returns
    -------
    float
        Average Euclidean nearest-neighbor distance excluding self. Returns 0.0 if n < 2.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, np.newaxis]
    elif X_arr.ndim != 2:
        msg = f"Expected 1D or 2D array, got shape {X_arr.shape}"
        raise ValueError(msg)

    n = X_arr.shape[0]
    if n < 2:
        return 0.0

    if n <= sample:
        indices = np.arange(n)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed]))
        indices = rng.choice(n, size=sample, replace=False)

    nn_dists: list[float] = []
    for idx in indices:
        diff = X_arr - X_arr[idx]
        dists = np.linalg.norm(diff, axis=1)
        dists[idx] = np.inf
        min_dist = float(np.min(dists))
        nn_dists.append(min_dist)

    return float(np.mean(nn_dists))


def rng_for(seed: int, step: int) -> np.random.Generator:
    """Generate an isolated, deterministic NumPy Generator for a specific simulation step.

    Guarantees that pseudo-random outputs depend solely on the (seed, step) pair
    and never on query evaluation order or call sequence.

    Parameters
    ----------
    seed : int
        Base seed for the simulation model.
    step : int
        Discrete simulation step index.

    Returns
    -------
    np.random.Generator
        Deterministic random generator seeded via SeedSequence([seed, step]).
    """
    return np.random.default_rng(np.random.SeedSequence([seed, step]))
