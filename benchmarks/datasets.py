"""Dataset loaders and synthetic fallbacks for K-HNSW benchmarks (Member 4).

Supported sources
-----------------
- ``synthetic_gaussian`` — simple i.i.d. Gaussian cloud
- ``load_sift``          — ANN-benchmarks SIFT1M (.fvecs binary format)
- ``load_glove``         — Stanford GloVe word vectors (.txt format)
- ``generate_sift_like`` — offline clustered synthetic fallback (SIFT-like
                           statistics: non-negative, sparse) used when the
                           real datasets are not available locally

All loaders return ``np.float32`` arrays of shape ``(N, d)``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Simple Gaussian synthetic dataset
# ---------------------------------------------------------------------------


def synthetic_gaussian(n: int, d: int, seed: int = 42) -> NDArray[np.float32]:
    """Return *n* i.i.d. standard-normal vectors of dimension *d*.

    Parameters
    ----------
    n : int
        Number of vectors.
    d : int
        Dimensionality.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ndarray, shape (n, d), float32
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# SIFT1M loader (.fvecs)
# ---------------------------------------------------------------------------


def load_sift(path: Union[str, Path]) -> NDArray[np.float32]:
    """Load a ``.fvecs`` file (ANN-benchmarks / SIFT format).

    Format per vector:
        [dim : int32][v₀ : float32][v₁ : float32]…[v_{dim-1} : float32]

    Parameters
    ----------
    path : str or Path
        Path to the ``.fvecs`` file.

    Returns
    -------
    ndarray, shape (N, dim), float32

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is empty or the format is inconsistent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SIFT file not found: {p}")

    with p.open("rb") as fh:
        raw = fh.read()

    if len(raw) < 4:
        raise ValueError(f"File too small to contain even one vector: {p}")

    # Read dimension from first 4 bytes
    (dim,) = struct.unpack_from("<i", raw, 0)
    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} in {p}")

    record_size = 4 + dim * 4  # 4 bytes for dim + dim floats
    if len(raw) % record_size != 0:
        raise ValueError(
            f"File size {len(raw)} is not a multiple of record size "
            f"{record_size} (dim={dim}) in {p}"
        )

    n = len(raw) // record_size
    # Interpret the entire file as int32 + float32 interleaved
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, dim + 1)
    # First column is the dimension integer reinterpreted as float — drop it
    vectors = arr[:, 1:].copy()
    return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# GloVe loader (.txt)
# ---------------------------------------------------------------------------


def load_glove(
    path: Union[str, Path],
    d: int,
    max_vectors: int | None = None,
) -> NDArray[np.float32]:
    """Load a GloVe ``.txt`` file and return the ``d``-dimensional vectors.

    Each line in a GloVe file has the format::

        <word> <f1> <f2> … <fd>

    Parameters
    ----------
    path : str or Path
        Path to the GloVe ``.txt`` file.
    d : int
        Expected dimensionality.  Lines with a different number of floats
        are silently skipped (they may be header lines or corrupt).
    max_vectors : int or None
        If set, return at most this many vectors (first ``max_vectors`` valid
        lines).

    Returns
    -------
    ndarray, shape (N, d), float32

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If no valid vectors were found.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GloVe file not found: {p}")

    vectors: list[NDArray[np.float32]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(" ")
            # parts[0] is the word token; parts[1:] are the floats
            floats = parts[1:]
            if len(floats) != d:
                continue
            try:
                vec = np.array(floats, dtype=np.float32)
            except ValueError:
                continue
            vectors.append(vec)
            if max_vectors is not None and len(vectors) >= max_vectors:
                break

    if not vectors:
        raise ValueError(
            f"No valid {d}-dimensional vectors found in {p}. "
            f"Check that 'd={d}' matches the file's dimensionality."
        )

    return np.stack(vectors, axis=0)


# ---------------------------------------------------------------------------
# SIFT-like clustered synthetic fallback
# ---------------------------------------------------------------------------


def generate_sift_like(
    n: int,
    d: int = 128,
    seed: int = 42,
    n_clusters: int = 20,
) -> NDArray[np.float32]:
    """Generate a SIFT-like synthetic dataset as an offline fallback.

    Mimics SIFT1M statistics:
    - Non-negative values (absolute value of Gaussian draws)
    - Clustered structure (``n_clusters`` Gaussian cluster centres)
    - Sparse-ish histogram-like shape

    This is *not* real SIFT data.  It is provided only as a reproducible
    offline substitute for sandboxed or disconnected environments.

    Parameters
    ----------
    n : int
        Number of vectors.
    d : int
        Dimensionality (default 128, matching SIFT1M).
    seed : int
        RNG seed.
    n_clusters : int
        Number of synthetic cluster centres.

    Returns
    -------
    ndarray, shape (n, d), float32
    """
    rng = np.random.default_rng(seed)

    # Sample cluster centres
    centres = np.abs(rng.standard_normal((n_clusters, d))).astype(np.float32) * 50.0

    # Assign each vector to a random cluster
    assignments = rng.integers(0, n_clusters, size=n)

    # Draw vectors as centre + small noise, then clip to non-negative
    noise = rng.standard_normal((n, d)).astype(np.float32) * 10.0
    X = centres[assignments] + noise
    X = np.clip(X, 0.0, None)  # SIFT descriptors are non-negative
    return X.astype(np.float32)
