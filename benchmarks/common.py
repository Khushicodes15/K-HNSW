"""Shared benchmark toolkit for K-HNSW Member 4 evaluation suite.

Provides:
- Synthetic drift data generation with strictly decoupled PRNG streams
- Brute-force k-NN ground truth
- Recall@k evaluation
- Pre-registered decision rule checker (Claim 13 / member4.md §4)
- JSON result serialisation

PRNG SAFETY NOTE
----------------
Every function that uses randomness receives its own independent
``np.random.Generator`` instance seeded from a *distinct* integer offset
so that drawing samples for one stream never advances another stream's
state. This is the "indep_dir" protocol used throughout the repo; see
``scripts/measure_missing_controls.py`` for the established pattern.

Violation of this protocol causes all velocity directions to become
*orthogonal to* the initial position (the "cos ≡ 0 hazard") or other
subtle correlations that invalidate experiment results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from khnsw.core.movement.base import mean_nn_distance

# ---------------------------------------------------------------------------
# Synthetic drift data
# ---------------------------------------------------------------------------


def generate_isotropic_drift_data(
    n: int,
    d: int,
    seed: int,
    drift_frac: float = 0.3,
    displacement_mult: float = 1.0,
    t_final: float = 1.0,
    steps: int = 10,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[int]]:
    """Generate a point cloud and velocity field for a synthetic drift experiment.

    Uses **three strictly decoupled PRNG streams** seeded at ``seed``,
    ``seed + 1``, and ``seed + 2`` to avoid any cross-stream correlation:

    * ``rng_x``  — draws initial positions  (seed)
    * ``rng_id`` — selects which nodes move  (seed + 1)
    * ``rng_v``  — draws velocity directions (seed + 2)

    Speed is calibrated so that the *average* displacement after ``t_final``
    equals ``displacement_mult × mean_nn_distance(X₀)``.  This makes the
    multiplier interpretable across different ``(n, d)`` settings.

    Parameters
    ----------
    n : int
        Number of nodes.
    d : int
        Embedding dimensionality.
    seed : int
        Base seed.  Streams are seeded at seed, seed+1, seed+2.
    drift_frac : float
        Fraction of nodes that drift (in [0, 1]).
    displacement_mult : float
        Target displacement at ``t_final`` as a multiple of mean-NN distance.
    t_final : float
        Simulation end time.
    steps : int
        Number of discrete time steps.

    Returns
    -------
    X0 : ndarray, shape (n, d), float32
        Initial positions (Gaussian, zero-mean, unit variance).
    velocities : ndarray, shape (n, d), float32
        Per-node velocity vectors.  Static nodes have velocity **0**.
    moving_ids : list[int]
        Indices of drifting nodes.
    """
    if not (0.0 <= drift_frac <= 1.0):
        raise ValueError(f"drift_frac must be in [0, 1], got {drift_frac}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    # --- Stream 1: initial positions ---
    rng_x = np.random.default_rng(seed)
    X0 = rng_x.standard_normal((n, d)).astype(np.float32)

    # --- Stream 2: select moving node IDs ---
    rng_id = np.random.default_rng(seed + 1)
    n_moving = max(1, int(math.ceil(drift_frac * n)))
    moving_ids: list[int] = sorted(rng_id.choice(n, size=n_moving, replace=False).tolist())

    # --- Calibrate speed ---
    scale = mean_nn_distance(X0)
    if scale <= 0.0:
        scale = 1.0
    # speed such that ||v|| * t_final = displacement_mult * scale
    speed = (displacement_mult * scale) / max(t_final, 1e-9)

    # --- Stream 3: velocity directions (isotropic unit vectors) ---
    rng_v = np.random.default_rng(seed + 2)
    raw = rng_v.standard_normal((n_moving, d)).astype(np.float64)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    directions = (raw / norms).astype(np.float32)

    velocities = np.zeros((n, d), dtype=np.float32)
    for local_i, nid in enumerate(moving_ids):
        velocities[nid] = directions[local_i] * speed

    return X0, velocities, moving_ids


def advance_positions(
    X: NDArray[np.float32],
    velocities: NDArray[np.float32],
    dt: float,
) -> NDArray[np.float32]:
    """Return X advanced by one time step of size *dt*.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Current positions.
    velocities : ndarray, shape (n, d)
        Per-node velocities.
    dt : float
        Time-step size.

    Returns
    -------
    ndarray, shape (n, d), float32
        Updated positions.
    """
    return (X + velocities * dt).astype(np.float32)


# ---------------------------------------------------------------------------
# Brute-force ground truth
# ---------------------------------------------------------------------------


def brute_force_knn(
    X: NDArray[np.float32],
    queries: NDArray[np.float32],
    k: int,
) -> NDArray[np.intp]:
    """Exact k-nearest-neighbour indices using squared L2 distance.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Database vectors.
    queries : ndarray, shape (q, d)
        Query vectors.
    k : int
        Number of neighbours to return.

    Returns
    -------
    ndarray, shape (q, k), dtype intp
        Row ``i`` contains the indices in ``X`` of the k nearest
        neighbours of ``queries[i]``, sorted nearest-first.
    """
    X_f = np.asarray(X, dtype=np.float32)
    Q_f = np.asarray(queries, dtype=np.float32)
    k = min(k, len(X_f))

    results = np.empty((len(Q_f), k), dtype=np.intp)
    for qi, q in enumerate(Q_f):
        diff = X_f - q
        sq_dists = (diff * diff).sum(axis=1)
        n_db = len(sq_dists)
        if k >= n_db:
            # Return all indices sorted by distance
            order = np.argsort(sq_dists)
            results[qi] = order[:k]
        else:
            # argpartition is O(n) instead of O(n log n) full sort
            part = np.argpartition(sq_dists, k)[:k]
            # sort only the k candidates
            order = np.argsort(sq_dists[part])
            results[qi] = part[order]
    return results


# ---------------------------------------------------------------------------
# Recall evaluation
# ---------------------------------------------------------------------------


def evaluate_recall(
    index: Any,
    queries: NDArray[np.float32],
    ground_truth: NDArray[np.intp],
    k: int = 10,
    ef: int = 50,
) -> float:
    """Compute mean Recall@k of *index* over all queries.

    Recall@k for a single query = |returned ∩ true_k| / k.

    Parameters
    ----------
    index : HNSWIndex or compatible
        Must expose a ``search(query, k, ef)`` method returning a list/array
        of (node_id, distance) pairs or just node IDs.
    queries : ndarray, shape (q, d)
        Query vectors.
    ground_truth : ndarray, shape (q, k)
        True k-NN indices from :func:`brute_force_knn`.
    k : int
        Number of neighbours to retrieve.
    ef : int
        Search expansion factor (passed to index.search).

    Returns
    -------
    float
        Mean Recall@k in [0, 1].
    """
    total_recall = 0.0
    q = len(queries)
    for qi in range(q):
        raw_res = index.search(queries[qi], k=k, ef=ef)

        # Handle tuple of (labels, distances) as returned by HNSWIndex and hnswlib
        if isinstance(raw_res, tuple) and len(raw_res) == 2 and hasattr(raw_res[0], "__iter__"):
            retrieved = {int(x) for x in raw_res[0]}
        elif isinstance(raw_res, (list, np.ndarray)) and len(raw_res) > 0:
            first = raw_res[0]
            if isinstance(first, (tuple, list)) and len(first) >= 2:
                # list of (id, dist) pairs
                retrieved = {int(item[0]) for item in raw_res}
            else:
                # list/array of IDs
                retrieved = {int(x) for x in raw_res}
        else:
            retrieved = set()

        true_set = set(int(x) for x in ground_truth[qi])
        total_recall += len(retrieved & true_set) / k
    return total_recall / q if q > 0 else 0.0


# ---------------------------------------------------------------------------
# Pre-registered decision rule (member4.md §4 / Claim 13)
# ---------------------------------------------------------------------------


def check_decision_rule(
    recall_a: float,
    recall_b: float,
    cost_a: float | None = None,
    cost_b: float | None = None,
) -> bool:
    """Apply the pre-registered decision rule for Claim 13.

    **Rule (pre-registered in member4.md §4)**:

    Returns ``True`` (arm A wins) if **either**:

    1. ``recall_a - recall_b >= 0.02``  (≥ 2 pp recall advantage), OR
    2. ``recall_a >= recall_b - 0.01``  AND  ``cost_a <= cost_b * 0.5``
       (within 1 pp recall AND at least 2× cheaper)

    Returns ``False`` otherwise (no win / inconclusive).

    Parameters
    ----------
    recall_a : float
        Mean Recall@k for arm A (e.g. ``round_robin``).
    recall_b : float
        Mean Recall@k for arm B (e.g. ``periodic_k5``).
    cost_a : float or None
        Total distance computations for arm A.  Required for rule 2.
    cost_b : float or None
        Total distance computations for arm B.  Required for rule 2.

    Returns
    -------
    bool
        ``True`` if arm A satisfies either branch of the decision rule.
    """
    # Rule 1: recall advantage >= 2 percentage points
    if recall_a - recall_b >= 0.02:
        return True

    # Rule 2: near-parity recall with >= 2x cost saving
    if cost_a is not None and cost_b is not None and cost_b > 0:
        if recall_a >= recall_b - 0.01 and cost_a <= cost_b * 0.5:
            return True

    return False


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def save_results(results: dict[str, Any], path: str | Path) -> None:
    """Serialise *results* to a pretty-printed JSON file.

    Creates parent directories as needed.  If the file already exists it
    is **overwritten**.  All NumPy scalars are coerced to native Python
    types so that ``json.dump`` never raises ``TypeError``.

    Parameters
    ----------
    results : dict
        Benchmark result dictionary (arbitrary nesting).
    path : str or Path
        Output file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_coerce(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with out.open("w", encoding="utf-8") as fh:
        json.dump(_coerce(results), fh, indent=2)
