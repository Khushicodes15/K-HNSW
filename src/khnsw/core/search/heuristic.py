"""HNSW Algorithm 4: greedy diverse-neighbor selection heuristic.

Pure Python standard library and NumPy only (browser/Pyodide safe).

Reference: Malkov & Yashunin 2018, Algorithm 4.
Certificate relationship: docs/algorithm/certificates.md §1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from khnsw.core.graph.store import GraphStore


def _sq_l2(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Squared Euclidean distance between two float32 1-D arrays."""
    diff = a - b
    return float(np.dot(diff, diff))


def select_neighbors_heuristic(
    store: GraphStore,
    base: int | NDArray[np.float32],
    candidates: Sequence[int],
    m: int,
    *,
    keep_pruned_connections: bool = False,
    extend_candidates: bool = False,
    level: int = 0,
) -> list[int]:
    """Select up to m neighbors for a node using the HNSW relative-neighborhood heuristic.

    Implements HNSW Algorithm 4 (Malkov & Yashunin 2018):
    1. Sort candidates by ascending squared-L2 distance to the base vector.
    2. Accept a candidate *e* only if it is closer to the base than to every
       already-accepted neighbor (relative-neighborhood diversity condition).
    3. When ``keep_pruned_connections=True``, fill the gap between the accepted
       set and *m* from the discarded (pruned) candidates, in distance order.
    4. The base node itself is excluded from the result regardless of its presence
       in the candidate list.

    Distance metric: squared Euclidean (||base - e||²), consistent with
    docs/contracts/graph_index.md §Distance Metric.

    Parameters
    ----------
    store : GraphStore
        Graph store providing node vectors.
    base : int | NDArray[np.float32]
        Either a node ID present in *store*, or a raw float32 1-D vector of
        shape (store.dim,) representing the query position.
    candidates : Sequence[int]
        Node IDs to evaluate as potential neighbors.
    m : int
        Maximum number of neighbors to return (m >= 1).
    keep_pruned_connections : bool, optional
        When True, fill the result with pruned candidates (in distance order)
        until the result reaches min(m, len(unique_valid_candidates)).
        Default False.
    extend_candidates : bool, optional
        When True, expand the working candidate set with the existing neighbors
        of each candidate at *level* (Algorithm 3 extension, unused by default).
        Default False.
    level : int, optional
        Graph layer used for neighbour lookup when ``extend_candidates=True``.
        Ignored when ``extend_candidates=False``.  Default 0.

    Returns
    -------
    list[int]
        Up to *m* selected neighbor node IDs, ordered by ascending distance to
        the base vector (nearest first).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    # Resolve base vector
    if isinstance(base, (int, np.integer)):
        base_id = int(base)
        if base_id not in store:
            raise KeyError(f"base node {base_id} does not exist in graph")
        base_vec = store.vector(base_id)
        exclude_id: int | None = base_id
    else:
        base_vec = np.asarray(base, dtype=np.float32)
        if base_vec.shape != (store.dim,):
            raise ValueError(f"base vector shape must be ({store.dim},), got {base_vec.shape}")
        exclude_id = None

    if not candidates:
        return []

    # De-duplicate and exclude the base node from consideration
    seen: set[int] = set()
    working: list[int] = []
    for c in candidates:
        c_int = int(c)
        if c_int in seen:
            continue
        if c_int == exclude_id:
            continue
        if c_int not in store:
            raise KeyError(f"candidate node {c_int} does not exist in graph")
        seen.add(c_int)
        working.append(c_int)

    if not working:
        return []

    # Optionally extend the working set with neighbors of candidates
    if extend_candidates:
        extended: set[int] = set(working)
        for c in list(working):
            if level <= store.node_level(c):
                for nbr in store.neighbors(c, level):
                    nbr_int = int(nbr)
                    if nbr_int != exclude_id and nbr_int not in extended:
                        if nbr_int in store:
                            extended.add(nbr_int)
                            working.append(nbr_int)

    # Compute distances from base to all candidates in one batched call
    candidate_vecs = np.stack([store.vector(c) for c in working])
    diff = candidate_vecs - base_vec
    dists = np.sum(diff * diff, axis=1)  # squared L2

    # Sort by ascending distance
    order = np.argsort(dists, kind="stable")
    sorted_pairs = [(working[int(i)], float(dists[int(i)])) for i in order]

    result: list[int] = []  # accepted neighbors
    result_vecs: list[NDArray[np.float32]] = []  # their vectors (for diversity check)
    pruned: list[tuple[float, int]] = []  # (dist, node_id) of discarded candidates

    for c_node, d_base in sorted_pairs:
        if len(result) >= m:
            break

        c_vec = store.vector(c_node)

        # Diversity condition (Algorithm 4):
        # Accept c only if dist(base, c) < dist(c, r) for every already-accepted r.
        dominated = False
        for r_vec in result_vecs:
            r_diff = c_vec - r_vec
            d_c_r = float(np.dot(r_diff, r_diff))
            if d_c_r < d_base:
                # c is closer to r than to base → c is "dominated" by r; prune it
                dominated = True
                break

        if dominated:
            pruned.append((d_base, c_node))
        else:
            result.append(c_node)
            result_vecs.append(c_vec)

    # keepPrunedConnections: fill up to m with discarded candidates (distance order)
    if keep_pruned_connections and len(result) < m and pruned:
        pruned.sort(key=lambda x: x[0])
        for _d_p, p_node in pruned:
            if len(result) >= m:
                break
            result.append(p_node)

    return result
