"""Layer-level beam search for Hierarchical Navigable Small World graphs.

Implements HNSW Algorithm 2 (Malkov & Yashunin, 2018) using pure Python stdlib + NumPy.
"""

from __future__ import annotations

import heapq
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from khnsw.core.graph.store import GraphStore


class SearchResult(NamedTuple):
    """Container for search result candidate with squared L2 distance and node ID."""

    distance: float
    node_id: int


class DistanceCounter:
    """Mutable counter that accumulates distance computation counts across calls.

    Pass the same instance to multiple ``search_layer`` invocations to
    accumulate a running total without changing the function's return type.
    """

    __slots__ = ("value",)

    def __init__(self, value: int = 0) -> None:
        self.value = value


def search_layer(
    store: GraphStore,
    query: NDArray[np.float32],
    entry_points: Sequence[int] | set[int] | int,
    ef: int,
    level: int,
    counter: DistanceCounter | None = None,
) -> list[SearchResult]:
    """Perform beam search at a single graph hierarchy layer (HNSW Algorithm 2).

    Explores neighbors at the specified level using a candidate min-heap and a
    result max-heap of capacity `ef`. Terminates when the nearest unexplored
    candidate is farther from the query than the furthest candidate currently
    in the result set.

    Distance Metric:
        Squared Euclidean distance (||q - v||_2^2), consistent with project contracts.

    Parameters
    ----------
    store : GraphStore
        Underlying graph store containing node vectors and adjacency lists.
    query : NDArray[np.float32]
        1D query vector of shape (dim,).
    entry_points : Sequence[int] | set[int] | int
        One or more node identifiers to begin traversal from.
    ef : int
        Size of the dynamic candidate exploration list (ef >= 1).
    level : int
        Hierarchy layer level to traverse (level >= 0).
    counter : DistanceCounter | None, optional
        If provided, ``counter.value`` is incremented by the number of
        pairwise distance computations actually performed during this call.
        Existing callers that omit this argument are unaffected.

    Returns
    -------
    list[SearchResult]
        Up to `ef` nearest elements found, sorted in ascending order of distance.
    """
    if ef < 1:
        raise ValueError(f"ef must be >= 1, got {ef}")
    if level < 0:
        raise ValueError(f"level must be non-negative, got {level}")

    q = np.asarray(query, dtype=np.float32)
    if q.shape != (store.dim,):
        raise ValueError(f"query shape must be ({store.dim},), got {q.shape}")

    # Standardize entry points to a unique list
    if isinstance(entry_points, (int, np.integer)):
        ep_list = [int(entry_points)]
    else:
        ep_list = [int(ep) for ep in entry_points]

    if not ep_list or len(store) == 0:
        return []

    # Validate entry points
    for ep in ep_list:
        if ep not in store:
            raise KeyError(f"Entry point {ep} does not exist in graph store")
        if level > store.node_level(ep):
            raise ValueError(
                f"Entry point {ep} max level {store.node_level(ep)} < search level {level}"
            )

    visited: set[int] = set()
    # Min-heap of candidates to explore: (distance, node_id)
    candidates: list[tuple[float, int]] = []
    # Max-heap of best ef results: (-distance, node_id)
    w_max_heap: list[tuple[float, int]] = []

    # Initialize with unique entry points
    seen_eps: set[int] = set()
    for ep in ep_list:
        if ep in seen_eps:
            continue
        seen_eps.add(ep)
        visited.add(ep)

        ep_vec = store.vector(ep)
        d = float(np.sum((ep_vec - q) ** 2))
        if counter is not None:
            counter.value += 1

        heapq.heappush(candidates, (d, ep))
        heapq.heappush(w_max_heap, (-d, ep))

    # Cap initial results to ef best
    while len(w_max_heap) > ef:
        heapq.heappop(w_max_heap)

    # Main beam search loop
    while candidates:
        c_dist, c_node = heapq.heappop(candidates)
        furthest_dist = -w_max_heap[0][0]

        # Stopping condition: nearest candidate is farther than the worst result
        if c_dist > furthest_dist:
            break

        # Collect unvisited neighbors at this level
        nbrs = store.neighbors(c_node, level)
        unvisited = [nbr for nbr in nbrs if nbr not in visited]
        if not unvisited:
            continue

        for nbr in unvisited:
            visited.add(nbr)

        # Batched squared L2 distance evaluation
        nbr_vecs = np.stack([store.vector(nbr) for nbr in unvisited])
        diff = nbr_vecs - q
        dists = np.sum(diff * diff, axis=-1)
        if counter is not None:
            counter.value += len(unvisited)

        for i, nbr in enumerate(unvisited):
            d = float(dists[i])
            furthest_dist = -w_max_heap[0][0]

            if d < furthest_dist or len(w_max_heap) < ef:
                heapq.heappush(candidates, (d, nbr))
                heapq.heappush(w_max_heap, (-d, nbr))

                if len(w_max_heap) > ef:
                    heapq.heappop(w_max_heap)

    # Convert results to SearchResult and sort ascending by distance
    results = [SearchResult(distance=-neg_d, node_id=nid) for neg_d, nid in w_max_heap]
    results.sort(key=lambda r: (r.distance, r.node_id))
    return results
