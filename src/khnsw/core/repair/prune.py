"""Neighbor selection and pruning primitives for K-HNSW kinetic repair."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from khnsw.core.detection.cost import CostMeter


def rank_pool(
    base: int,
    pool: Iterable[int],
    ef: int | None,
    meter: CostMeter,
    vector_of: Callable[[int], NDArray[np.floating[Any]]],
) -> list[tuple[float, int]]:
    """Deduplicate candidate pool, drop base node, compute distances, and rank candidates.

    Evaluation and Sorting:
    -----------------------
    1. Candidates are deduplicated and the base node is excluded.
    2. Candidates are evaluated deterministically in ascending ID order so distance computations
       through `meter` are order-invariant.
    3. Pairwise distance D(base, c) is evaluated exactly once per candidate using `meter.dist`.
    4. Candidates are sorted primarily by distance ASCENDING (nearest first), with ties broken
       by node ID DESCENDING (mirroring hnswlib's max-heap priority queue extraction order).
    5. If ef is specified, only the top ef ranked candidates are retained. If ef is None,
       all ranked candidates are kept.

    Parameters
    ----------
    base : int
        Reference source node ID from which candidate distances are evaluated.
    pool : Iterable[int]
        Collection of candidate node IDs.
    ef : int | None
        Exploration budget cap. If None, no cap is applied.
    meter : CostMeter
        Computational distance counter for exact accounting.
    vector_of : Callable[[int], NDArray[np.floating]]
        Callable returning the 1D spatial vector for a given node ID.

    Returns
    -------
    list[tuple[float, int]]
        List of (distance, candidate_id) pairs sorted by distance asc, id desc.
    """
    base_id = int(base)
    unique_candidates = sorted({int(c) for c in pool if int(c) != base_id})

    if not unique_candidates:
        return []

    base_vec = vector_of(base_id)
    evaluated: list[tuple[float, int]] = []
    for cand_id in unique_candidates:
        d = meter.dist(base_vec, vector_of(cand_id))
        evaluated.append((d, cand_id))

    # Sort: distance ascending, candidate ID descending (mirrors hnswlib queue pop order)
    evaluated.sort(key=lambda item: (item[0], -item[1]))

    if ef is not None and ef >= 0:
        return evaluated[:ef]
    return evaluated


def prune(
    ranked: list[tuple[float, int]],
    base: int,
    cap: int,
    alpha: float,
    skip_if_below_cap: bool,
    meter: CostMeter,
    vector_of: Callable[[int], NDArray[np.floating[Any]]],
) -> list[tuple[float, int]]:
    r"""Prune ranked candidate connections using the alpha-RNG geometric heuristic.

    Mathematical Formulation and Properties:
    ----------------------------------------
    - The relaxation parameter alpha operates strictly on SQUARED Euclidean distances:
          \alpha \cdot D(r, e) < D(\text{base}, e)
      Consequently, \alpha = 1.1 on squared distances corresponds to
      \sqrt{1.1} \approx 1.0488088 on true unsquared Euclidean distances.
    - Early Return Semantics:
      When `skip_if_below_cap=True` and `len(ranked) < cap`, the heuristic returns `ranked`
      unchanged without evaluating any pairwise inter-candidate distances. This behavior
      directly matches hnswlib's `getNeighborsByHeuristic2`.
      IMPORTANT: This early return can leave edges in the graph that statically violate the
      empty-ellipse certificate condition.
    - Greedy Selection:
      Candidates are evaluated in `ranked` order. Candidate e is rejected if and only if
      there exists some already-accepted neighbor r (checked in strict acceptance order,
      stopping at the first hit) satisfying:
          \alpha \cdot D(r, e) < D(\text{base}, e)   (strict inequality; ties are accepted)
      Each inter-candidate distance D(r, e) evaluated constitutes exactly one meter count.

    Parameters
    ----------
    ranked : list[tuple[float, int]]
        Pre-ranked (distance_to_base, candidate_id) pairs.
    base : int
        Source node ID.
    cap : int
        Maximum number of surviving connections (e.g. max_degree(level)).
    alpha : float
        Heuristic pruning factor applied to squared Euclidean distances.
    skip_if_below_cap : bool
        If True and len(ranked) < cap, returns ranked immediately without pruning.
    meter : CostMeter
        Computational distance counter.
    vector_of : Callable[[int], NDArray[np.floating]]
        Callable returning the 1D spatial vector for a given node ID.

    Returns
    -------
    list[tuple[float, int]]
        List of accepted (distance_to_base, candidate_id) pairs in acceptance order,
        having length at most cap.
    """
    if cap <= 0 or not ranked:
        return []

    # Early return when candidate count is strictly below capacity
    if skip_if_below_cap and len(ranked) < cap:
        return list(ranked)

    selected: list[tuple[float, int]] = []
    selected_vecs: list[NDArray[np.floating[Any]]] = []

    for d_base_e, e_id in ranked:
        if len(selected) >= cap:
            break

        vec_e = vector_of(e_id)
        reject = False

        for r_vec in selected_vecs:
            d_re = meter.dist(r_vec, vec_e)
            if alpha * d_re < d_base_e:
                reject = True
                break

        if not reject:
            selected.append((d_base_e, e_id))
            selected_vecs.append(vec_e)

    return selected
