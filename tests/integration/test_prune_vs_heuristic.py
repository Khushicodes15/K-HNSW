"""Differential testing of core/repair prune vs Member 2's select_neighbors_heuristic.

Verifies mathematical equivalence of alpha-RNG geometric pruning on real HNSWIndex
candidate pools. Validates that on tie-free pools, prune and select_neighbors_heuristic
produce identical neighbor lists, and on pools with distance ties, produce identical
sets, with strict inequality (< vs <=) handling explicitly verified.
"""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.detection.cost import CostMeter
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.repair.prune import prune, rank_pool
from khnsw.core.search.heuristic import select_neighbors_heuristic


def _run_prune(idx: HNSWIndex, base: int, pool: list[int], cap: int) -> list[int]:
    """Thin adapter evaluating core/repair prune with alpha=1.0 and skip_if_below_cap=False."""
    meter = CostMeter()
    ranked = rank_pool(
        base=base,
        pool=pool,
        ef=None,
        meter=meter,
        vector_of=idx.vector,
    )
    selected = prune(
        ranked=ranked,
        base=base,
        cap=cap,
        alpha=1.0,
        skip_if_below_cap=False,
        meter=meter,
        vector_of=idx.vector,
    )
    return [nid for _, nid in selected]


@pytest.mark.parametrize("d", [2, 8])
def test_differential_prune_vs_heuristic(d: int) -> None:
    """Compare prune vs select_neighbors_heuristic on 150 random pools per dimension."""
    rng = np.random.default_rng(123 + d)
    n = 100
    m = 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    idx = HNSWIndex(dim=d, m=m, seed=123 + d)
    idx.add(np.arange(n, dtype=np.int64), X)

    tie_cases = 0
    exact_matches = 0

    for trial in range(150):
        base = int(rng.choice(idx.ids()))
        pool_size = int(rng.integers(5, 35))
        pool = list(rng.choice(idx.ids(), size=pool_size, replace=True))
        cap = int(rng.integers(2, 12))

        # Member 2's heuristic output
        h_out = select_neighbors_heuristic(
            idx._store,
            base=base,
            candidates=pool,
            m=cap,
            keep_pruned_connections=False,
        )

        # core/repair prune output
        p_out = _run_prune(idx, base=base, pool=pool, cap=cap)

        # Detect if candidate distances to base contain numerical ties
        unique_cands = [c for c in set(pool) if c != base]
        dists = [
            round(float(np.sum((idx.vector(base) - idx.vector(c)) ** 2)), 6) for c in unique_cands
        ]
        has_distance_tie = len(dists) != len(set(dists))

        if has_distance_tie:
            tie_cases += 1
            if set(h_out) != set(p_out):
                raise AssertionError(
                    f"Disagreement on pool with ties (d={d}, trial={trial}):\n"
                    f"  base: {base}\n"
                    f"  pool: {pool}\n"
                    f"  heuristic output: {h_out}\n"
                    f"  prune output:     {p_out}"
                )
        else:
            if h_out != p_out:
                raise AssertionError(
                    f"Disagreement on tie-free pool (d={d}, trial={trial}):\n"
                    f"  base: {base}\n"
                    f"  pool: {pool}\n"
                    f"  heuristic output: {h_out}\n"
                    f"  prune output:     {p_out}"
                )
            exact_matches += 1

    print(f"\n[Differential d={d}] total=150, exact_matches={exact_matches}, tie_cases={tie_cases}")
    assert exact_matches + tie_cases == 150


def test_strict_inequality_exact_boundary() -> None:
    """Verifies strict inequality (< vs <=) on exact integer boundary where D(r, c) == D(base, c).

    Coordinates:
    - Base node 0 at (0, 0)
    - Node 1 at (2, 0): D(base, 1) = 4
    - Node 2 at (1, 2): D(base, 2) = 1 + 4 = 5, and D(1, 2) = (1 - 2)^2 + 2^2 = 5

    Node 1 is accepted first (D=4).
    For Node 2, D(base, 2) = 5 and D(1, 2) = 5.
    Under strict inequality (<): 1.0 * D(1, 2) < D(base, 2) is False (5 < 5 is False),
    so Node 2 is accepted.
    Under non-strict (<=): 5 <= 5 is True -> Node 2 would be pruned!
    """
    idx = HNSWIndex(dim=2, m=4, seed=42)
    pts = np.array(
        [
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [1.0, 2.0],  # 2
        ],
        dtype=np.float32,
    )
    idx.add(np.array([0, 1, 2], dtype=np.int64), pts)

    h_out = select_neighbors_heuristic(
        idx._store,
        base=0,
        candidates=[1, 2],
        m=2,
        keep_pruned_connections=False,
    )
    p_out = _run_prune(idx, base=0, pool=[1, 2], cap=2)

    assert h_out == [1, 2]
    assert p_out == [1, 2]
