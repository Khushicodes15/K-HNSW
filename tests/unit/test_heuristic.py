"""Unit tests for select_neighbors_heuristic (HNSW Algorithm 4)."""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.graph.store import GraphStore
from khnsw.core.search.heuristic import select_neighbors_heuristic


def _make_line_store(n: int, dim: int = 1, m: int = 16, m0: int = 32) -> GraphStore:
    """Build a GraphStore with n nodes at positions 0.0, 1.0, ..., n-1 (level 0)."""
    store = GraphStore(dim=dim, m=m, m0=m0)
    for i in range(n):
        store.add_node(i, np.array([float(i)] * dim, dtype=np.float32), level=0)
    return store


# ---------------------------------------------------------------------------
# 1. Empty candidates
# ---------------------------------------------------------------------------


def test_empty_candidates_returns_empty() -> None:
    store = _make_line_store(5)
    result = select_neighbors_heuristic(store, base=0, candidates=[], m=4)
    assert result == []


# ---------------------------------------------------------------------------
# 2. Result size <= m
# ---------------------------------------------------------------------------


def test_result_size_at_most_m() -> None:
    store = _make_line_store(20)
    for m in (1, 3, 5, 10, 25):
        result = select_neighbors_heuristic(store, base=0, candidates=list(range(1, 20)), m=m)
        assert len(result) <= m


# ---------------------------------------------------------------------------
# 3. No duplicates in result
# ---------------------------------------------------------------------------


def test_no_duplicates() -> None:
    store = _make_line_store(10)
    result = select_neighbors_heuristic(store, base=0, candidates=list(range(1, 10)), m=8)
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# 4. Base node is excluded even when present in candidates
# ---------------------------------------------------------------------------


def test_base_node_excluded() -> None:
    store = _make_line_store(6)
    # Include the base node (id=2) explicitly in candidates
    result = select_neighbors_heuristic(store, base=2, candidates=[0, 1, 2, 3, 4, 5], m=5)
    assert 2 not in result


# ---------------------------------------------------------------------------
# 5. Deterministic: same input → same output
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    store = _make_line_store(15)
    candidates = list(range(1, 15))
    r1 = select_neighbors_heuristic(store, base=0, candidates=candidates, m=5)
    r2 = select_neighbors_heuristic(store, base=0, candidates=candidates, m=5)
    assert r1 == r2


# ---------------------------------------------------------------------------
# 6. Diversity property: each accepted node is closer to base than to every
#    previously accepted neighbor (without keepPruned)
# ---------------------------------------------------------------------------


def test_diversity_property() -> None:
    # 2-D vectors arranged so some candidates are "dominated"
    # Node 0 = base at (0, 0)
    # Node 1 = (1, 0)  — close, should be selected
    # Node 2 = (2, 0)  — collinear; dominated by node 1 when checking dist(2,1) < dist(0,2)
    # Node 3 = (0, 1)  — orthogonal to node 1; should be selected (not dominated)
    dim = 2
    store = GraphStore(dim=dim, m=10, m0=10)
    positions = {
        0: [0.0, 0.0],
        1: [1.0, 0.0],
        2: [2.0, 0.0],
        3: [0.0, 1.0],
    }
    for nid, pos in positions.items():
        store.add_node(nid, np.array(pos, dtype=np.float32), level=0)

    result = select_neighbors_heuristic(store, base=0, candidates=[1, 2, 3], m=3)

    # Node 1 must be selected (closest)
    assert 1 in result
    # Node 3 must be selected (orthogonal — not dominated)
    assert 3 in result
    # Node 2 should be pruned: dist(2,1) = 1.0 < dist(0,2) = 4.0  ✓ dominated
    assert 2 not in result


# ---------------------------------------------------------------------------
# 7. keepPruned fills up to min(m, candidates)
# ---------------------------------------------------------------------------


def test_keep_pruned_fills_up_to_m() -> None:
    # Same collinear setup as above: without keepPruned, node 2 is pruned.
    dim = 2
    store = GraphStore(dim=dim, m=10, m0=10)
    for i in range(5):
        store.add_node(i, np.array([float(i), 0.0], dtype=np.float32), level=0)

    # Without keepPruned — collinear nodes 2, 3, 4 are pruned after 1 is selected
    result_no_prune = select_neighbors_heuristic(
        store, base=0, candidates=[1, 2, 3, 4], m=4, keep_pruned_connections=False
    )
    # Without keepPruned, only node 1 should be accepted (others are dominated)
    assert result_no_prune == [1]

    # With keepPruned — should fill up to m=4 from pruned pool
    result_pruned = select_neighbors_heuristic(
        store, base=0, candidates=[1, 2, 3, 4], m=4, keep_pruned_connections=True
    )
    assert len(result_pruned) == 4
    assert 1 in result_pruned  # accepted first
    # Pruned ones (2, 3, 4) fill the remaining slots
    for nid in [2, 3, 4]:
        assert nid in result_pruned


# ---------------------------------------------------------------------------
# 8. Raw base vector (not a node id) works correctly
# ---------------------------------------------------------------------------


def test_base_as_raw_vector() -> None:
    store = _make_line_store(6)
    # Base at 0.5 (not an actual node); nodes 0 and 1 should be nearest
    base_vec = np.array([0.5], dtype=np.float32)
    result = select_neighbors_heuristic(store, base=base_vec, candidates=[0, 1, 2, 3, 4, 5], m=2)
    assert len(result) <= 2
    # Both 0 and 1 equidistant; at least one of them should be in result
    assert 0 in result or 1 in result


# ---------------------------------------------------------------------------
# 9. All candidates identical to base → empty result (all excluded or dominated)
# ---------------------------------------------------------------------------


def test_candidates_all_same_as_base_excluded() -> None:
    store = _make_line_store(3)
    # Only the base node itself in candidates — must be excluded
    result = select_neighbors_heuristic(store, base=0, candidates=[0], m=5)
    assert result == []


# ---------------------------------------------------------------------------
# 10. Duplicate candidates de-duplicated
# ---------------------------------------------------------------------------


def test_duplicate_candidates_deduped() -> None:
    store = _make_line_store(4)
    result = select_neighbors_heuristic(store, base=0, candidates=[1, 1, 2, 2, 3, 3], m=10)
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# 11. m=1 always returns exactly one candidate (the nearest)
# ---------------------------------------------------------------------------


def test_m_equals_one_returns_nearest() -> None:
    store = _make_line_store(10)
    # Base at 5; nearest non-base candidate from [1,3,6,9] is 6 (dist=1)
    result = select_neighbors_heuristic(store, base=5, candidates=[1, 3, 6, 9], m=1)
    assert len(result) == 1
    assert result[0] == 6


# ---------------------------------------------------------------------------
# 12. Validation errors
# ---------------------------------------------------------------------------


def test_validation_errors() -> None:
    store = _make_line_store(3)

    with pytest.raises(ValueError, match="m must be >= 1"):
        select_neighbors_heuristic(store, base=0, candidates=[1, 2], m=0)

    with pytest.raises(KeyError, match="does not exist"):
        select_neighbors_heuristic(store, base=999, candidates=[1], m=1)

    with pytest.raises(KeyError, match="does not exist"):
        select_neighbors_heuristic(store, base=0, candidates=[999], m=1)

    with pytest.raises(ValueError, match="base vector shape must be"):
        select_neighbors_heuristic(store, base=np.zeros(5, dtype=np.float32), candidates=[1], m=1)
