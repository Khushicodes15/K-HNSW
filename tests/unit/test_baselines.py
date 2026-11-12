"""Unit tests for baseline index maintenance routines (src/khnsw/core/graph/baselines.py).

Covers:
  - HNSWIndex.delete() on leaf, intermediate, and entry-point nodes.
  - Invariants hold after every deletion operation.
  - Removed ID is removed from all neighbor and in_neighbor lists.
  - delete_and_reinsert leaves the moved node searchable at its new coordinates.
  - rebuild produces an index matching a fresh build from the same data and seed.
  - Hypothesis property test verifying invariants across random delete/insert sequences.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from khnsw.core.graph.audit import validate_invariants
from khnsw.core.graph.baselines import delete_and_reinsert, rebuild
from khnsw.core.graph.hnsw import HNSWIndex


def test_hnsw_delete_leaf_middle_and_entry_point() -> None:
    """Test delete() on leaf, middle, and entry-point nodes, verifying invariants."""
    rng = np.random.default_rng(42)
    dim = 8
    m = 4
    idx = HNSWIndex(dim=dim, m=m, ef_construction=32, seed=42)
    n = 25
    ids = np.arange(n, dtype=np.int64)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    idx.add(ids, vecs)

    assert validate_invariants(idx) == []
    initial_ep = idx.entry_point()
    assert initial_ep is not None

    # Identify nodes at different layer levels
    leaf_nodes = [nid for nid in ids if idx.node_level(int(nid)) == 0]
    mid_nodes = [nid for nid in ids if 0 < idx.node_level(int(nid)) < idx.max_level()]

    # 1. Delete a leaf node (level 0)
    leaf_to_del = int(leaf_nodes[0])
    idx.delete(leaf_to_del)
    assert not idx.contains(leaf_to_del)
    assert leaf_to_del not in idx.ids()
    with pytest.raises(KeyError):
        idx.version(leaf_to_del)
    assert validate_invariants(idx) == []

    # Verify leaf_to_del is gone from every node's forward and reverse lists
    for rem_id in idx.ids():
        for lvl in range(idx.node_level(rem_id) + 1):
            assert leaf_to_del not in idx.neighbors(rem_id, lvl)
            assert leaf_to_del not in idx.in_neighbors(rem_id, lvl)

    # 2. Delete an intermediate node (if present)
    if mid_nodes:
        mid_to_del = int(mid_nodes[0])
        idx.delete(mid_to_del)
        assert not idx.contains(mid_to_del)
        assert validate_invariants(idx) == []

    # 3. Delete the entry point node
    ep_to_del = idx.entry_point()
    assert ep_to_del is not None
    idx.delete(ep_to_del)
    assert not idx.contains(ep_to_del)
    assert ep_to_del not in idx.ids()
    assert validate_invariants(idx) == []

    # New entry point must be valid at the new max_level
    new_ep = idx.entry_point()
    assert new_ep is not None
    assert idx.node_level(new_ep) == idx.max_level()

    # 4. Deleting a non-existent node raises KeyError
    with pytest.raises(KeyError):
        idx.delete(99999)

    # 5. Delete all remaining nodes one by one until empty
    for rem_id in list(idx.ids()):
        idx.delete(rem_id)
        assert validate_invariants(idx) == []

    assert idx.size() == 0
    assert idx.max_level() == -1
    assert idx.entry_point() is None


def test_delete_and_reinsert() -> None:
    """delete_and_reinsert updates the node vector and makes it searchable at new position."""
    rng = np.random.default_rng(123)
    dim = 8
    idx = HNSWIndex(dim=dim, m=4, ef_construction=32, seed=123)
    n = 20
    idx.add(np.arange(n, dtype=np.int64), rng.standard_normal((n, dim)).astype(np.float32))

    target_id = 5
    old_vec = idx.vector(target_id).copy()

    # Move target_id far away: e.g. [100.0, ...]
    new_vec = np.full(dim, 100.0, dtype=np.float32)
    delete_and_reinsert(idx, target_id, new_vec)

    assert idx.size() == n
    assert idx.contains(target_id)
    np.testing.assert_array_equal(idx.vector(target_id), new_vec)
    assert not np.array_equal(idx.vector(target_id), old_vec)

    assert validate_invariants(idx) == []

    # Query near the new vector: target_id must be the #1 nearest neighbor
    ret_ids, dists = idx.search(new_vec, k=1, ef=10)
    assert ret_ids[0] == target_id
    assert dists[0] == pytest.approx(0.0, abs=1e-5)


def test_rebuild_matches_fresh_build() -> None:
    """rebuild produces an identical graph topology and search results to fresh build."""
    seed = 77
    n = 40
    dim = 8
    m = 4
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    # 1. Base build
    idx1 = HNSWIndex(dim=dim, m=m, ef_construction=32, seed=seed)
    idx1.add(ids, data)

    # 2. Rebuild without vector overrides
    rebuilt = rebuild(idx1)

    assert validate_invariants(rebuilt) == []
    assert rebuilt.size() == idx1.size()
    assert rebuilt.ids() == idx1.ids()
    assert rebuilt.max_level() == idx1.max_level()
    assert rebuilt.entry_point() == idx1.entry_point()

    for nid in ids:
        nid_int = int(nid)
        assert rebuilt.node_level(nid_int) == idx1.node_level(nid_int)
        for lvl in range(rebuilt.node_level(nid_int) + 1):
            assert rebuilt.neighbors(nid_int, lvl) == idx1.neighbors(nid_int, lvl)
            assert rebuilt.in_neighbors(nid_int, lvl) == idx1.in_neighbors(nid_int, lvl)

    # 3. Rebuild with vectors_by_id override
    updated_vec = np.ones(dim, dtype=np.float32) * 50.0
    rebuilt_override = rebuild(idx1, vectors_by_id={3: updated_vec})
    assert validate_invariants(rebuilt_override) == []
    np.testing.assert_array_equal(rebuilt_override.vector(3), updated_vec)

    # 4. Rebuild on empty index
    empty_idx = HNSWIndex(dim=dim, m=m)
    empty_rebuilt = rebuild(empty_idx)
    assert empty_rebuilt.size() == 0
    assert empty_rebuilt.entry_point() is None


# ---------------------------------------------------------------------------
# Hypothesis: random delete/insert sequences maintain invariants
# ---------------------------------------------------------------------------


class TestHypothesisBaselines:
    @given(
        n_initial=st.integers(min_value=5, max_value=25),
        n_ops=st.integers(min_value=5, max_value=20),
        dim=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_delete_insert_sequence_maintains_invariants(
        self, n_initial: int, n_ops: int, dim: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        idx = HNSWIndex(dim=dim, m=4, ef_construction=16, seed=seed)

        # Initial populate
        active_ids = list(range(n_initial))
        initial_vecs = rng.standard_normal((n_initial, dim)).astype(np.float32)
        idx.add(np.array(active_ids, dtype=np.int64), initial_vecs)

        next_id = n_initial

        for _ in range(n_ops):
            # If we have active nodes, randomly choose to delete or insert
            action = rng.choice(["delete", "insert", "move"]) if active_ids else "insert"

            if action == "delete" and len(active_ids) > 1:
                target = int(rng.choice(active_ids))
                idx.delete(target)
                active_ids.remove(target)
            elif action == "move" and active_ids:
                target = int(rng.choice(active_ids))
                new_v = rng.standard_normal(dim).astype(np.float32)
                delete_and_reinsert(idx, target, new_v)
            else:
                new_id = next_id
                next_id += 1
                new_v = rng.standard_normal(dim).astype(np.float32)
                idx.add(np.array([new_id], dtype=np.int64), new_v.reshape(1, -1))
                active_ids.append(new_id)

            assert validate_invariants(idx) == []
