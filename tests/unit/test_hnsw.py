"""Unit tests for HNSWIndex (src/khnsw/core/graph/hnsw.py).

Covers:
  - Structural invariants after random inserts (layer hierarchy, degree bounds,
    forward/reverse consistency, valid entry point).
  - Recall@10 >= 0.9 versus brute-force on a seeded random dataset.
  - Search returns at most min(k, n) results sorted by distance.
  - Empty-index edge case.
  - move() updates the vector and version, changes nothing else.
  - Hypothesis property test on random insert batches keeping invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from khnsw.core.graph.hnsw import HNSWIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brute_force_knn(
    vectors: np.ndarray, query: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, sq_l2_dists) for exact k-NN (ids == row indices)."""
    diffs = vectors - query
    dists = np.sum(diffs * diffs, axis=1)
    order = np.argsort(dists)[:k]
    return order.astype(np.int64), dists[order].astype(np.float32)


def _check_invariants(index: HNSWIndex) -> None:
    """Assert every structural invariant that must hold after any insert batch."""
    all_ids = index.ids()
    n = index.size()
    assert len(all_ids) == n

    if n == 0:
        assert index.max_level() == -1
        assert index.entry_point() is None
        return

    ep = index.entry_point()
    assert ep is not None
    assert index.contains(ep)
    assert index.node_level(ep) == index.max_level()

    for nid in all_ids:
        nl = index.node_level(nid)
        assert 0 <= nl <= index.max_level()

        for lc in range(nl + 1):
            nbrs = index.neighbors(nid, lc)
            max_deg = index.max_degree(lc)

            # Degree bound
            assert len(nbrs) <= max_deg, (
                f"node {nid} level {lc}: degree {len(nbrs)} > max {max_deg}"
            )

            # No self-loops
            assert nid not in nbrs, f"self-loop at node {nid} level {lc}"

            # Forward/reverse consistency
            for nbr in nbrs:
                in_nbrs = index.in_neighbors(nbr, lc)
                assert nid in in_nbrs, f"edge {nid}->{nbr} at level {lc} missing reverse entry"

        # Reverse edges also point to valid forward edges
        for lc in range(nl + 1):
            in_nbrs = index.in_neighbors(nid, lc)
            for src in in_nbrs:
                fwd = index.neighbors(src, lc)
                assert nid in fwd, f"reverse edge {src}->{nid} at level {lc} has no forward"


# ---------------------------------------------------------------------------
# Test: empty index
# ---------------------------------------------------------------------------


class TestEmptyIndex:
    def test_empty_properties(self) -> None:
        idx = HNSWIndex(dim=8, m=4)
        assert idx.size() == 0
        assert idx.max_level() == -1
        assert idx.entry_point() is None
        assert idx.ids() == []
        assert idx.dim == 8

    def test_empty_search(self) -> None:
        idx = HNSWIndex(dim=8, m=4)
        q = np.zeros(8, dtype=np.float32)
        ids, dists = idx.search(q, k=5, ef=10)
        assert len(ids) == 0
        assert len(dists) == 0

    def test_empty_snapshot(self) -> None:
        idx = HNSWIndex(dim=8, m=4)
        snap = idx.snapshot()
        assert snap["num_nodes"] == 0
        assert snap["max_level"] == -1


# ---------------------------------------------------------------------------
# Test: precondition validation
# ---------------------------------------------------------------------------


class TestPreconditions:
    def test_add_mismatched_lengths(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(ValueError, match="len\\(ids\\)"):
            idx.add(np.array([0, 1], dtype=np.int64), np.zeros((3, 4), dtype=np.float32))

    def test_add_wrong_dim(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(ValueError, match="shape"):
            idx.add(np.array([0], dtype=np.int64), np.zeros((1, 5), dtype=np.float32))

    def test_add_negative_id(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(ValueError, match="non-negative"):
            idx.add(np.array([-1], dtype=np.int64), np.zeros((1, 4), dtype=np.float32))

    def test_add_duplicate_ids_in_batch(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(ValueError, match="unique"):
            idx.add(np.array([0, 0], dtype=np.int64), np.zeros((2, 4), dtype=np.float32))

    def test_add_existing_id(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        idx.add(np.array([0], dtype=np.int64), np.zeros((1, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="already exists"):
            idx.add(np.array([0], dtype=np.int64), np.ones((1, 4), dtype=np.float32))

    def test_search_ef_lt_k(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(ValueError, match="ef"):
            idx.search(np.zeros(4, dtype=np.float32), k=5, ef=3)

    def test_search_k_lt_1(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(ValueError, match="k"):
            idx.search(np.zeros(4, dtype=np.float32), k=0, ef=1)

    def test_search_wrong_dim(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(ValueError, match="shape"):
            idx.search(np.zeros(5, dtype=np.float32), k=1, ef=1)

    def test_move_nonexistent(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        with pytest.raises(KeyError):
            idx.move(999, np.zeros(4, dtype=np.float32))

    def test_move_wrong_dim(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        idx.add(np.array([0], dtype=np.int64), np.zeros((1, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="shape"):
            idx.move(0, np.zeros(5, dtype=np.float32))


# ---------------------------------------------------------------------------
# Test: invariants after random inserts
# ---------------------------------------------------------------------------


class TestInvariantsAfterInserts:
    @pytest.mark.parametrize("n", [1, 5, 50, 200])
    def test_invariants(self, n: int) -> None:
        rng = np.random.default_rng(42)
        dim = 8
        m = 8
        idx = HNSWIndex(dim=dim, m=m, ef_construction=32, seed=42)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        ids = np.arange(n, dtype=np.int64)
        idx.add(ids, vecs)
        assert idx.size() == n
        _check_invariants(idx)

    def test_incremental_inserts(self) -> None:
        """Insert nodes one-by-one and check invariants at every step."""
        rng = np.random.default_rng(123)
        dim = 8
        m = 4
        idx = HNSWIndex(dim=dim, m=m, ef_construction=16, seed=123)
        for i in range(30):
            v = rng.standard_normal(dim).astype(np.float32)
            idx.add(np.array([i], dtype=np.int64), v.reshape(1, -1))
            _check_invariants(idx)


# ---------------------------------------------------------------------------
# Test: keep_pruned_connections parameter and degree contrast
# ---------------------------------------------------------------------------


class TestKeepPrunedConnections:
    def test_default_keep_pruned_connections_is_false(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        assert idx.keep_pruned_connections is False

    def test_keep_pruned_connections_degree_contrast_and_invariants(self) -> None:
        """With False (default), level 0 min/mean degree is strictly below cap M0.
        With True, nodes top up pruned connections and reach cap M0."""
        from khnsw.core.graph.audit import degree_stats, validate_invariants

        seed = 7
        n = 800
        dim = 16
        m = 16
        m0 = 32

        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n, dim)).astype(np.float32)
        ids_arr = np.arange(n, dtype=np.int64)

        # Default: keep_pruned_connections=False
        idx_default = HNSWIndex(dim=dim, m=m, ef_construction=100, seed=seed)
        idx_default.add(ids_arr, data)

        violations_default = validate_invariants(idx_default)
        assert violations_default == []

        stats_default = degree_stats(idx_default)
        assert stats_default[0]["min"] < m0
        assert stats_default[0]["mean"] < m0

        # Explicit: keep_pruned_connections=True
        idx_true = HNSWIndex(
            dim=dim, m=m, ef_construction=100, seed=seed, keep_pruned_connections=True
        )
        idx_true.add(ids_arr, data)

        violations_true = validate_invariants(idx_true)
        assert violations_true == []

        stats_true = degree_stats(idx_true)
        # With True, all nodes top up to cap M0
        assert stats_true[0]["min"] == m0
        assert stats_true[0]["mean"] == pytest.approx(float(m0))


# ---------------------------------------------------------------------------
# Test: recall@10 >= 0.9 on seeded random data
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_at_10(self) -> None:
        rng = np.random.default_rng(7)
        n = 800
        dim = 16
        k = 10
        ef_search = 64
        num_queries = 100

        idx = HNSWIndex(dim=dim, m=16, ef_construction=100, seed=7)
        data = rng.standard_normal((n, dim)).astype(np.float32)
        ids_arr = np.arange(n, dtype=np.int64)
        idx.add(ids_arr, data)

        queries = rng.standard_normal((num_queries, dim)).astype(np.float32)
        total_recall = 0.0

        for qi in range(num_queries):
            q = queries[qi]
            ann_ids, ann_dists = idx.search(q, k=k, ef=ef_search)
            bf_ids, _ = _brute_force_knn(data, q, k)

            # Recall = |intersection| / k
            recall = len(set(ann_ids.tolist()) & set(bf_ids.tolist())) / k
            total_recall += recall

        avg_recall = total_recall / num_queries
        print(f"\n=== Recall@{k} (n={n}, dim={dim}, ef={ef_search}): {avg_recall:.4f} ===")
        assert avg_recall >= 0.9, f"Recall@{k} = {avg_recall:.4f} < 0.9"


# ---------------------------------------------------------------------------
# Test: search returns <= min(k, n) sorted results
# ---------------------------------------------------------------------------


class TestSearchResults:
    def test_returns_at_most_min_k_n(self) -> None:
        dim = 4
        n = 3
        idx = HNSWIndex(dim=dim, m=4, ef_construction=16, seed=0)
        vecs = np.eye(n, dim, dtype=np.float32)
        idx.add(np.arange(n, dtype=np.int64), vecs)

        # k > n: should return only n results
        ids, dists = idx.search(np.zeros(dim, dtype=np.float32), k=10, ef=10)
        assert len(ids) == n
        assert len(dists) == n

    def test_results_sorted_by_distance(self) -> None:
        rng = np.random.default_rng(99)
        dim = 8
        n = 50
        idx = HNSWIndex(dim=dim, m=8, ef_construction=32, seed=99)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        idx.add(np.arange(n, dtype=np.int64), vecs)

        q = rng.standard_normal(dim).astype(np.float32)
        ids, dists = idx.search(q, k=10, ef=32)
        # Distances must be non-decreasing
        for i in range(1, len(dists)):
            assert dists[i] >= dists[i - 1], (
                f"results not sorted: dist[{i - 1}]={dists[i - 1]} > dist[{i}]={dists[i]}"
            )

    def test_single_point_search(self) -> None:
        dim = 4
        idx = HNSWIndex(dim=dim, m=4, ef_construction=16, seed=0)
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        idx.add(np.array([42], dtype=np.int64), v.reshape(1, -1))

        ids, dists = idx.search(v, k=1, ef=1)
        assert len(ids) == 1
        assert ids[0] == 42
        assert dists[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: distance computation counter
# ---------------------------------------------------------------------------


class TestDistanceComputations:
    def test_search_counter_bounds_and_determinism(self) -> None:
        dim = 16
        n = 500
        ef = 32
        k = 10
        seed = 42

        # Build index 1
        rng1 = np.random.default_rng(seed)
        vecs1 = rng1.standard_normal((n, dim)).astype(np.float32)
        idx1 = HNSWIndex(dim=dim, m=16, ef_construction=64, seed=seed)
        idx1.add(np.arange(n, dtype=np.int64), vecs1)

        counts_after_build1 = idx1.distance_computations
        assert counts_after_build1 > 0

        query = rng1.standard_normal(dim).astype(np.float32)
        idx1.search(query, k=k, ef=ef)
        search_dists1 = idx1.distance_computations - counts_after_build1

        # Counter increases by more than ef but less than n
        assert search_dists1 > ef, f"Expected > ef ({ef}), got {search_dists1}"
        assert search_dists1 < n, f"Expected < n ({n}), got {search_dists1}"

        # Determinism check with fresh index built with identical seed and query
        rng2 = np.random.default_rng(seed)
        vecs2 = rng2.standard_normal((n, dim)).astype(np.float32)
        idx2 = HNSWIndex(dim=dim, m=16, ef_construction=64, seed=seed)
        idx2.add(np.arange(n, dtype=np.int64), vecs2)

        counts_after_build2 = idx2.distance_computations
        assert counts_after_build2 == counts_after_build1

        idx2.search(query, k=k, ef=ef)
        search_dists2 = idx2.distance_computations - counts_after_build2
        assert search_dists1 == search_dists2


# ---------------------------------------------------------------------------
# Test: move updates vector and version only
# ---------------------------------------------------------------------------


class TestMove:
    def test_move_updates_vector(self) -> None:
        dim = 4
        idx = HNSWIndex(dim=dim, m=4, ef_construction=16, seed=0)
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.add(np.array([0], dtype=np.int64), v0.reshape(1, -1))
        assert idx.version(0) == 0

        v1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        idx.move(0, v1)
        np.testing.assert_array_equal(idx.vector(0), v1)
        assert idx.version(0) == 1

    def test_move_does_not_change_topology(self) -> None:
        rng = np.random.default_rng(55)
        dim = 8
        n = 30
        idx = HNSWIndex(dim=dim, m=8, ef_construction=32, seed=55)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        idx.add(np.arange(n, dtype=np.int64), vecs)

        node_id = 5
        # Snapshot topology before move
        nl = idx.node_level(node_id)
        nbrs_before = {lc: idx.neighbors(node_id, lc) for lc in range(nl + 1)}
        in_nbrs_before = {lc: idx.in_neighbors(node_id, lc) for lc in range(nl + 1)}
        ep_before = idx.entry_point()

        new_vec = rng.standard_normal(dim).astype(np.float32)
        idx.move(node_id, new_vec)

        # Topology unchanged
        assert idx.node_level(node_id) == nl
        assert idx.entry_point() == ep_before
        for lc in range(nl + 1):
            assert idx.neighbors(node_id, lc) == nbrs_before[lc]
            assert idx.in_neighbors(node_id, lc) == in_nbrs_before[lc]

    def test_move_increments_version_monotonically(self) -> None:
        dim = 4
        idx = HNSWIndex(dim=dim, m=4, ef_construction=16, seed=0)
        idx.add(np.array([10], dtype=np.int64), np.ones((1, 4), dtype=np.float32))
        for i in range(5):
            idx.move(10, np.full(4, float(i), dtype=np.float32))
            assert idx.version(10) == i + 1


# ---------------------------------------------------------------------------
# Hypothesis property test: random insert batches maintain invariants
# ---------------------------------------------------------------------------


class TestHypothesis:
    @given(
        n=st.integers(min_value=1, max_value=80),
        dim=st.integers(min_value=2, max_value=16),
        m=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_inserts_keep_invariants(self, n: int, dim: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        idx = HNSWIndex(dim=dim, m=m, ef_construction=max(m, 16), seed=seed)

        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        ids_arr = np.arange(n, dtype=np.int64)
        idx.add(ids_arr, vecs)

        assert idx.size() == n
        _check_invariants(idx)
