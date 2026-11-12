"""Unit and property tests for ReferenceGraph test double."""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.detection.certificates import is_violated_static
from khnsw.core.interfaces import ANNIndex, GraphIndex, MutableGraphIndex
from tests.fixtures.reference_graph import ReferenceGraph

# =============================================================================
# 1. Protocol Conformance & Empty Graph Behavior
# =============================================================================


def test_protocol_conformance_and_real_calls() -> None:
    """Exercise EVERY method of MutableGraphIndex with real invocations."""
    g = ReferenceGraph(dim=3, M=4)

    # Runtime checkable protocol assertions
    assert isinstance(g, ANNIndex)
    assert isinstance(g, GraphIndex)
    assert isinstance(g, MutableGraphIndex)

    # 1. dim property
    assert g.dim == 3

    # 2. size, contains, max_level, entry_point on empty graph
    assert g.size() == 0
    assert not g.contains(10)
    assert g.max_level() == -1
    assert g.entry_point() is None
    assert g.ids() == []

    # 3. search on empty graph
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    top_ids, top_dists = g.search(q, k=5, ef=10)
    assert len(top_ids) == 0
    assert len(top_dists) == 0

    # 4. add nodes
    ids = np.array([10, 20], dtype=np.int64)
    vecs = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    g.add(ids, vecs)
    assert g.size() == 2
    assert g.contains(10)
    assert g.contains(20)
    assert g.ids() == [10, 20]
    assert g.node_level(10) == 0
    assert g.max_level() == 0
    assert g.entry_point() == 10  # lowest-id at max level

    # 5. vector and version
    np.testing.assert_array_equal(g.vector(10), vecs[0])
    assert g.version(10) == 0

    # 6. move node
    g.move(10, np.array([0.5, 0.5, 0.5], dtype=np.float32))
    assert g.version(10) == 1
    assert g.version(20) == 0
    np.testing.assert_array_equal(g.vector(10), np.array([0.5, 0.5, 0.5], dtype=np.float32))

    # 7. max_degree
    assert g.max_degree(0) == 8  # 2 * M
    assert g.max_degree(1) == 4  # M

    # 8. add_edge, neighbors, in_neighbors
    g.add_edge(10, 20, 0)
    assert g.neighbors(10, 0) == [20]
    assert g.in_neighbors(20, 0) == [10]
    assert g.neighbors(20, 0) == []
    assert g.in_neighbors(10, 0) == []

    # 9. set_neighbors
    g.set_neighbors(10, 0, [])
    assert g.neighbors(10, 0) == []
    assert g.in_neighbors(20, 0) == []

    # 10. remove_edge
    g.add_edge(10, 20, 0)
    assert g.neighbors(10, 0) == [20]
    g.remove_edge(10, 20, 0)
    assert g.neighbors(10, 0) == []
    assert g.in_neighbors(20, 0) == []

    # 11. distance_computations and search on non-empty graph
    comps_before = g.distance_computations
    top_ids, top_dists = g.search(q, k=1, ef=5)
    assert len(top_ids) == 1
    assert g.distance_computations > comps_before

    # 12. snapshot
    snap = g.snapshot()
    assert snap["size"] == 2
    assert snap["dim"] == 3
    assert snap["M"] == 4
    assert snap["max_level"] == 0
    assert snap["entry_point"] == 10
    assert snap["distance_computations"] == g.distance_computations
    assert snap["nodes"] == [10, 20]
    assert "levels" in snap
    assert "degrees" in snap

    # 13. remove
    g.remove(10)
    assert not g.contains(10)
    assert g.size() == 1
    assert g.entry_point() == 20
    g.remove(20)
    assert g.size() == 0
    assert g.entry_point() is None


# =============================================================================
# 2. Construction: from_adjacency and Round-Trip Validation
# =============================================================================


def test_from_adjacency_round_trip() -> None:
    """from_adjacency reconstructs topology and validates reverse links."""
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([1.0, 1.0], dtype=np.float32),
    }
    levels = {0: 1, 1: 0, 2: 1, 3: 0}
    adjacency = {
        (0, 0): [1, 2],
        (0, 1): [2],
        (1, 0): [0, 3],
        (2, 0): [0],
        (2, 1): [0],
        (3, 0): [1],
    }

    g = ReferenceGraph.from_adjacency(vectors, adjacency, levels=levels, M=2)
    assert g.size() == 4
    assert g.dim == 2
    assert g.max_level() == 1
    assert g.entry_point() == 0  # node 0 and 2 at level 1; min is 0

    for (u, lvl), expected_nbrs in adjacency.items():
        assert g.neighbors(u, lvl) == expected_nbrs
        for v in expected_nbrs:
            assert u in g.in_neighbors(v, lvl)

    # Empty from_adjacency
    empty_g = ReferenceGraph.from_adjacency({}, {}, M=4)
    assert empty_g.size() == 0
    assert empty_g.max_level() == -1
    assert empty_g.entry_point() is None


# =============================================================================
# 3. Validation Errors: Degree Limits, Self-Loops, and Endpoints
# =============================================================================


def test_validation_errors_degree_self_loop_endpoints() -> None:
    """Verify ValueError is raised on degree overflow, self-loop, or invalid endpoints."""
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([2.0, 0.0], dtype=np.float32),
    }
    levels = {0: 0, 1: 0, 2: 1}
    g = ReferenceGraph.from_adjacency(vectors, {}, levels=levels, M=1)
    # level 0 allows 2*M = 2; level 1 allows M = 1

    # 1. Self-loop rejection
    with pytest.raises(ValueError, match="Self-loop"):
        g.add_edge(0, 0, 0)
    with pytest.raises(ValueError, match="Self-loop"):
        g.set_neighbors(0, 0, [0])

    # 2. Endpoint existence and layer membership
    with pytest.raises(ValueError, match="must exist"):
        g.add_edge(0, 999, 0)
    with pytest.raises(ValueError, match="must exist at layer"):
        # node 0 has max level 0, cannot be connected at level 1
        g.add_edge(0, 2, 1)
    with pytest.raises(ValueError, match="does not exist at layer"):
        g.set_neighbors(2, 1, [0])

    # 3. Degree limits
    g.add_edge(0, 1, 0)
    g.add_edge(0, 2, 0)
    with pytest.raises(ValueError, match="degree limit"):
        # node 0 at level 0 already has 2 neighbors; adding third exceeds limit of 2*1=2
        g.add(np.array([3]), np.array([[3.0, 0.0]], dtype=np.float32))
        g.add_edge(0, 3, 0)

    # 4. set_neighbors degree overflow
    with pytest.raises(ValueError, match="exceeds max_degree"):
        g.set_neighbors(2, 1, [0, 1])  # limit at level 1 is M=1

    # 5. from_adjacency validation errors
    with pytest.raises(ValueError, match="Self-loop"):
        ReferenceGraph.from_adjacency(vectors, {(0, 0): [0]}, levels=levels, M=2)
    with pytest.raises(ValueError, match="exceeds limit"):
        ReferenceGraph.from_adjacency(vectors, {(0, 0): [1, 2, 3]}, levels=levels, M=1)


# =============================================================================
# 4. Move Semantics
# =============================================================================


def test_move_bumps_version_and_preserves_edges() -> None:
    """move() increments version(node_id) by exactly 1 and changes no topology."""
    rng = np.random.default_rng(101)
    X = rng.standard_normal((10, 4))
    g = ReferenceGraph.from_points(X, M=4, seed=1)

    # Record initial topology and versions
    initial_versions = {nid: g.version(nid) for nid in g.ids()}
    initial_nbrs = {
        (u, lvl): g.neighbors(u, lvl)
        for lvl in range(g.max_level() + 1)
        for u in g.ids()
        if g.node_level(u) >= lvl
    }
    initial_in_nbrs = {
        (u, lvl): g.in_neighbors(u, lvl)
        for lvl in range(g.max_level() + 1)
        for u in g.ids()
        if g.node_level(u) >= lvl
    }

    target_node = 3
    new_vec = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    g.move(target_node, new_vec)

    # Assert version bump
    assert g.version(target_node) == initial_versions[target_node] + 1
    for nid in g.ids():
        if nid != target_node:
            assert g.version(nid) == initial_versions[nid]

    # Assert no edge changed
    for (u, lvl), nbrs in initial_nbrs.items():
        assert g.neighbors(u, lvl) == nbrs
    for (u, lvl), in_nbrs in initial_in_nbrs.items():
        assert g.in_neighbors(u, lvl) == in_nbrs


# =============================================================================
# 5. Remove Semantics
# =============================================================================


def test_remove_cleans_all_forward_and_reverse_edges() -> None:
    """remove(node_id) completely removes node from all adjacency and reverse tables."""
    rng = np.random.default_rng(202)
    X = rng.standard_normal((15, 3))
    g = ReferenceGraph.from_points(X, M=4, seed=42)

    target_node = 5
    assert g.contains(target_node)

    g.remove(target_node)

    assert not g.contains(target_node)
    assert target_node not in g.ids()

    for lvl in range(g.max_level() + 1):
        for u in g.ids():
            assert target_node not in g.neighbors(u, lvl)
            assert target_node not in g.in_neighbors(u, lvl)


# =============================================================================
# 6. from_points Rule Validation, Equilateral Ties, and Degree Bounds
# =============================================================================


def test_from_points_independent_rule_match_and_degree() -> None:
    """from_points matches an independent reference selection rule and respects degree."""
    rng = np.random.default_rng(303)
    n = 12
    dim = 3
    M = 3
    X = rng.standard_normal((n, dim))

    # Force all nodes to level 0 for direct comparison
    levels = {i: 0 for i in range(n)}
    g = ReferenceGraph.from_points(X, M=M, levels=levels, mimic_reverse_append=False)

    limit = g.max_degree(0)
    assert limit == 2 * M

    # Independent re-implementation of the heuristic
    for q in range(n):
        candidates = [c for c in range(n) if c != q]
        candidates.sort(key=lambda c: (float(np.sum((X[q] - X[c]) ** 2)), c))

        expected_accepted: list[int] = []
        for c in candidates:
            d_qc = float(np.sum((X[q] - X[c]) ** 2))
            reject = False
            for r in expected_accepted:
                d_rc = float(np.sum((X[r] - X[c]) ** 2))
                if d_rc < d_qc:
                    reject = True
                    break
            if not reject:
                expected_accepted.append(c)
                if len(expected_accepted) == limit:
                    break

        assert g.neighbors(q, 0) == expected_accepted


def test_from_points_explicit_tie_acceptance() -> None:
    """Ties d(r, e) == d(q, e) are accepted (mirroring hnswlib getNeighborsByHeuristic2)."""
    # Equilateral triangle: q at (0, 0), r at (1, 0), e at (0.5, sqrt(3)/2)
    # d(q, r) = 1, d(q, e) = 1, d(r, e) = 1
    # When evaluated from q:
    # r is nearest (or tie-break index 1 < 2); r is accepted first.
    # Next candidate e has d(q, e) = 1. d(r, e) = 1.
    # Strict inequality test d(r, e) < d(q, e) is 1 < 1 (False) -> NOT rejected -> accepted!
    sqrt3_2 = float(np.sqrt(3) / 2)
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, sqrt3_2],
        ],
        dtype=np.float32,
    )
    g = ReferenceGraph.from_points(X, M=2, levels=[0, 0, 0], mimic_reverse_append=False)

    # Node 0 should accept both 1 and 2
    nbrs_0 = g.neighbors(0, 0)
    assert 1 in nbrs_0
    assert 2 in nbrs_0


# =============================================================================
# 7. mimic_reverse_append: Certificate Violation Counts in d=2 and d=16
# =============================================================================


def _count_violating_edges(graph: ReferenceGraph) -> int:
    """Count directed edges u -> v violated by any same-level witness w in graph."""
    violated_count = 0
    for lvl in range(graph.max_level() + 1):
        for u in graph.ids():
            nbrs = graph.neighbors(u, lvl)
            if len(nbrs) < 2:
                continue
            vec_u = graph.vector(u)
            for v in nbrs:
                vec_v = graph.vector(v)
                d_uv = float(np.sum((vec_u - vec_v) ** 2))
                is_edge_violated = False
                for w in nbrs:
                    if w == v:
                        continue
                    vec_w = graph.vector(w)
                    d_uw = float(np.sum((vec_u - vec_w) ** 2))
                    d_vw = float(np.sum((vec_v - vec_w) ** 2))
                    if is_violated_static(d_uv, d_uw, d_vw):
                        is_edge_violated = True
                        break
                if is_edge_violated:
                    violated_count += 1
    return violated_count


def test_mimic_reverse_append_produces_strictly_more_violations() -> None:
    """mimic_reverse_append=True generates strictly more kinetic certificate violations."""
    results: dict[int, tuple[int, int]] = {}

    for d in [2, 16]:
        rng = np.random.default_rng(12345)
        X = rng.standard_normal((100, d))
        g_no_rev = ReferenceGraph.from_points(X, M=8, seed=42, mimic_reverse_append=False)
        g_rev = ReferenceGraph.from_points(X, M=8, seed=42, mimic_reverse_append=True)

        count_no_rev = _count_violating_edges(g_no_rev)
        count_rev = _count_violating_edges(g_rev)

        results[d] = (count_no_rev, count_rev)
        print(
            f"\n[Dimension d={d}] Without reverse append: {count_no_rev}, "
            f"With reverse append: {count_rev}"
        )

        assert count_rev > count_no_rev, f"Expected count_rev > count_no_rev for d={d}"


# =============================================================================
# 8. Forward / Reverse Consistency Invariant Under Random Mutation Sequences
# =============================================================================


def test_randomized_mutations_preserve_forward_reverse_invariants() -> None:
    """Symmetric forward/reverse consistency holds across random sequence of mutations."""
    rng = np.random.default_rng(777)
    dim = 4
    M = 4
    init_n = 20
    X = rng.standard_normal((init_n, dim))

    g = ReferenceGraph.from_points(X, M=M, seed=99, mimic_reverse_append=True)

    def check_invariants() -> None:
        top_lvl = g.max_level()
        if top_lvl < 0:
            assert g.size() == 0
            assert g.entry_point() is None
            return

        active_ids = set(g.ids())
        assert len(active_ids) == g.size()

        for lvl in range(top_lvl + 1):
            limit = g.max_degree(lvl)
            for u in active_ids:
                if g.node_level(u) < lvl:
                    assert g.neighbors(u, lvl) == []
                    assert g.in_neighbors(u, lvl) == []
                    continue

                nbrs = g.neighbors(u, lvl)
                # Degree limit
                assert len(nbrs) <= limit
                # No self-loops
                assert u not in nbrs

                # Forward-reverse symmetry
                for v in nbrs:
                    assert v in active_ids
                    assert g.node_level(v) >= lvl
                    assert u in g.in_neighbors(v, lvl)

                in_nbrs = g.in_neighbors(u, lvl)
                for in_u in in_nbrs:
                    assert in_u in active_ids
                    assert g.node_level(in_u) >= lvl
                    assert u in g.neighbors(in_u, lvl)

    check_invariants()

    next_node_id = 100
    # Execute 300 random operations
    for _ in range(300):
        op = rng.choice(["add", "move", "remove", "add_edge", "remove_edge", "set_neighbors"])
        active = g.ids()

        if op == "add" or len(active) < 3:
            nid = next_node_id
            next_node_id += 1
            vec = rng.standard_normal((1, dim)).astype(np.float32)
            g.add(np.array([nid], dtype=np.int64), vec)
        elif op == "move":
            nid = int(rng.choice(active))
            new_vec = rng.standard_normal(dim).astype(np.float32)
            g.move(nid, new_vec)
        elif op == "remove":
            nid = int(rng.choice(active))
            g.remove(nid)
        elif op == "add_edge":
            u = int(rng.choice(active))
            lvl = int(rng.integers(0, g.node_level(u) + 1))
            eligible = [v for v in active if v != u and g.node_level(v) >= lvl]
            if eligible and len(g.neighbors(u, lvl)) < g.max_degree(lvl):
                v = int(rng.choice(eligible))
                g.add_edge(u, v, lvl)
        elif op == "remove_edge":
            u = int(rng.choice(active))
            lvl = int(rng.integers(0, g.node_level(u) + 1))
            nbrs = g.neighbors(u, lvl)
            if nbrs:
                v = int(rng.choice(nbrs))
                g.remove_edge(u, v, lvl)
        elif op == "set_neighbors":
            u = int(rng.choice(active))
            lvl = int(rng.integers(0, g.node_level(u) + 1))
            eligible = [v for v in active if v != u and g.node_level(v) >= lvl]
            degree_limit = g.max_degree(lvl)
            k = int(rng.integers(0, min(len(eligible), degree_limit) + 1))
            selected = list(rng.choice(eligible, size=k, replace=False)) if eligible else []
            g.set_neighbors(u, lvl, [int(x) for x in selected])

        check_invariants()
