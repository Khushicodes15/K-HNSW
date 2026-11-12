"""Unit tests for graph audit and invariant validation (src/khnsw/core/graph/audit.py).

Covers:
  - Valid HNSWIndex produces zero invariant violations (empty, single-node, multi-node).
  - Hand-built broken graphs are detected via FakeGraphIndex:
    * Over-degree node
    * Missing reverse edge (asymmetric link)
    * Bad entry points (None on non-empty, wrong level, absent ID, non-empty when empty)
    * Unreachable node detection via reachable_from_entry and zero_in_degree_count
  - Reachability and connected component counts on small hand-built topologies.
  - Property-based testing with Hypothesis verifying that random HNSWIndex builds
    always satisfy all invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from khnsw.core.graph.audit import (
    check_degree_bounds,
    connected_components,
    degree_stats,
    reachable_from_entry,
    validate_invariants,
    zero_in_degree_count,
)
from khnsw.core.graph.hnsw import HNSWIndex


class FakeGraphIndex:
    """Mock implementation of the read-only GraphIndex protocol for test verification.

    Allows constructing deliberately invalid or disconnected graph states that
    GraphStore normally forbids.
    """

    def __init__(
        self,
        node_levels: dict[int, int],
        forward_edges: dict[int, dict[int, list[int]]],
        reverse_edges: dict[int, dict[int, list[int]]],
        entry_point: int | None,
        max_level: int,
        m: int = 4,
        m0: int = 8,
    ) -> None:
        self._node_levels = dict(node_levels)
        self._forward = {
            u: {lvl: list(nbrs) for lvl, nbrs in lvls.items()} for u, lvls in forward_edges.items()
        }
        self._reverse = {
            u: {lvl: list(nbrs) for lvl, nbrs in lvls.items()} for u, lvls in reverse_edges.items()
        }
        self._entry_point = entry_point
        self._max_level = max_level
        self.m = m
        self.m0 = m0

    def ids(self) -> list[int]:
        return sorted(self._node_levels.keys())

    def max_level(self) -> int:
        return self._max_level

    def entry_point(self) -> int | None:
        return self._entry_point

    def node_level(self, node_id: int) -> int:
        if node_id not in self._node_levels:
            raise KeyError(f"node_id {node_id} not found")
        return self._node_levels[node_id]

    def neighbors(self, node_id: int, level: int) -> list[int]:
        return self._forward.get(node_id, {}).get(level, [])

    def in_neighbors(self, node_id: int, level: int) -> list[int]:
        return self._reverse.get(node_id, {}).get(level, [])

    def max_degree(self, level: int) -> int:
        return self.m0 if level == 0 else self.m


# ---------------------------------------------------------------------------
# Test: Valid HNSWIndex invariant validation
# ---------------------------------------------------------------------------


class TestValidHNSWIndex:
    def test_empty_index_has_no_violations(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        violations = validate_invariants(idx)
        assert violations == []

    def test_single_node_index_has_no_violations(self) -> None:
        idx = HNSWIndex(dim=4, m=4)
        idx.add(np.array([42], dtype=np.int64), np.zeros((1, 4), dtype=np.float32))
        violations = validate_invariants(idx)
        assert violations == []

    def test_populated_hnsw_index_has_no_violations(self) -> None:
        rng = np.random.default_rng(100)
        n, dim, m = 60, 8, 4
        idx = HNSWIndex(dim=dim, m=m, ef_construction=32, seed=100)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        idx.add(np.arange(n, dtype=np.int64), vecs)

        violations = validate_invariants(idx)
        assert violations == []

        bounds_violations = check_degree_bounds(idx)
        assert bounds_violations == []


# ---------------------------------------------------------------------------
# Test: Hand-built broken graphs
# ---------------------------------------------------------------------------


class TestHandBuiltBrokenGraphs:
    def test_detects_over_degree_node(self) -> None:
        # Node 0 has 5 neighbors at level 0, but m0 bound is 4
        fake = FakeGraphIndex(
            node_levels={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            forward_edges={
                0: {0: [1, 2, 3, 4, 5]},
                1: {0: [0]},
                2: {0: [0]},
                3: {0: [0]},
                4: {0: [0]},
                5: {0: [0]},
            },
            reverse_edges={
                0: {0: [1, 2, 3, 4, 5]},
                1: {0: [0]},
                2: {0: [0]},
                3: {0: [0]},
                4: {0: [0]},
                5: {0: [0]},
            },
            entry_point=0,
            max_level=0,
            m=4,
            m0=4,
        )
        bound_violations = check_degree_bounds(fake, m0=4, m=4)
        assert len(bound_violations) == 1
        assert "node 0 at level 0 has out-degree 5 > max allowed 4" in bound_violations[0]

        all_violations = validate_invariants(fake, m0=4, m=4)
        assert any("Degree bound violation" in v for v in all_violations)

    def test_detects_missing_reverse_edge(self) -> None:
        # Edge 0 -> 1 exists in forward, but 0 is NOT in reverse of 1
        fake = FakeGraphIndex(
            node_levels={0: 0, 1: 0},
            forward_edges={0: {0: [1]}, 1: {0: []}},
            reverse_edges={0: {0: []}, 1: {0: []}},  # Missing 0 in in_neighbors(1)
            entry_point=0,
            max_level=0,
            m=4,
            m0=8,
        )
        violations = validate_invariants(fake)
        assert len(violations) > 0
        assert any("Bidirectional consistency violation" in v and "0->1" in v for v in violations)

    def test_detects_missing_forward_edge(self) -> None:
        # Reverse edge 0 -> 1 exists in reverse of 1, but 1 is NOT in neighbors(0)
        fake = FakeGraphIndex(
            node_levels={0: 0, 1: 0},
            forward_edges={0: {0: []}, 1: {0: []}},
            reverse_edges={0: {0: []}, 1: {0: [0]}},
            entry_point=0,
            max_level=0,
            m=4,
            m0=8,
        )
        violations = validate_invariants(fake)
        assert any("Bidirectional consistency violation" in v for v in violations)

    def test_detects_bad_entry_point_none_on_non_empty(self) -> None:
        fake = FakeGraphIndex(
            node_levels={0: 0},
            forward_edges={0: {0: []}},
            reverse_edges={0: {0: []}},
            entry_point=None,
            max_level=0,
        )
        violations = validate_invariants(fake)
        assert any("entry_point == None" in v for v in violations)

    def test_detects_bad_entry_point_wrong_level(self) -> None:
        fake = FakeGraphIndex(
            node_levels={0: 0, 1: 1},
            forward_edges={0: {0: []}, 1: {0: [], 1: []}},
            reverse_edges={0: {0: []}, 1: {0: [], 1: []}},
            entry_point=0,  # Level 0, but max_level is 1
            max_level=1,
        )
        violations = validate_invariants(fake)
        assert any("level 0 != max_level 1" in v for v in violations)

    def test_detects_bad_entry_point_nonexistent_id(self) -> None:
        fake = FakeGraphIndex(
            node_levels={0: 0},
            forward_edges={0: {0: []}},
            reverse_edges={0: {0: []}},
            entry_point=999,
            max_level=0,
        )
        violations = validate_invariants(fake)
        assert any("entry_point 999 not found" in v for v in violations)

    def test_detects_empty_index_with_bad_max_level_or_ep(self) -> None:
        fake = FakeGraphIndex(
            node_levels={},
            forward_edges={},
            reverse_edges={},
            entry_point=0,
            max_level=0,
        )
        violations = validate_invariants(fake)
        assert any("empty index must have max_level == -1" in v for v in violations)
        assert any("empty index must have entry_point == None" in v for v in violations)

    def test_detects_self_loop(self) -> None:
        fake = FakeGraphIndex(
            node_levels={0: 0},
            forward_edges={0: {0: [0]}},
            reverse_edges={0: {0: [0]}},
            entry_point=0,
            max_level=0,
        )
        violations = validate_invariants(fake)
        assert any("self-loop detected" in v for v in violations)

    def test_detects_unreachable_node(self) -> None:
        # Nodes 0 and 1 connected: 0 <-> 1.
        # Node 2 is isolated and unreachable from entry point 0.
        fake = FakeGraphIndex(
            node_levels={0: 0, 1: 0, 2: 0},
            forward_edges={
                0: {0: [1]},
                1: {0: [0]},
                2: {0: []},
            },
            reverse_edges={
                0: {0: [1]},
                1: {0: [0]},
                2: {0: []},
            },
            entry_point=0,
            max_level=0,
        )
        # Reachability from entry point 0 should not include node 2
        reachable = reachable_from_entry(fake, level=0)
        assert reachable == {0, 1}
        assert 2 not in reachable

        # zero_in_degree_count detects node 2 as having in-degree 0
        zero_in = zero_in_degree_count(fake, level=0)
        assert zero_in == 1

        # connected_components identifies 2 distinct components
        comps = connected_components(fake, level=0)
        assert len(comps) == 2
        assert {0, 1} in comps
        assert {2} in comps


# ---------------------------------------------------------------------------
# Test: Degree statistics and component metrics
# ---------------------------------------------------------------------------


class TestAuditMetrics:
    def test_degree_stats_calculation(self) -> None:
        fake = FakeGraphIndex(
            node_levels={0: 1, 1: 1, 2: 0},
            forward_edges={
                0: {0: [1, 2], 1: [1]},
                1: {0: [0], 1: [0]},
                2: {0: [0]},
            },
            reverse_edges={
                0: {0: [1, 2], 1: [1]},
                1: {0: [0], 1: [0]},
                2: {0: [0]},
            },
            entry_point=0,
            max_level=1,
        )
        stats = degree_stats(fake)
        assert 0 in stats and 1 in stats

        # Level 0 has 3 nodes: degrees are 2, 1, 1 -> min=1, max=2, mean=4/3
        assert stats[0]["count"] == 3
        assert stats[0]["min"] == 1
        assert stats[0]["max"] == 2
        assert stats[0]["mean"] == pytest.approx(4.0 / 3.0)

        # Level 1 has 2 nodes: degrees are 1, 1 -> min=1, max=1, mean=1.0
        assert stats[1]["count"] == 2
        assert stats[1]["min"] == 1
        assert stats[1]["max"] == 1
        assert stats[1]["mean"] == pytest.approx(1.0)

    def test_reachability_and_connected_components_multi_layer(self) -> None:
        # Multi-layer graph:
        # Layer 1: 0 <-> 2
        # Layer 0: 0 <-> 1 <-> 2 <-> 3
        fake = FakeGraphIndex(
            node_levels={0: 1, 1: 0, 2: 1, 3: 0},
            forward_edges={
                0: {0: [1], 1: [2]},
                1: {0: [0, 2]},
                2: {0: [1, 3], 1: [0]},
                3: {0: [2]},
            },
            reverse_edges={
                0: {0: [1], 1: [2]},
                1: {0: [0, 2]},
                2: {0: [1, 3], 1: [0]},
                3: {0: [2]},
            },
            entry_point=0,
            max_level=1,
        )
        # At layer 0, all nodes {0, 1, 2, 3} are reachable
        assert reachable_from_entry(fake, level=0) == {0, 1, 2, 3}
        assert len(connected_components(fake, level=0)) == 1

        # At layer 1, only nodes present at level 1 are {0, 2}
        assert reachable_from_entry(fake, level=1) == {0, 2}
        assert len(connected_components(fake, level=1)) == 1

    def test_explicit_ids_parameter_handling(self) -> None:
        class MinimalIndexWithoutIds:
            def max_level(self) -> int:
                return 0

            def entry_point(self) -> int:
                return 0

            def node_level(self, nid: int) -> int:
                return 0

            def neighbors(self, nid: int, lvl: int) -> list[int]:
                return []

            def in_neighbors(self, nid: int, lvl: int) -> list[int]:
                return []

            def max_degree(self, lvl: int) -> int:
                return 4

        min_idx = MinimalIndexWithoutIds()
        # Should raise ValueError if ids is None and ids() method is missing
        with pytest.raises(ValueError, match="Node IDs must be explicitly provided"):
            validate_invariants(min_idx)

        # Should succeed when explicit ids iterable is provided
        violations = validate_invariants(min_idx, ids=[0])
        assert violations == []


# ---------------------------------------------------------------------------
# Hypothesis: random HNSW builds maintain all invariants
# ---------------------------------------------------------------------------


class TestHypothesisAudit:
    @given(
        n=st.integers(min_value=1, max_value=60),
        dim=st.integers(min_value=2, max_value=12),
        m=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_hnsw_always_satisfies_invariants(
        self, n: int, dim: int, m: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        idx = HNSWIndex(dim=dim, m=m, ef_construction=max(m, 16), seed=seed)

        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        ids_arr = np.arange(n, dtype=np.int64)
        idx.add(ids_arr, vecs)

        violations = validate_invariants(idx)
        assert violations == [], f"Invariant violations found: {violations}"
