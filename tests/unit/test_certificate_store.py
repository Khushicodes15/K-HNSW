"""Unit and property tests for CertificateStore, transitions, cost tracking, and determinism."""

from __future__ import annotations

import numpy as np

from khnsw.core.detection.affected_scan import (
    affected_after_move,
    affected_after_neighbor_change,
)
from khnsw.core.detection.certificate_store import (
    CertificateStore,
    EdgeStatus,
)
from khnsw.core.detection.cost import CostMeter
from tests.fixtures.reference_graph import ReferenceGraph

# =============================================================================
# T4: Incremental == From-Scratch Equality
# =============================================================================


def test_incremental_equals_from_scratch_across_mutations() -> None:
    """T4: Incremental on_move and on_neighbor_change match fresh build() exactly."""
    rng = np.random.default_rng(888)
    n = 20
    d = 4
    M = 4
    X = rng.standard_normal((n, d)).astype(np.float32)
    g = ReferenceGraph.from_points(X, M=M, seed=888, mimic_reverse_append=True)

    meter = CostMeter()
    store = CertificateStore()
    store.build(g, meter)

    # Helper comparing store against fresh ground-truth store
    def assert_store_matches_scratch() -> None:
        fresh_meter = CostMeter()
        scratch_store = CertificateStore()
        scratch_store.build(g, fresh_meter)

        assert store.counts() == scratch_store.counts()
        for e in store._edge_status:
            assert store.status(e) == scratch_store.status(e)
            assert store.witness(e) == scratch_store.witness(e)
        for e in scratch_store._edge_status:
            assert e in store._edge_status

    assert_store_matches_scratch()

    # Sequence of random moves and rewiring operations
    for _ in range(50):
        op = rng.choice(["move", "set_neighbors", "add_edge", "remove_edge"])
        active = g.ids()

        if op == "move":
            u = int(rng.choice(active))
            new_vec = rng.standard_normal(d).astype(np.float32)
            g.move(u, new_vec)
            store.on_move(g, u, meter)
        elif op == "set_neighbors":
            u = int(rng.choice(active))
            lvl = int(rng.integers(0, g.node_level(u) + 1))
            eligible = [v for v in active if v != u and g.node_level(v) >= lvl]
            limit = g.max_degree(lvl)
            k = int(rng.integers(0, min(len(eligible), limit) + 1))
            chosen = list(rng.choice(eligible, size=k, replace=False)) if eligible else []
            g.set_neighbors(u, lvl, [int(x) for x in chosen])
            store.on_neighbor_change(g, u, lvl, meter)
        elif op == "add_edge":
            u = int(rng.choice(active))
            lvl = int(rng.integers(0, g.node_level(u) + 1))
            curr = g.neighbors(u, lvl)
            eligible = [v for v in active if v != u and g.node_level(v) >= lvl and v not in curr]
            if eligible and len(curr) < g.max_degree(lvl):
                v = int(rng.choice(eligible))
                g.add_edge(u, v, lvl)
                store.on_neighbor_change(g, u, lvl, meter)
        elif op == "remove_edge":
            u = int(rng.choice(active))
            lvl = int(rng.integers(0, g.node_level(u) + 1))
            curr = g.neighbors(u, lvl)
            if curr:
                v = int(rng.choice(curr))
                g.remove_edge(u, v, lvl)
                store.on_neighbor_change(g, u, lvl, meter)

        assert_store_matches_scratch()


# =============================================================================
# T5: Transition Semantics (Legacy, Failure, Healed, Forward-Only)
# =============================================================================


def test_transition_semantics_and_hand_built_scenarios() -> None:
    """T5: Legacy produces no failures; moves produce failure and healed transitions."""
    # 1. Forward-only graph builds with 0 violated edges
    rng = np.random.default_rng(123)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    g_forward = ReferenceGraph.from_points(X, M=4, seed=123, mimic_reverse_append=False)
    meter = CostMeter()
    store_fwd = CertificateStore()
    store_fwd.build(g_forward, meter)
    assert store_fwd.counts()["violated"] == 0
    assert store_fwd.baseline_violation_rate() == 0.0

    # 2. Graph with mimic_reverse_append has legacy violations but produces no failures at build
    g_rev = ReferenceGraph.from_points(X, M=4, seed=123, mimic_reverse_append=True)
    store_rev = CertificateStore()
    build_events = store_rev.build(g_rev, meter)
    assert build_events == []
    assert store_rev.counts()["violated"] > 0
    assert store_rev.baseline_violation_rate() > 0.0

    # 3. Hand-built scenario: known VALID -> VIOLATED move produces failure with witness
    # u at (0, 0), v at (2, 0), w initially far at (1, 10)
    # Edge (u, v, 0): D(u, v) = 4, D(u, w) = 101, D(v, w) = 101 -> VALID
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([2.0, 0.0], dtype=np.float32),
        2: np.array([1.0, 10.0], dtype=np.float32),
    }
    levels = {0: 0, 1: 0, 2: 0}
    adjacency = {
        (0, 0): [1, 2],
        (1, 0): [0],
        (2, 0): [0],
    }
    g_triangle = ReferenceGraph.from_adjacency(vectors, adjacency, levels=levels, M=2)
    store_tri = CertificateStore()
    store_tri.build(g_triangle, meter)
    assert store_tri.status((0, 1, 0)) == EdgeStatus.VALID.value
    assert store_tri.witness((0, 1, 0)) is None

    # Move w (node 2) in between u and v to (1.0, 0.1)
    # D(u, v) = 4, D(u, w) = 1.01, D(v, w) = 1.01 -> both <= 4 and < 4 -> VIOLATED!
    g_triangle.move(2, np.array([1.0, 0.1], dtype=np.float32))
    transitions = store_tri.on_move(g_triangle, 2, meter)

    # Exactly one failure transition for edge (0, 1, 0) with witness 2
    failures = [t for t in transitions if t.kind == "failure"]
    assert len(failures) == 1
    assert failures[0].edge == (0, 1, 0)
    assert failures[0].old == EdgeStatus.VALID.value
    assert failures[0].new == EdgeStatus.VIOLATED.value
    assert failures[0].witness == 2

    # 4. Hand-built scenario: VIOLATED -> VALID yields healed transition
    # Move w (node 2) back out to (1.0, 10.0)
    g_triangle.move(2, np.array([1.0, 10.0], dtype=np.float32))
    transitions_healed = store_tri.on_move(g_triangle, 2, meter)

    healed = [t for t in transitions_healed if t.kind == "healed"]
    assert len(healed) == 1
    assert healed[0].edge == (0, 1, 0)
    assert healed[0].old == EdgeStatus.VIOLATED.value
    assert healed[0].new == EdgeStatus.VALID.value
    assert healed[0].witness is None


# =============================================================================
# T6: Cost Accounting and Exact Formulas
# =============================================================================


def test_cost_accounting_exact_formulas() -> None:
    """T6: Affected scan does 0 distances; certify_source matches k*(k+1)/2 exactly."""
    meter = CostMeter()
    rng = np.random.default_rng(999)
    X = rng.standard_normal((25, 4)).astype(np.float32)
    g = ReferenceGraph.from_points(X, M=4, seed=999, mimic_reverse_append=True)

    # 1. Affected scan uses ZERO distance computations
    meter.reset()
    affected_after_move(g, 0)
    affected_after_neighbor_change(g, 0, 0)
    assert meter.count == 0

    # 2. certify_source cost formula D(k) = k * (k + 1) / 2
    store = CertificateStore()
    for u in g.ids()[:5]:
        lvl = 0
        k = len(g.neighbors(u, lvl))
        expected_cost = k * (k + 1) // 2

        meter.reset()
        store.certify_source(g, u, lvl, meter)
        assert meter.count == expected_cost, (
            f"Cost formula mismatch for degree k={k}: got {meter.count}, expected {expected_cost}"
        )

    # 3. build() cost equals sum of D(k) over all sources
    meter.reset()
    expected_build_cost = sum(
        len(g.neighbors(u, lvl)) * (len(g.neighbors(u, lvl)) + 1) // 2
        for lvl in range(g.max_level() + 1)
        for u in g.ids()
        if g.node_level(u) >= lvl
    )
    store.build(g, meter)
    assert meter.count == expected_build_cost

    # 4. on_move() cost equals sum of D(k) over the affected sources
    meter.reset()
    moved_node = 3
    aff_edges = affected_after_move(g, moved_node)
    sources = sorted({(u, lvl) for (u, _, lvl) in aff_edges})
    expected_move_cost = sum(
        len(g.neighbors(u, lvl)) * (len(g.neighbors(u, lvl)) + 1) // 2 for (u, lvl) in sources
    )
    g.move(moved_node, g.vector(moved_node) + np.float32(0.5))
    store.on_move(g, moved_node, meter)
    assert meter.count == expected_move_cost


# =============================================================================
# T7: Determinism
# =============================================================================


def test_determinism_identical_transitions_and_counts() -> None:
    """T7: Deterministic execution gives identical transition orders and meter counts."""
    for seed in [1, 42, 999]:
        rng1 = np.random.default_rng(seed)
        X1 = rng1.standard_normal((15, 3)).astype(np.float32)
        g1 = ReferenceGraph.from_points(X1, M=3, seed=seed, mimic_reverse_append=True)
        m1 = CostMeter()
        s1 = CertificateStore()
        s1.build(g1, m1)

        rng2 = np.random.default_rng(seed)
        X2 = rng2.standard_normal((15, 3)).astype(np.float32)
        g2 = ReferenceGraph.from_points(X2, M=3, seed=seed, mimic_reverse_append=True)
        m2 = CostMeter()
        s2 = CertificateStore()
        s2.build(g2, m2)

        assert m1.count == m2.count
        assert s1.counts() == s2.counts()

        # Move same node by same displacement
        g1.move(2, g1.vector(2) + np.float32(1.5))
        g2.move(2, g2.vector(2) + np.float32(1.5))

        t1 = s1.on_move(g1, 2, m1)
        t2 = s2.on_move(g2, 2, m2)

        assert t1 == t2
        assert m1.count == m2.count
