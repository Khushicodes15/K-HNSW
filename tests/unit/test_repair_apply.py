"""Unit and property tests for applying repair plans (apply_plan)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.plan import (
    RepairPlan,
    Rewrite,
    khnsw_plan,
)
from tests.fixtures.reference_graph import ReferenceGraph

# ---------------------------------------------------------------------------
# T6: Hypothesis Property Test (khnsw end-to-end)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=12, max_value=25),
    d=st.sampled_from([2, 8]),
    M=st.integers(min_value=3, max_value=6),
    mimic=st.booleans(),
    seed=st.integers(min_value=0, max_value=100000),
)
def test_khnsw_apply_end_to_end_property(
    n: int,
    d: int,
    M: int,
    mimic: bool,
    seed: int,
) -> None:
    """T6: khnsw end-to-end property test with strict non-vacuity guards.

    Verifies:
    (a) Every rewritten (u, level) has zero VIOLATED edges in the store.
    (b) Synchronized store exactly equals a fresh from-scratch build.
    (c) Degree bounds hold and the reverse index (in_neighbors) is consistent.
    (d) Planning again on the repaired graph yields an empty plan (idempotency).
    (e) apply_plan does not increment index distance computations.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    g = ReferenceGraph.from_points(X, M=M, seed=seed, mimic_reverse_append=mimic)

    # Initial build of certificate store
    meter = CostMeter()
    store = CertificateStore()
    store.build(g, meter)

    # Pick a connected node and displace it towards/past a neighbor to guarantee violations
    moved = max(g.ids(), key=lambda i: len(g.neighbors(i, 0)))
    nbr = g.neighbors(moved, 0)[0]
    new_vec = (g.vector(nbr) + (g.vector(nbr) - g.vector(moved)) * 0.5).astype(np.float32)
    g.move(moved, new_vec)

    # Detect violations after move using store on_move
    store.on_move(g, moved, meter)

    # Collect violated edges using real EdgeStatus enum
    violated: dict[tuple[int, int], list[int]] = {}
    for u in g.ids():
        for lvl in range(g.node_level(u) + 1):
            for v in g.neighbors(u, lvl):
                if store.status((u, v, lvl)) == EdgeStatus.VIOLATED:
                    violated.setdefault((u, lvl), []).append(v)

    # Non-vacuity guard: must have at least one violated edge before repair
    assert len(violated) > 0, "Non-vacuity guard: expected at least one violated edge before repair"

    # Plan K-HNSW repair
    plan = khnsw_plan(g, violated=violated, meter=meter)
    assert len(plan.rewrites) > 0, "Non-vacuity guard: expected at least one planned rewrite"

    # Track graph distance computations prior to apply
    initial_graph_dist = g.distance_computations

    # Apply repair plan
    apply_plan(g, plan, store=store, meter=meter)

    # (e) apply_plan should not perform graph distance evaluations
    assert g.distance_computations == initial_graph_dist

    # (a) Every rewritten (u, level) has zero VIOLATED edges in the store
    for rewrite in plan.rewrites:
        u = rewrite.node
        lvl = rewrite.level
        for v in g.neighbors(u, lvl):
            assert store.status((u, v, lvl)) == EdgeStatus.VALID

    # (b) Store equals a fresh from-scratch build
    fresh_meter = CostMeter()
    fresh_store = CertificateStore()
    fresh_store.build(g, fresh_meter)

    for u in g.ids():
        for lvl in range(g.node_level(u) + 1):
            for v in g.neighbors(u, lvl):
                edge = (u, v, lvl)
                assert store.status(edge) == fresh_store.status(edge)
                assert store.witness(edge) == fresh_store.witness(edge)

    # (c) Degree bounds hold and reverse index is consistent
    for u in g.ids():
        for lvl in range(g.node_level(u) + 1):
            nbrs = g.neighbors(u, lvl)
            cap = g.max_degree(lvl)

            assert len(nbrs) <= cap
            assert u not in nbrs
            assert len(nbrs) == len(set(nbrs))

            for v in nbrs:
                assert u in g.in_neighbors(v, lvl)

    # (d) Planning again on the repaired graph yields an empty plan (idempotency)
    violated_after: dict[tuple[int, int], list[int]] = {}
    for u in g.ids():
        for lvl in range(g.node_level(u) + 1):
            for v in g.neighbors(u, lvl):
                if store.status((u, v, lvl)) == EdgeStatus.VIOLATED:
                    violated_after.setdefault((u, lvl), []).append(v)

    # Non-vacuity guard: post-repair violations must be strictly smaller than initial
    assert len(violated_after) < len(violated)

    re_plan = khnsw_plan(g, violated=violated_after, meter=meter)
    assert len(re_plan.rewrites) == 0


# ---------------------------------------------------------------------------
# Targeted checks killing M10 and M11
# ---------------------------------------------------------------------------


def test_apply_plan_synchronizes_store_per_rewrite_spy() -> None:
    """Spy test verifying store.on_neighbor_change is called once per rewrite (kills M11)."""
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([2.0, 2.0], dtype=np.float32),
    }
    levels = {0: 0, 1: 0, 2: 0, 3: 0}
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0],
        (2, 0): [0],
        (3, 0): [],
    }
    g = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=3)

    plan = RepairPlan(
        rule="test_rule",
        moved=None,
        rewrites=(
            Rewrite(node=0, level=0, old=(1, 2), new=(1, 3)),
            Rewrite(node=2, level=0, old=(0,), new=(1,)),
        ),
        distance_computations=0,
    )

    meter = CostMeter()
    store = CertificateStore()
    store.build(g, meter)

    # Track on_neighbor_change invocations
    calls: list[tuple[int, int]] = []
    real_on_change = store.on_neighbor_change

    def spy_on_change(idx: Any, u: int, lvl: int, m: CostMeter) -> Any:
        calls.append((u, lvl))
        return real_on_change(idx, u, lvl, m)

    store.on_neighbor_change = spy_on_change  # type: ignore[method-assign]

    apply_plan(g, plan, store=store, meter=meter)

    # store.on_neighbor_change must be called exactly once per rewrite with (u, lvl)
    expected_calls = [(r.node, r.level) for r in plan.rewrites]
    assert len(calls) == 2
    assert calls == expected_calls


def test_khnsw_plan_prunes_violations_even_when_below_cap() -> None:
    """K-HNSW must actively prune violating edges even when len(ranked) < cap (kills M10).

    Setup:
    Node 0 at (0.0, 0.0), node 1 at (2.0, 0.0), node 2 at (1.0, 0.0).
    Edge (0, 1) is violated because node 2 is directly between 0 and 1.
    Cap at level 0 for M=5 is 10.
    Ranked candidates: [(1.0, 2), (4.0, 1)]. len(ranked) = 2 < 10.

    - Under skip_if_below_cap=False (K-HNSW):
      Candidate 1 is evaluated against 2: D(2, 1) = 1.0 < 4.0 -> candidate 1 is PRUNED.
      New neighbors for node 0 is (2,).
      Rewrite: old=(1, 2), new=(2,). plan.rewrites has 1 rewrite.
    - Under skip_if_below_cap=True (M10):
      Returns candidates unchanged without pruning -> new=(2, 1) -> set(new) == set(old).
      plan.rewrites is EMPTY ()!
    """
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([2.0, 0.0], dtype=np.float32),
        2: np.array([1.0, 0.0], dtype=np.float32),
    }
    levels = {0: 0, 1: 0, 2: 0}
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0],
        (2, 0): [0],
    }
    g = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=5)

    violated = {(0, 0): [1]}
    plan = khnsw_plan(g, violated=violated)

    # Must contain exactly 1 rewrite pruning violating node 1
    assert len(plan.rewrites) == 1
    r = plan.rewrites[0]
    assert r.node == 0
    assert r.level == 0
    assert r.new == (2,)
    assert 1 not in r.new


# ---------------------------------------------------------------------------
# T7: Applied list match, event emission, and error handling
# ---------------------------------------------------------------------------


def test_apply_plan_matches_new_and_emits_events() -> None:
    """T7.1: Applied neighbor lists match plan.new and emit proper trace events."""
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([2.0, 2.0], dtype=np.float32),
    }
    levels = {0: 0, 1: 0, 2: 0, 3: 0}
    adj = {
        (0, 0): [1, 2],
        (1, 0): [0],
        (2, 0): [0],
        (3, 0): [],
    }
    g = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=3)

    plan = RepairPlan(
        rule="test_rule",
        moved=None,
        rewrites=(Rewrite(node=0, level=0, old=(1, 2), new=(1, 3)),),
        distance_computations=0,
    )

    sink = ListSink()
    emitter = EventEmitter(run_id="test-run", sink=sink)
    meter = CostMeter()
    store = CertificateStore()
    store.build(g, meter)

    apply_plan(
        g,
        plan,
        emitter=emitter,
        store=store,
        reason="heuristic_prune",
        meter=meter,
    )

    # Repaired node's neighbor list matches plan.new exactly
    assert list(g.neighbors(0, 0)) == [1, 3]

    events = sink.events
    remove_events = [e for e in events if e.type == EventType.edge_remove]
    add_events = [e for e in events if e.type == EventType.edge_add]

    assert len(remove_events) == 1
    assert remove_events[0].node == 0
    assert remove_events[0].level == 0
    assert remove_events[0].version == g.version(0)
    assert remove_events[0].payload == {"target_id": 2, "reason": "heuristic_prune"}

    assert len(add_events) == 1
    assert add_events[0].node == 0
    assert add_events[0].level == 0
    assert add_events[0].version == g.version(0)
    assert add_events[0].payload == {"target_id": 3}


def test_apply_plan_validation_errors() -> None:
    """T7.2: apply_plan raises ValueError on integrity or bounds violations."""
    vectors = {
        0: np.array([0.0, 0.0], dtype=np.float32),
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([1.0, 1.0], dtype=np.float32),
        4: np.array([2.0, 0.0], dtype=np.float32),
        5: np.array([0.0, 2.0], dtype=np.float32),
    }
    levels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    adj = {
        (0, 0): [1],
        (1, 0): [0],
        (2, 0): [],
        (3, 0): [],
        (4, 0): [],
        (5, 0): [],
    }
    # M=2 -> max_degree(0) = 2*M = 4
    g = ReferenceGraph.from_adjacency(vectors, adj, levels=levels, M=2)

    # 1. Invalid source node ID
    p_bad_source = RepairPlan(
        rule="test",
        moved=None,
        rewrites=(Rewrite(node=99, level=0, old=(), new=(1,)),),
        distance_computations=0,
    )
    with pytest.raises(ValueError, match="Source node 99 does not exist"):
        apply_plan(g, p_bad_source)

    # 2. Level out of valid bounds
    p_bad_lvl = RepairPlan(
        rule="test",
        moved=None,
        rewrites=(Rewrite(node=0, level=5, old=(1,), new=(1,)),),
        distance_computations=0,
    )
    with pytest.raises(ValueError, match="Level 5 out of valid bounds"):
        apply_plan(g, p_bad_lvl)

    # 3. Self-loop
    p_self_loop = RepairPlan(
        rule="test",
        moved=None,
        rewrites=(Rewrite(node=0, level=0, old=(1,), new=(0, 1)),),
        distance_computations=0,
    )
    with pytest.raises(ValueError, match="Self-loop detected"):
        apply_plan(g, p_self_loop)

    # 4. Duplicate targets
    p_duplicate = RepairPlan(
        rule="test",
        moved=None,
        rewrites=(Rewrite(node=0, level=0, old=(1,), new=(1, 1)),),
        distance_computations=0,
    )
    with pytest.raises(ValueError, match="Duplicate neighbor IDs"):
        apply_plan(g, p_duplicate)

    # 5. Invalid target node ID
    p_bad_target = RepairPlan(
        rule="test",
        moved=None,
        rewrites=(Rewrite(node=0, level=0, old=(1,), new=(99,)),),
        distance_computations=0,
    )
    with pytest.raises(ValueError, match="Target neighbor 99 does not exist"):
        apply_plan(g, p_bad_target)

    # 6. Exceeding max degree (5 neighbors exceeds max_degree(0) = 4)
    p_overflow = RepairPlan(
        rule="test",
        moved=None,
        rewrites=(Rewrite(node=0, level=0, old=(1,), new=(1, 2, 3, 4, 5)),),
        distance_computations=0,
    )
    with pytest.raises(ValueError, match="exceeds max_degree"):
        apply_plan(g, p_overflow)
