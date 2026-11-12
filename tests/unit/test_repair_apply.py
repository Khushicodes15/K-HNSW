"""Unit and property tests for applying repair plans (apply_plan)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from khnsw.core.detection.certificate_store import CertificateStore
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
    """T6: khnsw end-to-end property test.

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

    # Move a randomly selected node
    moved = int(rng.integers(0, n))
    new_vec = (g.vector(moved) + rng.standard_normal(d) * 0.5).astype(np.float32)
    g.move(moved, new_vec)

    # Detect violations after move using store on_move
    store.on_move(g, moved, meter)

    # Collect violated edges
    violated: dict[tuple[int, int], list[int]] = {}
    for u in g.ids():
        for lvl in range(g.node_level(u) + 1):
            for v in g.neighbors(u, lvl):
                if store.status((u, v, lvl)) == "violated":
                    violated.setdefault((u, lvl), []).append(v)

    if not violated:
        return

    # Plan K-HNSW repair
    plan = khnsw_plan(g, violated=violated, meter=meter)
    if not plan.rewrites:
        return

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
            assert store.status((u, v, lvl)) == "valid"

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
                if store.status((u, v, lvl)) == "violated":
                    violated_after.setdefault((u, lvl), []).append(v)

    re_plan = khnsw_plan(g, violated=violated_after, meter=meter)
    assert len(re_plan.rewrites) == 0


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
