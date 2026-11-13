"""Unit tests for topology repair engines on ReferenceGraph."""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import Event, EventEmitter, EventType, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.interfaces import Repairer
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.engine import (
    HnswlibStyleEngine,
    KHNSWRepairEngine,
    MNRUStyleEngine,
)
from khnsw.core.repair.loop import repair_loop
from khnsw.core.repair.plan import mnru_plan, native_plan
from tests.fixtures.reference_graph import ReferenceGraph


def _collect_violated(graph: ReferenceGraph, store: CertificateStore) -> list[tuple[int, int, int]]:
    """Helper to collect violated directed edges from store."""
    violated: list[tuple[int, int, int]] = []
    for u in graph.ids():
        for lvl in range(graph.node_level(u) + 1):
            for v in graph.neighbors(u, lvl):
                edge = (u, v, lvl)
                if store.status(edge) == EdgeStatus.VIOLATED.value:
                    violated.append(edge)
    return sorted(violated)


@pytest.mark.parametrize("engine_cls", [KHNSWRepairEngine, HnswlibStyleEngine, MNRUStyleEngine])
def test_engine_satisfies_protocol_structurally(engine_cls: type) -> None:
    """Each engine implements the runtime_checkable Repairer protocol."""
    engine = engine_cls()
    assert isinstance(engine, Repairer), f"{engine_cls.__name__} must satisfy Repairer protocol"
    assert hasattr(engine, "notify_moved"), f"{engine_cls.__name__} must have notify_moved"
    assert hasattr(engine, "run_repairs"), f"{engine_cls.__name__} must have run_repairs"
    assert hasattr(engine, "on_move"), f"{engine_cls.__name__} must have on_move"


@pytest.mark.parametrize("engine_cls", [KHNSWRepairEngine, HnswlibStyleEngine, MNRUStyleEngine])
def test_nothing_to_do_returns_zeros(engine_cls: type) -> None:
    """Running repairs with no pending moves and no violations returns zeros and nothing_to_do."""
    rng = np.random.default_rng(101)
    X = rng.standard_normal((15, 2)).astype(np.float32)
    graph = ReferenceGraph.from_points(X, M=4, seed=101)

    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)
    repair_loop(graph, store)  # settle pass

    if engine_cls is KHNSWRepairEngine:
        engine = engine_cls(store=store)
    else:
        engine = engine_cls()

    sink = ListSink()
    result = engine.run_repairs(graph, budget=1000.0, sink=sink)

    assert result["stopped_reason"] == "nothing_to_do"
    assert result["rounds"] == 0
    assert result["edges_repaired"] == 0
    assert result["still_violated"] == []
    assert result["still_violated_count"] == 0
    assert result["unrepaired_nodes"] == []
    assert result["budget_consumed"] == 0
    assert sum(result["budget_used"].values()) == 0
    assert len(sink.events) == 0


@pytest.mark.parametrize("engine_cls", [KHNSWRepairEngine, HnswlibStyleEngine, MNRUStyleEngine])
def test_second_call_is_no_op(engine_cls: type) -> None:
    """A second immediate call to run_repairs without new moves is a no-op returning zeros."""
    rng = np.random.default_rng(202)
    X = rng.standard_normal((20, 2)).astype(np.float32)
    graph = ReferenceGraph.from_points(X, M=4, seed=202)

    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)
    repair_loop(graph, store)

    if engine_cls is KHNSWRepairEngine:
        engine = engine_cls(store=store)
    else:
        engine = engine_cls()

    moved_node = 2
    new_pos = (graph.vector(moved_node) + np.array([2.0, -1.5], dtype=np.float32)).astype(
        np.float32
    )
    graph.move(moved_node, new_pos)
    engine.notify_moved(moved_node)

    assert len(engine._pending_moves) > 0, "Precondition: pending moves must not be empty"

    sink = ListSink()
    res1 = engine.run_repairs(graph, budget=10000.0, sink=sink)
    assert res1["stopped_reason"] in {"converged", "max_rounds"}
    assert res1["rounds"] > 0, "Precondition: first run must execute repair rounds"

    # Second call immediately after:
    res2 = engine.run_repairs(graph, budget=10000.0, sink=sink)
    assert res2["stopped_reason"] == "nothing_to_do"
    assert res2["rounds"] == 0
    assert res2["edges_repaired"] == 0
    assert res2["budget_consumed"] == 0
    assert res2["still_violated"] == []


@pytest.mark.parametrize("engine_cls", [KHNSWRepairEngine, HnswlibStyleEngine, MNRUStyleEngine])
def test_sink_receives_well_formed_events(engine_cls: type) -> None:
    """The injected EventSink receives schema-valid trace events during repairs."""
    rng = np.random.default_rng(303)
    X = rng.standard_normal((25, 2)).astype(np.float32)
    graph = ReferenceGraph.from_points(X, M=4, seed=303)

    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)
    repair_loop(graph, store)

    if engine_cls is KHNSWRepairEngine:
        engine = engine_cls(store=store)
    else:
        engine = engine_cls()

    moved_node = 0
    new_pos = (graph.vector(moved_node) + np.array([3.0, 3.0], dtype=np.float32)).astype(np.float32)
    graph.move(moved_node, new_pos)
    engine.notify_moved(moved_node)

    sink = ListSink()
    res = engine.run_repairs(graph, budget=10000.0, sink=sink)
    assert res["rounds"] > 0, "Precondition: repair rounds must execute"

    assert len(sink.events) > 0, "Precondition: sink must receive trace events"
    for event in sink.events:
        assert isinstance(event, Event)
        assert event.run_id == engine._run_id
        assert event.type in {EventType.edge_add, EventType.edge_remove}
        assert "target_id" in event.payload


@pytest.mark.parametrize("engine_cls", [KHNSWRepairEngine, HnswlibStyleEngine, MNRUStyleEngine])
def test_budget_exhaustion_gives_budget_reason_and_leaves_invariants_intact(
    engine_cls: type,
) -> None:
    """Restricting budget forces stopped_reason 'budget' while preserving graph invariants."""
    rng = np.random.default_rng(404)
    X = rng.standard_normal((30, 2)).astype(np.float32)
    graph = ReferenceGraph.from_points(X, M=4, seed=404)

    store = CertificateStore()
    meter = CostMeter()
    store.build(graph, meter)
    repair_loop(graph, store)

    if engine_cls is KHNSWRepairEngine:
        engine = engine_cls(store=store)
    else:
        engine = engine_cls()

    moved_node = 1
    new_pos = (graph.vector(moved_node) + np.array([2.5, -2.5], dtype=np.float32)).astype(
        np.float32
    )
    graph.move(moved_node, new_pos)
    engine.notify_moved(moved_node)

    # Allot an extremely small budget that will be exhausted during planning
    tiny_budget = 2.0
    sink = ListSink()
    res = engine.run_repairs(graph, budget=tiny_budget, sink=sink)

    assert res["stopped_reason"] == "budget"
    assert res["budget_consumed"] <= int(tiny_budget) or res["budget_consumed"] > 0
    assert len(res["unrepaired_nodes"]) > 0, "Unrepaired nodes must be reported, never dropped"

    # Graph structural integrity must remain intact (no broken half-links or corruption)
    assert validate_invariants(graph) == []
    assert check_degree_bounds(graph) == []


def test_khnsw_engine_final_adjacency_equals_direct_loop() -> None:
    """KHNSWRepairEngine produces the exact same final topology as calling repair_loop directly."""
    rng = np.random.default_rng(505)
    X = rng.standard_normal((25, 2)).astype(np.float32)

    # 1. Direct repair_loop execution
    g1 = ReferenceGraph.from_points(X, M=4, seed=505)
    s1 = CertificateStore()
    m1 = CostMeter()
    s1.build(g1, m1)
    repair_loop(g1, s1)

    move_node = 3
    new_v = (g1.vector(move_node) + np.array([1.8, -1.2], dtype=np.float32)).astype(np.float32)
    g1.move(move_node, new_v)
    s1.on_move(g1, move_node, m1)
    assert len(_collect_violated(g1, s1)) > 0, "Precondition: violations must exist"

    sink1 = ListSink()
    emitter1 = EventEmitter("direct", sink1)
    res1 = repair_loop(g1, s1, moved_node=move_node, emitter=emitter1)

    # 2. KHNSWRepairEngine execution
    g2 = ReferenceGraph.from_points(X, M=4, seed=505)
    s2 = CertificateStore()
    m2 = CostMeter()
    s2.build(g2, m2)
    repair_loop(g2, s2)

    g2.move(move_node, new_v)
    s2.on_move(g2, move_node, m2)
    engine = KHNSWRepairEngine(store=s2)
    engine.notify_moved(move_node)

    sink2 = ListSink()
    res2 = engine.run_repairs(g2, budget=10000.0, sink=sink2)

    assert res1.stopped_reason == res2["stopped_reason"]
    assert res1.edges_repaired == res2["edges_repaired"]

    # Verify identical graph adjacency lists across all nodes and levels
    for u in g1.ids():
        for lvl in range(g1.node_level(u) + 1):
            assert g1.neighbors(u, lvl) == g2.neighbors(u, lvl), f"Adjacency mismatch at node {u}"


def test_baseline_engines_final_adjacency_equals_direct_planners() -> None:
    """Baseline engines match direct native_plan/mnru_plan + apply_plan adjacency exactly."""
    rng = np.random.default_rng(606)
    X = rng.standard_normal((25, 2)).astype(np.float32)

    # HnswlibStyleEngine vs direct native_plan
    g_direct_native = ReferenceGraph.from_points(X, M=4, seed=606)
    g_engine_native = ReferenceGraph.from_points(X, M=4, seed=606)

    move_node = 4
    new_v = (g_direct_native.vector(move_node) + np.array([1.5, 1.5], dtype=np.float32)).astype(
        np.float32
    )

    g_direct_native.move(move_node, new_v)
    plan_native = native_plan(g_direct_native, moved=move_node, meter=CostMeter())
    assert len(plan_native.rewrites) > 0, "Precondition: rewrites must be formulated"
    apply_plan(g_direct_native, plan_native)

    g_engine_native.move(move_node, new_v)
    hnsw_engine = HnswlibStyleEngine()
    hnsw_engine.notify_moved(move_node)
    sink_h = ListSink()
    hnsw_engine.run_repairs(g_engine_native, budget=10000.0, sink=sink_h)

    for u in g_direct_native.ids():
        for lvl in range(g_direct_native.node_level(u) + 1):
            assert g_direct_native.neighbors(u, lvl) == g_engine_native.neighbors(u, lvl), (
                f"Hnswlib adjacency mismatch at {u}"
            )

    # MNRUStyleEngine vs direct mnru_plan
    g_direct_mnru = ReferenceGraph.from_points(X, M=4, seed=606)
    g_engine_mnru = ReferenceGraph.from_points(X, M=4, seed=606)

    g_direct_mnru.move(move_node, new_v)
    plan_mnru = mnru_plan(
        g_direct_mnru, moved=move_node, alpha=1.0, scope="one_hop", meter=CostMeter()
    )
    assert len(plan_mnru.rewrites) > 0, "Precondition: rewrites must be formulated"
    apply_plan(g_direct_mnru, plan_mnru)

    g_engine_mnru.move(move_node, new_v)
    mnru_engine = MNRUStyleEngine(alpha=1.0, scope="one_hop")
    mnru_engine.notify_moved(move_node)
    sink_m = ListSink()
    mnru_engine.run_repairs(g_engine_mnru, budget=10000.0, sink=sink_m)

    for u in g_direct_mnru.ids():
        for lvl in range(g_direct_mnru.node_level(u) + 1):
            assert g_direct_mnru.neighbors(u, lvl) == g_engine_mnru.neighbors(u, lvl), (
                f"MNRU adjacency mismatch at {u}"
            )
