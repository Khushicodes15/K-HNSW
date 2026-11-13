"""Unit tests for drift-moved node handling in K-HNSW repair engine on ReferenceGraph."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import khnsw.core.repair.loop as loop_mod
from khnsw.core.detection.certificate_store import CertificateStore
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.loop import repair_loop
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.search_layer import search_layer
from tests.fixtures.reference_graph import ReferenceGraph


def make_rg_sync(graph: ReferenceGraph):
    """Callback syncing ReferenceGraph vectors in float32."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        graph._vectors[node_id] = pos.astype(np.float32)

    return _sync


def _settle_rg(graph: ReferenceGraph, store: CertificateStore) -> None:
    """Run khnsw_plan settle pass to eliminate initial edge violations."""
    repair_loop(graph, store)


def test_drifting_node_rebuilt_with_reverse_links_t0_zero() -> None:
    """Test (1a): Drifting node gets rebuilt out-links and >=1 reverse link at t0=0."""
    rng = np.random.default_rng(42)
    n = 20
    d = 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=4, seed=42, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)
    _settle_rg(rg, store)

    # Node 0 moves fast, other nodes are static
    drift_node = 0
    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[drift_node] = np.array([50.0, 50.0], dtype=np.float64)

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    emitter = EventEmitter("drift_test", sink)
    heap = KineticEventHeap(version_of=rg.version)
    det_meter = CostMeter()
    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=det_meter,
        horizon=1.0,
        sync_vector=make_rg_sync(rg),
        emitter=emitter,
    )
    detector.schedule_all(t0=0.0)

    # Advance to t=0.10: node 0 moves far away, failing its incident edges
    t_step = 0.10
    cur_t = t_step
    # Sync vectors in graph
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t_step).astype(np.float32)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=lambda: cur_t,
        search_layer=search_layer,
    )

    res = engine.run_repairs(rg, budget=10000.0, sink=sink)

    # Preconditions: engine processed drift failure
    assert res["moved_in_drift"] >= 1, "Precondition: drifting node must be recognized in drift"
    assert res["stopped_reason"] in {"converged", "max_rounds"}

    # Assert 1: Out-links of node 0 were rebuilt
    out_nbrs = rg.neighbors(drift_node, 0)
    assert len(out_nbrs) > 0, "Drifting node must have rebuilt out-neighbors"

    # Assert 2: At least one reverse link was formed (in-degree >= 1 and reciprocal edge)
    in_nbrs = rg.in_neighbors(drift_node, 0)
    assert len(in_nbrs) >= 1, f"Drifting node in-degree must be >= 1, got {len(in_nbrs)}"
    reciprocal = [w for w in out_nbrs if drift_node in rg.neighbors(w, 0)]
    assert len(reciprocal) >= 1, f"Expected at least one reciprocal reverse link, got {reciprocal}"


def test_drifting_node_rebuilt_with_reverse_links_t0_nonzero() -> None:
    """Test (1b): Drifting node gets rebuilt out-links and >=1 reverse link at t0!=0."""
    rng = np.random.default_rng(123)
    n = 20
    d = 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=4, seed=123, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)
    _settle_rg(rg, store)

    drift_node = 1
    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[drift_node] = np.array([40.0, -40.0], dtype=np.float64)

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    t0 = 0.15
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t0).astype(np.float32)

    sink = ListSink()
    emitter = EventEmitter("drift_test_nonzero", sink)
    heap = KineticEventHeap(version_of=rg.version)
    det_meter = CostMeter()
    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=det_meter,
        horizon=1.0,
        sync_vector=make_rg_sync(rg),
        emitter=emitter,
    )
    # Schedule certificates starting at t0 != 0
    detector.schedule_all(t0=t0)

    # Step to t1 = 0.25
    t1 = 0.25
    cur_t = t1
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t1).astype(np.float32)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=lambda: cur_t,
        search_layer=search_layer,
    )

    res = engine.run_repairs(rg, budget=10000.0, sink=sink)
    assert res["moved_in_drift"] >= 1, "Precondition: drifting node must be recognized at t0 != 0"

    in_nbrs = rg.in_neighbors(drift_node, 0)
    assert len(in_nbrs) >= 1, f"Drifting node in-degree must be >= 1 at t0 != 0, got {len(in_nbrs)}"
    out_nbrs = rg.neighbors(drift_node, 0)
    reciprocal = [w for w in out_nbrs if drift_node in rg.neighbors(w, 0)]
    assert len(reciprocal) >= 1, "Expected reciprocal reverse link at t0 != 0"


def test_static_node_involved_in_failure_never_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test (2): A static node involved in a failure is never treated as moved or rebuilt."""
    rng = np.random.default_rng(200)
    n = 20
    d = 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=4, seed=200, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)
    _settle_rg(rg, store)

    drift_node = 0
    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[drift_node] = np.array([60.0, 60.0], dtype=np.float64)

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    emitter = EventEmitter("drift_test_static", sink)
    heap = KineticEventHeap(version_of=rg.version)
    det_meter = CostMeter()
    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=det_meter,
        horizon=1.0,
        sync_vector=make_rg_sync(rg),
        emitter=emitter,
    )
    detector.schedule_all(t0=0.0)

    # Track which nodes have _rebuild_moved_out_links called
    rebuilt_nodes: list[int] = []
    orig_rebuild = loop_mod._rebuild_moved_out_links

    def tracking_rebuild(index: Any, moved: int, *args: Any, **kwargs: Any) -> Any:
        rebuilt_nodes.append(int(moved))
        return orig_rebuild(index, moved, *args, **kwargs)

    monkeypatch.setattr(loop_mod, "_rebuild_moved_out_links", tracking_rebuild)

    t_step = 0.10
    cur_t = t_step
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t_step).astype(np.float32)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=lambda: cur_t,
        search_layer=search_layer,
    )

    res = engine.run_repairs(rg, budget=10000.0, sink=sink)
    assert res["moved_in_drift"] == 1, (
        "Only the 1 drifting node should be counted as moved_in_drift"
    )

    # Precondition: At least one failure edge involved static nodes
    assert len(rebuilt_nodes) > 0, "Precondition: rebuild must have been called"
    # Core assertion: All rebuilt nodes must be drift_node (never any static node)
    for node in rebuilt_nodes:
        assert node == drift_node, f"Static node {node} was incorrectly rebuilt!"


def test_one_rebuild_per_node_per_step_across_multiple_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test (3): One rebuild per node per step even if it appears in several failures."""
    rng = np.random.default_rng(300)
    n = 25
    d = 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=4, seed=300, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)
    _settle_rg(rg, store)

    drift_node = 2
    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[drift_node] = np.array([45.0, 45.0], dtype=np.float64)

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    emitter = EventEmitter("drift_test_dedup", sink)
    heap = KineticEventHeap(version_of=rg.version)
    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=CostMeter(),
        horizon=1.0,
        sync_vector=make_rg_sync(rg),
        emitter=emitter,
    )
    detector.schedule_all(t0=0.0)

    rebuild_call_counts: dict[int, int] = {}
    orig_rebuild = loop_mod._rebuild_moved_out_links

    def tracking_rebuild(index: Any, moved: int, *args: Any, **kwargs: Any) -> Any:
        rebuild_call_counts[int(moved)] = rebuild_call_counts.get(int(moved), 0) + 1
        return orig_rebuild(index, moved, *args, **kwargs)

    monkeypatch.setattr(loop_mod, "_rebuild_moved_out_links", tracking_rebuild)

    t_step = 0.12
    cur_t = t_step
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t_step).astype(np.float32)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=lambda: cur_t,
        search_layer=search_layer,
    )

    res = engine.run_repairs(rg, budget=10000.0, sink=sink)
    assert res["moved_in_drift"] == 1

    # Precondition: Drift node had multiple neighbors before move
    assert rebuild_call_counts.get(drift_node, 0) == 1, (
        f"Drift node {drift_node} was rebuilt {rebuild_call_counts.get(drift_node, 0)} times"
    )


def test_multi_step_drift_rebuilds_each_step_not_once_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test (mutation c): Node continuing to drift is rebuilt once per step, NOT once per run."""
    rng = np.random.default_rng(400)
    n = 20
    d = 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=4, seed=400, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)
    _settle_rg(rg, store)

    # 5 drifting nodes so failures occur continuously across steps
    velocities = np.zeros((n, d), dtype=np.float64)
    for i in range(5):
        velocities[i] = rng.standard_normal(d).astype(np.float64) * 20.0

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    emitter = EventEmitter("drift_test_multistep", sink)
    heap = KineticEventHeap(version_of=rg.version)
    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=CostMeter(),
        horizon=1.0,
        sync_vector=make_rg_sync(rg),
        emitter=emitter,
    )
    detector.schedule_all(t0=0.0)

    step_rebuilds: list[int] = []
    orig_rebuild = loop_mod._rebuild_moved_out_links

    def tracking_rebuild(index: Any, moved: int, *args: Any, **kwargs: Any) -> Any:
        step_rebuilds.append(int(moved))
        return orig_rebuild(index, moved, *args, **kwargs)

    monkeypatch.setattr(loop_mod, "_rebuild_moved_out_links", tracking_rebuild)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=0.0,
        search_layer=search_layer,
    )

    # Step 1
    t1 = 0.08
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t1).astype(np.float32)
    engine._time_source = t1
    engine.run_repairs(rg, budget=10000.0, sink=sink)
    assert len(step_rebuilds) >= 1, "Precondition: step 1 must trigger rebuild"

    # Step 2
    step1_count = len(step_rebuilds)
    t2 = 0.16
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t2).astype(np.float32)
    engine._time_source = t2
    engine.run_repairs(rg, budget=10000.0, sink=sink)

    # Mutation (c) would prevent step 2 rebuild because node was already rebuilt in step 1
    assert len(step_rebuilds) > step1_count, (
        "Node moving across steps was not rebuilt in step 2 (violation of 'once per step')"
    )


def test_invariants_store_agreement_and_event_replay() -> None:
    """Test (4): After drift run, invariants hold, store agrees, and events replay."""
    rng = np.random.default_rng(500)
    n = 25
    d = 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=4, seed=500, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)
    _settle_rg(rg, store)

    initial_adj = {u: list(rg.neighbors(u, 0)) for u in rg.ids()}

    # 2 drifting nodes
    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[0] = np.array([30.0, 30.0], dtype=np.float64)
    velocities[1] = np.array([-30.0, 30.0], dtype=np.float64)

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    emitter = EventEmitter("drift_replay", sink)
    heap = KineticEventHeap(version_of=rg.version)
    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=CostMeter(),
        horizon=1.0,
        sync_vector=make_rg_sync(rg),
        emitter=emitter,
    )
    detector.schedule_all(t0=0.0)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=0.0,
        search_layer=search_layer,
        run_id="drift_replay",
    )

    for t in [0.05, 0.10, 0.15]:
        for i in range(n):
            rg._vectors[i] = (X[i] + velocities[i] * t).astype(np.float32)
        engine._time_source = t
        engine.run_repairs(rg, budget=10000.0, sink=sink)

    # 1. Invariants and degree bounds
    assert validate_invariants(rg) == [], "Invariants must hold"
    assert check_degree_bounds(rg) == [], "Degree bounds must hold"

    # 2. From-scratch store agreement
    fresh_store = CertificateStore()
    fresh_meter = CostMeter()
    fresh_store.build(rg, fresh_meter)
    for u in rg.ids():
        for lvl in range(rg.node_level(u) + 1):
            for v in rg.neighbors(u, lvl):
                edge = (u, v, lvl)
                inc_status = store.status(edge)
                fresh_status = fresh_store.status(edge)
                assert inc_status == fresh_status, (
                    f"Store status mismatch at edge {edge}: inc={inc_status}, fresh={fresh_status}"
                )

    # 3. Event replay reproduces final graph adjacency
    replayed_adj: dict[int, list[int]] = {u: list(nbrs) for u, nbrs in initial_adj.items()}
    for ev in sink.events:
        if ev.type == EventType.edge_add:
            u = ev.node
            v = ev.payload["target_id"]
            if v not in replayed_adj[u]:
                replayed_adj[u].append(v)
        elif ev.type == EventType.edge_remove:
            u = ev.node
            v = ev.payload["target_id"]
            if v in replayed_adj[u]:
                replayed_adj[u].remove(v)

    for u in rg.ids():
        assert sorted(rg.neighbors(u, 0)) == sorted(replayed_adj[u]), (
            f"Adjacency replay mismatch at node {u}"
        )


def test_budget_exhaustion_leaves_nothing_half_applied_and_reports_unrepaired() -> None:
    """Test (6): Restricting budget halts cleanly and reports unrepaired moving nodes."""
    rng = np.random.default_rng(600)
    n = 20
    d = 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    levels = {i: 0 for i in range(n)}
    rg = ReferenceGraph.from_points(X, M=4, seed=600, levels=levels)

    store = CertificateStore()
    meter = CostMeter()
    store.build(rg, meter)
    _settle_rg(rg, store)

    # 3 nodes drifting
    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[0] = np.array([40.0, 40.0], dtype=np.float64)
    velocities[1] = np.array([-40.0, 40.0], dtype=np.float64)
    velocities[2] = np.array([40.0, -40.0], dtype=np.float64)

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    heap = KineticEventHeap(version_of=rg.version)
    detector = KineticDetector(
        index=rg,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=CostMeter(),
        horizon=1.0,
        sync_vector=make_rg_sync(rg),
    )
    detector.schedule_all(t0=0.0)

    t_step = 0.10
    cur_t = t_step
    for i in range(n):
        rg._vectors[i] = (X[i] + velocities[i] * t_step).astype(np.float32)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=lambda: cur_t,
        search_layer=search_layer,
    )

    # Allot an extremely small budget
    tiny_budget = 2.0
    res = engine.run_repairs(rg, budget=tiny_budget, sink=sink)

    assert res["stopped_reason"] == "budget", (
        f"Expected stopped_reason 'budget', got {res['stopped_reason']}"
    )
    assert len(res["unrepaired_nodes"]) > 0, "Unrepaired nodes must be reported"
    assert validate_invariants(rg) == [], (
        "Invariants must be preserved even under budget exhaustion"
    )
    assert check_degree_bounds(rg) == [], "Degree bounds must be preserved under budget exhaustion"
