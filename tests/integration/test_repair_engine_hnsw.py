"""Integration tests for topology repair engines on Member 2's HNSWIndex."""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import Event, EventEmitter, EventType, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.interfaces import Repairer
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.engine import (
    HnswlibStyleEngine,
    KHNSWRepairEngine,
    MNRUStyleEngine,
)
from khnsw.core.repair.loop import repair_loop
from khnsw.core.repair.plan import mnru_plan, native_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap


def make_hnsw_sync(index: HNSWIndex):
    """Callback syncing HNSWIndex vectors to float32."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def _settle_hnsw(index: HNSWIndex, store: CertificateStore, meter: CostMeter) -> None:
    """Run initial settle pass on HNSWIndex to eliminate construction-time violations."""
    repair_loop(index, store)


def _collect_violated(index: HNSWIndex, store: CertificateStore) -> list[tuple[int, int, int]]:
    """Helper to collect violated directed edges from store."""
    violated: list[tuple[int, int, int]] = []
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                edge = (u, v, lvl)
                if store.status(edge) == EdgeStatus.VIOLATED.value:
                    violated.append(edge)
    return sorted(violated)


@pytest.mark.parametrize("d", [2, 8])
@pytest.mark.parametrize("engine_cls", [KHNSWRepairEngine, HnswlibStyleEngine, MNRUStyleEngine])
def test_hnsw_engine_satisfies_protocol_and_no_op(d: int, engine_cls: type) -> None:
    """Engines implement Repairer and return zeros when nothing is to be done."""
    n = 30
    m = 4
    seed = 42 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    if engine_cls is KHNSWRepairEngine:
        engine = engine_cls(store=store)
    else:
        engine = engine_cls()

    assert isinstance(engine, Repairer)

    sink = ListSink()
    res = engine.run_repairs(idx, budget=5000.0, sink=sink)
    assert res["stopped_reason"] == "nothing_to_do"
    assert res["rounds"] == 0
    assert res["edges_repaired"] == 0
    assert res["budget_consumed"] == 0
    assert len(sink.events) == 0


@pytest.mark.parametrize("d", [2, 8])
@pytest.mark.parametrize("engine_cls", [KHNSWRepairEngine, HnswlibStyleEngine, MNRUStyleEngine])
def test_hnsw_budget_exhaustion_preserves_invariants(d: int, engine_cls: type) -> None:
    """Restricting computational budget halts cleanly without corrupting HNSW index."""
    n = 40
    m = 4
    seed = 100 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    if engine_cls is KHNSWRepairEngine:
        engine = engine_cls(store=store)
    else:
        engine = engine_cls()

    moved_node = 2
    new_v = (idx.vector(moved_node) + np.ones(d, dtype=np.float32) * 2.0).astype(np.float32)
    idx.move(moved_node, new_v)
    engine.notify_moved(moved_node)

    assert len(engine._pending_moves) > 0, "Precondition: pending moves must exist"

    # Allot an extremely small budget
    tiny_budget = 2.0
    sink = ListSink()
    res = engine.run_repairs(idx, budget=tiny_budget, sink=sink)

    assert res["stopped_reason"] == "budget"
    assert len(res["unrepaired_nodes"]) > 0, "Unrepaired nodes must be preserved"
    assert validate_invariants(idx) == []
    assert check_degree_bounds(idx) == []


@pytest.mark.parametrize("d", [2, 8])
def test_hnsw_khnsw_engine_matches_direct_loop(d: int) -> None:
    """KHNSWRepairEngine produces identical topology to direct repair_loop on HNSWIndex."""
    n = 35
    m = 4
    seed = 200 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    # 1. Direct repair_loop
    idx1 = HNSWIndex(dim=d, m=m, seed=seed)
    idx1.add(np.arange(n, dtype=np.int64), X)
    store1 = CertificateStore()
    m1 = CostMeter()
    store1.build(idx1, m1)
    _settle_hnsw(idx1, store1, m1)

    move_node = 1
    new_v = (idx1.vector(move_node) + np.ones(d, dtype=np.float32) * 1.5).astype(np.float32)
    idx1.move(move_node, new_v)
    store1.on_move(idx1, move_node, m1)
    assert len(_collect_violated(idx1, store1)) > 0, "Precondition: violations must exist"

    sink1 = ListSink()
    emitter1 = EventEmitter("direct", sink1)
    res1 = repair_loop(idx1, store1, moved_node=move_node, emitter=emitter1)

    # 2. KHNSWRepairEngine
    idx2 = HNSWIndex(dim=d, m=m, seed=seed)
    idx2.add(np.arange(n, dtype=np.int64), X)
    store2 = CertificateStore()
    m2 = CostMeter()
    store2.build(idx2, m2)
    _settle_hnsw(idx2, store2, m2)

    idx2.move(move_node, new_v)
    store2.on_move(idx2, move_node, m2)
    engine = KHNSWRepairEngine(store=store2)
    engine.notify_moved(move_node)

    sink2 = ListSink()
    res2 = engine.run_repairs(idx2, budget=10000.0, sink=sink2)

    assert res1.stopped_reason == res2["stopped_reason"]
    assert res1.edges_repaired == res2["edges_repaired"]

    for u in idx1.ids():
        for lvl in range(idx1.node_level(u) + 1):
            assert idx1.neighbors(u, lvl) == idx2.neighbors(u, lvl), f"Mismatch at {u}"


@pytest.mark.parametrize("d", [2, 8])
def test_hnsw_baselines_match_direct_planners(d: int) -> None:
    """HnswlibStyleEngine and MNRUStyleEngine match direct planners on HNSWIndex."""
    n = 35
    m = 4
    seed = 300 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    move_node = 3
    new_v = (X[move_node] + np.ones(d, dtype=np.float32) * 1.8).astype(np.float32)

    # 1. HnswlibStyleEngine vs direct native_plan
    idx_h_dir = HNSWIndex(dim=d, m=m, seed=seed)
    idx_h_dir.add(np.arange(n, dtype=np.int64), X)
    idx_h_dir.move(move_node, new_v)
    plan_native = native_plan(idx_h_dir, moved=move_node, meter=CostMeter())
    assert len(plan_native.rewrites) > 0, "Precondition: rewrites must be formulated"
    apply_plan(idx_h_dir, plan_native)

    idx_h_eng = HNSWIndex(dim=d, m=m, seed=seed)
    idx_h_eng.add(np.arange(n, dtype=np.int64), X)
    idx_h_eng.move(move_node, new_v)
    h_eng = HnswlibStyleEngine()
    h_eng.notify_moved(move_node)
    sink_h = ListSink()
    h_eng.run_repairs(idx_h_eng, budget=10000.0, sink=sink_h)

    for u in idx_h_dir.ids():
        for lvl in range(idx_h_dir.node_level(u) + 1):
            assert idx_h_dir.neighbors(u, lvl) == idx_h_eng.neighbors(u, lvl), (
                f"Hnswlib mismatch at {u}"
            )

    # 2. MNRUStyleEngine vs direct mnru_plan
    idx_m_dir = HNSWIndex(dim=d, m=m, seed=seed)
    idx_m_dir.add(np.arange(n, dtype=np.int64), X)
    idx_m_dir.move(move_node, new_v)
    plan_mnru = mnru_plan(idx_m_dir, moved=move_node, alpha=1.0, scope="one_hop", meter=CostMeter())
    assert len(plan_mnru.rewrites) > 0, "Precondition: rewrites must be formulated"
    apply_plan(idx_m_dir, plan_mnru)

    idx_m_eng = HNSWIndex(dim=d, m=m, seed=seed)
    idx_m_eng.add(np.arange(n, dtype=np.int64), X)
    idx_m_eng.move(move_node, new_v)
    m_eng = MNRUStyleEngine(alpha=1.0, scope="one_hop")
    m_eng.notify_moved(move_node)
    sink_m = ListSink()
    m_eng.run_repairs(idx_m_eng, budget=10000.0, sink=sink_m)

    for u in idx_m_dir.ids():
        for lvl in range(idx_m_dir.node_level(u) + 1):
            assert idx_m_dir.neighbors(u, lvl) == idx_m_eng.neighbors(u, lvl), (
                f"MNRU mismatch at {u}"
            )


@pytest.mark.parametrize("d", [2, 8])
def test_hnsw_khnsw_engine_with_kinetic_detector(d: int) -> None:
    """KHNSWRepairEngine integrates with KineticDetector under LinearDrift."""
    n = 50
    m = 4
    seed = 400 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)
    _settle_hnsw(idx, store, meter)

    drift = LinearDrift(X, seed=seed, speed=5.0)
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=drift.trajectory,
        horizon=10.0,
        sync_vector=make_hnsw_sync(idx),
        meter=meter,
    )
    detector.schedule_all(t0=0.0)

    t_eval = 0.20
    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=lambda: t_eval,
    )

    sink = ListSink()
    res = engine.run_repairs(idx, budget=50000.0, sink=sink)

    assert res["stopped_reason"] == "converged"
    assert res["edges_repaired"] > 0, "Precondition: kinetic drift must trigger edge repairs"
    assert len(sink.events) > 0, "Trace events must be emitted"
    for ev in sink.events:
        assert isinstance(ev, Event)
        assert ev.type in {EventType.edge_add, EventType.edge_remove}
        assert "target_id" in ev.payload
