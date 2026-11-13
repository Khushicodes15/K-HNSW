"""Integration tests for drift-moved node handling on HNSWIndex (d in {2, 8})."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.loop import repair_loop
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.search_layer import search_layer


def make_hnsw_sync(index: HNSWIndex):
    """Callback syncing HNSWIndex vectors in float32."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def _settle_hnsw(index: HNSWIndex, store: CertificateStore) -> None:
    """Run initial settle pass on HNSWIndex to eliminate construction-time violations."""
    repair_loop(index, store)


@pytest.mark.parametrize("d", [2, 8])
def test_hnsw_drift_moved_rebuilt_with_reverse_links_and_invariants(d: int) -> None:
    """Tests (1, 2, 4): Drifting nodes on HNSWIndex get rebuilt out-links and reverse links."""
    n = 35
    m = 4
    seed = 42 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)
    _settle_hnsw(idx, store)

    initial_adj = {
        (u, lvl): list(idx.neighbors(u, lvl))
        for u in idx.ids()
        for lvl in range(idx.node_level(u) + 1)
    }

    # 1 drifting node, 34 static
    drift_node = 0
    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[drift_node] = np.ones(d, dtype=np.float64) * 35.0

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    emitter = EventEmitter("hnsw_drift_test", sink)
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=CostMeter(),
        horizon=1.0,
        sync_vector=make_hnsw_sync(idx),
        emitter=emitter,
    )
    detector.schedule_all(t0=0.0)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=0.0,
        search_layer=search_layer,
        run_id="hnsw_drift_test",
    )

    t_step = 0.10
    for i in range(n):
        idx._store.set_vector(i, (X[i] + velocities[i] * t_step).astype(np.float32))
    engine._time_source = t_step

    res = engine.run_repairs(idx, budget=10000.0, sink=sink)

    assert res["moved_in_drift"] == 1, "Precondition: exactly 1 drifting node must be recognized"
    assert res["stopped_reason"] in {"converged", "max_rounds"}

    # 1. In-degree >= 1 and reciprocal reverse link
    in_nbrs = idx.in_neighbors(drift_node, 0)
    assert len(in_nbrs) >= 1, f"Drifting node in-degree must be >= 1, got {len(in_nbrs)}"
    out_nbrs = idx.neighbors(drift_node, 0)
    assert len(out_nbrs) > 0, "Drifting node must have rebuilt out-links"
    reciprocal = [w for w in out_nbrs if drift_node in idx.neighbors(w, 0)]
    assert len(reciprocal) >= 1, f"Expected reciprocal reverse link, got {reciprocal}"

    # 2. Structural invariants and degree bounds
    assert validate_invariants(idx) == [], "Invariants must hold after drift repair"
    assert check_degree_bounds(idx) == [], "Degree bounds must hold after drift repair"

    # 3. From-scratch store agreement
    fresh_store = CertificateStore()
    fresh_meter = CostMeter()
    fresh_store.build(idx, fresh_meter)
    for u in idx.ids():
        for lvl in range(idx.node_level(u) + 1):
            for v in idx.neighbors(u, lvl):
                edge = (u, v, lvl)
                inc_st = store.status(edge)
                fresh_st = fresh_store.status(edge)
                assert inc_st == fresh_st, f"Store status mismatch at edge {edge}"

    # 4. Event replay reproduces final graph adjacency
    replayed_adj: dict[tuple[int, int], list[int]] = {
        k: list(nbrs) for k, nbrs in initial_adj.items()
    }
    for ev in sink.events:
        if ev.type == EventType.edge_add:
            u = ev.node
            lvl = ev.level
            v = ev.payload["target_id"]
            if (u, lvl) in replayed_adj and v not in replayed_adj[(u, lvl)]:
                replayed_adj[(u, lvl)].append(v)
        elif ev.type == EventType.edge_remove:
            u = ev.node
            lvl = ev.level
            v = ev.payload["target_id"]
            if (u, lvl) in replayed_adj and v in replayed_adj[(u, lvl)]:
                replayed_adj[(u, lvl)].remove(v)

    for u in idx.ids():
        for lvl in range(idx.node_level(u) + 1):
            assert sorted(idx.neighbors(u, lvl)) == sorted(replayed_adj[(u, lvl)]), (
                f"Adjacency replay mismatch at node {u}, level {lvl}"
            )


@pytest.mark.parametrize("d", [2, 8])
def test_hnsw_budget_exhaustion_reports_unrepaired(d: int) -> None:
    """Test (6): Restricting computational budget reports unrepaired moving nodes cleanly."""
    n = 30
    m = 4
    seed = 100 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)
    _settle_hnsw(idx, store)

    velocities = np.zeros((n, d), dtype=np.float64)
    velocities[0] = np.ones(d, dtype=np.float64) * 30.0
    velocities[1] = np.ones(d, dtype=np.float64) * -30.0

    def traj_of(node_id: int) -> tuple[np.ndarray, np.ndarray]:
        return X[node_id].astype(np.float64), velocities[node_id]

    sink = ListSink()
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=traj_of,
        meter=CostMeter(),
        horizon=1.0,
        sync_vector=make_hnsw_sync(idx),
    )
    detector.schedule_all(t0=0.0)

    t_step = 0.10
    for i in range(n):
        idx._store.set_vector(i, (X[i] + velocities[i] * t_step).astype(np.float32))

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=t_step,
        search_layer=search_layer,
    )

    tiny_budget = 2.0
    res = engine.run_repairs(idx, budget=tiny_budget, sink=sink)

    assert res["stopped_reason"] == "budget"
    assert len(res["unrepaired_nodes"]) > 0, "Unrepaired nodes must be reported"
    assert validate_invariants(idx) == []
    assert check_degree_bounds(idx) == []


@pytest.mark.parametrize("d", [2, 8])
def test_hnsw_discrete_move_equivalence_unchanged(d: int) -> None:
    """Test (7): Discrete-move behaviour stays identical to direct repair_loop."""
    n = 35
    m = 4
    seed = 300 + d
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    # 1. Direct repair_loop
    idx1 = HNSWIndex(dim=d, m=m, seed=seed)
    idx1.add(np.arange(n, dtype=np.int64), X)
    store1 = CertificateStore()
    meter1 = CostMeter()
    store1.build(idx1, meter1)
    _settle_hnsw(idx1, store1)

    move_node = 2
    new_v = (idx1.vector(move_node) + np.ones(d, dtype=np.float32) * 2.0).astype(np.float32)
    idx1.move(move_node, new_v)
    store1.on_move(idx1, move_node, meter1)

    sink1 = ListSink()
    emitter1 = EventEmitter("direct", sink1)
    res1 = repair_loop(idx1, store1, moved_node=move_node, emitter=emitter1)

    # 2. KHNSWRepairEngine with discrete notify_moved
    idx2 = HNSWIndex(dim=d, m=m, seed=seed)
    idx2.add(np.arange(n, dtype=np.int64), X)
    store2 = CertificateStore()
    meter2 = CostMeter()
    store2.build(idx2, meter2)
    _settle_hnsw(idx2, store2)

    idx2.move(move_node, new_v)
    store2.on_move(idx2, move_node, meter2)
    engine = KHNSWRepairEngine(store=store2)
    engine.notify_moved(move_node)

    sink2 = ListSink()
    res2 = engine.run_repairs(idx2, budget=10000.0, sink=sink2)

    assert res1.stopped_reason == res2["stopped_reason"]
    assert res1.edges_repaired == res2["edges_repaired"]
    assert res2["moved_in_drift"] == 0, "Discrete move must have moved_in_drift == 0"

    for u in idx1.ids():
        for lvl in range(idx1.node_level(u) + 1):
            assert idx1.neighbors(u, lvl) == idx2.neighbors(u, lvl), f"Mismatch at node {u}"


def test_hnsw_self_findability_at_30x_drift() -> None:
    """Test (5): On HNSWIndex at 30x drift: self-findability >= no_repair."""
    n = 500
    m = 8
    d = 8
    speed = 30.0
    T = 0.30
    steps = 10
    dt = T / steps
    seeds = [42, 43, 44]

    for seed in seeds:
        rng = np.random.default_rng(seed)
        X0 = rng.standard_normal((n, d)).astype(np.float32)

        base_index = HNSWIndex(dim=d, m=m, seed=seed)
        base_index.add(np.arange(n, dtype=np.int64), X0)
        settled_store = CertificateStore()
        meter = CostMeter()
        settled_store.build(base_index, meter)
        _settle_hnsw(base_index, settled_store)

        drift = LinearDrift(X0, seed=seed, speed=speed)
        drifting_nodes = sorted(rng.choice(n, size=int(0.10 * n), replace=False).tolist())
        non_drift = [i for i in range(n) if i not in drifting_nodes]
        drift._velocities[non_drift] = 0.0

        pos_T = drift.positions(T)

        # 1. No repair control
        idx_ctrl = copy.deepcopy(base_index)
        for nid in range(n):
            idx_ctrl._store.set_vector(nid, pos_T[nid])

        ctrl_self_find = 0
        for u in drifting_nodes:
            ids, _ = idx_ctrl.search(pos_T[u], k=1, ef=64)
            if len(ids) > 0 and ids[0] == u:
                ctrl_self_find += 1
        ctrl_rate = ctrl_self_find / len(drifting_nodes)

        # 2. Ours with drift-moved rebuild
        idx_ours = copy.deepcopy(base_index)
        store_ours = copy.deepcopy(settled_store)
        heap = KineticEventHeap(version_of=idx_ours.version)
        detector = KineticDetector(
            index=idx_ours,
            store=store_ours,
            heap=heap,
            trajectory_of=drift.trajectory,
            meter=CostMeter(),
            horizon=0.50,
            sync_vector=make_hnsw_sync(idx_ours),
        )
        detector.schedule_all(t0=0.0)

        engine = KHNSWRepairEngine(
            store=store_ours,
            detector=detector,
            time_source=0.0,
            search_layer=search_layer,
        )

        for step in range(1, steps + 1):
            t_curr = round(step * dt, 5)
            pos_t = drift.positions(t_curr)
            for nid in range(n):
                idx_ours._store.set_vector(nid, pos_t[nid])
            engine._time_source = t_curr
            engine.run_repairs(idx_ours, budget=1e9, sink=ListSink())

        ours_self_find = 0
        for u in drifting_nodes:
            ids, _ = idx_ours.search(pos_T[u], k=1, ef=64)
            if len(ids) > 0 and ids[0] == u:
                ours_self_find += 1
        ours_rate = ours_self_find / len(drifting_nodes)

        assert ours_rate >= ctrl_rate, (
            f"Seed {seed}: ours self-findability ({ours_rate:.3f}) < no_repair ({ctrl_rate:.3f})"
        )
