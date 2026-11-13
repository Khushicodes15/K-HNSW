"""End-to-end integration test driving HNSWIndex through drift, trajectory changes, and repairs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.plan import khnsw_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVENT_SCHEMA_PATH = REPO_ROOT / "docs" / "contracts" / "event.schema.json"
SCHEMAS_EVENTS_DIR = REPO_ROOT / "docs" / "contracts" / "schemas" / "events"


def make_hnsw_sync(index: HNSWIndex):
    """Callback syncing HNSWIndex vectors to float32 without bumping version."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def settle_hnsw(index: HNSWIndex, store: CertificateStore, meter: CostMeter) -> None:
    """Run khnsw_plan to settle any construction-time edge violations."""
    violated: dict[tuple[int, int], list[int]] = {}
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                edge = (u, v, lvl)
                if store._edge_status.get(edge) == EdgeStatus.VIOLATED.value:
                    violated.setdefault((u, lvl), []).append(v)
    if violated:
        plan = khnsw_plan(index, violated=violated, meter=meter)
        apply_plan(index, plan, store=store, meter=meter)


def brute_force_knn(X: np.ndarray, queries: np.ndarray, k: int = 10) -> list[set[int]]:
    """Compute exact k nearest neighbors via brute-force pairwise Euclidean distance."""
    exact_neighbors: list[set[int]] = []
    for q in queries:
        diffs = X - q
        dists = np.sum(diffs**2, axis=1)
        top_k = np.argpartition(dists, k)[:k]
        exact_neighbors.append(set(int(idx) for idx in top_k))
    return exact_neighbors


def evaluate_recall(
    index: HNSWIndex, queries: np.ndarray, exact_sets: list[set[int]], k: int = 10, ef: int = 32
) -> float:
    """Calculate mean recall@k across all query points."""
    recalls: list[float] = []
    for q, exact in zip(queries, exact_sets, strict=False):
        ids, _ = index.search(q, k=k, ef=ef)
        found = set(int(nid) for nid in ids[:k])
        recall = len(found & exact) / float(k)
        recalls.append(recall)
    return float(np.mean(recalls))


def test_end_to_end_simulation() -> None:
    """Drive HNSWIndex (n=500, M=8, d=8) through 3 kinetic phases with complete audit checks."""
    n = 500
    m = 8
    d = 8
    seed = 42
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    # 1. Build and settle HNSW index
    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)
    settle_hnsw(idx, store, meter)

    # Initial invariant checks
    assert validate_invariants(idx) == [], "Precondition: initial graph invariants must pass"
    assert check_degree_bounds(idx) == [], "Precondition: initial degree bounds must pass"

    scale = mean_nn_distance(X, seed=seed)
    sink = ListSink()
    emitter = EventEmitter(run_id="e2e-simulation", sink=sink)

    drift = LinearDrift(X, seed=seed, speed=5.0)
    heap = KineticEventHeap(version_of=idx.version)
    detector = KineticDetector(
        index=idx,
        store=store,
        heap=heap,
        trajectory_of=drift.trajectory,
        meter=meter,
        sync_vector=make_hnsw_sync(idx),
        emitter=emitter,
        horizon=1.0,
    )
    detector.schedule_all(t0=0.0)

    engine = KHNSWRepairEngine(
        store=store,
        detector=detector,
        time_source=0.0,
        ef=64,
        run_id="e2e-simulation",
    )

    initial_versions = {u: idx.version(u) for u in idx.ids()}
    initial_adjacency = {
        (u, lvl): set(idx.neighbors(u, lvl))
        for u in idx.ids()
        for lvl in range(idx.node_level(u) + 1)
    }
    all_moved_nodes: set[int] = set()

    # PHASE 1: Drift in steps
    t_drift_steps = [0.05, 0.10, 0.15]
    for t_step in t_drift_steps:
        detector.sync_nodes(idx.ids(), t_step)
        engine._time_source = t_step
        engine.run_repairs(idx, budget=100000.0, sink=sink)

        assert validate_invariants(idx) == [], f"Invariants failed during drift at t={t_step}"
        assert check_degree_bounds(idx) == [], f"Degree bounds failed during drift at t={t_step}"

    # PHASE 2: Trajectory change at t=0.20 then more drift
    t_change = 0.20
    detector.sync_nodes(idx.ids(), t_change)
    engine._time_source = t_change
    trajectory_changed_nodes = [10, 25, 50, 100, 200]
    all_moved_nodes.update(trajectory_changed_nodes)

    for u in trajectory_changed_nodes:
        curr_pos = drift.positions(t_change)[u]
        v_dir = rng.standard_normal(d).astype(np.float64)
        v_dir /= np.linalg.norm(v_dir)
        new_v = v_dir * (10.0 * scale)
        drift._base[u] = curr_pos.astype(np.float64) - new_v * t_change
        drift._velocities[u] = new_v
        idx.move(u, curr_pos.astype(np.float32))
        detector.on_trajectory_change(u, t=t_change)
        engine.notify_moved(u, t=t_change)

    engine.run_repairs(idx, budget=100000.0, sink=sink)
    assert validate_invariants(idx) == [], "Invariants failed after trajectory change"
    assert check_degree_bounds(idx) == [], "Degree bounds failed after trajectory change"

    for t_step in [0.25, 0.30]:
        detector.sync_nodes(idx.ids(), t_step)
        engine._time_source = t_step
        engine.run_repairs(idx, budget=100000.0, sink=sink)

        assert validate_invariants(idx) == [], f"Invariants failed during drift at t={t_step}"
        assert check_degree_bounds(idx) == [], f"Degree bounds failed during drift at t={t_step}"

    # PHASE 3: Discrete-move phase with LARGE displacement on 5% of nodes
    final_t = 0.30
    ctrl_idx = copy.deepcopy(idx)

    num_moved = int(0.05 * n)  # 25 nodes
    perm = rng.permutation(n)
    # Exclude trajectory changed nodes to maintain clear tracking
    candidates = [x for x in perm if x not in trajectory_changed_nodes]
    moved_2x = candidates[: num_moved // 2]
    moved_5x = candidates[num_moved // 2 : num_moved]
    discrete_moved = list(moved_2x) + list(moved_5x)
    all_moved_nodes.update(discrete_moved)

    for u in moved_2x:
        disp = rng.standard_normal(d).astype(np.float32)
        disp /= np.linalg.norm(disp)
        new_p = idx.vector(u) + disp * (2.0 * scale)
        drift._base[u] = new_p.astype(np.float64) - drift._velocities[u] * final_t
        idx.move(u, new_p)
        ctrl_idx.move(u, new_p)
        detector.on_trajectory_change(u, t=final_t)
        engine.notify_moved(u, t=final_t)

    for u in moved_5x:
        disp = rng.standard_normal(d).astype(np.float32)
        disp /= np.linalg.norm(disp)
        new_p = idx.vector(u) + disp * (5.0 * scale)
        drift._base[u] = new_p.astype(np.float64) - drift._velocities[u] * final_t
        idx.move(u, new_p)
        ctrl_idx.move(u, new_p)
        detector.on_trajectory_change(u, t=final_t)
        engine.notify_moved(u, t=final_t)

    res_discrete = engine.run_repairs(idx, budget=200000.0, sink=sink)
    assert res_discrete["stopped_reason"] in {"converged", "max_rounds"}
    assert res_discrete["edges_repaired"] > 0

    # Final Invariants Check
    assert validate_invariants(idx) == [], "Final invariants check failed"
    assert check_degree_bounds(idx) == [], "Final degree bounds check failed"

    # Final Oracle Agreement Check: from-scratch store matches incremental store exactly
    oracle_store = CertificateStore()
    oracle_meter = CostMeter()
    oracle_store.build(idx, oracle_meter)

    oracle_mismatches: list[str] = []
    for u in idx.ids():
        for lvl in range(idx.node_level(u) + 1):
            for v in idx.neighbors(u, lvl):
                edge = (u, v, lvl)
                st_store = store._edge_status.get(edge)
                st_oracle = oracle_store._edge_status.get(edge)
                if st_store != st_oracle:
                    oracle_mismatches.append(f"{edge}: store={st_store}, oracle={st_oracle}")
    assert oracle_mismatches == [], (
        f"Final oracle store disagreed with incremental store on {len(oracle_mismatches)} edges:\n"
        + "\n".join(oracle_mismatches[:10])
    )

    # Monotonic and Isolated Versions Check
    repaired_nodes: set[int] = set()
    for ev in sink.events:
        if ev.type in {EventType.edge_add, EventType.edge_remove} and ev.node is not None:
            repaired_nodes.add(ev.node)

    for u in idx.ids():
        v_init = initial_versions[u]
        v_final = idx.version(u)
        assert v_final >= v_init, (
            f"Version decreased monotonically on node {u}: {v_init} -> {v_final}"
        )
        if v_final > v_init:
            assert u in all_moved_nodes or u in repaired_nodes, (
                f"Version changed on node {u} without undergoing move or repair!"
            )

    # Event Schema Validation & cert_fail Count Check
    with open(EVENT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        generic_schema = json.load(f)

    per_type_schemas: dict[str, dict[str, Any]] = {}
    for schema_file in SCHEMAS_EVENTS_DIR.glob("*.schema.json"):
        with open(schema_file, "r", encoding="utf-8") as sf:
            per_type_schemas[schema_file.stem.replace(".schema", "")] = json.load(sf)

    cert_fail_event_count = 0
    for ev in sink.events:
        ev_dict = ev.to_dict()
        jsonschema.validate(instance=ev_dict, schema=generic_schema)
        type_str = ev.type.value if isinstance(ev.type, EventType) else str(ev.type)
        if type_str in per_type_schemas:
            jsonschema.validate(instance=ev_dict, schema=per_type_schemas[type_str])
        if ev.type == EventType.cert_fail:
            cert_fail_event_count += 1

    assert cert_fail_event_count == detector._confirmed, (
        f"cert_fail count ({cert_fail_event_count}) != detector._confirmed ({detector._confirmed})"
    )

    # REPLAY Check: applying edge_add / edge_remove in order reproduces exact final adjacency
    replayed_adjacency: dict[tuple[int, int], set[int]] = {
        k: set(v) for k, v in initial_adjacency.items()
    }
    for ev in sink.events:
        if ev.type == EventType.edge_add:
            assert ev.node is not None and ev.level is not None
            replayed_adjacency[(ev.node, ev.level)].add(ev.payload["target_id"])
        elif ev.type == EventType.edge_remove:
            assert ev.node is not None and ev.level is not None
            replayed_adjacency[(ev.node, ev.level)].discard(ev.payload["target_id"])

    replay_mismatches: list[str] = []
    for u in idx.ids():
        for lvl in range(idx.node_level(u) + 1):
            expected = set(idx.neighbors(u, lvl))
            actual = replayed_adjacency.get((u, lvl), set())
            if expected != actual:
                replay_mismatches.append(f"({u}, {lvl}): idx={expected}, replayed={actual}")
    assert replay_mismatches == [], (
        f"Replay mismatch on {len(replay_mismatches)} source-layers:\n"
        + "\n".join(replay_mismatches[:10])
    )

    # Recall Evaluation: Control vs Ours
    X_final = np.array([idx.vector(u) for u in idx.ids()], dtype=np.float32)
    # Queries targeting moved nodes and global distribution
    q_moved = [idx.vector(u) for u in discrete_moved]
    q_rand = [rng.standard_normal(d).astype(np.float32) for _ in range(75)]
    queries = np.array(q_moved + q_rand, dtype=np.float32)

    exact_sets = brute_force_knn(X_final, queries, k=10)
    recall_ours_32 = evaluate_recall(idx, queries, exact_sets, k=10, ef=32)
    recall_ctrl_32 = evaluate_recall(ctrl_idx, queries, exact_sets, k=10, ef=32)

    # For queries centered at moved nodes with significant displacements, control drops recall
    exact_moved = brute_force_knn(X_final, np.array(q_moved, dtype=np.float32), k=10)
    recall_moved_ours = evaluate_recall(
        idx, np.array(q_moved, dtype=np.float32), exact_moved, k=10, ef=32
    )
    recall_moved_ctrl = evaluate_recall(
        ctrl_idx, np.array(q_moved, dtype=np.float32), exact_moved, k=10, ef=32
    )

    # Control must lose recall
    assert recall_moved_ctrl < 1.0, (
        f"Control should exhibit routing degradation: {recall_moved_ctrl}"
    )
    lost_str = f"{1.0 - recall_moved_ctrl:.4f}"
    print(
        f"\nEnd-to-end Recall Table:\n"
        f"  Overall:  Ours={recall_ours_32:.4f}, Control={recall_ctrl_32:.4f}\n"
        f"  Moved:    Ours={recall_moved_ours:.4f}, Control={recall_moved_ctrl:.4f} "
        f"(Control lost {lost_str})"
    )
