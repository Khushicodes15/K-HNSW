"""Deterministic script demonstrating end-to-end kinetic trace event generation.

Outputs:
- events.jsonl: Line-delimited JSON stream of trace events conforming to event schemas.
- initial_snapshot.json: Graph topology and vector snapshot before kinetic drift.
- final_snapshot.json: Graph topology and vector snapshot after drift and repair.

Prints summary table of event counts partitioned by event type.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.plan import khnsw_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap


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


def export_snapshot(index: HNSWIndex, filepath: str | Path) -> None:
    """Serialize complete graph structure, layers, and coordinates to JSON."""
    snap = index.snapshot()
    # Add coordinates for frontend visualization
    snap["vectors"] = {str(u): index.vector(u).tolist() for u in index.ids()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)


def main() -> None:
    n = 500
    m = 8
    d = 8
    seed = 42
    run_id = "demo-run-500"
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    # 1. Build and settle HNSW index
    idx = HNSWIndex(dim=d, m=m, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X)

    store = CertificateStore()
    meter = CostMeter()
    store.build(idx, meter)
    settle_hnsw(idx, store, meter)

    # Export initial snapshot
    export_snapshot(idx, "initial_snapshot.json")
    print("Exported initial_snapshot.json")

    scale = mean_nn_distance(X, seed=seed)
    sink = ListSink()
    emitter = EventEmitter(run_id=run_id, sink=sink)

    # Emit run_start
    emitter.emit(
        type=EventType.run_start,
        node=None,
        level=None,
        version=None,
        payload={"total_nodes": n, "dim": d, "dataset": "gaussian_synthetic"},
        t=0.0,
    )

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
        run_id=run_id,
    )

    # PHASE 1: Drift in steps
    t_drift_steps = [0.05, 0.10, 0.15]
    for t_step in t_drift_steps:
        detector.sync_nodes(idx.ids(), t_step)
        engine._time_source = t_step
        engine.run_repairs(idx, budget=100000.0, sink=sink)

    # PHASE 2: Trajectory change at t=0.20
    t_change = 0.20
    detector.sync_nodes(idx.ids(), t_change)
    engine._time_source = t_change
    trajectory_changed_nodes = [10, 25, 50, 100, 200]

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

    for t_step in [0.25, 0.30]:
        detector.sync_nodes(idx.ids(), t_step)
        engine._time_source = t_step
        engine.run_repairs(idx, budget=100000.0, sink=sink)

    # PHASE 3: Discrete moves on 5% of nodes
    final_t = 0.30
    num_moved = int(0.05 * n)  # 25 nodes
    perm = rng.permutation(n)
    candidates = [x for x in perm if x not in trajectory_changed_nodes]
    moved_2x = candidates[: num_moved // 2]
    moved_5x = candidates[num_moved // 2 : num_moved]

    for u in moved_2x:
        disp = rng.standard_normal(d).astype(np.float32)
        disp /= np.linalg.norm(disp)
        new_p = idx.vector(u) + disp * (2.0 * scale)
        drift._base[u] = new_p.astype(np.float64) - drift._velocities[u] * final_t
        idx.move(u, new_p)
        detector.on_trajectory_change(u, t=final_t)
        engine.notify_moved(u, t=final_t)

    for u in moved_5x:
        disp = rng.standard_normal(d).astype(np.float32)
        disp /= np.linalg.norm(disp)
        new_p = idx.vector(u) + disp * (5.0 * scale)
        drift._base[u] = new_p.astype(np.float64) - drift._velocities[u] * final_t
        idx.move(u, new_p)
        detector.on_trajectory_change(u, t=final_t)
        engine.notify_moved(u, t=final_t)

    engine.run_repairs(idx, budget=200000.0, sink=sink)

    # Emit run_end
    emitter.emit(
        type=EventType.run_end,
        node=None,
        level=None,
        version=None,
        payload={"status": "completed", "total_events": len(sink.events) + 1},
        t=final_t,
    )

    # Export final snapshot
    export_snapshot(idx, "final_snapshot.json")
    print("Exported final_snapshot.json")

    # Write events.jsonl
    with open("events.jsonl", "w", encoding="utf-8") as f:
        for ev in sink.events:
            line = json.dumps(ev.to_dict(), allow_nan=False)
            f.write(line + "\n")
    print(f"Exported events.jsonl ({len(sink.events)} events)")

    # Print summary table of event counts by type
    counts: collections.Counter[str] = collections.Counter()
    for ev in sink.events:
        type_str = ev.type.value if isinstance(ev.type, EventType) else str(ev.type)
        counts[type_str] += 1

    print("\nTrace Event Summary:")
    print("-" * 35)
    print(f"{'Event Type':<20} | {'Count':>10}")
    print("-" * 35)
    for etype in sorted(counts.keys()):
        print(f"{etype:<20} | {counts[etype]:>10}")
    print("-" * 35)
    print(f"{'Total Events':<20} | {len(sink.events):>10}")


if __name__ == "__main__":
    main()
