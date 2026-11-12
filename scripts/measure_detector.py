"""Measurement script evaluating KineticDetector performance vs naive periodic recertification.

Evaluates on HNSWIndex (n=200, M=8, d in {2, 8}) under LinearDrift and ClusterDrift:
- Confirmed failures, false alarms, and boundary cases.
- Pure-math failure-time evaluations (fail_time_evals) split into:
  - Initial schedule evaluations
  - Reschedule evaluations (trajectory changes and false alarms)
  - Window refresh evaluations
- Confirmation distance computations (CostMeter).
- Cost comparison against naive periodic full recertification baseline charging
  evaluations at 0x (confirmation only), 1x, and 3x.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.cluster import ClusterDrift
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.plan import khnsw_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap


def make_hnsw_sync(index: HNSWIndex) -> Callable[[int, np.ndarray], None]:
    """Test callback syncing HNSWIndex vectors to float32."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def _settle_hnsw(index: HNSWIndex, store: CertificateStore, meter: CostMeter) -> None:
    """Run khnsw_plan to settle construction-time violations."""
    violated: dict[tuple[int, int], list[int]] = {}
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                if store.status((u, v, lvl)) == EdgeStatus.VIOLATED.value:
                    violated.setdefault((u, lvl), []).append(v)
    if violated:
        plan = khnsw_plan(index, violated=violated, meter=meter)
        apply_plan(index, plan, store=store, meter=meter)


def measure_configuration(
    model_name: str,
    d: int,
    n: int = 200,
    m: int = 8,
    seed: int = 42,
    horizon: float = 0.05,
    T: float = 0.30,
    steps: int = 10,
) -> dict[str, Any]:
    """Run detector benchmark and naive baseline comparison for a model and dimension."""
    rng = np.random.default_rng(seed + d)
    X = rng.standard_normal((n, d), dtype=np.float32)

    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X)

    init_meter = CostMeter()
    store = CertificateStore()
    store.build(index, init_meter)
    _settle_hnsw(index, store, init_meter)

    total_edges = len(store._edge_status)

    dt = T / steps
    if model_name == "LinearDrift":
        motion_model: LinearDrift | ClusterDrift = LinearDrift(X, seed=seed, speed=1.0)
    else:
        motion_model = ClusterDrift(X, seed=seed, n_clusters=4, speed=1.0, noise=0.1)

    trajs = {i: motion_model.trajectory(i) for i in range(n)}

    # 1. Kinetic Detector Run
    heap = KineticEventHeap(version_of=index.version)
    detector_meter = CostMeter()
    detector = KineticDetector(
        index=index,
        store=store,
        heap=heap,
        trajectory_of=lambda nid: trajs[nid],
        meter=detector_meter,
        horizon=horizon,
        sync_vector=make_hnsw_sync(index),
    )
    detector.schedule_all(t0=0.0)

    cumulative_confirmed: set[tuple[int, int, int]] = set()
    for step in range(1, steps + 1):
        t = round(step * dt, 5)
        # Mid-run trajectory change at step 5
        if step == 5:
            moved_node = max(index.ids(), key=lambda i: len(index.neighbors(i, 0)))
            old_a, old_b = trajs[moved_node]
            pos_t = old_a + old_b * t
            new_vel = -old_b * 1.5
            trajs[moved_node] = (pos_t - new_vel * t, new_vel)
            index.move(moved_node, pos_t.astype(np.float32))
            detector.on_trajectory_change(moved_node, t)

        new_conf = detector.advance(t)
        for e in new_conf:
            cumulative_confirmed.add(e)

    stats = detector.stats()
    confirmed_count = stats["confirmed"]
    pct_failed = (confirmed_count / total_edges) * 100.0 if total_edges > 0 else 0.0
    false_alarms = stats["false_alarms"]
    fail_time_evals = stats["fail_time_evals"]
    initial_evals = stats["initial_evals"]
    reschedule_evals = stats["reschedule_evals"]
    refresh_evals = stats["refresh_evals"]
    window_refreshes = stats["window_refreshes"]
    det_dist_comps = detector_meter.count

    # 2. Naive Periodic Recertification Baseline
    baseline_meter = CostMeter()
    baseline_store = CertificateStore()

    for step in range(1, steps + 1):
        t = round(step * dt, 5)
        for nid in range(n):
            a, b = trajs[nid]
            index._store.set_vector(nid, (a + b * t).astype(np.float32))
        for lvl in range(index.max_level() + 1):
            for u in index.ids():
                if index.node_level(u) >= lvl:
                    baseline_store.certify_source(index, u, lvl, baseline_meter)

    base_dist_comps = baseline_meter.count

    # Three-way cost accounting:
    cost_conf_only = det_dist_comps
    cost_evals_x1 = det_dist_comps + fail_time_evals * 1
    cost_evals_x3 = det_dist_comps + fail_time_evals * 3

    savings_conf = base_dist_comps / cost_conf_only if cost_conf_only > 0 else float("inf")
    savings_x1 = base_dist_comps / cost_evals_x1 if cost_evals_x1 > 0 else float("inf")
    savings_x3 = base_dist_comps / cost_evals_x3 if cost_evals_x3 > 0 else float("inf")

    # Boundary cases: compare final oracle vs detector
    oracle_store = CertificateStore()
    oracle_meter = CostMeter()
    for lvl in range(index.max_level() + 1):
        for u in index.ids():
            if index.node_level(u) >= lvl:
                oracle_store.certify_source(index, u, lvl, oracle_meter)

    oracle_violated = {
        e for e, s in oracle_store._edge_status.items() if s == EdgeStatus.VIOLATED.value
    }
    boundary_cases = len(oracle_violated - cumulative_confirmed)

    return {
        "model": model_name,
        "d": d,
        "total_edges": total_edges,
        "confirmed": confirmed_count,
        "pct_failed": pct_failed,
        "false_alarms": false_alarms,
        "boundary_cases": boundary_cases,
        "fail_time_evals": fail_time_evals,
        "initial_evals": initial_evals,
        "reschedule_evals": reschedule_evals,
        "refresh_evals": refresh_evals,
        "window_refreshes": window_refreshes,
        "det_dist_comps": det_dist_comps,
        "base_dist_comps": base_dist_comps,
        "cost_conf_only": cost_conf_only,
        "savings_conf": savings_conf,
        "cost_evals_x1": cost_evals_x1,
        "savings_x1": savings_x1,
        "cost_evals_x3": cost_evals_x3,
        "savings_x3": savings_x3,
    }


def main() -> None:
    """Run detector measurement across models and dimensions."""
    print("=" * 115)
    print("  KINETIC DETECTOR MEASUREMENT (n=200, M=8, T=0.30, steps=10, horizon=0.05)")
    print("=" * 115)

    runs: list[dict[str, Any]] = []
    for model_name in ["LinearDrift", "ClusterDrift"]:
        for d in [2, 8]:
            res = measure_configuration(model_name=model_name, d=d)
            runs.append(res)

    header = (
        f"{'MODEL':<13} {'D':<3} {'EDGES':<6} {'CONF':<6} {'FA':<4} {'BOUND':<6} "
        f"{'INIT_EV':<8} {'RESCH_EV':<9} {'REFR_EV':<8} {'TOTAL_EV':<9} "
        f"{'CONF_DIST':<10} {'BASE_DIST':<10} {'SAV(0x)':<8} {'SAV(1x)':<8} {'SAV(3x)':<8}"
    )
    print(header)
    print("-" * 115)

    for r in runs:
        s0 = f"{r['savings_conf']:.1f}x"
        s1 = f"{r['savings_x1']:.1f}x"
        s3 = f"{r['savings_x3']:.1f}x"
        line = (
            f"{r['model']:<13} {r['d']:<3} {r['total_edges']:<6} {r['confirmed']:<6} "
            f"{r['false_alarms']:<4} {r['boundary_cases']:<6} {r['initial_evals']:<8} "
            f"{r['reschedule_evals']:<9} {r['refresh_evals']:<8} {r['fail_time_evals']:<9} "
            f"{r['det_dist_comps']:<10} {r['base_dist_comps']:<10} {s0:<8} {s1:<8} {s3:<8}"
        )
        print(line)
    print("=" * 115)


if __name__ == "__main__":
    main()
