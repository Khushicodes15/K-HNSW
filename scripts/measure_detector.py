"""Measurement script evaluating KineticDetector performance vs naive periodic recertification.

Evaluates on HNSWIndex (n=200, M=8, d in {2, 8}) under LinearDrift motion over time T:
- Failures confirmed and failure percentage (~10-30% range).
- False alarms and boundary cases.
- Pure-math failure-time evaluations (fail_time_evals).
- Reschedules from on_trajectory_change and false alarms.
- Detection cost reported three ways:
  1. Confirmation distance computations only (CostMeter).
  2. + fail_time_evals x 1 (1 distance equivalent per witness evaluation).
  3. + fail_time_evals x 3 (3 dot products equivalent per quadratic polynomial setup).
- Cost comparison against naive periodic full recertification baseline.
"""

from __future__ import annotations

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.plan import khnsw_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap


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


def measure_dimension(d: int, n: int = 200, m: int = 8, seed: int = 42) -> dict[str, object]:
    """Run detector benchmark and naive baseline comparison for a given dimension."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)

    index = HNSWIndex(dim=d, m=m, seed=seed)
    index.add(np.arange(n, dtype=np.int64), X)

    init_meter = CostMeter()
    store = CertificateStore()
    store.build(index, init_meter)
    _settle_hnsw(index, store, init_meter)

    total_edges = len(store._edge_status)

    # Simulation setup: T=0.30, 10 steps (~15-27% failures)
    steps = 10
    T = 0.30
    dt = T / steps
    model = LinearDrift(X, seed=seed, speed=1.0)
    trajs = {i: model.trajectory(i) for i in range(n)}

    # 1. Kinetic Detector Run
    heap = KineticEventHeap(version_of=index.version)
    detector_meter = CostMeter()
    detector = KineticDetector(
        index=index,
        store=store,
        heap=heap,
        trajectory_of=lambda nid: trajs[nid],
        meter=detector_meter,
    )
    detector.schedule_all(t0=0.0)

    cumulative_confirmed: set[tuple[int, int, int]] = set()
    for step in range(1, steps + 1):
        t = round(step * dt, 5)
        # Mid-run trajectory change on a connected node
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
    rescheduled_count = stats["rescheduled"]
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
        "d": d,
        "total_edges": total_edges,
        "confirmed": confirmed_count,
        "pct_failed": pct_failed,
        "false_alarms": false_alarms,
        "boundary_cases": boundary_cases,
        "fail_time_evals": fail_time_evals,
        "rescheduled": rescheduled_count,
        "cost_conf_only": cost_conf_only,
        "savings_conf": savings_conf,
        "cost_evals_x1": cost_evals_x1,
        "savings_x1": savings_x1,
        "cost_evals_x3": cost_evals_x3,
        "savings_x3": savings_x3,
        "base_dist_comps": base_dist_comps,
    }


def main() -> None:
    """Run detector measurement across d in {2, 8} and print comparison table."""
    print("=" * 86)
    print("  KINETIC DETECTOR THREE-WAY MEASUREMENT (n=200, M=8, LinearDrift, 10 steps)")
    print("=" * 86)

    results = []
    for d in [2, 8]:
        res = measure_dimension(d=d)
        results.append(res)

    r0, r1 = results[0], results[1]
    print("\n| Metric | d = 2 | d = 8 |")
    print("| :--- | :---: | :---: |")
    print(f"| Total Edges (E) | {r0['total_edges']} | {r1['total_edges']} |")
    print(
        f"| Confirmed Failures | {r0['confirmed']} ({r0['pct_failed']:.1f}%) | "
        f"{r1['confirmed']} ({r1['pct_failed']:.1f}%) |"
    )
    print(f"| False Alarms | {r0['false_alarms']} | {r1['false_alarms']} |")
    print(f"| Boundary Cases | {r0['boundary_cases']} | {r1['boundary_cases']} |")
    print(
        f"| Pure-Math Evaluations (`fail_time_evals`) | {r0['fail_time_evals']} | "
        f"{r1['fail_time_evals']} |"
    )
    print(
        f"| Total Rescheduled (inc. trajectory move) | {r0['rescheduled']} | {r1['rescheduled']} |"
    )
    print(
        f"| Naive Baseline Distance Comps (Full) | {r0['base_dist_comps']} | "
        f"{r1['base_dist_comps']} |"
    )
    s_conf_0, s_conf_1 = f"{r0['savings_conf']:.1f}x", f"{r1['savings_conf']:.1f}x"
    s_x1_0, s_x1_1 = f"{r0['savings_x1']:.1f}x", f"{r1['savings_x1']:.1f}x"
    s_x3_0, s_x3_1 = f"{r0['savings_x3']:.1f}x", f"{r1['savings_x3']:.1f}x"
    print(
        f"| 1. Confirmation Only Cost | {r0['cost_conf_only']} ({s_conf_0}) | "
        f"{r1['cost_conf_only']} ({s_conf_1}) |"
    )
    print(
        f"| 2. Cost (+ fail_time_evals x 1) | {r0['cost_evals_x1']} ({s_x1_0}) | "
        f"{r1['cost_evals_x1']} ({s_x1_1}) |"
    )
    print(
        f"| 3. Cost (+ fail_time_evals x 3) | {r0['cost_evals_x3']} ({s_x3_0}) | "
        f"{r1['cost_evals_x3']} ({s_x3_1}) |"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
