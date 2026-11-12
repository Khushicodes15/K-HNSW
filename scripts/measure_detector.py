"""Measurement script evaluating KineticDetector performance vs naive periodic recertification.

Evaluates on HNSWIndex (n=200, M=8, d in {2, 8}) under LinearDrift motion over time T:
- Failures confirmed and failure percentage (~10-30% range).
- False alarms and boundary cases.
- Pure-math failure-time evaluations (fail_time_evals).
- Confirmation distance computations recorded by CostMeter.
- Computational cost of naive baseline recertifying every source at every time step.
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

    # Choose simulation horizon and steps: T=0.30, 10 steps (yielding ~14-26% edge failures)
    steps = 10
    T = 0.30
    dt = T / steps
    model = LinearDrift(X, seed=seed, speed=1.0)

    # 1. Kinetic Detector Run
    heap = KineticEventHeap(version_of=index.version)
    detector_meter = CostMeter()
    detector = KineticDetector(
        index=index,
        store=store,
        heap=heap,
        trajectory_of=model.trajectory,
        meter=detector_meter,
    )
    detector.schedule_all(t0=0.0)

    cumulative_confirmed: set[tuple[int, int, int]] = set()
    for step in range(1, steps + 1):
        t = round(step * dt, 5)
        new_conf = detector.advance(t)
        for e in new_conf:
            cumulative_confirmed.add(e)

    stats = detector.stats()
    confirmed_count = stats["confirmed"]
    pct_failed = (confirmed_count / total_edges) * 100.0 if total_edges > 0 else 0.0
    false_alarms = stats["false_alarms"]
    fail_time_evals = stats["fail_time_evals"]
    det_dist_comps = detector_meter.count

    # 2. Naive Periodic Recertification Baseline
    # At each time step, baseline updates all node coordinates and recertifies all sources
    baseline_meter = CostMeter()
    baseline_store = CertificateStore()

    for step in range(1, steps + 1):
        t = round(step * dt, 5)
        pos_t = model.positions(t)
        # Non-versioning vector update to timestamp t
        for nid in range(n):
            index._store.set_vector(nid, pos_t[nid])
        for lvl in range(index.max_level() + 1):
            for u in index.ids():
                if index.node_level(u) >= lvl:
                    baseline_store.certify_source(index, u, lvl, baseline_meter)

    base_dist_comps = baseline_meter.count
    savings_factor = base_dist_comps / det_dist_comps if det_dist_comps > 0 else float("inf")

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
        "det_dist_comps": det_dist_comps,
        "base_dist_comps": base_dist_comps,
        "savings_factor": savings_factor,
    }


def main() -> None:
    """Run detector measurement across d in {2, 8} and print comparison table."""
    print("=" * 80)
    print("  KINETIC DETECTOR MEASUREMENT (n=200, M=8, LinearDrift, 10 steps)")
    print("=" * 80)

    results = []
    for d in [2, 8]:
        res = measure_dimension(d=d)
        results.append(res)

    print("\n| Metric | d = 2 | d = 8 |")
    print("| :--- | :---: | :---: |")
    r0, r1 = results[0], results[1]
    print(f"| Total Edges (E) | {r0['total_edges']} | {r1['total_edges']} |")
    print(
        f"| Confirmed Failures | {r0['confirmed']} ({r0['pct_failed']:.1f}%) | "
        f"{r1['confirmed']} ({r1['pct_failed']:.1f}%) |"
    )
    print(f"| False Alarms | {r0['false_alarms']} | {r1['false_alarms']} |")
    print(f"| Boundary Cases | {r0['boundary_cases']} | {r1['boundary_cases']} |")
    print(f"| Pure-Math Evaluations (evals) | {r0['fail_time_evals']} | {r1['fail_time_evals']} |")
    print(
        f"| Confirmation Distance Comps (Detector) | {r0['det_dist_comps']} | "
        f"{r1['det_dist_comps']} |"
    )
    print(
        f"| Baseline Distance Comps (Periodic Full) | {r0['base_dist_comps']} | "
        f"{r1['base_dist_comps']} |"
    )
    print(
        f"| Distance Cost Reduction (Savings Factor) | {r0['savings_factor']:.1f}x | "
        f"{r1['savings_factor']:.1f}x |"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
