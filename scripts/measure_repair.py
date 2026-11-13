"""Measurement script comparing K-HNSW iterative repair loop vs baselines.

Evaluates on HNSWIndex (n in {200, 1000}, M=8, d in {2, 8}) under:
1. Discrete moves: perturbing m random vectors by displacement scale in {0.01, 0.05, 0.2}
   of the mean nearest-neighbor distance.
2. Continuous LinearDrift: points moving along constant velocity trajectories.

Methods compared:
- "ours" (K-HNSW iterative repair loop)
- "our emulation of hnswlib-update-style" (native_plan + apply_plan)
- "our emulation of MN-RU-style (beta)" (mnru_plan with alpha=1.0, scope="one_hop" + apply_plan)

Reports:
- Recall@10 against brute force on 100 test queries after movement and repair
- Distance computations partitioned by category: search, planning, store, detector, total
- Number of VIOLATED edges remaining in a from-scratch CertificateStore
- Number of nodes repaired
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.loop import repair_loop
from khnsw.core.repair.plan import khnsw_plan, mnru_plan, native_plan
from khnsw.core.scheduling.event_heap import KineticEventHeap


def make_hnsw_sync(index: HNSWIndex) -> Callable[[int, np.ndarray], None]:
    """Callback syncing HNSWIndex vectors to float32."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def _settle_hnsw(index: HNSWIndex, store: CertificateStore, meter: CostMeter) -> None:
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
    index: HNSWIndex, queries: np.ndarray, exact_sets: list[set[int]], k: int = 10
) -> float:
    """Calculate mean recall@k across all query points."""
    recalls: list[float] = []
    for q, exact in zip(queries, exact_sets, strict=False):
        ids, _ = index.search(q, k=k, ef=64)
        found = set(int(nid) for nid in ids[:k])
        recall = len(found & exact) / float(k)
        recalls.append(recall)
    return float(np.mean(recalls))


def count_from_scratch_violations(index: HNSWIndex) -> int:
    """Construct a fresh CertificateStore from scratch and count total VIOLATED edges."""
    fresh_store = CertificateStore()
    fresh_meter = CostMeter()
    fresh_store.build(index, fresh_meter)
    violations = 0
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                if fresh_store._edge_status.get((u, v, lvl)) == EdgeStatus.VIOLATED.value:
                    violations += 1
    return violations


def run_discrete_benchmark(
    n: int,
    d: int,
    scale_factor: float,
    m: int = 8,
    seed: int = 42,
    num_queries: int = 100,
) -> list[dict[str, Any]]:
    """Run discrete move benchmark comparing ours vs baselines."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((num_queries, d)).astype(np.float32)

    # Number of moved nodes
    num_moves = max(5, int(0.05 * n))
    moved_nodes = rng.choice(n, size=num_moves, replace=False)

    scale_nn = mean_nn_distance(X)
    displacement = float(scale_factor * scale_nn)

    # Compute perturbed vectors for each moved node
    perturbations: dict[int, np.ndarray] = {}
    for node in moved_nodes:
        direction = rng.standard_normal(d).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction /= norm
        perturbations[int(node)] = (X[node] + direction * displacement).astype(np.float32)

    methods = [
        "ours",
        "our emulation of hnswlib-update-style",
        "our emulation of MN-RU-style (beta)",
    ]

    results: list[dict[str, Any]] = []

    for method in methods:
        # Build fresh index for each method from identical initial points
        idx = HNSWIndex(dim=d, m=m, seed=seed)
        idx.add(np.arange(n, dtype=np.int64), X)

        meter_store = CostMeter()
        store = CertificateStore()
        store.build(idx, meter_store)
        _settle_hnsw(idx, store, meter_store)

        cat_dist: dict[str, int] = {"search": 0, "planning": 0, "store": 0, "detector": 0}
        repaired_nodes_set: set[int] = set()

        # Apply movements
        for node_id in moved_nodes:
            new_v = perturbations[node_id]
            idx.move(node_id, new_v)
            m_before = meter_store.count
            store.on_move(idx, node_id, meter_store)
            cat_dist["store"] += meter_store.count - m_before

        # Execute repairs
        if method == "ours":
            for node_id in moved_nodes:
                res = repair_loop(idx, store, moved_node=node_id, budget=None)
                cat_dist["search"] += res.budget_used["search"]
                cat_dist["planning"] += res.budget_used["planning"]
                cat_dist["store"] += res.budget_used["store"]
                cat_dist["detector"] += res.budget_used["detector"]
                if res.edges_repaired > 0:
                    repaired_nodes_set.add(node_id)
        elif method == "our emulation of hnswlib-update-style":
            meter_plan = CostMeter()
            for node_id in moved_nodes:
                plan = native_plan(idx, moved=node_id, meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    m_before = meter_store.count
                    apply_plan(idx, plan, store=store, meter=meter_store)
                    cat_dist["store"] += meter_store.count - m_before
        elif method == "our emulation of MN-RU-style (beta)":
            meter_plan = CostMeter()
            for node_id in moved_nodes:
                plan = mnru_plan(idx, moved=node_id, alpha=1.0, scope="one_hop", meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    m_before = meter_store.count
                    apply_plan(idx, plan, store=store, meter=meter_store)
                    cat_dist["store"] += meter_store.count - m_before

        # Ground truth coordinates after movements
        X_final = np.array([idx.vector(i) for i in range(n)], dtype=np.float32)
        exact_sets = brute_force_knn(X_final, queries, k=10)
        rec10 = evaluate_recall(idx, queries, exact_sets, k=10)
        violations = count_from_scratch_violations(idx)
        total_dist = sum(cat_dist.values())

        results.append(
            {
                "scenario": "discrete",
                "n": n,
                "d": d,
                "scale": scale_factor,
                "displacement": displacement,
                "method": method,
                "recall10": rec10,
                "search_dist": cat_dist["search"],
                "plan_dist": cat_dist["planning"],
                "store_dist": cat_dist["store"],
                "det_dist": cat_dist["detector"],
                "total_dist": total_dist,
                "violations": violations,
                "nodes_repaired": len(repaired_nodes_set),
            }
        )

    return results


def run_linear_drift_benchmark(
    n: int,
    d: int,
    m: int = 8,
    seed: int = 42,
    num_queries: int = 100,
    t_step: float = 0.10,
) -> list[dict[str, Any]]:
    """Run LinearDrift benchmark comparing ours vs baselines."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((num_queries, d)).astype(np.float32)

    methods = [
        "ours",
        "our emulation of hnswlib-update-style",
        "our emulation of MN-RU-style (beta)",
    ]

    results: list[dict[str, Any]] = []

    for method in methods:
        idx = HNSWIndex(dim=d, m=m, seed=seed)
        idx.add(np.arange(n, dtype=np.int64), X)

        meter_store = CostMeter()
        store = CertificateStore()
        store.build(idx, meter_store)
        _settle_hnsw(idx, store, meter_store)

        drift = LinearDrift(X, seed=seed, speed=1.5)
        heap = KineticEventHeap(version_of=idx.version)
        meter_det = CostMeter()
        detector = KineticDetector(
            index=idx,
            store=store,
            heap=heap,
            trajectory_of=drift.trajectory,
            horizon=10.0,
            sync_vector=make_hnsw_sync(idx),
            meter=meter_det,
        )
        detector.schedule_all(t0=0.0)

        # Advance to t_step
        confirmed_failures = detector.advance(t_step)

        cat_dist: dict[str, int] = {
            "search": 0,
            "planning": 0,
            "store": 0,
            "detector": meter_det.count,
        }
        repaired_nodes_set: set[int] = set()

        if method == "ours":
            res = repair_loop(idx, store, detector=detector, failures=confirmed_failures, t=t_step)
            cat_dist["search"] += res.budget_used["search"]
            cat_dist["planning"] += res.budget_used["planning"]
            cat_dist["store"] += res.budget_used["store"]
            cat_dist["detector"] += res.budget_used["detector"]
            repaired_nodes_set.update({u for (u, _, _) in confirmed_failures})
        elif method == "our emulation of hnswlib-update-style":
            meter_plan = CostMeter()
            # In update-style emulation, sources of confirmed failures are repaired
            failure_sources = sorted({u for (u, _, _) in confirmed_failures})
            for node_id in failure_sources:
                plan = native_plan(idx, moved=node_id, meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    m_before = meter_store.count
                    apply_plan(idx, plan, store=store, meter=meter_store)
                    cat_dist["store"] += meter_store.count - m_before
        elif method == "our emulation of MN-RU-style (beta)":
            meter_plan = CostMeter()
            failure_sources = sorted({u for (u, _, _) in confirmed_failures})
            for node_id in failure_sources:
                plan = mnru_plan(idx, moved=node_id, alpha=1.0, scope="one_hop", meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    m_before = meter_store.count
                    apply_plan(idx, plan, store=store, meter=meter_store)
                    cat_dist["store"] += meter_store.count - m_before

        # Sync all points to t_step to evaluate ground truth recall
        pos_t = drift.positions(t_step).astype(np.float32)
        exact_sets = brute_force_knn(pos_t, queries, k=10)
        rec10 = evaluate_recall(idx, queries, exact_sets, k=10)
        violations = count_from_scratch_violations(idx)
        total_dist = sum(cat_dist.values())

        results.append(
            {
                "scenario": "linear_drift",
                "n": n,
                "d": d,
                "scale": 1.0,
                "displacement": 0.0,
                "method": method,
                "recall10": rec10,
                "search_dist": cat_dist["search"],
                "plan_dist": cat_dist["planning"],
                "store_dist": cat_dist["store"],
                "det_dist": cat_dist["detector"],
                "total_dist": total_dist,
                "violations": violations,
                "nodes_repaired": len(repaired_nodes_set),
            }
        )

    return results


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Print formatted markdown table."""
    print(f"\n### {title}\n")
    headers = [
        ("Method", 36),
        ("Recall@10", 9),
        ("Search", 8),
        ("Planning", 9),
        ("Store", 8),
        ("Detector", 8),
        ("Total Dist", 10),
        ("Violations", 10),
        ("Repaired", 8),
    ]
    hdr_str = " | ".join(f"{h:<{w}}" for h, w in headers)
    sep_str = "|:".join("-" * w + ":|" for _, w in headers)
    print(f"| {hdr_str} |")
    print(f"|:{sep_str}")
    for r in rows:
        vals = [
            f"{r['method']:<36}",
            f"{r['recall10']:<9.4f}",
            f"{r['search_dist']:<8}",
            f"{r['plan_dist']:<9}",
            f"{r['store_dist']:<8}",
            f"{r['det_dist']:<8}",
            f"{r['total_dist']:<10}",
            f"{r['violations']:<10}",
            f"{r['nodes_repaired']:<8}",
        ]
        print(f"| {' | '.join(vals)} |")


def main() -> None:
    """Run full measurement suite across configs and scenarios."""
    print("=" * 100)
    print("K-HNSW REPAIR LOOP BENCHMARK AND EMPIRICAL EVALUATION")
    print("=" * 100)

    # 1. Discrete Moves Benchmark (n in {200, 1000}, d in {2, 8}, displacement in {0.01, 0.05, 0.2})
    for n in [200, 1000]:
        for d in [2, 8]:
            for scale in [0.01, 0.05, 0.2]:
                seed = 42 + d
                rows = run_discrete_benchmark(n=n, d=d, scale_factor=scale, m=8, seed=seed)
                disp = rows[0]["displacement"]
                title = (
                    f"Config: Discrete Moves (n={n}, M=8, d={d}, seed={seed}, "
                    f"displacement_scale={scale} [disp={disp:.4f}])"
                )
                print_table(title, rows)

    # 2. LinearDrift Benchmark (n in {200, 1000}, d in {2, 8})
    for n in [200, 1000]:
        for d in [2, 8]:
            seed = 42 + d
            rows = run_linear_drift_benchmark(n=n, d=d, m=8, seed=seed)
            title = f"Config: LinearDrift (n={n}, M=8, d={d}, seed={seed}, speed=1.5, t=0.10)"
            print_table(title, rows)


if __name__ == "__main__":
    main()
