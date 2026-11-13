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
        "no-repair control",
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
        assert count_from_scratch_violations(idx) == 0, (
            "Precondition failed: pre-move index must have 0 violations"
        )

        cat_dist: dict[str, int] = {
            "search": 0,
            "planning": 0,
            "store": 0,
            "detector": 0,
            "reverse_link": 0,
        }
        repaired_nodes_set: set[int] = set()

        if method == "no-repair control":
            # Apply movements without running repair or maintaining store
            for node_id in moved_nodes:
                idx.move(node_id, perturbations[node_id])
        elif method == "ours":
            # Apply movements and maintain CertificateStore
            for node_id in moved_nodes:
                new_v = perturbations[node_id]
                idx.move(node_id, new_v)
                m_before = meter_store.count
                store.on_move(idx, node_id, meter_store)
                cat_dist["store"] += meter_store.count - m_before

            for node_id in moved_nodes:
                res = repair_loop(idx, store, moved_node=node_id, budget=None)
                cat_dist["search"] += res.budget_used.get("search", 0)
                cat_dist["planning"] += res.budget_used.get("planning", 0)
                cat_dist["store"] += res.budget_used.get("store", 0)
                cat_dist["detector"] += res.budget_used.get("detector", 0)
                cat_dist["reverse_link"] += res.budget_used.get("reverse_link", 0)
                if res.edges_repaired > 0:
                    repaired_nodes_set.add(node_id)
        elif method == "our emulation of hnswlib-update-style":
            # Apply movements directly to index; real hnswlib has no CertificateStore
            for node_id in moved_nodes:
                idx.move(node_id, perturbations[node_id])

            meter_plan = CostMeter()
            for node_id in moved_nodes:
                plan = native_plan(idx, moved=node_id, meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    apply_plan(idx, plan, store=None, meter=None)
        elif method == "our emulation of MN-RU-style (beta)":
            # Apply movements directly to index; real MN-RU has no CertificateStore
            for node_id in moved_nodes:
                idx.move(node_id, perturbations[node_id])

            meter_plan = CostMeter()
            for node_id in moved_nodes:
                plan = mnru_plan(idx, moved=node_id, alpha=1.0, scope="one_hop", meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    apply_plan(idx, plan, store=None, meter=None)

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

    drift = LinearDrift(X, seed=seed, speed=1.5)
    pos_t = drift.positions(t_step).astype(np.float32)
    exact_sets = brute_force_knn(pos_t, queries, k=10)

    # First discover confirmed failure sources at t_step using reference detector
    idx_ref = HNSWIndex(dim=d, m=m, seed=seed)
    idx_ref.add(np.arange(n, dtype=np.int64), X)
    store_ref = CertificateStore()
    meter_ref = CostMeter()
    store_ref.build(idx_ref, meter_ref)
    _settle_hnsw(idx_ref, store_ref, meter_ref)
    assert count_from_scratch_violations(idx_ref) == 0, (
        "Precondition failed: pre-move index must have 0 violations"
    )

    heap_ref = KineticEventHeap(version_of=idx_ref.version)
    detector_ref = KineticDetector(
        index=idx_ref,
        store=store_ref,
        heap=heap_ref,
        trajectory_of=drift.trajectory,
        horizon=10.0,
        sync_vector=make_hnsw_sync(idx_ref),
        meter=CostMeter(),
    )
    detector_ref.schedule_all(t0=0.0)
    ref_failures = detector_ref.advance(t_step)
    failure_sources = sorted({u for (u, _, _) in ref_failures})

    methods = [
        "no-repair control",
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

        sync_fn = make_hnsw_sync(idx)

        cat_dist: dict[str, int] = {
            "search": 0,
            "planning": 0,
            "store": 0,
            "detector": 0,
            "reverse_link": 0,
        }
        repaired_nodes_set: set[int] = set()

        if method == "no-repair control":
            # Sync all points to t_step without running repair
            for u in range(n):
                sync_fn(u, pos_t[u])
        elif method == "ours":
            heap = KineticEventHeap(version_of=idx.version)
            meter_det = CostMeter()
            detector = KineticDetector(
                index=idx,
                store=store,
                heap=heap,
                trajectory_of=drift.trajectory,
                horizon=10.0,
                sync_vector=sync_fn,
                meter=meter_det,
            )
            detector.schedule_all(t0=0.0)
            confirmed_failures = detector.advance(t_step)
            cat_dist["detector"] = meter_det.count

            res = repair_loop(idx, store, detector=detector, failures=confirmed_failures, t=t_step)
            cat_dist["search"] += res.budget_used["search"]
            cat_dist["planning"] += res.budget_used["planning"]
            cat_dist["store"] += res.budget_used["store"]
            cat_dist["detector"] += res.budget_used["detector"]
            cat_dist["reverse_link"] += res.budget_used.get("reverse_link", 0)
            repaired_nodes_set.update({u for (u, _, _) in confirmed_failures})

            # Sync all remaining nodes to t_step for ground truth evaluation
            for u in range(n):
                sync_fn(u, pos_t[u])
        elif method == "our emulation of hnswlib-update-style":
            # Sync all points to t_step; baselines repair failure sources
            for u in range(n):
                sync_fn(u, pos_t[u])

            meter_plan = CostMeter()
            for node_id in failure_sources:
                plan = native_plan(idx, moved=node_id, meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    apply_plan(idx, plan, store=None, meter=None)
        elif method == "our emulation of MN-RU-style (beta)":
            # Sync all points to t_step; baselines repair failure sources
            for u in range(n):
                sync_fn(u, pos_t[u])

            meter_plan = CostMeter()
            for node_id in failure_sources:
                plan = mnru_plan(idx, moved=node_id, alpha=1.0, scope="one_hop", meter=meter_plan)
                cat_dist["planning"] += plan.distance_computations
                if plan.rewrites:
                    for r in plan.rewrites:
                        repaired_nodes_set.add(r.node)
                    apply_plan(idx, plan, store=None, meter=None)

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
    """Print formatted markdown table with caption and notes."""
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
    ctrl_recall = 0.0
    for r in rows:
        if r["method"] == "no-repair control":
            ctrl_recall = r["recall10"]
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

    print("\n*Definitions and accounting:*")
    print(
        "- **Total Dist**: Strictly algorithm cost (search + planning + store + detector) "
        "incurred during repair. Baselines have no CertificateStore and pay only planning cost. "
        "Measurement overhead (from-scratch store validation and recall evaluation) is excluded."
    )
    print(
        "- **Violations**: Global count of VIOLATED directed edges in a CertificateStore "
        "built from scratch on the index at measurement time "
        "(pre-move baseline is 0 across all configs)."
    )
    print("- **Repaired**: Number of unique nodes modified or rewritten by the repair algorithm.")
    if ctrl_recall >= 0.999:
        print(
            "- *Note on recall*: The no-repair control achieves recall ~1.0000 because small "
            "displacements in a dense graph preserve global reachability; repair actively restores "
            "the empty-ellipse Delaunay/certificate invariant to prevent structural degradation "
            "over extended trajectories."
        )


def run_reverse_link_diagnostic_sweep(
    n: int = 500,
    d: int = 8,
    m: int = 8,
    scales: tuple[float, ...] = (0.2, 1.0, 3.0, 12.0, 30.0),
    seeds: tuple[int, ...] = (42, 43, 44),
    num_queries: int = 100,
) -> list[dict[str, Any]]:
    """Run diagnostic sweep for moved node isolation and reverse-link repair.

    Evaluates:
    - Control (no repair)
    - Ours (before: reverse-links disabled)
    - Ours (after: reverse-links active)
    - Hnswlib-style baseline
    - MN-RU-style baseline
    - Delete+Reinsert upper bound
    """
    all_results: list[dict[str, Any]] = []
    num_moves = max(5, int(0.05 * n))

    variants = [
        "control",
        "ours_before",
        "ours_after",
        "hnswlib",
        "mnru",
        "reinsert",
    ]

    for scale in scales:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((n, d)).astype(np.float32)
            scale_nn = mean_nn_distance(X)
            disp = float(scale * scale_nn)

            moved_nodes = rng.choice(n, size=num_moves, replace=False)

            perturbations: dict[int, np.ndarray] = {}
            for node in moved_nodes:
                direction = rng.standard_normal(d).astype(np.float32)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-9:
                    direction /= norm
                perturbations[int(node)] = (X[node] + direction * disp).astype(np.float32)

            global_queries = rng.standard_normal((num_queries, d)).astype(np.float32)

            for var in variants:
                idx = HNSWIndex(dim=d, m=m, seed=seed)
                idx.add(np.arange(n, dtype=np.int64), X)
                store = CertificateStore()
                m_settle = CostMeter()
                store.build(idx, m_settle)
                _settle_hnsw(idx, store, m_settle)

                cat_dist: dict[str, int] = {
                    "search": 0,
                    "planning": 0,
                    "store": 0,
                    "detector": 0,
                    "reverse_link": 0,
                }

                if var == "control":
                    for u in moved_nodes:
                        idx.move(u, perturbations[u])
                elif var == "ours_before":
                    m_store = CostMeter()
                    for u in moved_nodes:
                        idx.move(u, perturbations[u])
                        m_b = m_store.count
                        store.on_move(idx, u, m_store)
                        cat_dist["store"] += m_store.count - m_b
                    for u in moved_nodes:
                        res = repair_loop(idx, store, moved_node=u, reverse_links=False)
                        for k in ["search", "planning", "store", "detector", "reverse_link"]:
                            cat_dist[k] += res.budget_used.get(k, 0)
                elif var == "ours_after":
                    m_store = CostMeter()
                    for u in moved_nodes:
                        idx.move(u, perturbations[u])
                        m_b = m_store.count
                        store.on_move(idx, u, m_store)
                        cat_dist["store"] += m_store.count - m_b
                    for u in moved_nodes:
                        res = repair_loop(idx, store, moved_node=u, reverse_links=True)
                        for k in ["search", "planning", "store", "detector", "reverse_link"]:
                            cat_dist[k] += res.budget_used.get(k, 0)
                elif var == "hnswlib":
                    for u in moved_nodes:
                        idx.move(u, perturbations[u])
                    m_plan = CostMeter()
                    for u in moved_nodes:
                        plan = native_plan(idx, moved=u, meter=m_plan)
                        cat_dist["planning"] += plan.distance_computations
                        apply_plan(idx, plan)
                elif var == "mnru":
                    for u in moved_nodes:
                        idx.move(u, perturbations[u])
                    m_plan = CostMeter()
                    for u in moved_nodes:
                        plan = mnru_plan(idx, moved=u, alpha=1.0, scope="one_hop", meter=m_plan)
                        cat_dist["planning"] += plan.distance_computations
                        apply_plan(idx, plan)
                elif var == "reinsert":
                    c_before = idx.distance_computations
                    for u in moved_nodes:
                        idx.delete(u)
                        idx.add(
                            np.array([u], dtype=np.int64),
                            np.array([perturbations[u]], dtype=np.float32),
                        )
                    cat_dist["planning"] = idx.distance_computations - c_before

                X_final = np.array([idx.vector(i) for i in range(n)], dtype=np.float32)

                in_degs = [len(idx.in_neighbors(u, 0)) for u in moved_nodes]
                mean_in_deg = float(np.mean(in_degs))

                self_finds = []
                for u in moved_nodes:
                    ids, _ = idx.search(idx.vector(u), k=1, ef=64)
                    self_finds.append(1.0 if len(ids) > 0 and ids[0] == u else 0.0)
                self_find = float(np.mean(self_finds))

                has_in_from_10nn = []
                for u in moved_nodes:
                    diffs = X_final - idx.vector(u)
                    sq_dists = np.sum(diffs**2, axis=1)
                    sq_dists[u] = np.inf
                    true_10 = np.argpartition(sq_dists, 10)[:10]
                    has_edge = any(u in idx.neighbors(int(v), 0) for v in true_10)
                    has_in_from_10nn.append(1.0 if has_edge else 0.0)
                frac_10nn = float(np.mean(has_in_from_10nn))

                moved_queries = np.array([idx.vector(u) for u in moved_nodes], dtype=np.float32)
                exact_moved = brute_force_knn(X_final, moved_queries, k=10)
                rec_moved = evaluate_recall(idx, moved_queries, exact_moved, k=10)

                exact_global = brute_force_knn(X_final, global_queries, k=10)
                rec_global = evaluate_recall(idx, global_queries, exact_global, k=10)

                total_cost = sum(cat_dist.values())

                row = {
                    "scale": scale,
                    "seed": seed,
                    "var": var,
                    "disp": disp,
                    "in_deg": mean_in_deg,
                    "self_find": self_find,
                    "frac_10nn": frac_10nn,
                    "rec_moved": rec_moved,
                    "rec_global": rec_global,
                    "search_cost": cat_dist["search"],
                    "plan_cost": cat_dist["planning"],
                    "store_cost": cat_dist["store"],
                    "det_cost": cat_dist["detector"],
                    "rev_cost": cat_dist["reverse_link"],
                    "total_cost": total_cost,
                }
                all_results.append(row)

    print("\n" + "=" * 115)
    print("REVERSE-LINK REPAIR DIAGNOSTIC SWEEP (n=500, M=8, d=8, 25 nodes moved)")
    print("=" * 115)
    print(
        f"{'Scale':<7} | {'Variant':<14} | {'In-Deg':<6} | {'Self-Find':<9} | {'10-NN In':<8} | "
        f"{'Rec (Moved)':<11} | {'Rec (Glob)':<10} | {'Search':<7} | {'Plan':<7} | "
        f"{'Store':<7} | {'Rev':<5} | {'Total':<7}"
    )
    print("-" * 115)

    for scale in scales:
        for var in variants:
            rows = [r for r in all_results if r["scale"] == scale and r["var"] == var]
            in_d = float(np.mean([r["in_deg"] for r in rows]))
            sf = float(np.mean([r["self_find"] for r in rows]))
            nn10 = float(np.mean([r["frac_10nn"] for r in rows]))
            rm = float(np.mean([r["rec_moved"] for r in rows]))
            rg = float(np.mean([r["rec_global"] for r in rows]))
            sc = float(np.mean([r["search_cost"] for r in rows]))
            pc = float(np.mean([r["plan_cost"] for r in rows]))
            stc = float(np.mean([r["store_cost"] for r in rows]))
            rc = float(np.mean([r["rev_cost"] for r in rows]))
            tc = float(np.mean([r["total_cost"] for r in rows]))

            print(
                f"{scale:<7.1f} | {var:<14} | {in_d:<6.2f} | {sf:<9.3f} | {nn10:<8.3f} | "
                f"{rm:<11.3f} | {rg:<10.3f} | {sc:<7.0f} | {pc:<7.0f} | "
                f"{stc:<7.0f} | {rc:<5.0f} | {tc:<7.0f}"
            )
        print("-" * 115)

    return all_results


def main() -> None:
    """Run full measurement suite across configs and scenarios."""
    import sys

    if "--reverse-link-sweep" in sys.argv:
        run_reverse_link_diagnostic_sweep()
        return

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
