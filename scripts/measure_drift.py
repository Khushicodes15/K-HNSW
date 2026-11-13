"""Comprehensive drift benchmark script reconciling drift-moved node handling.

Measures HNSWIndex (n=500, M=8, d=8) under continuous kinetic drift (LinearDrift,
T=0.30 in 10 steps, seeds 42, 43, 44).

Covers:
- Part A: 10% drift sweep (10x, 30x, 100x; ef in {16, 32}) comparing no_repair,
  ours-before, and ours-after.
- Part C: Ablation on 10% drift comparing (1) no_repair, (2) ours-before,
  (3) ours-after, and (4) ours-after with certificate-driven edge drops disabled
  (rebuild + reverse links only, violating edges kept).
- Part D: Miss classification of drifting query errors under ours-after at ef=16,
  and 5 concrete miss examples.
- Part E: 100% drift at 30x speed evaluating global recall on N(0, I) queries vs
  queries sampled near current drifting node coordinates, plus single-rebuild vs
  single-insert cost.

Outputs a complete JSON file containing all raw measurements and summary statistics.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

import khnsw.core.repair.loop as loop_mod
from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.loop import repair_loop
from khnsw.core.repair.plan import RepairPlan
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.search_layer import search_layer


def make_hnsw_sync(index: HNSWIndex) -> Callable[[int, np.ndarray], None]:
    """Callback syncing HNSWIndex vectors to float32."""

    def _sync(node_id: int, pos: np.ndarray) -> None:
        index._store.set_vector(node_id, pos.astype(np.float32))

    return _sync


def settle_hnsw(index: HNSWIndex, store: CertificateStore) -> None:
    """Run initial repair loop to eliminate construction-time violations."""
    repair_loop(index, store)


def brute_force_knn(X: np.ndarray, queries: np.ndarray, k: int = 10) -> list[set[int]]:
    """Compute exact k-nearest neighbors via brute-force pairwise Euclidean distance."""
    exact_neighbors: list[set[int]] = []
    for q in queries:
        diffs = X - q
        dists = np.sum(diffs**2, axis=1)
        top_k = np.argpartition(dists, k)[:k]
        exact_neighbors.append({int(idx) for idx in top_k})
    return exact_neighbors


def evaluate_recall(
    index: HNSWIndex, queries: np.ndarray, exact_sets: list[set[int]], k: int = 10, ef: int = 16
) -> float:
    """Calculate mean recall@k across all query points."""
    recalls: list[float] = []
    for q, exact in zip(queries, exact_sets, strict=False):
        ids, _ = index.search(q, k=k, ef=ef)
        found = {int(nid) for nid in ids[:k]}
        recalls.append(len(found & exact) / float(k))
    return float(np.mean(recalls))


def evaluate_reachability(index: HNSWIndex, target: int) -> bool:
    """Check if target node is reachable from entry point in index at level 0."""
    ep = index.entry_point()
    if ep is None:
        return False
    if ep == target:
        return True
    visited = {ep}
    queue = [ep]
    while queue:
        curr = queue.pop(0)
        for lvl in range(index.node_level(curr) + 1):
            for nbr in index.neighbors(curr, lvl):
                if nbr == target:
                    return True
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
    return False


def run_benchmark() -> dict[str, Any]:
    """Execute complete drift benchmark across Part A, C, D, and E."""
    n = 500
    m = 8
    d = 8
    t_final = 0.30
    steps = 10
    dt = t_final / steps
    seeds = [42, 43, 44]
    speeds = [10.0, 30.0, 100.0]
    conditions = ["no_repair", "ours_before", "ours_after", "ours_after_no_drop"]

    results: dict[str, Any] = {
        "sweep_10pct": [],
        "miss_classification": {},
        "miss_examples_d": [],
        "all_drift_30x": [],
        "insert_cost_scale": {},
    }

    # ---------------------------------------------------------
    # PART A & C: 10% Drift Sweep & Ablation Across 4 Arms
    # ---------------------------------------------------------
    for speed in speeds:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X0 = rng.standard_normal((n, d)).astype(np.float32)

            base_index = HNSWIndex(dim=d, m=m, seed=seed)
            base_index.add(np.arange(n, dtype=np.int64), X0)
            settled_store = CertificateStore()
            settled_store.build(base_index, CostMeter())
            settle_hnsw(base_index, settled_store)

            initial_adj = {
                (u, lvl): set(base_index.neighbors(u, lvl))
                for u in base_index.ids()
                for lvl in range(base_index.node_level(u) + 1)
            }

            drift = LinearDrift(X0, seed=seed, speed=speed)
            drifting_nodes = sorted(rng.choice(n, size=int(0.10 * n), replace=False).tolist())
            static_nodes = [i for i in range(n) if i not in drifting_nodes]
            drift._velocities[static_nodes] = 0.0

            pos_t_final = drift.positions(t_final).astype(np.float32)

            # Ground truth queries
            # 1. 100 global queries from standard normal
            rng_q = np.random.default_rng(seed + 1000)
            q_global = rng_q.standard_normal((100, d)).astype(np.float32)
            exact_global = brute_force_knn(pos_t_final, q_global, k=10)

            # 2. 50 queries at drifting nodes' current positions
            q_drift = pos_t_final[drifting_nodes]
            exact_drift = brute_force_knn(pos_t_final, q_drift, k=10)

            # 3. 100 queries at static nodes' current positions
            rng_s = np.random.default_rng(seed + 500)
            eval_static_nodes = sorted(rng_s.choice(static_nodes, size=100, replace=False).tolist())
            q_static = pos_t_final[eval_static_nodes]
            exact_static = brute_force_knn(pos_t_final, q_static, k=10)

            for condition in conditions:
                idx = copy.deepcopy(base_index)
                cost = {
                    "detector": 0,
                    "search": 0,
                    "planning": 0,
                    "store": 0,
                    "reverse_link": 0,
                    "total": 0,
                }
                moved_in_drift_total = 0

                if condition == "no_repair":
                    for nid in range(n):
                        idx._store.set_vector(nid, pos_t_final[nid])

                elif condition == "ours_before":
                    store = copy.deepcopy(settled_store)
                    heap = KineticEventHeap(version_of=idx.version)
                    det_meter = CostMeter()
                    detector = KineticDetector(
                        index=idx,
                        store=store,
                        heap=heap,
                        trajectory_of=drift.trajectory,
                        meter=det_meter,
                        horizon=0.50,
                        sync_vector=make_hnsw_sync(idx),
                    )
                    detector.schedule_all(t0=0.0)

                    for step in range(1, steps + 1):
                        t_curr = round(step * dt, 5)
                        pos_step = drift.positions(t_curr)
                        for nid in range(n):
                            idx._store.set_vector(nid, pos_step[nid].astype(np.float32))
                        detector_failures = detector.advance(t_curr)
                        res = repair_loop(
                            index=idx,
                            store=store,
                            detector=detector,
                            moved_node=None,
                            failures=detector_failures,
                            t=t_curr,
                            search_layer=search_layer,
                            budget=1e9,
                            max_rounds=10,
                            ef=64,
                        )
                        for k in ["search", "planning", "store", "reverse_link"]:
                            cost[k] += res.budget_used.get(k, 0)
                    cost["detector"] = det_meter.count
                    cost["total"] = cost["detector"] + sum(
                        cost[k] for k in ["search", "planning", "store", "reverse_link"]
                    )

                elif condition == "ours_after":
                    store = copy.deepcopy(settled_store)
                    heap = KineticEventHeap(version_of=idx.version)
                    det_meter = CostMeter()
                    detector = KineticDetector(
                        index=idx,
                        store=store,
                        heap=heap,
                        trajectory_of=drift.trajectory,
                        meter=det_meter,
                        horizon=0.50,
                        sync_vector=make_hnsw_sync(idx),
                    )
                    detector.schedule_all(t0=0.0)

                    engine = KHNSWRepairEngine(
                        store=store,
                        detector=detector,
                        time_source=0.0,
                        search_layer=search_layer,
                        ef=64,
                    )

                    for step in range(1, steps + 1):
                        t_curr = round(step * dt, 5)
                        pos_step = drift.positions(t_curr)
                        for nid in range(n):
                            idx._store.set_vector(nid, pos_step[nid].astype(np.float32))
                        engine._time_source = t_curr
                        res_engine = engine.run_repairs(idx, budget=1e9, sink=ListSink())
                        for k in ["search", "planning", "store", "reverse_link"]:
                            cost[k] += res_engine["budget_used"].get(k, 0)
                        moved_in_drift_total += res_engine.get("moved_in_drift", 0)
                    cost["detector"] = det_meter.count
                    cost["total"] = cost["detector"] + sum(
                        cost[k] for k in ["search", "planning", "store", "reverse_link"]
                    )

                elif condition == "ours_after_no_drop":
                    # Ablation arm: monkeypatch khnsw_plan to disable certificate-driven edge drops
                    store = copy.deepcopy(settled_store)
                    heap = KineticEventHeap(version_of=idx.version)
                    det_meter = CostMeter()
                    detector = KineticDetector(
                        index=idx,
                        store=store,
                        heap=heap,
                        trajectory_of=drift.trajectory,
                        meter=det_meter,
                        horizon=0.50,
                        sync_vector=make_hnsw_sync(idx),
                    )
                    detector.schedule_all(t0=0.0)

                    engine = KHNSWRepairEngine(
                        store=store,
                        detector=detector,
                        time_source=0.0,
                        search_layer=search_layer,
                        ef=64,
                    )

                    orig_khnsw_plan = loop_mod.khnsw_plan
                    try:
                        loop_mod.khnsw_plan = lambda index, violated, *a, **kw: RepairPlan(
                            "khnsw", None, (), 0
                        )
                        for step in range(1, steps + 1):
                            t_curr = round(step * dt, 5)
                            pos_step = drift.positions(t_curr)
                            for nid in range(n):
                                idx._store.set_vector(nid, pos_step[nid].astype(np.float32))
                            engine._time_source = t_curr
                            res_engine = engine.run_repairs(idx, budget=1e9, sink=ListSink())
                            for k in ["search", "planning", "store", "reverse_link"]:
                                cost[k] += res_engine["budget_used"].get(k, 0)
                            moved_in_drift_total += res_engine.get("moved_in_drift", 0)
                        cost["detector"] = det_meter.count
                        cost["total"] = cost["detector"] + sum(
                            cost[k] for k in ["search", "planning", "store", "reverse_link"]
                        )
                    finally:
                        loop_mod.khnsw_plan = orig_khnsw_plan

                # Violated edges from a fresh CertificateStore built on final state
                fresh_store = CertificateStore()
                fresh_meter = CostMeter()
                fresh_store.build(idx, fresh_meter)
                violated = sum(
                    1 for s in fresh_store._edge_status.values() if s == EdgeStatus.VIOLATED.value
                )

                # Self-findability
                self_find_count = sum(
                    1
                    for u in drifting_nodes
                    if len(idx.search(pos_t_final[u], k=1, ef=64)[0]) > 0
                    and idx.search(pos_t_final[u], k=1, ef=64)[0][0] == u
                )
                self_find = self_find_count / float(len(drifting_nodes))

                # Mean in-degree at level 0
                mean_in_deg = float(np.mean([len(idx.in_neighbors(u, 0)) for u in idx.ids()]))

                # Recalls
                rec16_global = evaluate_recall(idx, q_global, exact_global, k=10, ef=16)
                rec32_global = evaluate_recall(idx, q_global, exact_global, k=10, ef=32)
                rec16_drift = evaluate_recall(idx, q_drift, exact_drift, k=10, ef=16)
                rec32_drift = evaluate_recall(idx, q_drift, exact_drift, k=10, ef=32)
                rec16_static = evaluate_recall(idx, q_static, exact_static, k=10, ef=16)
                rec32_static = evaluate_recall(idx, q_static, exact_static, k=10, ef=32)

                record = {
                    "speed": speed,
                    "seed": seed,
                    "condition": condition,
                    "rec16_global": rec16_global,
                    "rec32_global": rec32_global,
                    "rec16_drift": rec16_drift,
                    "rec32_drift": rec32_drift,
                    "rec16_static": rec16_static,
                    "rec32_static": rec32_static,
                    "violated": violated,
                    "self_find": self_find,
                    "mean_in_deg": mean_in_deg,
                    "cost": cost,
                    "moved_in_drift": moved_in_drift_total,
                }
                results["sweep_10pct"].append(record)

    # ---------------------------------------------------------
    # PART D: Miss Classification and 5 Concrete Miss Examples
    # ---------------------------------------------------------
    miss_examples: list[dict[str, Any]] = []
    for speed in [10.0, 30.0]:
        cat_counts = {
            "a_unreachable": 0,
            "b_reverse_pruned": 0,
            "c_dropped_by_rebuild": 0,
            "d_other_beam_search": 0,
        }
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X0 = rng.standard_normal((n, d)).astype(np.float32)

            base_index = HNSWIndex(dim=d, m=m, seed=seed)
            base_index.add(np.arange(n, dtype=np.int64), X0)
            settled_store = CertificateStore()
            settled_store.build(base_index, CostMeter())
            settle_hnsw(base_index, settled_store)

            initial_adj = {
                (u, lvl): set(base_index.neighbors(u, lvl))
                for u in base_index.ids()
                for lvl in range(base_index.node_level(u) + 1)
            }

            drift = LinearDrift(X0, seed=seed, speed=speed)
            drifting_nodes = sorted(rng.choice(n, size=int(0.10 * n), replace=False).tolist())
            static_nodes = [i for i in range(n) if i not in drifting_nodes]
            drift._velocities[static_nodes] = 0.0

            pos_t_final = drift.positions(t_final).astype(np.float32)
            q_drift = pos_t_final[drifting_nodes]
            exact_drift = brute_force_knn(pos_t_final, q_drift, k=10)

            # Control
            idx_ctrl = copy.deepcopy(base_index)
            for nid in range(n):
                idx_ctrl._store.set_vector(nid, pos_t_final[nid])

            # Ours
            idx_ours = copy.deepcopy(base_index)
            store_ours = copy.deepcopy(settled_store)
            heap = KineticEventHeap(version_of=idx_ours.version)
            det = KineticDetector(
                index=idx_ours,
                store=store_ours,
                heap=heap,
                trajectory_of=drift.trajectory,
                meter=CostMeter(),
                horizon=0.50,
                sync_vector=make_hnsw_sync(idx_ours),
            )
            det.schedule_all(t0=0.0)
            engine = KHNSWRepairEngine(
                store=store_ours,
                detector=det,
                time_source=0.0,
                search_layer=search_layer,
                ef=64,
            )

            for step in range(1, steps + 1):
                t_curr = round(step * dt, 5)
                pos_step = drift.positions(t_curr)
                for nid in range(n):
                    idx_ours._store.set_vector(nid, pos_step[nid].astype(np.float32))
                engine._time_source = t_curr
                engine.run_repairs(idx_ours, budget=1e9, sink=ListSink())

            for i, u in enumerate(drifting_nodes):
                q = q_drift[i]
                exact = exact_drift[i]
                ids_ctrl, _ = idx_ctrl.search(q, k=10, ef=16)
                set_ctrl = {int(x) for x in ids_ctrl[:10]}
                ids_ours, _ = idx_ours.search(q, k=10, ef=16)
                set_ours = {int(x) for x in ids_ours[:10]}
                missed = (exact & set_ctrl) - set_ours

                for v in missed:
                    v_int = int(v)
                    # (a) Unreachable
                    in_deg_v = len(idx_ours.in_neighbors(v_int, 0))
                    reachable = evaluate_reachability(idx_ours, v_int)
                    if not reachable or in_deg_v == 0:
                        cat_counts["a_unreachable"] += 1
                        continue

                    # (b) Static node whose edges were pruned by reverse links
                    v_edges_now = set(idx_ours.neighbors(v_int, 0))
                    v_edges_orig = initial_adj.get((v_int, 0), set())
                    has_rev_link = any(dn in v_edges_now for dn in drifting_nodes)
                    lost_edges = v_edges_orig - v_edges_now
                    if (
                        v_int in static_nodes
                        and has_rev_link
                        and len(lost_edges) > 0
                        and len(v_edges_now) == idx_ours.max_degree(0)
                    ):
                        cat_counts["b_reverse_pruned"] += 1
                        continue

                    # (c) Direct edge dropped by rebuild
                    if v_int in initial_adj.get((u, 0), set()) and v_int not in idx_ours.neighbors(
                        u, 0
                    ):
                        cat_counts["c_dropped_by_rebuild"] += 1
                        continue

                    # (d) Other beam search miss
                    cat_counts["d_other_beam_search"] += 1
                    if len(miss_examples) < 5:
                        ids_ef64, _ = idx_ours.search(q, k=10, ef=64)
                        found_ef64 = v_int in {int(x) for x in ids_ef64[:10]}
                        miss_examples.append(
                            {
                                "seed": seed,
                                "speed": speed,
                                "query_node": u,
                                "missed_target": v_int,
                                "target_is_static": v_int in static_nodes,
                                "target_in_no_repair_out": v_int in idx_ctrl.neighbors(u, 0),
                                "target_in_no_repair_in": u in idx_ctrl.neighbors(v_int, 0),
                                "target_in_ours_out": v_int in idx_ours.neighbors(u, 0),
                                "target_in_ours_in": u in idx_ours.neighbors(v_int, 0),
                                "found_at_ef64": found_ef64,
                            }
                        )
        results["miss_classification"][f"{speed}x"] = cat_counts
    results["miss_examples_d"] = miss_examples

    # ---------------------------------------------------------
    # PART E: All-Drift 30x Setup
    # ---------------------------------------------------------
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X0 = rng.standard_normal((n, d)).astype(np.float32)

        base_index = HNSWIndex(dim=d, m=m, seed=seed)
        base_index.add(np.arange(n, dtype=np.int64), X0)
        settled_store = CertificateStore()
        settled_store.build(base_index, CostMeter())
        settle_hnsw(base_index, settled_store)

        drift = LinearDrift(X0, seed=seed, speed=30.0)
        pos_t_final = drift.positions(t_final).astype(np.float32)

        # Standard normal queries
        rng_q = np.random.default_rng(seed + 2000)
        q_norm = rng_q.standard_normal((100, d)).astype(np.float32)
        exact_norm = brute_force_knn(pos_t_final, q_norm, k=10)

        # Drifting queries: current node positions + small noise
        scale = mean_nn_distance(X0, seed=seed)
        rng_noise = np.random.default_rng(seed + 3000)
        noise = rng_noise.standard_normal((100, d)).astype(np.float32) * (0.05 * scale)
        sample_node_ids = rng_noise.choice(n, size=100, replace=False)
        q_drifting_nodes = pos_t_final[sample_node_ids] + noise
        exact_drifting_nodes = brute_force_knn(pos_t_final, q_drifting_nodes, k=10)

        for condition in ["no_repair", "ours_after"]:
            idx = copy.deepcopy(base_index)
            cost = {
                "detector": 0,
                "search": 0,
                "planning": 0,
                "store": 0,
                "reverse_link": 0,
                "total": 0,
            }
            moved_in_drift_total = 0

            if condition == "no_repair":
                for nid in range(n):
                    idx._store.set_vector(nid, pos_t_final[nid])
            elif condition == "ours_after":
                store = copy.deepcopy(settled_store)
                heap = KineticEventHeap(version_of=idx.version)
                det_meter = CostMeter()
                detector = KineticDetector(
                    index=idx,
                    store=store,
                    heap=heap,
                    trajectory_of=drift.trajectory,
                    meter=det_meter,
                    horizon=0.50,
                    sync_vector=make_hnsw_sync(idx),
                )
                detector.schedule_all(t0=0.0)

                engine = KHNSWRepairEngine(
                    store=store,
                    detector=detector,
                    time_source=0.0,
                    search_layer=search_layer,
                    ef=64,
                )

                for step in range(1, steps + 1):
                    t_curr = round(step * dt, 5)
                    pos_step = drift.positions(t_curr)
                    for nid in range(n):
                        idx._store.set_vector(nid, pos_step[nid].astype(np.float32))
                    engine._time_source = t_curr
                    res_engine = engine.run_repairs(idx, budget=1e9, sink=ListSink())
                    for k in ["search", "planning", "store", "reverse_link"]:
                        cost[k] += res_engine["budget_used"].get(k, 0)
                    moved_in_drift_total += res_engine.get("moved_in_drift", 0)
                cost["detector"] = det_meter.count
                cost["total"] = cost["detector"] + sum(
                    cost[k] for k in ["search", "planning", "store", "reverse_link"]
                )

            fresh_store = CertificateStore()
            fresh_meter = CostMeter()
            fresh_store.build(idx, fresh_meter)
            violated = sum(
                1 for s in fresh_store._edge_status.values() if s == EdgeStatus.VIOLATED.value
            )

            rec16_norm = evaluate_recall(idx, q_norm, exact_norm, k=10, ef=16)
            rec32_norm = evaluate_recall(idx, q_norm, exact_norm, k=10, ef=32)
            rec16_drift_q = evaluate_recall(
                idx, q_drifting_nodes, exact_drifting_nodes, k=10, ef=16
            )
            rec32_drift_q = evaluate_recall(
                idx, q_drifting_nodes, exact_drifting_nodes, k=10, ef=32
            )

            results["all_drift_30x"].append(
                {
                    "seed": seed,
                    "condition": condition,
                    "rec16_norm": rec16_norm,
                    "rec32_norm": rec32_norm,
                    "rec16_drift_queries": rec16_drift_q,
                    "rec32_drift_queries": rec32_drift_q,
                    "violated": violated,
                    "cost": cost,
                    "moved_in_drift": moved_in_drift_total,
                }
            )

    # Measure 1 fresh insert cost into settled n=500 index
    insert_costs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X0 = rng.standard_normal((n, d)).astype(np.float32)
        base_index = HNSWIndex(dim=d, m=m, seed=seed)
        base_index.add(np.arange(n, dtype=np.int64), X0)
        new_vec = rng.standard_normal((1, d)).astype(np.float32)
        before = base_index.distance_computations
        base_index.add(np.array([n], dtype=np.int64), new_vec)
        insert_costs.append(base_index.distance_computations - before)
    results["insert_cost_scale"] = {
        "per_seed": insert_costs,
        "mean": float(np.mean(insert_costs)),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure drift benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/data/drift_benchmark.json",
        help="Path to save JSON benchmark output",
    )
    args = parser.parse_args()

    print("Executing reproducible kinetic drift benchmark...")
    results = run_benchmark()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results successfully written to {out_path}")


if __name__ == "__main__":
    main()
