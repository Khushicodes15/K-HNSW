"""Comprehensive benchmark for Phase 5k: Harness Check & Repair Arms on Vanilla Graph.

Executes:
Item A: Harness check comparing reconcile_harness vs measure_vanilla_headroom
        for both settled (settle=True) and vanilla (settle=False) graphs.
Item B: Repair arms evaluated on the VANILLA graph (settle=False), speeds 10x/30x, seeds 42-44:
 1. no_repair
 2. ceiling (full rebuild of all 500, cost counted for all 500 inserts)
 3. in-place reinsert of 50 drifting nodes ONCE at final positions (fixed levels)
 4. as 3, but first re-select static in-neighbour lists from (nbrs + deleted nbrs)
 5. reinsert 50 drifting nodes every step (10 steps), (a) without and (b) with pre-repair
 6. periodic: reinsert drifting nodes with pre-repair every K steps, K=2 and K=5
 7. ours (existing engine, ef=16 and ef=64 rebuild) on vanilla graph:
    (a) t=0 violations repaired at t=0 (setup cost counted)
    (b) t=0 violations exempted from repair via scratch monkeypatch
Item C: Generates decision table marking arms reaching within 0.01 of ceiling at < 50% cost.
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events.schema import ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.engine import KHNSWRepairEngine
from khnsw.core.repair.loop import repair_loop
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.search.heuristic import select_neighbors_heuristic
from khnsw.core.search.search_layer import search_layer


def brute_force_knn(X: np.ndarray, queries: np.ndarray, k: int = 10) -> list[set[int]]:
    """Brute force exact top-k nearest neighbors by squared Euclidean distance."""
    gt: list[set[int]] = []
    for q in queries:
        dists = np.sum((X - q) ** 2, axis=1)
        topk = np.argsort(dists)[:k]
        gt.append(set(int(x) for x in topk))
    return gt


def evaluate_recall(
    index: HNSWIndex, queries: np.ndarray, ground_truth: list[set[int]], k: int = 10, ef: int = 16
) -> float:
    """Recall@k against precomputed ground truth sets."""
    hits = 0
    total = len(queries) * k
    for q, gt in zip(queries, ground_truth, strict=False):
        res_ids, _ = index.search(q, k=k, ef=ef)
        hits += len(set(int(x) for x in res_ids) & gt)
    return hits / total if total > 0 else 0.0


def check_violations(index: HNSWIndex) -> int:
    """Check violated certificates in from-scratch store."""
    fresh_store = CertificateStore()
    fresh_store.build(index, CostMeter())
    return sum(1 for s in fresh_store._edge_status.values() if s == EdgeStatus.VIOLATED.value)


def in_neighbour_pre_repair(
    index: HNSWIndex, drifting_nodes: list[int], static_set: set[int]
) -> int:
    """Re-select static in-neighbors' lists using standard heuristic.

    Returns the number of distance computations used.
    """
    heuristic_dists = 0
    for d_id in drifting_nodes:
        top_lvl = index.node_level(d_id)
        for lc in range(top_lvl + 1):
            static_in = [u for u in index.in_neighbors(d_id, lc) if u in static_set]
            d_nbrs = set(index.neighbors(d_id, lc))
            for u in static_in:
                u_nbrs = set(index.neighbors(u, lc))
                cands = (u_nbrs | d_nbrs) - {u, d_id}
                max_m = index.max_degree(lc)
                cand_list = list(cands)
                if cand_list:
                    heuristic_dists += len(cand_list)
                    pruned = select_neighbors_heuristic(index._store, u, cand_list, max_m, level=lc)
                    heuristic_dists += (len(pruned) * (len(pruned) - 1)) // 2
                    index._store.set_neighbors(u, lc, pruned)
    return heuristic_dists


def reinsert_nodes_fixed_levels(index: HNSWIndex, nodes: list[int], pos_current: np.ndarray) -> int:
    """Delete and reinsert nodes with fixed levels at pos_current.

    Returns distance computations incurred.
    """
    before_cost = index.distance_computations
    for nid in nodes:
        orig_lvl = index.node_level(nid)
        v_node = pos_current[nid].copy()
        index.delete(nid)

        orig_sample = index._level_gen.sample
        index._level_gen.sample = lambda lvl=orig_lvl: lvl  # type: ignore[method-assign]
        try:
            index._insert_one(nid, v_node)
        finally:
            index._level_gen.sample = orig_sample  # type: ignore[method-assign]

    return index.distance_computations - before_cost


def run_item_a_harness_check() -> dict[str, Any]:
    """Item A: Verify no_repair recall discrepancy between default_rng and RandomState."""
    n, d, m = 500, 8, 8
    t_final = 0.30
    speeds = [10.0, 30.0]
    seeds = [42, 43, 44]

    results: dict[str, Any] = {"default_rng": {}, "RandomState": {}}

    for rng_name in ["default_rng", "RandomState"]:
        speed_data: dict[str, Any] = {}
        for speed in speeds:
            per_seed = []
            for seed in seeds:
                if rng_name == "default_rng":
                    rng = np.random.default_rng(seed)
                    X0 = rng.standard_normal((n, d)).astype(np.float32)
                    drifting_nodes = sorted(rng.choice(n, size=50, replace=False).tolist())
                else:
                    rng = np.random.RandomState(seed)
                    X0 = rng.randn(n, d).astype(np.float32)
                    drifting_nodes = list(rng.choice(n, size=50, replace=False))

                static_nodes = [i for i in range(n) if i not in set(drifting_nodes)]
                eval_static = static_nodes[:100]

                drift = LinearDrift(X0, seed=seed, speed=speed)
                drift._velocities[static_nodes] = 0.0
                scale = drift._scale
                pos_t = drift.positions(t_final).astype(np.float32)

                q_drift = pos_t[drifting_nodes]
                gt_drift = brute_force_knn(pos_t, q_drift, k=10)
                q_static = pos_t[eval_static]
                gt_static = brute_force_knn(pos_t, q_static, k=10)

                # Total displacement in units of mean 1-NN
                disp = float(
                    np.mean(np.linalg.norm(pos_t[drifting_nodes] - X0[drifting_nodes], axis=1))
                )
                disp_units = disp / scale

                # 1. Vanilla (settle=False)
                idx_v = HNSWIndex(dim=d, m=m, seed=seed)
                idx_v.add(np.arange(n, dtype=np.int64), X0)
                for nid in range(n):
                    idx_v._store.set_vector(nid, pos_t[nid])
                rec16_d_v = evaluate_recall(idx_v, q_drift, gt_drift, k=10, ef=16)
                rec32_d_v = evaluate_recall(idx_v, q_drift, gt_drift, k=10, ef=32)
                rec16_s_v = evaluate_recall(idx_v, q_static, gt_static, k=10, ef=16)
                rec32_s_v = evaluate_recall(idx_v, q_static, gt_static, k=10, ef=32)

                # 2. Settled (settle=True)
                idx_s = HNSWIndex(dim=d, m=m, seed=seed)
                idx_s.add(np.arange(n, dtype=np.int64), X0)
                store_s = CertificateStore()
                store_s.build(idx_s, CostMeter())
                repair_loop(idx_s, store_s)
                for nid in range(n):
                    idx_s._store.set_vector(nid, pos_t[nid])
                rec16_d_s = evaluate_recall(idx_s, q_drift, gt_drift, k=10, ef=16)
                rec32_d_s = evaluate_recall(idx_s, q_drift, gt_drift, k=10, ef=32)
                rec16_s_s = evaluate_recall(idx_s, q_static, gt_static, k=10, ef=16)
                rec32_s_s = evaluate_recall(idx_s, q_static, gt_static, k=10, ef=32)

                per_seed.append(
                    {
                        "seed": seed,
                        "displacement_units": disp_units,
                        "vanilla": {
                            "drift_rec16": rec16_d_v,
                            "drift_rec32": rec32_d_v,
                            "static_rec16": rec16_s_v,
                            "static_rec32": rec32_s_v,
                        },
                        "settled": {
                            "drift_rec16": rec16_d_s,
                            "drift_rec32": rec32_d_s,
                            "static_rec16": rec16_s_s,
                            "static_rec32": rec32_s_s,
                        },
                    }
                )

            speed_data[f"{int(speed)}x"] = {
                "per_seed": per_seed,
                "mean": {
                    "displacement_units": float(
                        np.mean([p["displacement_units"] for p in per_seed])
                    ),
                    "vanilla": {
                        "drift_rec16": float(
                            np.mean([p["vanilla"]["drift_rec16"] for p in per_seed])
                        ),
                        "drift_rec32": float(
                            np.mean([p["vanilla"]["drift_rec32"] for p in per_seed])
                        ),
                        "static_rec16": float(
                            np.mean([p["vanilla"]["static_rec16"] for p in per_seed])
                        ),
                        "static_rec32": float(
                            np.mean([p["vanilla"]["static_rec32"] for p in per_seed])
                        ),
                    },
                    "settled": {
                        "drift_rec16": float(
                            np.mean([p["settled"]["drift_rec16"] for p in per_seed])
                        ),
                        "drift_rec32": float(
                            np.mean([p["settled"]["drift_rec32"] for p in per_seed])
                        ),
                        "static_rec16": float(
                            np.mean([p["settled"]["static_rec16"] for p in per_seed])
                        ),
                        "static_rec32": float(
                            np.mean([p["settled"]["static_rec32"] for p in per_seed])
                        ),
                    },
                },
            }
        results[rng_name] = speed_data

    return results


def run_seed_vanilla_arms(seed: int, speeds: list[float] | None = None) -> dict[str, Any]:
    """Execute all Item B arms for a single seed across speeds on VANILLA graph."""
    if speeds is None:
        speeds = [10.0, 30.0]

    n, d, m = 500, 8, 8
    t_final = 0.30
    steps = 10
    dt = t_final / steps

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, d)).astype(np.float32)

    drifting_nodes = sorted(rng.choice(n, size=50, replace=False).tolist())
    drifting_set = set(drifting_nodes)
    static_nodes = [i for i in range(n) if i not in drifting_set]
    static_set = set(static_nodes)
    eval_static = static_nodes[:100]

    # Vanilla base index
    base_vanilla = HNSWIndex(dim=d, m=m, seed=seed)
    base_vanilla.add(np.arange(n, dtype=np.int64), X0)

    # Initial t=0 store and violations
    init_store = CertificateStore()
    init_store.build(base_vanilla, CostMeter())
    t0_violated = {
        edge
        for edge, status in init_store._edge_status.items()
        if status == EdgeStatus.VIOLATED.value
    }
    t0_violation_count = len(t0_violated)

    seed_res: dict[str, Any] = {
        "seed": seed,
        "t0_initial_violations": t0_violation_count,
        "speeds": {},
    }

    for speed in speeds:
        drift = LinearDrift(X0, seed=seed, speed=speed)
        drift._velocities[static_nodes] = 0.0
        pos_t_final = drift.positions(t_final).astype(np.float32)

        q_drift = pos_t_final[drifting_nodes]
        gt_drift = brute_force_knn(pos_t_final, q_drift, k=10)
        q_static = pos_t_final[eval_static]
        gt_static = brute_force_knn(pos_t_final, q_static, k=10)

        def make_traj(
            v_mat: np.ndarray = drift._velocities,
        ) -> Callable[[int], tuple[np.ndarray, np.ndarray]]:
            def _traj(node_id: int) -> tuple[np.ndarray, np.ndarray]:
                return X0[node_id].astype(np.float64), v_mat[node_id]

            return _traj

        arm_results: dict[str, Any] = {}

        # -------------------------------------------------------------
        # Arm 1: no_repair
        # -------------------------------------------------------------
        idx1 = copy.deepcopy(base_vanilla)
        for nid in range(n):
            idx1._store.set_vector(nid, pos_t_final[nid])
        rec16_d1 = evaluate_recall(idx1, q_drift, gt_drift, k=10, ef=16)
        rec32_d1 = evaluate_recall(idx1, q_drift, gt_drift, k=10, ef=32)
        rec16_s1 = evaluate_recall(idx1, q_static, gt_static, k=10, ef=16)
        rec32_s1 = evaluate_recall(idx1, q_static, gt_static, k=10, ef=32)
        viol1 = check_violations(idx1)
        arm_results["arm1_no_repair"] = {
            "drift_rec16": rec16_d1,
            "drift_rec32": rec32_d1,
            "static_rec16": rec16_s1,
            "static_rec32": rec32_s1,
            "violated": viol1,
            "total_cost": 0,
            "categories": {
                "search": 0,
                "planning": 0,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 2: ceiling (full rebuild of all 500 at final positions)
        # -------------------------------------------------------------
        idx2 = HNSWIndex(dim=d, m=m, seed=seed)
        idx2.add(np.arange(n, dtype=np.int64), pos_t_final)
        cost2 = idx2.distance_computations
        rec16_d2 = evaluate_recall(idx2, q_drift, gt_drift, k=10, ef=16)
        rec32_d2 = evaluate_recall(idx2, q_drift, gt_drift, k=10, ef=32)
        rec16_s2 = evaluate_recall(idx2, q_static, gt_static, k=10, ef=16)
        rec32_s2 = evaluate_recall(idx2, q_static, gt_static, k=10, ef=32)
        viol2 = check_violations(idx2)
        arm_results["arm2_ceiling"] = {
            "drift_rec16": rec16_d2,
            "drift_rec32": rec32_d2,
            "static_rec16": rec16_s2,
            "static_rec32": rec32_s2,
            "violated": viol2,
            "total_cost": cost2,
            "categories": {
                "search": cost2,
                "planning": 0,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 3: in-place reinsert ONCE at final positions (fixed levels)
        # -------------------------------------------------------------
        idx3 = copy.deepcopy(base_vanilla)
        for nid in range(n):
            idx3._store.set_vector(nid, pos_t_final[nid])
        cost3 = reinsert_nodes_fixed_levels(idx3, drifting_nodes, pos_t_final)
        rec16_d3 = evaluate_recall(idx3, q_drift, gt_drift, k=10, ef=16)
        rec32_d3 = evaluate_recall(idx3, q_drift, gt_drift, k=10, ef=32)
        rec16_s3 = evaluate_recall(idx3, q_static, gt_static, k=10, ef=16)
        rec32_s3 = evaluate_recall(idx3, q_static, gt_static, k=10, ef=32)
        viol3 = check_violations(idx3)
        arm_results["arm3_inplace_reinsert_once"] = {
            "drift_rec16": rec16_d3,
            "drift_rec32": rec32_d3,
            "static_rec16": rec16_s3,
            "static_rec32": rec32_s3,
            "violated": viol3,
            "total_cost": cost3,
            "categories": {
                "search": cost3,
                "planning": 0,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 4: in-place reinsert ONCE with in-neighbour pre-repair
        # -------------------------------------------------------------
        idx4 = copy.deepcopy(base_vanilla)
        for nid in range(n):
            idx4._store.set_vector(nid, pos_t_final[nid])
        cost4_plan = in_neighbour_pre_repair(idx4, drifting_nodes, static_set)
        cost4_search = reinsert_nodes_fixed_levels(idx4, drifting_nodes, pos_t_final)
        rec16_d4 = evaluate_recall(idx4, q_drift, gt_drift, k=10, ef=16)
        rec32_d4 = evaluate_recall(idx4, q_drift, gt_drift, k=10, ef=32)
        rec16_s4 = evaluate_recall(idx4, q_static, gt_static, k=10, ef=16)
        rec32_s4 = evaluate_recall(idx4, q_static, gt_static, k=10, ef=32)
        viol4 = check_violations(idx4)
        arm_results["arm4_inplace_reinsert_with_prerepair_once"] = {
            "drift_rec16": rec16_d4,
            "drift_rec32": rec32_d4,
            "static_rec16": rec16_s4,
            "static_rec32": rec32_s4,
            "violated": viol4,
            "total_cost": cost4_plan + cost4_search,
            "categories": {
                "search": cost4_search,
                "planning": cost4_plan,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 5a: reinsert every step (10 steps) WITHOUT pre-repair
        # -------------------------------------------------------------
        idx5a = copy.deepcopy(base_vanilla)
        cost5a_search = 0
        for step in range(1, steps + 1):
            t_curr = round(step * dt, 5)
            pos_step = drift.positions(t_curr).astype(np.float32)
            for nid in range(n):
                idx5a._store.set_vector(nid, pos_step[nid])
            cost5a_search += reinsert_nodes_fixed_levels(idx5a, drifting_nodes, pos_step)
        rec16_d5a = evaluate_recall(idx5a, q_drift, gt_drift, k=10, ef=16)
        rec32_d5a = evaluate_recall(idx5a, q_drift, gt_drift, k=10, ef=32)
        rec16_s5a = evaluate_recall(idx5a, q_static, gt_static, k=10, ef=16)
        rec32_s5a = evaluate_recall(idx5a, q_static, gt_static, k=10, ef=32)
        viol5a = check_violations(idx5a)
        arm_results["arm5a_reinsert_every_step_no_prerepair"] = {
            "drift_rec16": rec16_d5a,
            "drift_rec32": rec32_d5a,
            "static_rec16": rec16_s5a,
            "static_rec32": rec32_s5a,
            "violated": viol5a,
            "total_cost": cost5a_search,
            "categories": {
                "search": cost5a_search,
                "planning": 0,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 5b: reinsert every step (10 steps) WITH in-neighbour pre-repair
        # -------------------------------------------------------------
        idx5b = copy.deepcopy(base_vanilla)
        cost5b_search = 0
        cost5b_plan = 0
        for step in range(1, steps + 1):
            t_curr = round(step * dt, 5)
            pos_step = drift.positions(t_curr).astype(np.float32)
            for nid in range(n):
                idx5b._store.set_vector(nid, pos_step[nid])
            cost5b_plan += in_neighbour_pre_repair(idx5b, drifting_nodes, static_set)
            cost5b_search += reinsert_nodes_fixed_levels(idx5b, drifting_nodes, pos_step)
        rec16_d5b = evaluate_recall(idx5b, q_drift, gt_drift, k=10, ef=16)
        rec32_d5b = evaluate_recall(idx5b, q_drift, gt_drift, k=10, ef=32)
        rec16_s5b = evaluate_recall(idx5b, q_static, gt_static, k=10, ef=16)
        rec32_s5b = evaluate_recall(idx5b, q_static, gt_static, k=10, ef=32)
        viol5b = check_violations(idx5b)
        arm_results["arm5b_reinsert_every_step_with_prerepair"] = {
            "drift_rec16": rec16_d5b,
            "drift_rec32": rec32_d5b,
            "static_rec16": rec16_s5b,
            "static_rec32": rec32_s5b,
            "violated": viol5b,
            "total_cost": cost5b_search + cost5b_plan,
            "categories": {
                "search": cost5b_search,
                "planning": cost5b_plan,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 6a: periodic reinsert with pre-repair every K=2 steps
        # -------------------------------------------------------------
        idx6a = copy.deepcopy(base_vanilla)
        cost6a_search = 0
        cost6a_plan = 0
        for step in range(1, steps + 1):
            t_curr = round(step * dt, 5)
            pos_step = drift.positions(t_curr).astype(np.float32)
            for nid in range(n):
                idx6a._store.set_vector(nid, pos_step[nid])
            if step % 2 == 0:
                cost6a_plan += in_neighbour_pre_repair(idx6a, drifting_nodes, static_set)
                cost6a_search += reinsert_nodes_fixed_levels(idx6a, drifting_nodes, pos_step)
        rec16_d6a = evaluate_recall(idx6a, q_drift, gt_drift, k=10, ef=16)
        rec32_d6a = evaluate_recall(idx6a, q_drift, gt_drift, k=10, ef=32)
        rec16_s6a = evaluate_recall(idx6a, q_static, gt_static, k=10, ef=16)
        rec32_s6a = evaluate_recall(idx6a, q_static, gt_static, k=10, ef=32)
        viol6a = check_violations(idx6a)
        arm_results["arm6a_periodic_k2_with_prerepair"] = {
            "drift_rec16": rec16_d6a,
            "drift_rec32": rec32_d6a,
            "static_rec16": rec16_s6a,
            "static_rec32": rec32_s6a,
            "violated": viol6a,
            "total_cost": cost6a_search + cost6a_plan,
            "categories": {
                "search": cost6a_search,
                "planning": cost6a_plan,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 6b: periodic reinsert with pre-repair every K=5 steps
        # -------------------------------------------------------------
        idx6b = copy.deepcopy(base_vanilla)
        cost6b_search = 0
        cost6b_plan = 0
        for step in range(1, steps + 1):
            t_curr = round(step * dt, 5)
            pos_step = drift.positions(t_curr).astype(np.float32)
            for nid in range(n):
                idx6b._store.set_vector(nid, pos_step[nid])
            if step % 5 == 0:
                cost6b_plan += in_neighbour_pre_repair(idx6b, drifting_nodes, static_set)
                cost6b_search += reinsert_nodes_fixed_levels(idx6b, drifting_nodes, pos_step)
        rec16_d6b = evaluate_recall(idx6b, q_drift, gt_drift, k=10, ef=16)
        rec32_d6b = evaluate_recall(idx6b, q_drift, gt_drift, k=10, ef=32)
        rec16_s6b = evaluate_recall(idx6b, q_static, gt_static, k=10, ef=16)
        rec32_s6b = evaluate_recall(idx6b, q_static, gt_static, k=10, ef=32)
        viol6b = check_violations(idx6b)
        arm_results["arm6b_periodic_k5_with_prerepair"] = {
            "drift_rec16": rec16_d6b,
            "drift_rec32": rec32_d6b,
            "static_rec16": rec16_s6b,
            "static_rec32": rec32_s6b,
            "violated": viol6b,
            "total_cost": cost6b_search + cost6b_plan,
            "categories": {
                "search": cost6b_search,
                "planning": cost6b_plan,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            },
            "setup_cost": 0,
        }

        # -------------------------------------------------------------
        # Arm 7a: ours with t=0 violations REPAIRED at t=0 (ef=16 and ef=64)
        # -------------------------------------------------------------
        for ef_rebuild in [16, 64]:
            idx7a = copy.deepcopy(base_vanilla)
            store7a = copy.deepcopy(init_store)

            # Repair at t=0
            res_settle = repair_loop(idx7a, store7a)
            setup_cost = (
                sum(res_settle.budget_used.values()) if hasattr(res_settle, "budget_used") else 0
            )

            det_meter7a = CostMeter()
            heap7a = KineticEventHeap(version_of=idx7a.version)
            det7a = KineticDetector(
                index=idx7a,
                store=store7a,
                heap=heap7a,
                trajectory_of=make_traj(),
                meter=det_meter7a,
                horizon=0.50,
                sync_vector=lambda nid, vec, _idx=idx7a: _idx._store.set_vector(
                    nid, np.asarray(vec, dtype=np.float32)
                ),
            )
            det7a.schedule_all(t0=0.0)

            engine7a = KHNSWRepairEngine(
                store=store7a,
                detector=det7a,
                time_source=0.0,
                search_layer=search_layer,
                ef=ef_rebuild,
            )

            drift_categories = {
                "search": 0,
                "planning": 0,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            }

            for step in range(1, steps + 1):
                t_curr = round(step * dt, 5)
                pos_step = drift.positions(t_curr).astype(np.float32)
                for nid in range(n):
                    idx7a._store.set_vector(nid, pos_step[nid])

                engine7a._time_source = t_curr
                det_before = det_meter7a.count
                res_eng = engine7a.run_repairs(idx7a, budget=1e9, sink=ListSink())
                det_cost = det_meter7a.count - det_before
                drift_categories["detector"] += det_cost
                for k in ["search", "planning", "store", "reverse_link"]:
                    drift_categories[k] += res_eng["budget_used"].get(k, 0)

            drift_total = sum(drift_categories.values())
            rec16_d7a = evaluate_recall(idx7a, q_drift, gt_drift, k=10, ef=16)
            rec32_d7a = evaluate_recall(idx7a, q_drift, gt_drift, k=10, ef=32)
            rec16_s7a = evaluate_recall(idx7a, q_static, gt_static, k=10, ef=16)
            rec32_s7a = evaluate_recall(idx7a, q_static, gt_static, k=10, ef=32)
            viol7a = check_violations(idx7a)
            arm_name = f"arm7a_ours_t0_repaired_ef{ef_rebuild}"
            arm_results[arm_name] = {
                "drift_rec16": rec16_d7a,
                "drift_rec32": rec32_d7a,
                "static_rec16": rec16_s7a,
                "static_rec32": rec32_s7a,
                "violated": viol7a,
                "total_cost": drift_total,
                "categories": drift_categories,
                "setup_cost": setup_cost,
            }

        # -------------------------------------------------------------
        # Arm 7b: ours with t=0 violations EXEMPTED via scratch monkeypatch
        # -------------------------------------------------------------
        for ef_rebuild in [16, 64]:
            idx7b = copy.deepcopy(base_vanilla)
            store7b = copy.deepcopy(init_store)

            # Scratch monkeypatch: wrap store._edge_status to treat t0_violated as VALID
            class ExemptDict(dict):
                def get(self, k, default=None):
                    if k in t0_violated:
                        return EdgeStatus.VALID.value
                    return super().get(k, default)

            store7b._edge_status = ExemptDict(store7b._edge_status)
            orig_status_fn = store7b.status
            store7b.status = lambda edge, _orig=orig_status_fn: (
                EdgeStatus.VALID.value if edge in t0_violated else _orig(edge)
            )

            det_meter7b = CostMeter()
            heap7b = KineticEventHeap(version_of=idx7b.version)
            det7b = KineticDetector(
                index=idx7b,
                store=store7b,
                heap=heap7b,
                trajectory_of=make_traj(),
                meter=det_meter7b,
                horizon=0.50,
                sync_vector=lambda nid, vec, _idx=idx7b: _idx._store.set_vector(
                    nid, np.asarray(vec, dtype=np.float32)
                ),
            )
            det7b.schedule_all(t0=0.0)

            engine7b = KHNSWRepairEngine(
                store=store7b,
                detector=det7b,
                time_source=0.0,
                search_layer=search_layer,
                ef=ef_rebuild,
            )

            drift_categories_7b = {
                "search": 0,
                "planning": 0,
                "store": 0,
                "detector": 0,
                "reverse_link": 0,
            }

            for step in range(1, steps + 1):
                t_curr = round(step * dt, 5)
                pos_step = drift.positions(t_curr).astype(np.float32)
                for nid in range(n):
                    idx7b._store.set_vector(nid, pos_step[nid])

                engine7b._time_source = t_curr
                det_before = det_meter7b.count
                res_eng = engine7b.run_repairs(idx7b, budget=1e9, sink=ListSink())
                det_cost = det_meter7b.count - det_before
                drift_categories_7b["detector"] += det_cost
                for k in ["search", "planning", "store", "reverse_link"]:
                    drift_categories_7b[k] += res_eng["budget_used"].get(k, 0)

            drift_total_7b = sum(drift_categories_7b.values())
            rec16_d7b = evaluate_recall(idx7b, q_drift, gt_drift, k=10, ef=16)
            rec32_d7b = evaluate_recall(idx7b, q_drift, gt_drift, k=10, ef=32)
            rec16_s7b = evaluate_recall(idx7b, q_static, gt_static, k=10, ef=16)
            rec32_s7b = evaluate_recall(idx7b, q_static, gt_static, k=10, ef=32)
            viol7b = check_violations(idx7b)
            arm_name = f"arm7b_ours_t0_exempted_ef{ef_rebuild}"
            arm_results[arm_name] = {
                "drift_rec16": rec16_d7b,
                "drift_rec32": rec32_d7b,
                "static_rec16": rec16_s7b,
                "static_rec32": rec32_s7b,
                "violated": viol7b,
                "total_cost": drift_total_7b,
                "categories": drift_categories_7b,
                "setup_cost": 0,
            }

        seed_res["speeds"][f"{int(speed)}x"] = arm_results

    return seed_res


def aggregate_means(per_seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed results into mean summaries across seeds."""
    speeds = ["10x", "30x"]
    arm_names = list(per_seed_results[0]["speeds"]["10x"].keys())

    mean_summary: dict[str, Any] = {}
    for spd in speeds:
        mean_summary[spd] = {}
        for arm in arm_names:
            arm_seeds = [s["speeds"][spd][arm] for s in per_seed_results]
            mean_summary[spd][arm] = {
                "drift_rec16": float(np.mean([a["drift_rec16"] for a in arm_seeds])),
                "drift_rec32": float(np.mean([a["drift_rec32"] for a in arm_seeds])),
                "static_rec16": float(np.mean([a["static_rec16"] for a in arm_seeds])),
                "static_rec32": float(np.mean([a["static_rec32"] for a in arm_seeds])),
                "violated": float(np.mean([a["violated"] for a in arm_seeds])),
                "total_cost": float(np.mean([a["total_cost"] for a in arm_seeds])),
                "setup_cost": float(np.mean([a["setup_cost"] for a in arm_seeds])),
                "categories": {
                    cat: float(np.mean([a["categories"][cat] for a in arm_seeds]))
                    for cat in ["search", "planning", "store", "detector", "reverse_link"]
                },
            }
    return mean_summary


def generate_decision_table(means: dict[str, Any]) -> str:
    """Format markdown decision table comparing recall vs cost across arms."""
    lines = [
        "| Speed | Arm | Drift Rec@10 ($ef=16$) | Drift Rec@10 ($ef=32$) | Static Rec@10 ($ef=16$) "
        "| Violated | Drift Cost | Setup Cost | Reaches $\\ge \\text{ceil}-0.01$ "
        "at $< 50\\%$ Cost? |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for spd in ["10x", "30x"]:
        ceiling_rec16 = means[spd]["arm2_ceiling"]["drift_rec16"]
        ceiling_cost = means[spd]["arm2_ceiling"]["total_cost"]
        cost_thresh = 0.5 * ceiling_cost
        target_rec16 = ceiling_rec16 - 0.01

        for arm, data in means[spd].items():
            clean_name = arm.replace("arm", "").replace("_", " ").strip()
            rec16 = data["drift_rec16"]
            rec32 = data["drift_rec32"]
            stat16 = data["static_rec16"]
            viol = data["violated"]
            cost = data["total_cost"]
            setup = data["setup_cost"]

            qualifies = (rec16 >= target_rec16) and (cost < cost_thresh)
            status_mark = "**YES**" if qualifies else "No"

            row = (
                f"| **{spd}** | {clean_name} | {rec16:.4f} | {rec32:.4f} | {stat16:.4f} | "
                f"{viol:.1f} | {int(round(cost)):,d} | {int(round(setup)):,d} | {status_mark} |"
            )
            lines.append(row)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5k Vanilla Arms Benchmark")
    parser.add_argument("--out", type=str, default="docs/data/vanilla_arms_benchmark.json")
    args = parser.parse_args()

    print("=== Running Item A: Harness Check ===")
    res_a = run_item_a_harness_check()
    print("Item A completed.")

    print("=== Running Item B: Repair Arms on Vanilla Graph (Parallel Seeds 42-44) ===")
    seeds = [42, 43, 44]
    with ProcessPoolExecutor(max_workers=3) as executor:
        per_seed_results = list(executor.map(run_seed_vanilla_arms, seeds))
    print("Item B completed.")

    means_b = aggregate_means(per_seed_results)
    decision_table = generate_decision_table(means_b)

    full_output = {
        "item_a_harness_check": res_a,
        "item_b_vanilla_arms": {
            "per_seed": per_seed_results,
            "mean": means_b,
        },
        "item_c_decision_table": decision_table,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"Results written to {out_path}")
    print("\n=== DECISION TABLE ===")
    print(decision_table)


if __name__ == "__main__":
    main()
