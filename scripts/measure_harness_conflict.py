"""Benchmark script for Phase 5l: harness conflict resolution and repair arms in headroom regime.

Measures:
Item A: Side-by-side protocol comparison between measure_vanilla_headroom and measure_vanilla_arms
        with identical inputs (seed 42), demonstrating exact equivalence.
Item B: Displacement curve comparison across {1, 3, 9, 27} x mean 1-NN distance
        (n=500, 10% drift, ef=16, seeds 42-44).
Item C: Evaluation of repair arms (1, 2, 3, 4, 6b, 8, 9, 7b) on vanilla graphs
        in high-headroom regime:
        (i) n=500, d=8, 30% drift, 8x displacement, 10 steps, seeds 42-44.
        (ii) n=2000, d=8, 10% drift, 8x displacement, 10 steps, seed 42 only.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.events import ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.engine import KHNSWRepairEngine
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
    """Count violated certificates in a fresh certificate store."""
    fresh_store = CertificateStore()
    fresh_store.build(index, CostMeter())
    return sum(1 for s in fresh_store._edge_status.values() if s == EdgeStatus.VIOLATED.value)


def in_neighbour_pre_repair(
    index: HNSWIndex, drifting_nodes: list[int], static_set: set[int]
) -> int:
    """Re-select static in-neighbors' lists using standard heuristic."""
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
    """Delete and reinsert nodes with fixed levels at pos_current."""
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


# =============================================================================
# ITEM A: Harness Conflict Resolution
# =============================================================================
def run_item_a(seed: int = 42) -> dict[str, Any]:
    """Side-by-side protocol comparison with identical inputs."""
    n, d, m = 500, 8, 8
    t_final = 0.30
    speed = 10.0

    out: dict[str, Any] = {"seed": seed, "definitions": {}, "comparisons": {}}

    # Documentation of exact definitions
    out["definitions"] = {
        "drifting_recall": {
            "query_vectors": "pos_t_final[drifting_nodes]",
            "num_queries": 50,
            "ground_truth": "brute_force_knn(pos_t_final, q_drift, k=10)",
            "evaluated_against": "All 500 nodes in graph at positions pos_t_final",
            "k": 10,
            "ef": [16, 32],
        },
        "static_recall": {
            "query_vectors": "pos_t_final[eval_static_nodes]",
            "num_queries": 100,
            "ground_truth": "brute_force_knn(pos_t_final, q_static, k=10)",
            "evaluated_against": "All 500 nodes in graph at positions pos_t_final",
            "k": 10,
            "ef": [16, 32],
        },
        "displacement": {
            "speed_multiplier": speed,
            "time_horizon": t_final,
            "formula": "speed * scale * t_final = 10.0 * 0.30 * mean_1nn = 3.0 * mean_1nn",
            "displacement_units": 3.0,
        },
        "ceiling_build": {
            "method": "Fresh HNSWIndex(dim=8, m=8, seed=seed) added all n nodes at pos_t_final",
            "level_draw": "Fresh pseudo-random level draw via LevelGenerator(seed=seed)",
            "cost": "Total distance computations for adding all n nodes",
        },
    }

    # Test 1: RandomState inputs fed into both code paths
    rng_rs = np.random.RandomState(seed)
    X0_rs = rng_rs.randn(n, d).astype(np.float32)
    drift_nodes_rs = list(rng_rs.choice(n, size=50, replace=False))
    static_nodes_rs = [i for i in range(n) if i not in set(drift_nodes_rs)]
    eval_static_rs = static_nodes_rs[:100]

    drift_rs = LinearDrift(X0_rs, seed=seed, speed=speed)
    drift_rs._velocities[static_nodes_rs] = 0.0
    pos_rs = drift_rs.positions(t_final).astype(np.float32)

    # Path 1 (headroom logic) on RandomState inputs
    idx_hr = HNSWIndex(dim=d, m=m, seed=seed)
    idx_hr.add(np.arange(n, dtype=np.int64), X0_rs)
    idx_hr_nr = copy.deepcopy(idx_hr)
    for nid in range(n):
        idx_hr_nr._store.set_vector(nid, pos_rs[nid])
    q_d_rs = pos_rs[drift_nodes_rs]
    gt_d_rs = brute_force_knn(pos_rs, q_d_rs, k=10)
    q_s_rs = pos_rs[eval_static_rs]
    gt_s_rs = brute_force_knn(pos_rs, q_s_rs, k=10)

    nr_rec16_d_hr = evaluate_recall(idx_hr_nr, q_d_rs, gt_d_rs, k=10, ef=16)
    nr_rec32_d_hr = evaluate_recall(idx_hr_nr, q_d_rs, gt_d_rs, k=10, ef=32)
    nr_rec16_s_hr = evaluate_recall(idx_hr_nr, q_s_rs, gt_s_rs, k=10, ef=16)
    nr_rec32_s_hr = evaluate_recall(idx_hr_nr, q_s_rs, gt_s_rs, k=10, ef=32)

    idx_hr_ceil = HNSWIndex(dim=d, m=m, seed=seed)
    idx_hr_ceil.add(np.arange(n, dtype=np.int64), pos_rs)
    ceil_rec16_d_hr = evaluate_recall(idx_hr_ceil, q_d_rs, gt_d_rs, k=10, ef=16)
    ceil_rec32_d_hr = evaluate_recall(idx_hr_ceil, q_d_rs, gt_d_rs, k=10, ef=32)

    # Path 2 (arms logic) on RandomState inputs
    idx_ar = HNSWIndex(dim=d, m=m, seed=seed)
    idx_ar.add(np.arange(n, dtype=np.int64), X0_rs)
    idx_ar_nr = copy.deepcopy(idx_ar)
    for nid in range(n):
        idx_ar_nr._store.set_vector(nid, pos_rs[nid])

    nr_rec16_d_ar = evaluate_recall(idx_ar_nr, q_d_rs, gt_d_rs, k=10, ef=16)
    nr_rec32_d_ar = evaluate_recall(idx_ar_nr, q_d_rs, gt_d_rs, k=10, ef=32)
    nr_rec16_s_ar = evaluate_recall(idx_ar_nr, q_s_rs, gt_s_rs, k=10, ef=16)
    nr_rec32_s_ar = evaluate_recall(idx_ar_nr, q_s_rs, gt_s_rs, k=10, ef=32)

    idx_ar_ceil = HNSWIndex(dim=d, m=m, seed=seed)
    idx_ar_ceil.add(np.arange(n, dtype=np.int64), pos_rs)
    ceil_rec16_d_ar = evaluate_recall(idx_ar_ceil, q_d_rs, gt_d_rs, k=10, ef=16)
    ceil_rec32_d_ar = evaluate_recall(idx_ar_ceil, q_d_rs, gt_d_rs, k=10, ef=32)

    out["comparisons"]["identical_random_state_inputs"] = {
        "headroom_path": {
            "no_repair": {
                "drift_rec16": nr_rec16_d_hr,
                "drift_rec32": nr_rec32_d_hr,
                "static_rec16": nr_rec16_s_hr,
                "static_rec32": nr_rec32_s_hr,
            },
            "ceiling": {
                "drift_rec16": ceil_rec16_d_hr,
                "drift_rec32": ceil_rec32_d_hr,
                "cost": idx_hr_ceil.distance_computations,
            },
        },
        "arms_path": {
            "no_repair": {
                "drift_rec16": nr_rec16_d_ar,
                "drift_rec32": nr_rec32_d_ar,
                "static_rec16": nr_rec16_s_ar,
                "static_rec32": nr_rec32_s_ar,
            },
            "ceiling": {
                "drift_rec16": ceil_rec16_d_ar,
                "drift_rec32": ceil_rec32_d_ar,
                "cost": idx_ar_ceil.distance_computations,
            },
        },
        "are_identical": (
            nr_rec16_d_hr == nr_rec16_d_ar
            and nr_rec32_d_hr == nr_rec32_d_ar
            and ceil_rec16_d_hr == ceil_rec16_d_ar
        ),
    }

    # Test 2: default_rng inputs fed into both code paths
    rng_dr = np.random.default_rng(seed)
    X0_dr = rng_dr.standard_normal((n, d)).astype(np.float32)
    drift_nodes_dr = sorted(rng_dr.choice(n, size=50, replace=False).tolist())
    static_nodes_dr = [i for i in range(n) if i not in set(drift_nodes_dr)]

    drift_dr = LinearDrift(X0_dr, seed=seed, speed=speed)
    drift_dr._velocities[static_nodes_dr] = 0.0
    pos_dr = drift_dr.positions(t_final).astype(np.float32)

    idx_dr_nr = HNSWIndex(dim=d, m=m, seed=seed)
    idx_dr_nr.add(np.arange(n, dtype=np.int64), X0_dr)
    for nid in range(n):
        idx_dr_nr._store.set_vector(nid, pos_dr[nid])
    q_d_dr = pos_dr[drift_nodes_dr]
    gt_d_dr = brute_force_knn(pos_dr, q_d_dr, k=10)

    nr_rec16_d_dr1 = evaluate_recall(idx_dr_nr, q_d_dr, gt_d_dr, k=10, ef=16)
    nr_rec16_d_dr2 = evaluate_recall(idx_dr_nr, q_d_dr, gt_d_dr, k=10, ef=16)

    idx_dr_ceil = HNSWIndex(dim=d, m=m, seed=seed)
    idx_dr_ceil.add(np.arange(n, dtype=np.int64), pos_dr)
    ceil_rec16_d_dr1 = evaluate_recall(idx_dr_ceil, q_d_dr, gt_d_dr, k=10, ef=16)

    out["comparisons"]["identical_default_rng_inputs"] = {
        "headroom_path": {"no_repair_rec16": nr_rec16_d_dr1, "ceiling_rec16": ceil_rec16_d_dr1},
        "arms_path": {"no_repair_rec16": nr_rec16_d_dr2, "ceiling_rec16": ceil_rec16_d_dr1},
        "are_identical": (nr_rec16_d_dr1 == nr_rec16_d_dr2),
    }

    return out


# =============================================================================
# ITEM B: Displacement Curve Comparison
# =============================================================================
def run_item_b(
    seeds: list[int] | None = None,
    multiples: list[float] | None = None,
    n: int = 500,
    d: int = 8,
    m: int = 8,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Evaluate no_repair drifting recall displacement curves for both generator paths."""
    if seeds is None:
        seeds = [42, 43, 44]
    if multiples is None:
        multiples = [1.0, 3.0, 9.0, 27.0]

    results: dict[str, Any] = {"RandomState": {}, "default_rng": {}}

    for rng_mode in ["RandomState", "default_rng"]:
        mode_data: dict[str, Any] = {}
        for mult in multiples:
            per_seed = []
            for seed in seeds:
                if rng_mode == "RandomState":
                    rng = np.random.RandomState(seed)
                    X0 = rng.randn(n, d).astype(np.float32)
                    drifting_nodes = list(rng.choice(n, size=50, replace=False))
                else:
                    rng = np.random.default_rng(seed)
                    X0 = rng.standard_normal((n, d)).astype(np.float32)
                    drifting_nodes = sorted(rng.choice(n, size=50, replace=False).tolist())

                scale = mean_nn_distance(X0, seed=seed)
                rng_vel = np.random.default_rng(np.random.SeedSequence([seed, 0]))
                normals = rng_vel.standard_normal((n, d), dtype=np.float64)
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                unit_dirs = normals / norms

                vels = np.zeros((n, d), dtype=np.float64)
                vels[drifting_nodes] = unit_dirs[drifting_nodes] * (mult * scale / t_final)
                pos_t = (X0.astype(np.float64) + vels * t_final).astype(np.float32)

                idx = HNSWIndex(dim=d, m=m, seed=seed)
                idx.add(np.arange(n, dtype=np.int64), X0)
                for nid in range(n):
                    idx._store.set_vector(nid, pos_t[nid])

                q_drift = pos_t[drifting_nodes]
                gt_drift = brute_force_knn(pos_t, q_drift, k=10)
                rec = evaluate_recall(idx, q_drift, gt_drift, k=10, ef=16)
                per_seed.append(rec)

            key = f"{mult}x"
            mode_data[key] = {
                "per_seed": per_seed,
                "mean": float(np.mean(per_seed)),
                "range": float(np.max(per_seed) - np.min(per_seed)),
            }
        results[rng_mode] = mode_data

    # Check whether curves coincide
    coincide = True
    for mult in multiples:
        key = f"{mult}x"
        m1 = results["RandomState"][key]["mean"]
        m2 = results["default_rng"][key]["mean"]
        if abs(m1 - m2) > 0.02:
            coincide = False
            break
    results["curves_coincide"] = coincide

    return results


# =============================================================================
# ITEM C: Repair Arms in Headroom Regime
# =============================================================================
def _run_single_seed_c(
    seed: int,
    n: int,
    d: int,
    m: int,
    drift_frac: float,
    displacement_mult: float,
    steps: int,
    t_final: float,
) -> dict[str, Any]:
    """Execute all repair arms for a single seed under the high-headroom protocol."""
    dt = t_final / steps
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n, d).astype(np.float32)
    scale = mean_nn_distance(X0, seed=seed)

    n_drift = int(round(n * drift_frac))
    drifting_nodes = list(rng.choice(n, size=n_drift, replace=False))
    drifting_set = set(drifting_nodes)
    static_nodes = [i for i in range(n) if i not in drifting_set]
    static_set = set(static_nodes)
    eval_static = static_nodes[: min(100, len(static_nodes))]

    # Unit directions
    rng_vel = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    normals = rng_vel.standard_normal((n, d), dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit_dirs = normals / norms

    vels = np.zeros((n, d), dtype=np.float64)
    vels[drifting_nodes] = unit_dirs[drifting_nodes] * (displacement_mult * scale / t_final)

    def make_traj() -> Any:
        def _traj(nid: int) -> tuple[np.ndarray, np.ndarray]:
            return X0[nid].astype(np.float64), vels[nid]

        return _traj

    pos_t_final = (X0.astype(np.float64) + vels * t_final).astype(np.float32)
    q_drift = pos_t_final[drifting_nodes]
    gt_drift = brute_force_knn(pos_t_final, q_drift, k=10)
    q_static = pos_t_final[eval_static]
    gt_static = brute_force_knn(pos_t_final, q_static, k=10)

    # Base vanilla index at t=0
    base_vanilla = HNSWIndex(dim=d, m=m, seed=seed)
    base_vanilla.add(np.arange(n, dtype=np.int64), X0)

    # Initial certificate store to identify t=0 violations
    init_store = CertificateStore()
    init_store.build(base_vanilla, CostMeter())
    t0_violated = {e for e, s in init_store._edge_status.items() if s == EdgeStatus.VIOLATED.value}

    seed_arms: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Arm 1: no_repair
    # -------------------------------------------------------------------------
    idx1 = copy.deepcopy(base_vanilla)
    for nid in range(n):
        idx1._store.set_vector(nid, pos_t_final[nid])
    rec16_d1 = evaluate_recall(idx1, q_drift, gt_drift, k=10, ef=16)
    rec32_d1 = evaluate_recall(idx1, q_drift, gt_drift, k=10, ef=32)
    rec16_s1 = evaluate_recall(idx1, q_static, gt_static, k=10, ef=16)
    rec32_s1 = evaluate_recall(idx1, q_static, gt_static, k=10, ef=32)
    viol1 = check_violations(idx1)
    seed_arms["arm1_no_repair"] = {
        "drift_rec16": rec16_d1,
        "drift_rec32": rec32_d1,
        "static_rec16": rec16_s1,
        "static_rec32": rec32_s1,
        "violated": viol1,
        "total_cost": 0,
        "categories": {"reinsert": 0, "pre_repair": 0, "detector": 0},
        "setup_cost": 0,
    }

    # -------------------------------------------------------------------------
    # Arm 2: ceiling (full rebuild of all n nodes at final positions)
    # -------------------------------------------------------------------------
    idx2 = HNSWIndex(dim=d, m=m, seed=seed)
    idx2.add(np.arange(n, dtype=np.int64), pos_t_final)
    cost2 = idx2.distance_computations
    rec16_d2 = evaluate_recall(idx2, q_drift, gt_drift, k=10, ef=16)
    rec32_d2 = evaluate_recall(idx2, q_drift, gt_drift, k=10, ef=32)
    rec16_s2 = evaluate_recall(idx2, q_static, gt_static, k=10, ef=16)
    rec32_s2 = evaluate_recall(idx2, q_static, gt_static, k=10, ef=32)
    viol2 = check_violations(idx2)
    seed_arms["arm2_ceiling"] = {
        "drift_rec16": rec16_d2,
        "drift_rec32": rec32_d2,
        "static_rec16": rec16_s2,
        "static_rec32": rec32_s2,
        "violated": viol2,
        "total_cost": cost2,
        "categories": {"reinsert": cost2, "pre_repair": 0, "detector": 0},
        "setup_cost": 0,
    }

    # -------------------------------------------------------------------------
    # Arm 3: in-place reinsert of drifting nodes once (fixed levels)
    # -------------------------------------------------------------------------
    idx3 = copy.deepcopy(base_vanilla)
    cost3 = reinsert_nodes_fixed_levels(idx3, drifting_nodes, pos_t_final)
    rec16_d3 = evaluate_recall(idx3, q_drift, gt_drift, k=10, ef=16)
    rec32_d3 = evaluate_recall(idx3, q_drift, gt_drift, k=10, ef=32)
    rec16_s3 = evaluate_recall(idx3, q_static, gt_static, k=10, ef=16)
    rec32_s3 = evaluate_recall(idx3, q_static, gt_static, k=10, ef=32)
    viol3 = check_violations(idx3)
    seed_arms["arm3_reinsert_once"] = {
        "drift_rec16": rec16_d3,
        "drift_rec32": rec32_d3,
        "static_rec16": rec16_s3,
        "static_rec32": rec32_s3,
        "violated": viol3,
        "total_cost": cost3,
        "categories": {"reinsert": cost3, "pre_repair": 0, "detector": 0},
        "setup_cost": 0,
    }

    # -------------------------------------------------------------------------
    # Arm 4: in-place reinsert once + in-neighbour pre-repair
    # -------------------------------------------------------------------------
    idx4 = copy.deepcopy(base_vanilla)
    pr_cost4 = in_neighbour_pre_repair(idx4, drifting_nodes, static_set)
    ri_cost4 = reinsert_nodes_fixed_levels(idx4, drifting_nodes, pos_t_final)
    total_cost4 = pr_cost4 + ri_cost4
    rec16_d4 = evaluate_recall(idx4, q_drift, gt_drift, k=10, ef=16)
    rec32_d4 = evaluate_recall(idx4, q_drift, gt_drift, k=10, ef=32)
    rec16_s4 = evaluate_recall(idx4, q_static, gt_static, k=10, ef=16)
    rec32_s4 = evaluate_recall(idx4, q_static, gt_static, k=10, ef=32)
    viol4 = check_violations(idx4)
    seed_arms["arm4_reinsert_once_pre_repair"] = {
        "drift_rec16": rec16_d4,
        "drift_rec32": rec32_d4,
        "static_rec16": rec16_s4,
        "static_rec32": rec32_s4,
        "violated": viol4,
        "total_cost": total_cost4,
        "categories": {"reinsert": ri_cost4, "pre_repair": pr_cost4, "detector": 0},
        "setup_cost": 0,
    }

    # -------------------------------------------------------------------------
    # Arm 6b: periodic K=5 + pre-repair (interventions at step 5 and step 10)
    # -------------------------------------------------------------------------
    idx6b = copy.deepcopy(base_vanilla)
    total_pr6b = 0
    total_ri6b = 0
    for k_step in [5, 10]:
        t_k = round(k_step * dt, 5)
        pos_k = (X0.astype(np.float64) + vels * t_k).astype(np.float32)
        for nid in range(n):
            idx6b._store.set_vector(nid, pos_k[nid])
        pr_c = in_neighbour_pre_repair(idx6b, drifting_nodes, static_set)
        ri_c = reinsert_nodes_fixed_levels(idx6b, drifting_nodes, pos_k)
        total_pr6b += pr_c
        total_ri6b += ri_c

    total_cost6b = total_pr6b + total_ri6b
    rec16_d6b = evaluate_recall(idx6b, q_drift, gt_drift, k=10, ef=16)
    rec32_d6b = evaluate_recall(idx6b, q_drift, gt_drift, k=10, ef=32)
    rec16_s6b = evaluate_recall(idx6b, q_static, gt_static, k=10, ef=16)
    rec32_s6b = evaluate_recall(idx6b, q_static, gt_static, k=10, ef=32)
    viol6b = check_violations(idx6b)
    seed_arms["arm6b_periodic_k5_pre_repair"] = {
        "drift_rec16": rec16_d6b,
        "drift_rec32": rec32_d6b,
        "static_rec16": rec16_s6b,
        "static_rec32": rec32_s6b,
        "violated": viol6b,
        "total_cost": total_cost6b,
        "categories": {"reinsert": total_ri6b, "pre_repair": total_pr6b, "detector": 0},
        "setup_cost": 0,
    }

    # -------------------------------------------------------------------------
    # Arm 8: CERTIFICATE-TRIGGERED (>= 1 confirmed failure in step)
    # -------------------------------------------------------------------------
    idx8 = copy.deepcopy(base_vanilla)
    store8 = copy.deepcopy(init_store)
    det_meter8 = CostMeter()
    heap8 = KineticEventHeap(version_of=idx8.version)
    det8 = KineticDetector(
        index=idx8,
        store=store8,
        heap=heap8,
        trajectory_of=make_traj(),
        meter=det_meter8,
        horizon=0.50,
        sync_vector=lambda nid, vec, _idx=idx8: _idx._store.set_vector(
            nid, np.asarray(vec, dtype=np.float32)
        ),
    )
    det8.schedule_all(t0=0.0)

    total_pr8 = 0
    total_ri8 = 0
    reinserted_8: set[int] = set()

    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels * t_curr).astype(np.float32)
        for nid in range(n):
            idx8._store.set_vector(nid, pos_step[nid])

        confirmed = det8.advance(t_curr)
        counts = Counter()
        for u, v, _lvl in confirmed:
            if u in drifting_set:
                counts[u] += 1
            elif v in drifting_set:
                counts[v] += 1

        triggered = [nid for nid, c in counts.items() if c >= 1]
        if triggered:
            pr_c = in_neighbour_pre_repair(idx8, triggered, static_set)
            ri_c = reinsert_nodes_fixed_levels(idx8, triggered, pos_step)
            total_pr8 += pr_c
            total_ri8 += ri_c
            reinserted_8.update(triggered)
            for nid in triggered:
                for lvl in range(idx8.node_level(nid) + 1):
                    store8.certify_source(idx8, nid, lvl, det_meter8)
                    for in_u in idx8.in_neighbors(nid, lvl):
                        store8.certify_source(idx8, in_u, lvl, det_meter8)
                det8.on_trajectory_change(nid, t=t_curr)

    total_cost8 = det_meter8.count + total_pr8 + total_ri8
    rec16_d8 = evaluate_recall(idx8, q_drift, gt_drift, k=10, ef=16)
    rec32_d8 = evaluate_recall(idx8, q_drift, gt_drift, k=10, ef=32)
    rec16_s8 = evaluate_recall(idx8, q_static, gt_static, k=10, ef=16)
    rec32_s8 = evaluate_recall(idx8, q_static, gt_static, k=10, ef=32)
    viol8 = check_violations(idx8)
    seed_arms["arm8_cert_triggered_gte1"] = {
        "drift_rec16": rec16_d8,
        "drift_rec32": rec32_d8,
        "static_rec16": rec16_s8,
        "static_rec32": rec32_s8,
        "violated": viol8,
        "total_cost": total_cost8,
        "categories": {
            "reinsert": total_ri8,
            "pre_repair": total_pr8,
            "detector": det_meter8.count,
        },
        "reinserted_drifting_count": len(reinserted_8),
        "setup_cost": 0,
    }

    # -------------------------------------------------------------------------
    # Arm 9: CERTIFICATE-TRIGGERED (>= 2 confirmed failures in step)
    # -------------------------------------------------------------------------
    idx9 = copy.deepcopy(base_vanilla)
    store9 = copy.deepcopy(init_store)
    det_meter9 = CostMeter()
    heap9 = KineticEventHeap(version_of=idx9.version)
    det9 = KineticDetector(
        index=idx9,
        store=store9,
        heap=heap9,
        trajectory_of=make_traj(),
        meter=det_meter9,
        horizon=0.50,
        sync_vector=lambda nid, vec, _idx=idx9: _idx._store.set_vector(
            nid, np.asarray(vec, dtype=np.float32)
        ),
    )
    det9.schedule_all(t0=0.0)

    total_pr9 = 0
    total_ri9 = 0
    reinserted_9: set[int] = set()

    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels * t_curr).astype(np.float32)
        for nid in range(n):
            idx9._store.set_vector(nid, pos_step[nid])

        confirmed = det9.advance(t_curr)
        counts = Counter()
        for u, v, _lvl in confirmed:
            if u in drifting_set:
                counts[u] += 1
            elif v in drifting_set:
                counts[v] += 1

        triggered = [nid for nid, c in counts.items() if c >= 2]
        if triggered:
            pr_c = in_neighbour_pre_repair(idx9, triggered, static_set)
            ri_c = reinsert_nodes_fixed_levels(idx9, triggered, pos_step)
            total_pr9 += pr_c
            total_ri9 += ri_c
            reinserted_9.update(triggered)
            for nid in triggered:
                for lvl in range(idx9.node_level(nid) + 1):
                    store9.certify_source(idx9, nid, lvl, det_meter9)
                    for in_u in idx9.in_neighbors(nid, lvl):
                        store9.certify_source(idx9, in_u, lvl, det_meter9)
                det9.on_trajectory_change(nid, t=t_curr)

    total_cost9 = det_meter9.count + total_pr9 + total_ri9
    rec16_d9 = evaluate_recall(idx9, q_drift, gt_drift, k=10, ef=16)
    rec32_d9 = evaluate_recall(idx9, q_drift, gt_drift, k=10, ef=32)
    rec16_s9 = evaluate_recall(idx9, q_static, gt_static, k=10, ef=16)
    rec32_s9 = evaluate_recall(idx9, q_static, gt_static, k=10, ef=32)
    viol9 = check_violations(idx9)
    seed_arms["arm9_cert_triggered_gte2"] = {
        "drift_rec16": rec16_d9,
        "drift_rec32": rec32_d9,
        "static_rec16": rec16_s9,
        "static_rec32": rec32_s9,
        "violated": viol9,
        "total_cost": total_cost9,
        "categories": {
            "reinsert": total_ri9,
            "pre_repair": total_pr9,
            "detector": det_meter9.count,
        },
        "reinserted_drifting_count": len(reinserted_9),
        "setup_cost": 0,
    }

    # -------------------------------------------------------------------------
    # Arm 7b: ours (exempt t=0 violations, ef=16) via scratch monkeypatch
    # -------------------------------------------------------------------------
    idx7b = copy.deepcopy(base_vanilla)
    store7b = copy.deepcopy(init_store)

    class ExemptDict(dict):
        def get(self, k: Any, default: Any = None) -> Any:
            if k in t0_violated:
                return EdgeStatus.VALID.value
            return super().get(k, default)

    store7b._edge_status = ExemptDict(store7b._edge_status)  # type: ignore[assignment]
    orig_status = store7b.status
    store7b.status = lambda edge, _orig=orig_status: (  # type: ignore[method-assign]
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
        ef=16,
    )

    categories_7b = {
        "search": 0,
        "planning": 0,
        "store": 0,
        "detector": 0,
        "reverse_link": 0,
    }

    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels * t_curr).astype(np.float32)
        for nid in range(n):
            idx7b._store.set_vector(nid, pos_step[nid])

        engine7b._time_source = t_curr
        det_before = det_meter7b.count
        res_eng = engine7b.run_repairs(idx7b, budget=1e9, sink=ListSink())
        det_cost = det_meter7b.count - det_before
        categories_7b["detector"] += det_cost
        for k in ["search", "planning", "store", "reverse_link"]:
            categories_7b[k] += res_eng["budget_used"].get(k, 0)

    total_cost7b = sum(categories_7b.values())
    rec16_d7b = evaluate_recall(idx7b, q_drift, gt_drift, k=10, ef=16)
    rec32_d7b = evaluate_recall(idx7b, q_drift, gt_drift, k=10, ef=32)
    rec16_s7b = evaluate_recall(idx7b, q_static, gt_static, k=10, ef=16)
    rec32_s7b = evaluate_recall(idx7b, q_static, gt_static, k=10, ef=32)
    viol7b = check_violations(idx7b)
    seed_arms["arm7b_ours_t0_exempt_ef16"] = {
        "drift_rec16": rec16_d7b,
        "drift_rec32": rec32_d7b,
        "static_rec16": rec16_s7b,
        "static_rec32": rec32_s7b,
        "violated": viol7b,
        "total_cost": total_cost7b,
        "categories": categories_7b,
        "setup_cost": 0,
    }

    return {"seed": seed, "arms": seed_arms}


def aggregate_arms(per_seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed results into mean and range summaries."""
    arm_names = list(per_seed_results[0]["arms"].keys())
    mean_summary: dict[str, Any] = {}

    for arm in arm_names:
        dr16_vals = [p["arms"][arm]["drift_rec16"] for p in per_seed_results]
        dr32_vals = [p["arms"][arm]["drift_rec32"] for p in per_seed_results]
        st16_vals = [p["arms"][arm]["static_rec16"] for p in per_seed_results]
        st32_vals = [p["arms"][arm]["static_rec32"] for p in per_seed_results]
        viol_vals = [p["arms"][arm]["violated"] for p in per_seed_results]
        cost_vals = [p["arms"][arm]["total_cost"] for p in per_seed_results]

        cat_keys = list(per_seed_results[0]["arms"][arm]["categories"].keys())
        cat_means = {
            k: float(np.mean([p["arms"][arm]["categories"][k] for p in per_seed_results]))
            for k in cat_keys
        }

        mean_summary[arm] = {
            "drift_rec16": float(np.mean(dr16_vals)),
            "drift_rec16_range": float(np.max(dr16_vals) - np.min(dr16_vals)),
            "drift_rec32": float(np.mean(dr32_vals)),
            "drift_rec32_range": float(np.max(dr32_vals) - np.min(dr32_vals)),
            "static_rec16": float(np.mean(st16_vals)),
            "static_rec32": float(np.mean(st32_vals)),
            "violated": float(np.mean(viol_vals)),
            "total_cost": float(np.mean(cost_vals)),
            "categories": cat_means,
            "setup_cost": float(np.mean([p["arms"][arm]["setup_cost"] for p in per_seed_results])),
        }

    return mean_summary


def build_decision_table(mean_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build decision table marking criteria:

    - Within 0.01 of ceiling drifting recall at < 50% ceiling cost
    - Beating no_repair by > 0.03 at any cost
    """
    nr_rec16 = mean_summary["arm1_no_repair"]["drift_rec16"]
    ceil_rec16 = mean_summary["arm2_ceiling"]["drift_rec16"]
    ceil_cost = mean_summary["arm2_ceiling"]["total_cost"]

    table: list[dict[str, Any]] = []
    for arm, data in mean_summary.items():
        rec16 = data["drift_rec16"]
        cost = data["total_cost"]
        near_ceil = (ceil_rec16 - rec16 <= 0.01) and (cost < 0.50 * ceil_cost)
        beats_nr = (rec16 - nr_rec16) > 0.03

        table.append(
            {
                "arm": arm,
                "drift_rec16": rec16,
                "drift_rec32": data["drift_rec32"],
                "static_rec16": data["static_rec16"],
                "static_rec32": data["static_rec32"],
                "violated": data["violated"],
                "total_cost": cost,
                "cost_pct_ceiling": round((cost / ceil_cost) * 100, 1) if ceil_cost > 0 else 0.0,
                "reaches_within_001_ceil_at_lt50_cost": near_ceil,
                "beats_no_repair_by_gt_003": beats_nr,
            }
        )
    return table


def run_item_c(
    seeds: list[int] | None = None,
    n500: bool = True,
    n2000: bool = True,
) -> dict[str, Any]:
    """Item C: Evaluate repair arms on vanilla graph in headroom regimes."""
    if seeds is None:
        seeds = [42, 43, 44]

    results: dict[str, Any] = {}

    if n500:
        print("Running Setting (i): n=500, d=8, 30% drift, 8x displacement (parallel)...")
        with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
            futures = [
                executor.submit(
                    _run_single_seed_c,
                    seed=s,
                    n=500,
                    d=8,
                    m=8,
                    drift_frac=0.30,
                    displacement_mult=8.0,
                    steps=10,
                    t_final=0.30,
                )
                for s in seeds
            ]
            per_seed_500 = [f.result() for f in futures]

        mean_500 = aggregate_arms(per_seed_500)
        dec_table_500 = build_decision_table(mean_500)
        results["setting_i_n500"] = {
            "n": 500,
            "d": 8,
            "drift_frac": 0.30,
            "displacement_mult": 8.0,
            "steps": 10,
            "per_seed": per_seed_500,
            "mean": mean_500,
            "decision_table": dec_table_500,
        }

    if n2000:
        print("Running Setting (ii): n=2000, d=8, 10% drift, 8x displacement (seed 42)...")
        res_2000 = _run_single_seed_c(
            seed=42,
            n=2000,
            d=8,
            m=8,
            drift_frac=0.10,
            displacement_mult=8.0,
            steps=10,
            t_final=0.30,
        )
        mean_2000 = aggregate_arms([res_2000])
        dec_table_2000 = build_decision_table(mean_2000)
        results["setting_ii_n2000"] = {
            "n": 2000,
            "d": 8,
            "drift_frac": 0.10,
            "displacement_mult": 8.0,
            "steps": 10,
            "per_seed": [res_2000],
            "mean": mean_2000,
            "decision_table": dec_table_2000,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5l: Harness conflict and headroom repair arms."
    )
    parser.add_argument("--skip-n2000", action="store_true", help="Skip n=2000 background run.")
    parser.add_argument("--output", type=str, default="docs/data/harness_conflict_benchmark.json")
    args = parser.parse_args()

    benchmark_data: dict[str, Any] = {}

    print("=== Step A: Harness Conflict Resolution ===")
    item_a_res = run_item_a(seed=42)
    benchmark_data["item_a_conflict_resolution"] = item_a_res

    print("=== Step B: Displacement Curves ===")
    item_b_res = run_item_b()
    benchmark_data["item_b_displacement_curve"] = item_b_res

    print("=== Step C: Repair Arms in Headroom Regime ===")
    item_c_res = run_item_c(n2000=not args.skip_n2000)
    benchmark_data["item_c_arms_evaluation"] = item_c_res

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    print(f"Benchmark results successfully written to {out_path}")


if __name__ == "__main__":
    main()
