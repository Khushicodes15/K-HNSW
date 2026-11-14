"""Measurement script for Phase 5n: Protocol check, controls, and budgeted priority.

Evaluates:
- Item A: Protocol audit (cos(v_i, X0_i) and no_repair drift recall at 1x/3x/9x/27x).
- Item B: Controls on n=500, 30% drift, 8x displacement, 10 steps across:
          - indep_dir (unconfounded protocol)
          - RandomState
          - default_rng
          Evaluating:
          1 no_repair
          2 periodic K in {1, 2, 5} (with pre-repair)
          3 lazy theta in {0.5, 1.0, 2.0} (no pre-repair)
          4 round_robin (matched count to lazy theta=1.0)
          5 fixed_distance (median radius threshold D)
          6 random (matched count to lazy theta=1.0)
          7 budgeted_priority (top B by score ||x-p||/r)
- Item C: Large-graph scaling on n=2000, 10% drift, 8x displacement, 10 steps,
          seeds 42-44, evaluated under indep_dir.
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.scheduling.tracker import (
    BudgetedPriorityTracker,
    NodeDisplacementTracker,
    in_neighbour_pre_repair,
    reinsert_node_fixed_level,
)


def brute_force_knn(X: NDArray[Any], queries: NDArray[Any], k: int = 10) -> list[set[int]]:
    """Brute-force exact top-k nearest neighbors by squared Euclidean distance."""
    gt: list[set[int]] = []
    for q in queries:
        dists = np.sum((X - q) ** 2, axis=1)
        topk = np.argsort(dists)[:k]
        gt.append(set(int(x) for x in topk))
    return gt


def evaluate_recall(
    index: HNSWIndex, queries: NDArray[Any], ground_truth: list[set[int]], k: int = 10, ef: int = 16
) -> float:
    """Recall@k against precomputed ground truth sets."""
    hits = 0
    total = len(queries) * k
    for q, gt in zip(queries, ground_truth, strict=False):
        res_ids, _ = index.search(q, k=k, ef=ef)
        hits += len(set(int(x) for x in res_ids) & gt)
    return hits / total if total > 0 else 0.0


def setup_protocol_data(
    protocol_name: str,
    n: int,
    d: int,
    seed: int,
    drift_frac: float,
    displacement_mult: float,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Generate X0, velocities, and node partition for a given protocol."""
    n_drift = int(round(n * drift_frac))

    if protocol_name == "RandomState":
        rng = np.random.RandomState(seed)
        X0 = rng.randn(n, d).astype(np.float32)
        drift_nodes = list(rng.choice(n, size=n_drift, replace=False))
        drift_set = set(drift_nodes)
        static_nodes = [i for i in range(n) if i not in drift_set]
        scale = mean_nn_distance(X0, seed=seed)
        drift = LinearDrift(X0, seed=seed, speed=displacement_mult / t_final)
        drift._velocities[static_nodes] = 0.0
        vels = drift._velocities

    elif protocol_name == "default_rng":
        rng = np.random.default_rng(seed)
        X0 = rng.standard_normal((n, d)).astype(np.float32)
        drift_nodes = sorted(rng.choice(n, size=n_drift, replace=False).tolist())
        drift_set = set(drift_nodes)
        static_nodes = [i for i in range(n) if i not in drift_set]
        scale = mean_nn_distance(X0, seed=seed)
        drift = LinearDrift(X0, seed=seed, speed=displacement_mult / t_final)
        drift._velocities[static_nodes] = 0.0
        vels = drift._velocities

    elif protocol_name == "indep_dir":
        rng_x = np.random.default_rng(seed)
        rng_id = np.random.default_rng(seed + 1)
        rng_v = np.random.default_rng(seed + 2)

        X0 = rng_x.standard_normal((n, d)).astype(np.float32)
        drift_nodes = sorted(rng_id.choice(n, size=n_drift, replace=False).tolist())
        drift_set = set(drift_nodes)
        static_nodes = [i for i in range(n) if i not in drift_set]
        scale = mean_nn_distance(X0, seed=seed)

        normals = rng_v.standard_normal((n, d), dtype=np.float64)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit_dirs = normals / norms

        vels = np.zeros((n, d), dtype=np.float64)
        vels[drift_nodes] = unit_dirs[drift_nodes] * (displacement_mult * scale / t_final)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    eval_static = static_nodes[: min(100, len(static_nodes))]

    return {
        "protocol": protocol_name,
        "seed": seed,
        "n": n,
        "d": d,
        "scale": scale,
        "X0": X0,
        "vels": vels,
        "drift_nodes": drift_nodes,
        "static_nodes": static_nodes,
        "eval_static": eval_static,
    }


# =============================================================================
# ITEM A: Protocol Check
# =============================================================================
def run_item_a(seeds: list[int] | None = None) -> dict[str, Any]:
    """Execute Item A: cos(v_i, X0_i) and no_repair drift recall across displacements."""
    if seeds is None:
        seeds = [42, 43, 44]

    n, d, m = 500, 8, 8
    drift_frac = 0.10  # 50 nodes
    displacements = [1.0, 3.0, 9.0, 27.0]
    protocols = ["RandomState", "default_rng", "indep_dir"]

    results: dict[str, Any] = {
        "linear_drift_velocity_sampling_quote": (
            "In LinearDrift.__init__ (src/khnsw/core/movement/linear.py lines 54-61):\n"
            "rng = np.random.default_rng(np.random.SeedSequence([self._seed, 0]))\n"
            "normals = rng.standard_normal((self._n, self._dim), dtype=np.float64)\n"
            "norms = np.linalg.norm(normals, axis=1, keepdims=True)\n"
            "unit_dirs = normals / norms\n"
            "self._velocities = unit_dirs * (self._speed * self._scale)\n"
            "Consequence: When X0 is sampled via np.random.default_rng(seed).standard_normal,\n"
            "NumPy SeedSequence([seed, 0]) produces the exact same PRNG variates as\n"
            "SeedSequence(seed). Thus normals == X0, unit_dirs == X0 / ||X0||, "
            "and cos(v, X0) == 1.0."
        ),
        "cosine_audit": {},
        "no_repair_recall_sweep": {},
    }

    for proto in protocols:
        proto_cos_list = []
        sweep_data: dict[str, Any] = {f"{int(mult)}x": [] for mult in displacements}

        for s in seeds:
            data = setup_protocol_data(
                protocol_name=proto,
                n=n,
                d=d,
                seed=s,
                drift_frac=drift_frac,
                displacement_mult=1.0,
                t_final=0.30,
            )
            X0 = data["X0"]
            v = data["vels"][data["drift_nodes"]]
            x = X0[data["drift_nodes"]]
            dot = np.sum(v * x, axis=1)
            norm_v = np.linalg.norm(v, axis=1)
            norm_x = np.linalg.norm(x, axis=1)
            cos_vals = dot / (norm_v * norm_x)

            proto_cos_list.append(
                {
                    "seed": s,
                    "mean_cos": float(np.mean(cos_vals)),
                    "min_cos": float(np.min(cos_vals)),
                    "max_cos": float(np.max(cos_vals)),
                    "mean_abs_cos": float(np.mean(np.abs(cos_vals))),
                }
            )

            # Build index once for sweep
            base_index = HNSWIndex(dim=d, m=m, seed=s)
            base_index.add(np.arange(n, dtype=np.int64), X0)

            scale = data["scale"]
            unit_v = v / np.linalg.norm(v, axis=1, keepdims=True)

            for mult in displacements:
                disp_vec = unit_v * (mult * scale)
                pos_final = X0.copy()
                pos_final[data["drift_nodes"]] = (x + disp_vec).astype(np.float32)

                idx_nr = copy.deepcopy(base_index)
                for nid in range(n):
                    idx_nr._store.set_vector(nid, pos_final[nid])

                q_d = pos_final[data["drift_nodes"]]
                gt_d = brute_force_knn(pos_final, q_d, k=10)
                rec16 = evaluate_recall(idx_nr, q_d, gt_d, k=10, ef=16)

                sweep_data[f"{int(mult)}x"].append({"seed": s, "recall_ef16": rec16})

        # Mean across seeds for sweep
        mean_sweep = {
            k: float(np.mean([item["recall_ef16"] for item in v])) for k, v in sweep_data.items()
        }

        results["cosine_audit"][proto] = proto_cos_list
        results["no_repair_recall_sweep"][proto] = {
            "per_seed": sweep_data,
            "mean_across_seeds": mean_sweep,
        }

    return results


# =============================================================================
# ITEM B & C: Simulators for Maintenance Arms & Controls
# =============================================================================
def _simulate_single_arm(
    arm_name: str,
    base_index: HNSWIndex,
    X0: NDArray[Any],
    vels: NDArray[Any],
    drift_nodes: list[int],
    static_nodes: list[int],
    eval_static: list[int],
    steps: int,
    t_final: float,
    seed: int,
    budget_schedule: list[int] | None = None,
    theta: float = 1.0,
    period_k: int = 1,
) -> dict[str, Any]:
    """Execute simulation of a single maintenance arm over discrete steps."""
    dt = t_final / steps
    n = len(X0)
    static_set = set(static_nodes)

    index = copy.deepcopy(base_index)
    categories = {"stale_check": 0, "pre_repair": 0, "reinsert": 0}
    total_reinsert_count = 0
    step_records = []

    # Setup arm specific objects
    lazy_tracker: NodeDisplacementTracker | None = None
    priority_tracker: BudgetedPriorityTracker | None = None
    median_radius: float = 0.0
    fixed_dist_anchors: dict[int, NDArray[np.float32]] = {}

    if arm_name.startswith("lazy_theta_"):
        lazy_tracker = NodeDisplacementTracker(index, theta=theta, pre_repair=False)
        lazy_tracker.register_all(X0)

    elif arm_name == "budgeted_priority":
        priority_tracker = BudgetedPriorityTracker(index, budget=1)
        priority_tracker.register_all(X0)

    elif arm_name == "fixed_distance":
        # Compute median over nodes of (1.0 * r_i at registration)
        reg_radii = []
        for nid in drift_nodes:
            p_vec = np.asarray(X0[nid], dtype=np.float32)
            nbrs = list(index.neighbors(nid, level=0))
            if nbrs:
                max_r = max(float(np.linalg.norm(p_vec - index.vector(nb))) for nb in nbrs)
            else:
                max_r = 0.0
            reg_radii.append(max_r)
            fixed_dist_anchors[nid] = p_vec.copy()
        median_radius = float(np.median(reg_radii))

    rr_cursor = 0
    rng_random = np.random.default_rng(np.random.SeedSequence([seed, 777]))

    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels * t_curr).astype(np.float32)
        b_t = budget_schedule[step - 1] if budget_schedule is not None else 0

        # Step intervention
        if arm_name == "no_repair":
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])

        elif arm_name.startswith("periodic_k"):
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            if step % period_k == 0:
                pr_cost = in_neighbour_pre_repair(index, drift_nodes, static_set)
                ri_cost = 0
                for nid in drift_nodes:
                    ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["pre_repair"] += pr_cost
                categories["reinsert"] += ri_cost
                total_reinsert_count += len(drift_nodes)

        elif arm_name.startswith("lazy_theta_"):
            assert lazy_tracker is not None
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            rep_res = lazy_tracker.check_and_repair(
                pos_step, moving_nodes=drift_nodes, static_nodes=static_nodes
            )
            categories["stale_check"] += rep_res["stale_check_cost"]
            categories["reinsert"] += rep_res["reinsert_cost"]
            total_reinsert_count += rep_res["reinsert_count"]

        elif arm_name == "round_robin":
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            if b_t > 0:
                selected_nodes = []
                for _ in range(b_t):
                    selected_nodes.append(drift_nodes[rr_cursor % len(drift_nodes)])
                    rr_cursor += 1
                ri_cost = 0
                for nid in selected_nodes:
                    ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["reinsert"] += ri_cost
                total_reinsert_count += len(selected_nodes)

        elif arm_name == "fixed_distance":
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            stale_nodes = []
            for nid in drift_nodes:
                p_anch = fixed_dist_anchors[nid]
                curr = pos_step[nid]
                d_val = float(np.linalg.norm(curr - p_anch))
                categories["stale_check"] += 1
                if d_val > median_radius:
                    stale_nodes.append(nid)
            ri_cost = 0
            for nid in stale_nodes:
                ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                fixed_dist_anchors[nid] = pos_step[nid].copy()
            categories["reinsert"] += ri_cost
            total_reinsert_count += len(stale_nodes)

        elif arm_name == "random":
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            if b_t > 0:
                selected_nodes = rng_random.choice(drift_nodes, size=b_t, replace=False).tolist()
                ri_cost = 0
                for nid in selected_nodes:
                    ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["reinsert"] += ri_cost
                total_reinsert_count += len(selected_nodes)

        elif arm_name == "budgeted_priority":
            assert priority_tracker is not None
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            priority_tracker._budget = b_t
            step_res = priority_tracker.step(pos_step, moving_nodes=drift_nodes)
            categories["stale_check"] += step_res["stale_check_cost"]
            categories["reinsert"] += step_res["reinsert_cost"]
            total_reinsert_count += step_res["reinsert_count"]

        # Evaluate recall
        q_d = pos_step[drift_nodes]
        gt_d = brute_force_knn(pos_step, q_d, k=10)
        q_s = pos_step[eval_static]
        gt_s = brute_force_knn(pos_step, q_s, k=10)

        rec16_d = evaluate_recall(index, q_d, gt_d, k=10, ef=16)
        rec32_d = evaluate_recall(index, q_d, gt_d, k=10, ef=32)
        rec16_s = evaluate_recall(index, q_s, gt_s, k=10, ef=16)
        rec32_s = evaluate_recall(index, q_s, gt_s, k=10, ef=32)

        step_records.append(
            {
                "step": step,
                "t": t_curr,
                "drift_rec16": rec16_d,
                "drift_rec32": rec32_d,
                "static_rec16": rec16_s,
                "static_rec32": rec32_s,
            }
        )

    mean_d16 = float(np.mean([r["drift_rec16"] for r in step_records]))
    mean_d32 = float(np.mean([r["drift_rec32"] for r in step_records]))
    mean_s16 = float(np.mean([r["static_rec16"] for r in step_records]))
    mean_s32 = float(np.mean([r["static_rec32"] for r in step_records]))

    final_d16 = step_records[-1]["drift_rec16"]
    final_d32 = step_records[-1]["drift_rec32"]
    final_s16 = step_records[-1]["static_rec16"]
    final_s32 = step_records[-1]["static_rec32"]

    total_cost = sum(categories.values())

    return {
        "arm": arm_name,
        "mean_drift_rec16": mean_d16,
        "mean_drift_rec32": mean_d32,
        "final_drift_rec16": final_d16,
        "final_drift_rec32": final_d32,
        "mean_static_rec16": mean_s16,
        "mean_static_rec32": mean_s32,
        "final_static_rec16": final_s16,
        "final_static_rec32": final_s32,
        "total_cost": total_cost,
        "categories": categories,
        "total_reinsertions": total_reinsert_count,
        "step_records": step_records,
    }


def _run_seed_item_b(
    protocol_name: str,
    seed: int,
    n: int = 500,
    d: int = 8,
    m: int = 8,
    drift_frac: float = 0.30,
    displacement_mult: float = 8.0,
    steps: int = 10,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Run all Item B arms for a single seed under a given protocol."""
    data = setup_protocol_data(
        protocol_name=protocol_name,
        n=n,
        d=d,
        seed=seed,
        drift_frac=drift_frac,
        displacement_mult=displacement_mult,
        t_final=t_final,
    )
    X0 = data["X0"]
    vels = data["vels"]
    drift_nodes = data["drift_nodes"]
    static_nodes = data["static_nodes"]
    eval_static = data["eval_static"]

    base_index = HNSWIndex(dim=d, m=m, seed=seed)
    base_index.add(np.arange(n, dtype=np.int64), X0)

    results: dict[str, Any] = {}

    # 1. Run lazy theta=1.0 first to determine exact reinsertion count
    lazy_1_res = _simulate_single_arm(
        arm_name="lazy_theta_1.0",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        theta=1.0,
    )
    results["lazy_theta_1.0"] = lazy_1_res
    target_count = lazy_1_res["total_reinsertions"]

    # Build budget schedule B_t across steps summing to target_count
    base_b = target_count // steps
    rem = target_count % steps
    budget_schedule = [base_b + (1 if s < rem else 0) for s in range(steps)]
    assert sum(budget_schedule) == target_count

    # 2. Run other existing arms
    results["no_repair"] = _simulate_single_arm(
        arm_name="no_repair",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
    )

    for k in [1, 2, 5]:
        results[f"periodic_k{k}"] = _simulate_single_arm(
            arm_name=f"periodic_k{k}",
            base_index=base_index,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            static_nodes=static_nodes,
            eval_static=eval_static,
            steps=steps,
            t_final=t_final,
            seed=seed,
            period_k=k,
        )

    for th in [0.5, 2.0]:
        results[f"lazy_theta_{th}"] = _simulate_single_arm(
            arm_name=f"lazy_theta_{th}",
            base_index=base_index,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            static_nodes=static_nodes,
            eval_static=eval_static,
            steps=steps,
            t_final=t_final,
            seed=seed,
            theta=th,
        )

    # 3. Controls matched to lazy_theta_1.0 count
    results["round_robin"] = _simulate_single_arm(
        arm_name="round_robin",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        budget_schedule=budget_schedule,
    )

    results["fixed_distance"] = _simulate_single_arm(
        arm_name="fixed_distance",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
    )

    results["random"] = _simulate_single_arm(
        arm_name="random",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        budget_schedule=budget_schedule,
    )

    results["budgeted_priority"] = _simulate_single_arm(
        arm_name="budgeted_priority",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        budget_schedule=budget_schedule,
    )

    return {"seed": seed, "target_count": target_count, "arms": results}


def _run_seed_item_c(
    seed: int,
    n: int = 2000,
    d: int = 8,
    m: int = 8,
    drift_frac: float = 0.10,
    displacement_mult: float = 8.0,
    steps: int = 10,
    t_final: float = 0.30,
    protocol_name: str = "indep_dir",
) -> dict[str, Any]:
    """Run Item C large-graph scaling arms for a single seed."""
    data = setup_protocol_data(
        protocol_name=protocol_name,
        n=n,
        d=d,
        seed=seed,
        drift_frac=drift_frac,
        displacement_mult=displacement_mult,
        t_final=t_final,
    )
    X0 = data["X0"]
    vels = data["vels"]
    drift_nodes = data["drift_nodes"]
    static_nodes = data["static_nodes"]
    eval_static = data["eval_static"]

    base_index = HNSWIndex(dim=d, m=m, seed=seed)
    base_index.add(np.arange(n, dtype=np.int64), X0)

    results: dict[str, Any] = {}

    # Run lazy theta=1.0 first to get budget target
    lazy_res = _simulate_single_arm(
        arm_name="lazy_theta_1.0",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        theta=1.0,
    )
    results["lazy_theta_1.0"] = lazy_res
    target_count = lazy_res["total_reinsertions"]

    base_b = target_count // steps
    rem = target_count % steps
    budget_schedule = [base_b + (1 if s < rem else 0) for s in range(steps)]

    results["no_repair"] = _simulate_single_arm(
        arm_name="no_repair",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
    )

    results["periodic_k1"] = _simulate_single_arm(
        arm_name="periodic_k1",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        period_k=1,
    )

    results["periodic_k5"] = _simulate_single_arm(
        arm_name="periodic_k5",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        period_k=5,
    )

    results["round_robin"] = _simulate_single_arm(
        arm_name="round_robin",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        budget_schedule=budget_schedule,
    )

    results["budgeted_priority"] = _simulate_single_arm(
        arm_name="budgeted_priority",
        base_index=base_index,
        X0=X0,
        vels=vels,
        drift_nodes=drift_nodes,
        static_nodes=static_nodes,
        eval_static=eval_static,
        steps=steps,
        t_final=t_final,
        seed=seed,
        budget_schedule=budget_schedule,
    )

    return {"seed": seed, "target_count": target_count, "arms": results}


def aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed arm results into means across seeds."""
    arm_names = list(seed_results[0]["arms"].keys())
    means: dict[str, Any] = {}

    for arm in arm_names:
        m_d16 = float(np.mean([s["arms"][arm]["mean_drift_rec16"] for s in seed_results]))
        f_d16 = float(np.mean([s["arms"][arm]["final_drift_rec16"] for s in seed_results]))
        m_d32 = float(np.mean([s["arms"][arm]["mean_drift_rec32"] for s in seed_results]))
        f_d32 = float(np.mean([s["arms"][arm]["final_drift_rec32"] for s in seed_results]))

        m_s16 = float(np.mean([s["arms"][arm]["mean_static_rec16"] for s in seed_results]))
        f_s16 = float(np.mean([s["arms"][arm]["final_static_rec16"] for s in seed_results]))
        m_s32 = float(np.mean([s["arms"][arm]["mean_static_rec32"] for s in seed_results]))
        f_s32 = float(np.mean([s["arms"][arm]["final_static_rec32"] for s in seed_results]))

        cost = float(np.mean([s["arms"][arm]["total_cost"] for s in seed_results]))
        reins = float(np.mean([s["arms"][arm]["total_reinsertions"] for s in seed_results]))

        cats = {
            "stale_check": float(
                np.mean([s["arms"][arm]["categories"]["stale_check"] for s in seed_results])
            ),
            "pre_repair": float(
                np.mean([s["arms"][arm]["categories"]["pre_repair"] for s in seed_results])
            ),
            "reinsert": float(
                np.mean([s["arms"][arm]["categories"]["reinsert"] for s in seed_results])
            ),
        }

        means[arm] = {
            "mean_drift_rec16": m_d16,
            "mean_drift_rec32": m_d32,
            "final_drift_rec16": f_d16,
            "final_drift_rec32": f_d32,
            "mean_static_rec16": m_s16,
            "mean_static_rec32": m_s32,
            "final_static_rec16": f_s16,
            "final_static_rec32": f_s32,
            "total_cost": cost,
            "categories": cats,
            "total_reinsertions": reins,
        }

    return means


def evaluate_controls_preregistered_rule(means: dict[str, Any]) -> dict[str, Any]:
    """Evaluate pre-registered rule comparing adaptive policies vs round_robin.

    Rule: Adaptive policies (lazy theta=1.0, budgeted_priority) count as better than
    round_robin only if mean-over-steps drifting recall (ef 16, mean of 3 seeds)
    is higher by >= 0.01 at the same reinsertion count, in the same protocol.
    """
    rr_rec = means["round_robin"]["mean_drift_rec16"]
    lazy_rec = means["lazy_theta_1.0"]["mean_drift_rec16"]
    bp_rec = means["budgeted_priority"]["mean_drift_rec16"]

    lazy_diff = lazy_rec - rr_rec
    bp_diff = bp_rec - rr_rec

    lazy_better = lazy_diff >= 0.01
    bp_better = bp_diff >= 0.01

    gain_attributable_to_staggering = not (lazy_better or bp_better)

    return {
        "round_robin_mean_rec16": rr_rec,
        "lazy_theta_1.0": {
            "mean_rec16": lazy_rec,
            "diff_vs_round_robin": lazy_diff,
            "better_than_round_robin": lazy_better,
        },
        "budgeted_priority": {
            "mean_rec16": bp_rec,
            "diff_vs_round_robin": bp_diff,
            "better_than_round_robin": bp_better,
        },
        "gain_attributable_to_staggering": gain_attributable_to_staggering,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5n protocol check and controls.")
    parser.add_argument("--only-a", action="store_true", help="Run Item A only.")
    parser.add_argument("--only-b", action="store_true", help="Run Item B only.")
    parser.add_argument("--only-c", action="store_true", help="Run Item C only.")
    parser.add_argument("--output", type=str, default="docs/data/controls_benchmark.json")
    args = parser.parse_args()

    out_file = Path(args.output)
    benchmark_data: dict[str, Any] = {}
    if out_file.exists():
        try:
            benchmark_data = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            benchmark_data = {}

    seeds = [42, 43, 44]

    run_all = not (args.only_a or args.only_b or args.only_c)

    if args.only_a or run_all:
        print("=== Step A: Protocol Audit ===")
        res_a = run_item_a(seeds=seeds)
        benchmark_data["item_a_protocol_check"] = res_a
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
        print(f"Step A written to {out_file}")

    if args.only_b or run_all:
        print("=== Step B: Controls Evaluation (n=500) ===")
        item_b_results: dict[str, Any] = {}
        for proto in ["indep_dir", "RandomState", "default_rng"]:
            print(f"  Running protocol: {proto} across seeds {seeds} in parallel...")
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(_run_seed_item_b, protocol_name=proto, seed=s) for s in seeds
                ]
                per_seed = [f.result() for f in futures]
            mean_stats = aggregate_seed_results(per_seed)
            rule_eval = evaluate_controls_preregistered_rule(mean_stats)
            item_b_results[proto] = {
                "per_seed": per_seed,
                "mean": mean_stats,
                "preregistered_rule": rule_eval,
            }
        benchmark_data["item_b_controls"] = item_b_results
        out_file.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
        print(f"Step B written to {out_file}")

    if args.only_c or run_all:
        print("=== Step C: Scaling Evaluation (n=2000, indep_dir) ===")
        print(f"  Running seeds {seeds} in parallel...")
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_run_seed_item_c, seed=s) for s in seeds]
            per_seed_c = [f.result() for f in futures]
        mean_c = aggregate_seed_results(per_seed_c)
        benchmark_data["item_c_n2000"] = {
            "protocol": "indep_dir",
            "per_seed": per_seed_c,
            "mean": mean_c,
        }
        out_file.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
        print(f"Step C written to {out_file}")

    print("Phase 5n execution complete.")


if __name__ == "__main__":
    main()
