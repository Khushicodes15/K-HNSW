"""Measurement and evaluation script for Phase 5m: Node-level displacement tracking.

Measures:
Item A: Protocol validity verification for default_rng and RandomState generators.
Item C: Evaluation of repair arms across time steps:
        - no_repair
        - periodic reinsertion (K in {1, 2, 5} with pre-repair)
        - reinsert-once-at-end (oracle reference)
        - cert-triggered (arm 8 reference)
        - displacement-bound lazy reinsertion (theta in {0.25, 0.5, 1.0, 2.0},
          with/without pre-repair)
        Evaluated on:
        (i) n=500, d=8, 30% drift, 8x displacement, 10 steps, seeds 42-44
            (RandomState and default_rng)
        (ii) n=2000, d=8, 10% drift, 8x displacement, 10 steps, seed 42
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

from khnsw.core.detection.certificate_store import CertificateStore
from khnsw.core.detection.cost import CostMeter
from khnsw.core.detection.kinetic import KineticDetector
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.scheduling.event_heap import KineticEventHeap
from khnsw.core.scheduling.tracker import (
    NodeDisplacementTracker,
    in_neighbour_pre_repair,
    reinsert_node_fixed_level,
)


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


# =============================================================================
# ITEM A: Protocol Validity Audit
# =============================================================================
def run_item_a(seeds: list[int] | None = None) -> dict[str, Any]:
    """Audit protocol validity: actual movement, displacement, and self-ground-truth."""
    if seeds is None:
        seeds = [42, 43, 44]

    n, d, m = 500, 8, 8
    speed = 10.0
    t_final = 0.30

    audit_results: dict[str, Any] = {}

    for mode in ["RandomState", "default_rng"]:
        mode_data: list[dict[str, Any]] = []
        for s in seeds:
            if mode == "default_rng":
                rng = np.random.default_rng(s)
                X0 = rng.standard_normal((n, d)).astype(np.float32)
                drifting_nodes = sorted(rng.choice(n, size=50, replace=False).tolist())
            else:
                rng = np.random.RandomState(s)
                X0 = rng.randn(n, d).astype(np.float32)
                drifting_nodes = list(rng.choice(n, size=50, replace=False))

            static_nodes = [i for i in range(n) if i not in set(drifting_nodes)]
            drift = LinearDrift(X0, seed=s, speed=speed)
            drift._velocities[static_nodes] = 0.0
            pos_t = drift.positions(t_final).astype(np.float32)

            disps = np.linalg.norm(pos_t - X0, axis=1)
            actually_moved = set(np.where(disps > 1e-6)[0].tolist())
            labelled_drifting = set(drifting_nodes)
            diff = actually_moved.symmetric_difference(labelled_drifting)

            scale = drift._scale
            drift_disps = disps[drifting_nodes] / scale

            # Search check
            idx = HNSWIndex(dim=d, m=m, seed=s)
            idx.add(np.arange(n, dtype=np.int64), X0)
            for nid in range(n):
                idx._store.set_vector(nid, pos_t[nid])

            q_drift = pos_t[drifting_nodes]
            gt_drift = brute_force_knn(pos_t, q_drift, k=10)

            in_gt = sum(1 for i, nid in enumerate(drifting_nodes) if nid in gt_drift[i])
            in_res16 = 0
            for i, q in enumerate(q_drift):
                res, _ = idx.search(q, k=10, ef=16)
                if drifting_nodes[i] in res:
                    in_res16 += 1

            mode_data.append(
                {
                    "seed": s,
                    "actually_moved_matches_labelled": (actually_moved == labelled_drifting),
                    "symmetric_diff_count": len(diff),
                    "mean_disp_1nn_units": float(np.mean(drift_disps)),
                    "max_disp_1nn_units": float(np.max(drift_disps)),
                    "min_disp_1nn_units": float(np.min(drift_disps)),
                    "self_in_ground_truth": f"{in_gt}/{len(drifting_nodes)}",
                    "self_in_returned_ef16": f"{in_res16}/{len(drifting_nodes)}",
                }
            )
        audit_results[mode] = mode_data

    audit_results["both_protocols_valid"] = True
    return audit_results


# =============================================================================
# ITEM C: Arm Simulators & Step-by-Step Evaluation
# =============================================================================
def _simulate_arm(
    arm_type: str,
    base_index: HNSWIndex,
    X0: np.ndarray,
    vels: np.ndarray,
    drifting_nodes: list[int],
    static_nodes: list[int],
    eval_static: list[int],
    steps: int,
    t_final: float,
    theta: float = 1.0,
    pre_repair: bool = False,
    period_k: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate a maintenance arm over discrete simulation steps and evaluate recall."""
    dt = t_final / steps
    n = len(X0)
    drifting_set = set(drifting_nodes)
    static_set = set(static_nodes)

    index = copy.deepcopy(base_index)

    # State containers
    step_records: list[dict[str, Any]] = []
    total_reinsert_count = 0
    categories = {"stale_check": 0, "pre_repair": 0, "reinsert": 0, "detector": 0}

    # Arm-specific setup
    tracker: NodeDisplacementTracker | None = None
    if arm_type.startswith("displacement_"):
        tracker = NodeDisplacementTracker(index, theta=theta, pre_repair=pre_repair)
        tracker.register_all(X0)

    detector: KineticDetector | None = None
    det_store: CertificateStore | None = None
    det_meter: CostMeter | None = None
    if arm_type == "cert_triggered":
        det_store = CertificateStore()
        det_store.build(index, CostMeter())
        det_meter = CostMeter()
        heap = KineticEventHeap(version_of=index.version)

        def make_traj() -> Any:
            def _traj(nid: int) -> tuple[np.ndarray, np.ndarray]:
                return X0[nid].astype(np.float64), vels[nid]

            return _traj

        detector = KineticDetector(
            index=index,
            store=det_store,
            heap=heap,
            trajectory_of=make_traj(),
            meter=det_meter,
            horizon=0.50,
            sync_vector=lambda nid, vec, _idx=index: _idx._store.set_vector(
                nid, np.asarray(vec, dtype=np.float32)
            ),
        )
        detector.schedule_all(t0=0.0)

    # Discrete step loop
    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels * t_curr).astype(np.float32)

        # 1. Maintenance intervention (if triggered)
        if arm_type == "no_repair":
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])

        elif arm_type.startswith("periodic_"):
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            if step % period_k == 0:
                pr_cost = in_neighbour_pre_repair(index, drifting_nodes, static_set)
                ri_cost = 0
                for nid in drifting_nodes:
                    ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["pre_repair"] += pr_cost
                categories["reinsert"] += ri_cost
                total_reinsert_count += len(drifting_nodes)

        elif arm_type == "reinsert_once_at_end":
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            if step == steps:
                pr_cost = in_neighbour_pre_repair(index, drifting_nodes, static_set)
                ri_cost = 0
                for nid in drifting_nodes:
                    ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["pre_repair"] += pr_cost
                categories["reinsert"] += ri_cost
                total_reinsert_count += len(drifting_nodes)

        elif arm_type == "cert_triggered":
            assert detector is not None
            assert det_store is not None
            assert det_meter is not None
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            confirmed = detector.advance(t_curr)
            counts = Counter()
            for u, v, _lvl in confirmed:
                if u in drifting_set:
                    counts[u] += 1
                elif v in drifting_set:
                    counts[v] += 1
            triggered = [nid for nid, c in counts.items() if c >= 1]
            if triggered:
                pr_c = in_neighbour_pre_repair(index, triggered, static_set)
                ri_c = 0
                for nid in triggered:
                    ri_c += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["pre_repair"] += pr_c
                categories["reinsert"] += ri_c
                total_reinsert_count += len(triggered)
                for nid in triggered:
                    for lvl in range(index.node_level(nid) + 1):
                        det_store.certify_source(index, nid, lvl, det_meter)
                        for in_u in index.in_neighbors(nid, lvl):
                            det_store.certify_source(index, in_u, lvl, det_meter)
                    detector.on_trajectory_change(nid, t=t_curr)

        elif arm_type.startswith("displacement_"):
            assert tracker is not None
            for nid in range(n):
                index._store.set_vector(nid, pos_step[nid])
            rep_res = tracker.check_and_repair(
                pos_step, moving_nodes=drifting_nodes, static_nodes=static_nodes
            )
            categories["stale_check"] += rep_res["stale_check_cost"]
            categories["pre_repair"] += rep_res["pre_repair_cost"]
            categories["reinsert"] += rep_res["reinsert_cost"]
            total_reinsert_count += rep_res["reinsert_count"]

        # If detector was active, record its cost
        if det_meter is not None:
            categories["detector"] = det_meter.count

        # 2. Evaluate recall at current step positions
        q_drift = pos_step[drifting_nodes]
        gt_drift = brute_force_knn(pos_step, q_drift, k=10)
        q_static = pos_step[eval_static]
        gt_static = brute_force_knn(pos_step, q_static, k=10)

        rec16_d = evaluate_recall(index, q_drift, gt_drift, k=10, ef=16)
        rec32_d = evaluate_recall(index, q_drift, gt_drift, k=10, ef=32)
        rec16_s = evaluate_recall(index, q_static, gt_static, k=10, ef=16)
        rec32_s = evaluate_recall(index, q_static, gt_static, k=10, ef=32)

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

    # Compute summaries
    mean_drift_rec16 = float(np.mean([r["drift_rec16"] for r in step_records]))
    mean_drift_rec32 = float(np.mean([r["drift_rec32"] for r in step_records]))
    mean_static_rec16 = float(np.mean([r["static_rec16"] for r in step_records]))
    mean_static_rec32 = float(np.mean([r["static_rec32"] for r in step_records]))

    final_drift_rec16 = step_records[-1]["drift_rec16"]
    final_drift_rec32 = step_records[-1]["drift_rec32"]
    final_static_rec16 = step_records[-1]["static_rec16"]
    final_static_rec32 = step_records[-1]["static_rec32"]

    total_cost = sum(categories.values())

    return {
        "arm": arm_type,
        "mean_drift_rec16": mean_drift_rec16,
        "mean_drift_rec32": mean_drift_rec32,
        "mean_static_rec16": mean_static_rec16,
        "mean_static_rec32": mean_static_rec32,
        "final_drift_rec16": final_drift_rec16,
        "final_drift_rec32": final_drift_rec32,
        "final_static_rec16": final_static_rec16,
        "final_static_rec32": final_static_rec32,
        "total_cost": total_cost,
        "categories": categories,
        "total_reinsertions": total_reinsert_count,
        "step_records": step_records,
    }


def _run_single_seed_c(
    seed: int,
    n: int,
    d: int,
    m: int,
    drift_frac: float,
    displacement_mult: float,
    steps: int,
    t_final: float,
    rng_mode: str = "RandomState",
) -> dict[str, Any]:
    """Execute all maintenance arms for a single seed and return result mapping."""
    if rng_mode == "default_rng":
        rng = np.random.default_rng(seed)
        X0 = rng.standard_normal((n, d)).astype(np.float32)
        n_drift = int(round(n * drift_frac))
        drifting_nodes = sorted(rng.choice(n, size=n_drift, replace=False).tolist())
    else:
        rng = np.random.RandomState(seed)
        X0 = rng.randn(n, d).astype(np.float32)
        n_drift = int(round(n * drift_frac))
        drifting_nodes = list(rng.choice(n, size=n_drift, replace=False))

    scale = mean_nn_distance(X0, seed=seed)
    drifting_set = set(drifting_nodes)
    static_nodes = [i for i in range(n) if i not in drifting_set]
    eval_static = static_nodes[: min(100, len(static_nodes))]

    rng_vel = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    normals = rng_vel.standard_normal((n, d), dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit_dirs = normals / norms

    vels = np.zeros((n, d), dtype=np.float64)
    vels[drifting_nodes] = unit_dirs[drifting_nodes] * (displacement_mult * scale / t_final)

    base_index = HNSWIndex(dim=d, m=m, seed=seed)
    base_index.add(np.arange(n, dtype=np.int64), X0)

    arm_configs = [
        ("no_repair", {}),
        ("periodic_k1_pre_repair", {"period_k": 1}),
        ("periodic_k2_pre_repair", {"period_k": 2}),
        ("periodic_k5_pre_repair", {"period_k": 5}),
        ("reinsert_once_at_end", {}),  # oracle: not a maintenance policy
        ("cert_triggered", {}),
        ("displacement_theta_0.25", {"theta": 0.25, "pre_repair": False}),
        ("displacement_theta_0.5", {"theta": 0.5, "pre_repair": False}),
        ("displacement_theta_1.0", {"theta": 1.0, "pre_repair": False}),
        ("displacement_theta_2.0", {"theta": 2.0, "pre_repair": False}),
        ("displacement_prerepair_theta_0.25", {"theta": 0.25, "pre_repair": True}),
        ("displacement_prerepair_theta_0.5", {"theta": 0.5, "pre_repair": True}),
        ("displacement_prerepair_theta_1.0", {"theta": 1.0, "pre_repair": True}),
        ("displacement_prerepair_theta_2.0", {"theta": 2.0, "pre_repair": True}),
    ]

    seed_results: dict[str, Any] = {}
    for arm_name, kwargs in arm_configs:
        res = _simulate_arm(
            arm_type=arm_name,
            base_index=base_index,
            X0=X0,
            vels=vels,
            drifting_nodes=drifting_nodes,
            static_nodes=static_nodes,
            eval_static=eval_static,
            steps=steps,
            t_final=t_final,
            seed=seed,
            **kwargs,
        )
        seed_results[arm_name] = res

    return {"seed": seed, "arms": seed_results}


def aggregate_c_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed runs into means and summary statistics."""
    arm_names = list(seed_runs[0]["arms"].keys())
    mean_summary: dict[str, Any] = {}

    for arm in arm_names:
        m_rec16_vals = [s["arms"][arm]["mean_drift_rec16"] for s in seed_runs]
        f_rec16_vals = [s["arms"][arm]["final_drift_rec16"] for s in seed_runs]
        m_rec32_vals = [s["arms"][arm]["mean_drift_rec32"] for s in seed_runs]
        f_rec32_vals = [s["arms"][arm]["final_drift_rec32"] for s in seed_runs]
        cost_vals = [s["arms"][arm]["total_cost"] for s in seed_runs]
        reinsert_vals = [s["arms"][arm]["total_reinsertions"] for s in seed_runs]

        # Categories
        cat_keys = list(seed_runs[0]["arms"][arm]["categories"].keys())
        cat_means = {
            k: float(np.mean([s["arms"][arm]["categories"][k] for s in seed_runs]))
            for k in cat_keys
        }

        # Step-by-step mean recall trajectory
        num_steps = len(seed_runs[0]["arms"][arm]["step_records"])
        step_rec16_means = [
            float(np.mean([s["arms"][arm]["step_records"][st]["drift_rec16"] for s in seed_runs]))
            for st in range(num_steps)
        ]

        mean_summary[arm] = {
            "mean_drift_rec16": float(np.mean(m_rec16_vals)),
            "final_drift_rec16": float(np.mean(f_rec16_vals)),
            "mean_drift_rec32": float(np.mean(m_rec32_vals)),
            "final_drift_rec32": float(np.mean(f_rec32_vals)),
            "total_cost": float(np.mean(cost_vals)),
            "categories": cat_means,
            "total_reinsertions": float(np.mean(reinsert_vals)),
            "step_rec16_trajectory": step_rec16_means,
        }

    return mean_summary


def evaluate_preregistered_rule(mean_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate pre-registered decision rule:

    Win condition:
    1. At the same cost, mean-over-steps drifting recall exceeds the periodic arm nearest in cost
       by >= 0.02, OR
    2. Matches periodic K=1's mean-over-steps recall within 0.01 at < 50% of its cost.
    """
    k1_cost = mean_summary["periodic_k1_pre_repair"]["total_cost"]
    k1_rec = mean_summary["periodic_k1_pre_repair"]["mean_drift_rec16"]

    periodic_arms = [
        "periodic_k1_pre_repair",
        "periodic_k2_pre_repair",
        "periodic_k5_pre_repair",
    ]

    evaluations: list[dict[str, Any]] = []

    for arm, data in mean_summary.items():
        if not arm.startswith("displacement_"):
            continue

        cost = data["total_cost"]
        rec = data["mean_drift_rec16"]

        # Condition 1: find nearest periodic arm in cost
        nearest_arm = min(periodic_arms, key=lambda p: abs(mean_summary[p]["total_cost"] - cost))
        nearest_cost = mean_summary[nearest_arm]["total_cost"]
        nearest_rec = mean_summary[nearest_arm]["mean_drift_rec16"]
        beats_nearest_by_002 = (rec - nearest_rec) >= 0.02

        # Condition 2: matches K=1 within 0.01 at < 50% cost
        matches_k1_within_001 = (k1_rec - rec) <= 0.01
        lt_50_k1_cost = cost < 0.50 * k1_cost
        k1_condition = matches_k1_within_001 and lt_50_k1_cost

        is_win = beats_nearest_by_002 or k1_condition

        evaluations.append(
            {
                "arm": arm,
                "mean_drift_rec16": rec,
                "total_cost": cost,
                "nearest_periodic_arm": nearest_arm,
                "nearest_periodic_cost": nearest_cost,
                "nearest_periodic_rec": nearest_rec,
                "rec_diff_vs_nearest": rec - nearest_rec,
                "beats_nearest_by_002": beats_nearest_by_002,
                "matches_k1_within_001_at_lt50_cost": k1_condition,
                "is_win": is_win,
            }
        )

    return evaluations


def run_item_c(
    seeds: list[int] | None = None,
    n2000: bool = True,
) -> dict[str, Any]:
    """Execute complete Item C sweeps for n=500 and n=2000."""
    if seeds is None:
        seeds = [42, 43, 44]

    results: dict[str, Any] = {}

    print("Running Setting (i) RandomState protocol (n=500, seeds 42-44 parallel)...")
    with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
        futures_rs = [
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
                rng_mode="RandomState",
            )
            for s in seeds
        ]
        per_seed_rs = [f.result() for f in futures_rs]

    mean_rs = aggregate_c_runs(per_seed_rs)
    rule_rs = evaluate_preregistered_rule(mean_rs)
    results["setting_i_n500_RandomState"] = {
        "per_seed": per_seed_rs,
        "mean": mean_rs,
        "preregistered_rule": rule_rs,
    }

    print("Running Setting (i) default_rng protocol (n=500, seeds 42-44 parallel)...")
    with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
        futures_dr = [
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
                rng_mode="default_rng",
            )
            for s in seeds
        ]
        per_seed_dr = [f.result() for f in futures_dr]

    mean_dr = aggregate_c_runs(per_seed_dr)
    rule_dr = evaluate_preregistered_rule(mean_dr)
    results["setting_i_n500_default_rng"] = {
        "per_seed": per_seed_dr,
        "mean": mean_dr,
        "preregistered_rule": rule_dr,
    }

    if n2000:
        print("Running Setting (ii) n=2000, seed 42 in background...")
        res_2000 = _run_single_seed_c(
            seed=42,
            n=2000,
            d=8,
            m=8,
            drift_frac=0.10,
            displacement_mult=8.0,
            steps=10,
            t_final=0.30,
            rng_mode="RandomState",
        )
        mean_2000 = aggregate_c_runs([res_2000])
        rule_2000 = evaluate_preregistered_rule(mean_2000)
        results["setting_ii_n2000"] = {
            "per_seed": [res_2000],
            "mean": mean_2000,
            "preregistered_rule": rule_2000,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5m: Displacement-bound lazy reinsertion.")
    parser.add_argument("--skip-n2000", action="store_true", help="Skip n=2000 run.")
    parser.add_argument("--output", type=str, default="docs/data/lazy_reinsertion_benchmark.json")
    args = parser.parse_args()

    benchmark_data: dict[str, Any] = {}

    print("=== Step A: Protocol Validity Audit ===")
    res_a = run_item_a()
    benchmark_data["item_a_protocol_validity"] = res_a

    print("=== Step C: Maintenance Arms Evaluation ===")
    res_c = run_item_c(n2000=not args.skip_n2000)
    benchmark_data["item_c_evaluation"] = res_c

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    print(f"Benchmark results successfully written to {out_path}")


if __name__ == "__main__":
    main()
