"""Missing control experiments for Phase 6 (Item B).

Evaluates under indep_dir:
- periodic K in {1, 2, 5} WITHOUT pre-repair
- round_robin WITH pre-repair
Matched reinsertion counts to 5n:
- n=500, 30% drift, 8x, 10 steps, seeds 42-44: target 312 reinsertions
- n=2000, 10% drift, 8x, 10 steps, seeds 42-44: target 443 reinsertions
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
from khnsw.core.scheduling.tracker import (
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


def setup_indep_dir_data(
    n: int,
    d: int,
    seed: int,
    drift_frac: float,
    displacement_mult: float,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Generate X0, velocities, and node partition for indep_dir protocol."""
    n_drift = int(round(n * drift_frac))
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
    eval_static = static_nodes[: min(100, len(static_nodes))]

    return {
        "n": n,
        "d": d,
        "seed": seed,
        "scale": scale,
        "X0": X0,
        "vels": vels,
        "drift_nodes": drift_nodes,
        "static_nodes": static_nodes,
        "eval_static": eval_static,
    }


def _simulate_arm(
    arm_name: str,
    base_index: HNSWIndex,
    X0: NDArray[Any],
    vels: NDArray[Any],
    drift_nodes: list[int],
    static_nodes: list[int],
    eval_static: list[int],
    steps: int,
    t_final: float,
    budget_schedule: list[int] | None = None,
    period_k: int = 1,
    pre_repair: bool = False,
) -> dict[str, Any]:
    """Simulate missing control arm."""
    dt = t_final / steps
    n = len(X0)
    static_set = set(static_nodes)

    index = copy.deepcopy(base_index)
    categories = {"stale_check": 0, "pre_repair": 0, "reinsert": 0}
    total_reinsert_count = 0
    step_records = []
    rr_cursor = 0

    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels * t_curr).astype(np.float32)
        b_t = budget_schedule[step - 1] if budget_schedule is not None else 0

        for nid in range(n):
            index._store.set_vector(nid, pos_step[nid])

        if arm_name.startswith("periodic_k"):
            if step % period_k == 0:
                pr_cost = 0
                if pre_repair:
                    pr_cost = in_neighbour_pre_repair(index, drift_nodes, static_set)
                ri_cost = 0
                for nid in drift_nodes:
                    ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["pre_repair"] += pr_cost
                categories["reinsert"] += ri_cost
                total_reinsert_count += len(drift_nodes)

        elif arm_name.startswith("round_robin"):
            if b_t > 0:
                selected_nodes = []
                for _ in range(b_t):
                    selected_nodes.append(drift_nodes[rr_cursor % len(drift_nodes)])
                    rr_cursor += 1
                pr_cost = 0
                if pre_repair:
                    pr_cost = in_neighbour_pre_repair(index, selected_nodes, static_set)
                ri_cost = 0
                for nid in selected_nodes:
                    ri_cost += reinsert_node_fixed_level(index, nid, pos_step[nid])
                categories["pre_repair"] += pr_cost
                categories["reinsert"] += ri_cost
                total_reinsert_count += len(selected_nodes)

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


def _run_seed_job(
    seed: int,
    n: int,
    d: int,
    m: int,
    drift_frac: float,
    displacement_mult: float,
    steps: int,
    t_final: float,
    target_count: int,
) -> dict[str, Any]:
    """Execute all missing control arms for a given seed."""
    data = setup_indep_dir_data(
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

    base_b = target_count // steps
    rem = target_count % steps
    budget_schedule = [base_b + (1 if s < rem else 0) for s in range(steps)]

    arm_configs = [
        ("periodic_k1_no_prerepair", {"period_k": 1, "pre_repair": False}),
        ("periodic_k2_no_prerepair", {"period_k": 2, "pre_repair": False}),
        ("periodic_k5_no_prerepair", {"period_k": 5, "pre_repair": False}),
        ("round_robin_no_prerepair", {"budget_schedule": budget_schedule, "pre_repair": False}),
        ("round_robin_with_prerepair", {"budget_schedule": budget_schedule, "pre_repair": True}),
    ]

    seed_arms: dict[str, Any] = {}
    for arm_name, kwargs in arm_configs:
        res = _simulate_arm(
            arm_name=arm_name,
            base_index=base_index,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            static_nodes=static_nodes,
            eval_static=eval_static,
            steps=steps,
            t_final=t_final,
            **kwargs,
        )
        seed_arms[arm_name] = res

    return {"seed": seed, "target_count": target_count, "arms": seed_arms}


def aggregate_arms(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed runs into mean metrics across seeds."""
    arm_names = list(seed_runs[0]["arms"].keys())
    means: dict[str, Any] = {}

    for arm in arm_names:
        m_d16 = float(np.mean([s["arms"][arm]["mean_drift_rec16"] for s in seed_runs]))
        f_d16 = float(np.mean([s["arms"][arm]["final_drift_rec16"] for s in seed_runs]))
        m_d32 = float(np.mean([s["arms"][arm]["mean_drift_rec32"] for s in seed_runs]))
        f_d32 = float(np.mean([s["arms"][arm]["final_drift_rec32"] for s in seed_runs]))

        m_s16 = float(np.mean([s["arms"][arm]["mean_static_rec16"] for s in seed_runs]))
        f_s16 = float(np.mean([s["arms"][arm]["final_static_rec16"] for s in seed_runs]))
        m_s32 = float(np.mean([s["arms"][arm]["mean_static_rec32"] for s in seed_runs]))
        f_s32 = float(np.mean([s["arms"][arm]["final_static_rec32"] for s in seed_runs]))

        cost = float(np.mean([s["arms"][arm]["total_cost"] for s in seed_runs]))
        reins = float(np.mean([s["arms"][arm]["total_reinsertions"] for s in seed_runs]))

        cats = {
            "stale_check": float(
                np.mean([s["arms"][arm]["categories"]["stale_check"] for s in seed_runs])
            ),
            "pre_repair": float(
                np.mean([s["arms"][arm]["categories"]["pre_repair"] for s in seed_runs])
            ),
            "reinsert": float(
                np.mean([s["arms"][arm]["categories"]["reinsert"] for s in seed_runs])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Item B: Missing control evaluation.")
    parser.add_argument("--only-n500", action="store_true", help="Run n=500 only.")
    parser.add_argument("--only-n2000", action="store_true", help="Run n=2000 only.")
    parser.add_argument("--output", type=str, default="docs/data/missing_controls_benchmark.json")
    args = parser.parse_args()

    out_file = Path(args.output)
    benchmark_data: dict[str, Any] = {}
    if out_file.exists():
        try:
            benchmark_data = json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            benchmark_data = {}

    seeds = [42, 43, 44]
    run_all = not (args.only_n500 or args.only_n2000)

    if args.only_n500 or run_all:
        print("=== Running n=500 Missing Controls (indep_dir, seeds 42-44 parallel) ===")
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures_500 = [
                executor.submit(
                    _run_seed_job,
                    seed=s,
                    n=500,
                    d=8,
                    m=8,
                    drift_frac=0.30,
                    displacement_mult=8.0,
                    steps=10,
                    t_final=0.30,
                    target_count=312,
                )
                for s in seeds
            ]
            per_seed_500 = [f.result() for f in futures_500]
        mean_500 = aggregate_arms(per_seed_500)
        benchmark_data["n500"] = {"per_seed": per_seed_500, "mean": mean_500}
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
        print(f"n=500 results written to {out_file}")

    if args.only_n2000 or run_all:
        print("=== Running n=2000 Missing Controls (indep_dir, seeds 42-44 parallel) ===")
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures_2000 = [
                executor.submit(
                    _run_seed_job,
                    seed=s,
                    n=2000,
                    d=8,
                    m=8,
                    drift_frac=0.10,
                    displacement_mult=8.0,
                    steps=10,
                    t_final=0.30,
                    target_count=443,
                )
                for s in seeds
            ]
            per_seed_2000 = [f.result() for f in futures_2000]
        mean_2000 = aggregate_arms(per_seed_2000)
        benchmark_data["n2000"] = {"per_seed": per_seed_2000, "mean": mean_2000}
        out_file.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
        print(f"n=2000 results written to {out_file}")

    print("Missing controls benchmark complete.")


if __name__ == "__main__":
    main()
