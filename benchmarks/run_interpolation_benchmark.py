"""On-manifold interpolation drift benchmark driver (Member 4 â€” Benchmarks Lead).

Evaluates HNSW maintenance policies under on-manifold interpolation drift:
Each drifting node moves linearly toward a randomly chosen other data point,
up to fraction f of the distance over 10 discrete steps.

Settings:
---------
- Real SIFT dataset (n=2000, d=128, ef_construction=200).
- Seeds: 110-114 (5 distinct evaluation seeds).
- Conditions:
  - f = 0.25 (quarter-way interpolation)
  - f = 0.50 (half-way interpolation)
  - f = 1.00 (full interpolation to target point)
  - f = 0.50 with small random jitter added to target (rules out duplicate ties)
- Arms:
  - no_repair: passive drift (zero maintenance)
  - periodic_k5: batch reinsertion every 5 steps
  - round_robin: budgeted incremental reinsertion each step (B = ceil(n_drift / 5))
  - rebuild: empirical ceiling from fresh HNSW construction on final positions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmarks.common import (
    brute_force_knn,
    evaluate_recall,
)
from benchmarks.run_real_embeddings import load_benchmark_dataset
from benchmarks.run_synthetic_sweep import (
    build_base_index,
    simulate_arm,
)
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance

T_STUDENT_95_N5 = 2.7764451051977987


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def generate_interpolation_drift(
    X0: NDArray[np.float32],
    seed: int,
    f_frac: float = 0.5,
    drift_frac: float = 0.3,
    t_final: float = 0.30,
    jitter: bool = False,
    jitter_scale: float = 0.05,
) -> tuple[NDArray[np.float32], list[int], dict[int, int], NDArray[np.float32]]:
    """Generate velocity vectors for linear interpolation drift toward random targets.

    Uses strictly decoupled PRNG streams:
    - seed + 1: drifting node selection (rng_id)
    - seed + 3: target node selection (rng_target)
    - seed + 4: jitter generation (rng_jitter, if enabled)
    """
    n, d = X0.shape
    rng_id = np.random.default_rng(seed + 1)
    rng_target = np.random.default_rng(seed + 3)
    rng_jitter = np.random.default_rng(seed + 4)

    n_drift = max(1, int(round(n * drift_frac)))
    drift_nodes = sorted(rng_id.choice(n, size=n_drift, replace=False).tolist())

    targets: dict[int, int] = {}
    for u in drift_nodes:
        choices = [j for j in range(n) if j != u]
        targets[u] = int(rng_target.choice(choices))

    target_matrix = np.empty((len(drift_nodes), d), dtype=np.float32)
    for i, u in enumerate(drift_nodes):
        target_matrix[i] = X0[targets[u]]

    if jitter:
        scale = mean_nn_distance(X0, sample=min(n, 500), seed=seed)
        if scale <= 0.0:
            scale = 1.0
        noise = rng_jitter.normal(
            loc=0.0, scale=jitter_scale * scale, size=(len(drift_nodes), d)
        ).astype(np.float32)
        target_matrix = target_matrix + noise

    vels = np.zeros((n, d), dtype=np.float32)
    for i, u in enumerate(drift_nodes):
        disp_vec = target_matrix[i] - X0[u]
        vels[u] = (disp_vec * f_frac) / max(t_final, 1e-9)

    return vels, drift_nodes, targets, target_matrix


def evaluate_rebuild_ceiling(
    pos_final: NDArray[np.float32],
    eval_drift: list[int],
    eval_static: list[int],
    ef_construction: int = 200,
    m: int = 8,
    m0: int = 16,
    seed: int = 42,
    k: int = 10,
    ef_values: list[int] | None = None,
) -> dict[str, Any]:
    """Build fresh HNSW on final positions to serve as an empirical upper bound."""
    if ef_values is None:
        ef_values = [16, 32]
    n, d = pos_final.shape
    idx = HNSWIndex(dim=d, m=m, m0=m0, ef_construction=ef_construction, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), pos_final)
    cost = idx.distance_computations

    q_d = pos_final[eval_drift]
    gt_d = brute_force_knn(pos_final, q_d, k=k)

    res: dict[str, Any] = {
        "arm": "rebuild",
        "total_cost": cost,
        "total_reinsertions": 0,
        "categories": {"rebuild_build": cost},
    }

    for ef in ef_values:
        rec_d = evaluate_recall(idx, q_d, gt_d, k=k, ef=ef)
        res[f"mean_drift_rec{ef}"] = rec_d
        res[f"final_drift_rec{ef}"] = rec_d

    if eval_static:
        q_s = pos_final[eval_static]
        gt_s = brute_force_knn(pos_final, q_s, k=k)
        for ef in ef_values:
            rec_s = evaluate_recall(idx, q_s, gt_s, k=k, ef=ef)
            res[f"mean_static_rec{ef}"] = rec_s
            res[f"final_static_rec{ef}"] = rec_s

    return res


def run_single_seed_task(
    seed: int,
    data_path: str,
    n: int = 2000,
    d: int = 128,
    ef_construction: int = 200,
    steps: int = 10,
    t_final: float = 0.30,
    drift_frac: float = 0.3,
    conditions: list[dict[str, Any]] | None = None,
    ef_values: list[int] | None = None,
    k: int = 10,
    max_eval_queries: int = 100,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Execute all interpolation conditions and arms for a single seed."""
    if conditions is None:
        conditions = [
            {"name": "f_0.25", "f_frac": 0.25, "jitter": False},
            {"name": "f_0.50", "f_frac": 0.50, "jitter": False},
            {"name": "f_1.00", "f_frac": 1.00, "jitter": False},
            {"name": "f_0.50_jitter", "f_frac": 0.50, "jitter": True},
        ]
    if ef_values is None:
        ef_values = [16, 32]

    X0, source_info = load_benchmark_dataset(
        "sift", data_path=data_path, n=n, d=d, seed=seed, allow_fallback=allow_fallback
    )
    base_index, setup_cost = build_base_index(
        n=n, d=d, seed=seed, X0=X0, m=8, m0=16, ef_construction=ef_construction
    )

    scale_1nn = mean_nn_distance(X0, sample=min(n, 500), seed=seed)
    dt = t_final / steps

    seed_record: dict[str, Any] = {
        "seed": seed,
        "setup_cost": setup_cost,
        "scale_1nn": float(scale_1nn),
        "source_info": source_info,
        "conditions": {},
    }

    for cond in conditions:
        c_name = cond["name"]
        f_frac = cond["f_frac"]
        jitter = cond["jitter"]

        vels, drift_nodes, targets, target_pts = generate_interpolation_drift(
            X0=X0,
            seed=seed,
            f_frac=f_frac,
            drift_frac=drift_frac,
            t_final=t_final,
            jitter=jitter,
        )

        drift_set = set(drift_nodes)
        static_nodes = [i for i in range(n) if i not in drift_set]
        eval_drift = drift_nodes[:max_eval_queries]
        eval_static = static_nodes[:max_eval_queries]
        b_matched = max(1, int(math.ceil(len(drift_nodes) / 5)))

        cond_arms: dict[str, Any] = {}

        # 1. no_repair
        cond_arms["no_repair"] = simulate_arm(
            arm_name="no_repair",
            base_index=base_index,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            static_nodes=static_nodes,
            eval_drift=eval_drift,
            eval_static=eval_static,
            steps=steps,
            dt=dt,
            policy_cfg={"policy": "none", "seed": seed},
            ef_values=ef_values,
            k=k,
        )

        # 2. periodic_k5
        cond_arms["periodic_k5"] = simulate_arm(
            arm_name="periodic_k5",
            base_index=base_index,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            static_nodes=static_nodes,
            eval_drift=eval_drift,
            eval_static=eval_static,
            steps=steps,
            dt=dt,
            policy_cfg={"policy": "periodic", "K": 5, "seed": seed},
            ef_values=ef_values,
            k=k,
        )

        # 3. round_robin
        cond_arms["round_robin"] = simulate_arm(
            arm_name="round_robin",
            base_index=base_index,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            static_nodes=static_nodes,
            eval_drift=eval_drift,
            eval_static=eval_static,
            steps=steps,
            dt=dt,
            policy_cfg={"policy": "round_robin", "B": b_matched, "seed": seed},
            ef_values=ef_values,
            k=k,
        )

        # 4. Fresh rebuild ceiling
        pos_final = (X0.astype(np.float64) + vels.astype(np.float64) * t_final).astype(np.float32)
        cond_arms["rebuild"] = evaluate_rebuild_ceiling(
            pos_final=pos_final,
            eval_drift=eval_drift,
            eval_static=eval_static,
            ef_construction=ef_construction,
            m=8,
            m0=16,
            seed=seed,
            k=k,
            ef_values=ef_values,
        )

        seed_record["conditions"][c_name] = {
            "f_frac": f_frac,
            "jitter": jitter,
            "arms": cond_arms,
        }

    return seed_record


def compute_paired_stats(
    diffs: list[float],
) -> dict[str, Any]:
    """Compute mean, sample std, positive count, and 95% Student's t interval."""
    arr = np.array(diffs, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    t_crit = T_STUDENT_95_N5 if n == 5 else 2.0
    margin = t_crit * se
    ci_lower = mean - margin
    ci_upper = mean + margin
    pos_count = int(np.sum(arr > 0))

    return {
        "per_seed_diffs": [float(x) for x in arr],
        "mean": mean,
        "std": std,
        "positive_seeds": pos_count,
        "n_seeds": n,
        "t_crit": t_crit,
        "t_interval_95": [float(ci_lower), float(ci_upper)],
        "excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
    }


def aggregate_benchmark_results(
    raw_seed_records: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    arms: list[str],
    ef_values: list[int],
) -> dict[str, Any]:
    """Aggregate per-condition and per-arm statistics across seeds."""
    agg_conditions: dict[str, Any] = {}

    for cond in conditions:
        c_name = cond["name"]
        agg_conditions[c_name] = {
            "f_frac": cond["f_frac"],
            "jitter": cond["jitter"],
            "arm_summary": {},
            "paired_differences": {},
        }

        # Arm summary
        for arm in arms:
            arm_data: dict[str, Any] = {}
            for metric in [
                "mean_drift_rec16",
                "final_drift_rec16",
                "mean_drift_rec32",
                "final_drift_rec32",
                "mean_static_rec16",
                "final_static_rec16",
                "mean_static_rec32",
                "final_static_rec32",
                "total_cost",
                "total_reinsertions",
            ]:
                vals = [
                    s["conditions"][c_name]["arms"][arm][metric]
                    for s in raw_seed_records
                    if metric in s["conditions"][c_name]["arms"][arm]
                ]
                if vals:
                    arm_data[metric] = {
                        "per_seed": vals,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    }
            agg_conditions[c_name]["arm_summary"][arm] = arm_data

        # Paired differences
        pairs = [
            ("periodic_k5_vs_no_repair", "periodic_k5", "no_repair"),
            ("round_robin_vs_no_repair", "round_robin", "no_repair"),
            ("round_robin_vs_periodic_k5", "round_robin", "periodic_k5"),
        ]

        for p_label, arm_a, arm_b in pairs:
            agg_conditions[c_name]["paired_differences"][p_label] = {}
            for ef in ef_values:
                for prefix in ["mean_drift", "final_drift"]:
                    mkey = f"{prefix}_rec{ef}"
                    diffs = [
                        s["conditions"][c_name]["arms"][arm_a][mkey]
                        - s["conditions"][c_name]["arms"][arm_b][mkey]
                        for s in raw_seed_records
                    ]
                    agg_conditions[c_name]["paired_differences"][p_label][mkey] = (
                        compute_paired_stats(diffs)
                    )

    return agg_conditions


def run_interpolation_benchmark(
    data_path: str,
    n: int = 2000,
    d: int = 128,
    ef_construction: int = 200,
    seeds: list[int] | None = None,
    conditions: list[dict[str, Any]] | None = None,
    output_path: str | None = None,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Execute the full interpolation benchmark suite and write JSON results."""
    t0 = time.time()
    if seeds is None:
        seeds = [110, 111, 112, 113, 114]
    if conditions is None:
        conditions = [
            {"name": "f_0.25", "f_frac": 0.25, "jitter": False},
            {"name": "f_0.50", "f_frac": 0.50, "jitter": False},
            {"name": "f_1.00", "f_frac": 1.00, "jitter": False},
            {"name": "f_0.50_jitter", "f_frac": 0.50, "jitter": True},
        ]
    arms = ["no_repair", "periodic_k5", "round_robin", "rebuild"]
    ef_values = [16, 32]

    file_sha256 = compute_file_sha256(data_path) if Path(data_path).exists() else "missing"

    raw_seeds: list[dict[str, Any]] = []
    print(
        f"Executing interpolation benchmark: n={n}, ef_c={ef_construction}, seeds={seeds}...",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=min(len(seeds), 5)) as pool:
        futures = [
            pool.submit(
                run_single_seed_task,
                seed=s,
                data_path=data_path,
                n=n,
                d=d,
                ef_construction=ef_construction,
                conditions=conditions,
                ef_values=ef_values,
                allow_fallback=allow_fallback,
            )
            for s in seeds
        ]
        for f in futures:
            res = f.result()
            raw_seeds.append(res)
            print(f"  Completed seed {res['seed']}", flush=True)

    # Sort seeds deterministically
    raw_seeds.sort(key=lambda x: x["seed"])

    aggregated = aggregate_benchmark_results(raw_seeds, conditions, arms, ef_values)

    results: dict[str, Any] = {
        "metadata": {
            "title": "K-HNSW On-Manifold Interpolation Drift Evaluation",
            "dataset": "siftsmall_base.fvecs",
            "file_sha256": file_sha256,
            "n": n,
            "d": d,
            "ef_construction": ef_construction,
            "seeds": seeds,
            "arms": arms,
            "ef_values": ef_values,
            "conditions": [c["name"] for c in conditions],
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "conditions": aggregated,
        "seed_records": raw_seeds,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote benchmark results to {out_p}", flush=True)

    return results


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-HNSW On-Manifold Interpolation Benchmark Driver"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run quick smoke test with small parameters"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[110, 111, 112, 113, 114], help="Random seeds"
    )
    parser.add_argument("--n", type=int, default=2000, help="Number of dataset vectors")
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="efConstruction for base index and rebuild",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "docs"
            / "data"
            / "siftsmall"
            / "siftsmall_base.fvecs"
        ),
        help="Path to siftsmall_base.fvecs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "docs"
            / "data"
            / "interpolation_benchmark.json"
        ),
        help="Path to write JSON results",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.smoke:
        print("Running in SMOKE mode...")
        run_interpolation_benchmark(
            data_path=args.data_path,
            n=50,
            d=128,
            ef_construction=16,
            seeds=[110],
            conditions=[{"name": "f_0.50", "f_frac": 0.50, "jitter": False}],
            output_path=None,
        )
        return

    run_interpolation_benchmark(
        data_path=args.data_path,
        n=args.n,
        d=128,
        ef_construction=args.ef_construction,
        seeds=args.seeds,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
