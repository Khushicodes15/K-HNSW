"""Run interpolation drift benchmark evaluating self-excluded recall alongside committed metric.

This runner evaluates all 5 interpolation drift conditions:
- f_0.25 (without jitter)
- f_0.50 (without jitter)
- f_1.00 (without jitter)
- f_0.50_jitter (with jitter)
- f_1.00_jitter (with jitter)

Across arms:
- no_repair
- periodic_k5
- round_robin
- rebuild

Metric definition:
- k=10, ef in [16, 32]
- Self-excluded drift recall: requests k+1 results at beam width ef, drops the query's
  own node ID by ID (never by distance), takes the first 10. Ground truth is the 10 nearest
  non-self points (self removed by ID).
- Self-hit rate at ef=16: fraction of the 100 evaluated drift queries where self appears in raw
  search results.
- Committed metric: self included on both sides (as evaluated by simulate_arm).
  Every condition, arm, and seed is asserted to match docs/data/interpolation_benchmark.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root and src/ are in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

import benchmarks.run_interpolation_benchmark as interp_mod  # noqa: E402
import benchmarks.run_synthetic_sweep as sweep_mod  # noqa: E402
from benchmarks.run_interpolation_benchmark import (  # noqa: E402
    compute_paired_stats,
    evaluate_rebuild_ceiling,
    generate_interpolation_drift,
)
from benchmarks.run_real_embeddings import load_benchmark_dataset  # noqa: E402
from benchmarks.run_synthetic_sweep import (  # noqa: E402
    build_base_index,
    simulate_arm,
)
from khnsw.core.movement.base import mean_nn_distance  # noqa: E402


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def get_git_commit(cwd: Path) -> str:
    """Get current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def evaluate_colocation(
    pos_final: NDArray[np.float32],
    eval_drift: list[int],
    mean_1nn: float,
    threshold: float = 1e-3,
) -> dict[str, Any]:
    """Evaluate duplicate co-location at final positions."""
    near_count = 0
    min_nonzero_dist = float("inf")

    for u in eval_drift:
        q = pos_final[u]
        d = np.linalg.norm(pos_final - q, axis=1)
        d[u] = np.inf  # exclude self

        min_d = np.min(d)
        if min_d < threshold:
            near_count += 1

        nonzero_d = d[d > 1e-6]
        if len(nonzero_d) > 0:
            min_nz = np.min(nonzero_d)
            if min_nz < min_nonzero_dist:
                min_nonzero_dist = float(min_nz)

    ratio = min_nonzero_dist / mean_1nn if mean_1nn > 0 else 0.0
    return {
        "near_count": near_count,
        "eval_queries": len(eval_drift),
        "smallest_nonzero_dist": min_nonzero_dist,
        "mean_1nn": mean_1nn,
        "ratio_to_mean_1nn": ratio,
    }


def compute_jitter_norm_ratios(
    X0: NDArray[np.float32],
    seed: int,
    mean_1nn: float,
    drift_frac: float = 0.3,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Measure exact jitter displacement norm relative to mean_1nn."""
    vels_nojitter, drift_nodes, _, _ = generate_interpolation_drift(
        X0=X0, seed=seed, f_frac=1.00, drift_frac=drift_frac, t_final=t_final, jitter=False
    )
    vels_jitter, _, _, _ = generate_interpolation_drift(
        X0=X0, seed=seed, f_frac=1.00, drift_frac=drift_frac, t_final=t_final, jitter=True
    )
    pos_final_nojitter = (
        X0.astype(np.float64) + vels_nojitter.astype(np.float64) * t_final
    ).astype(np.float32)
    pos_final_jitter = (X0.astype(np.float64) + vels_jitter.astype(np.float64) * t_final).astype(
        np.float32
    )

    diff = pos_final_jitter[drift_nodes] - pos_final_nojitter[drift_nodes]
    jitter_norms = np.linalg.norm(diff, axis=1)
    ratios = jitter_norms / mean_1nn

    m_r = float(np.mean(ratios))
    return {
        "seed": seed,
        "mean_ratio": m_r,
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
        "ratio_div_005": float(m_r / 0.05),
    }


def run_benchmark(
    repo_root: Path,
    output_path: Path,
    committed_json_path: Path,
    data_path: Path,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Execute the full self-excluded and committed comparison benchmark."""
    start_time = time.time()
    if seeds is None:
        seeds = [110, 111, 112, 113, 114]

    conditions = [
        {"name": "f_0.25", "f_frac": 0.25, "jitter": False},
        {"name": "f_0.50", "f_frac": 0.50, "jitter": False},
        {"name": "f_1.00", "f_frac": 1.00, "jitter": False},
        {"name": "f_0.50_jitter", "f_frac": 0.50, "jitter": True},
        {"name": "f_1.00_jitter", "f_frac": 1.00, "jitter": True},
    ]
    arms = ["no_repair", "periodic_k5", "round_robin", "rebuild"]
    ef_values = [16, 32]
    k_eval = 10
    n = 2000
    d = 128
    ef_construction = 200
    t_final = 0.30
    steps = 10
    dt = t_final / steps

    with open(committed_json_path, encoding="utf-8") as f:
        committed_data = json.load(f)

    file_sha256 = compute_file_sha256(data_path)
    git_commit = get_git_commit(repo_root)

    # Interception context for self-excluded metric evaluation during simulation
    interception_context: dict[str, Any] = {
        "eval_drift": None,
        "current_pos_step": None,
        "self_ex_records": {16: [], 32: []},
        "self_hits": [],
    }

    orig_evaluate_recall = sweep_mod.evaluate_recall
    orig_policy_step = sweep_mod.MaintenancePolicy.step

    def hooked_step(
        self: Any, index: Any, pos_step: NDArray[np.float32], *args: Any, **kwargs: Any
    ) -> Any:
        interception_context["current_pos_step"] = pos_step
        return orig_policy_step(self, index, pos_step, *args, **kwargs)

    def hooked_evaluate_recall(
        index: Any,
        queries: NDArray[np.float32],
        ground_truth: NDArray[np.intp],
        k: int = 10,
        ef: int = 50,
    ) -> float:
        res = orig_evaluate_recall(index, queries, ground_truth, k=k, ef=ef)

        pos_step = interception_context["current_pos_step"]
        eval_drift = interception_context["eval_drift"]

        # Intercept when evaluating drift queries at ef=16 or ef=32
        if (
            pos_step is not None
            and eval_drift is not None
            and len(queries) == len(eval_drift)
            and np.array_equal(queries, pos_step[eval_drift])
            and ef in ef_values
        ):
            diff = pos_step[np.newaxis, :, :] - queries[:, np.newaxis, :]
            sq_dists = np.sum(diff * diff, axis=2)

            recs = []
            hits = []
            for i, u in enumerate(eval_drift):
                row = sq_dists[i].copy()
                row[u] = np.inf  # ground truth: nearest non-self points
                part = np.argpartition(row, k_eval)[:k_eval]
                gt_no_self = set(part)

                # Request k+1 results at that beam width ef
                raw = index.search(queries[i], k=k_eval + 1, ef=ef)
                if isinstance(raw, tuple) and len(raw) == 2 and hasattr(raw[0], "__iter__"):
                    labels = [int(x) for x in raw[0]]
                elif isinstance(raw, (list, np.ndarray)) and len(raw) > 0:
                    first = raw[0]
                    if isinstance(first, (tuple, list)) and len(first) >= 2:
                        labels = [int(item[0]) for item in raw]
                    else:
                        labels = [int(x) for x in raw]
                else:
                    labels = []

                hit = 1 if u in labels else 0

                # Drop query's own node ID by ID (never by distance), take first 10
                filt = [x for x in labels if x != u][:k_eval]
                rec = len(set(filt) & gt_no_self) / k_eval

                recs.append(rec)
                if ef == 16:
                    hits.append(hit)

            interception_context["self_ex_records"][ef].append(float(np.mean(recs)))
            if ef == 16:
                interception_context["self_hits"].append(float(np.mean(hits)))

        return res

    # Install hooks
    sweep_mod.evaluate_recall = hooked_evaluate_recall
    interp_mod.evaluate_recall = hooked_evaluate_recall
    sweep_mod.MaintenancePolicy.step = hooked_step

    raw_seed_records: list[dict[str, Any]] = []
    diagnostics_colocation: dict[str, dict[str, Any]] = {c["name"]: {} for c in conditions}
    diagnostics_jitter: dict[str, Any] = {}

    print("=" * 110)
    print("RUNNING INTERPOLATION SELF-EXCLUDED & COMMITTED COMPARISON BENCHMARK")
    print(f"Git commit: {git_commit}")
    print(f"File SHA256: {file_sha256}")
    print("=" * 110)

    try:
        for seed in seeds:
            print(f"\n>>> Processing Seed {seed}...")
            X0, _ = load_benchmark_dataset("sift", data_path=str(data_path), n=n, d=d, seed=seed)
            base_index, _ = build_base_index(
                n=n, d=d, seed=seed, X0=X0, m=8, m0=16, ef_construction=ef_construction
            )

            mean_1nn = float(mean_nn_distance(X0, sample=min(len(X0), 500), seed=seed))
            diagnostics_jitter[str(seed)] = compute_jitter_norm_ratios(X0, seed, mean_1nn)

            seed_entry: dict[str, Any] = {"seed": seed, "conditions": {}}
            comm_seed_rec = next(s for s in committed_data["seed_records"] if s["seed"] == seed)

            for cond in conditions:
                c_name = cond["name"]
                f_frac = cond["f_frac"]
                jitter = cond["jitter"]

                vels, drift_nodes, targets, _ = generate_interpolation_drift(
                    X0=X0, seed=seed, f_frac=f_frac, drift_frac=0.3, t_final=t_final, jitter=jitter
                )
                drift_set = set(drift_nodes)
                static_nodes = [i for i in range(n) if i not in drift_set]
                eval_drift = drift_nodes[:100]
                eval_static = static_nodes[:100]
                b_matched = max(1, int(math.ceil(len(drift_nodes) / 5)))

                pos_final = (X0.astype(np.float64) + vels.astype(np.float64) * t_final).astype(
                    np.float32
                )
                diagnostics_colocation[c_name][str(seed)] = evaluate_colocation(
                    pos_final, eval_drift, mean_1nn
                )

                interception_context["eval_drift"] = eval_drift

                cond_arms_out: dict[str, Any] = {}

                for arm_name in arms:
                    interception_context["self_ex_records"] = {16: [], 32: []}
                    interception_context["self_hits"] = []

                    if arm_name == "no_repair":
                        p_cfg = {"policy": "none", "seed": seed}
                        arm_res = simulate_arm(
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
                            policy_cfg=p_cfg,
                            ef_values=ef_values,
                            k=k_eval,
                        )
                    elif arm_name == "periodic_k5":
                        p_cfg = {"policy": "periodic", "K": 5, "seed": seed}
                        arm_res = simulate_arm(
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
                            policy_cfg=p_cfg,
                            ef_values=ef_values,
                            k=k_eval,
                        )
                    elif arm_name == "round_robin":
                        p_cfg = {"policy": "round_robin", "B": b_matched, "seed": seed}
                        arm_res = simulate_arm(
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
                            policy_cfg=p_cfg,
                            ef_values=ef_values,
                            k=k_eval,
                        )
                    else:  # rebuild
                        interception_context["current_pos_step"] = pos_final
                        arm_res = evaluate_rebuild_ceiling(
                            pos_final=pos_final,
                            eval_drift=eval_drift,
                            eval_static=eval_static,
                            ef_construction=ef_construction,
                            m=8,
                            m0=16,
                            seed=seed,
                            k=k_eval,
                            ef_values=ef_values,
                        )
                        # Rebuild ceiling trajectory mean equals final step by construction
                        interception_context["self_ex_records"][16] = [
                            interception_context["self_ex_records"][16][-1]
                        ]
                        interception_context["self_ex_records"][32] = [
                            interception_context["self_ex_records"][32][-1]
                        ]

                    # Exact equality assertion against committed data
                    comm_arm_val = comm_seed_rec["conditions"][c_name]["arms"][arm_name][
                        "mean_drift_rec16"
                    ]
                    rerun_comm = arm_res["mean_drift_rec16"]
                    assert rerun_comm == comm_arm_val, (
                        f"FATAL: Exact match failure for {c_name} {arm_name} seed {seed}: "
                        f"rerun={rerun_comm} != committed={comm_arm_val}"
                    )

                    self_ex_mean_16 = float(np.mean(interception_context["self_ex_records"][16]))
                    self_ex_final_16 = float(interception_context["self_ex_records"][16][-1])
                    self_ex_mean_32 = float(np.mean(interception_context["self_ex_records"][32]))
                    self_ex_final_32 = float(interception_context["self_ex_records"][32][-1])
                    self_hit_16 = float(np.mean(interception_context["self_hits"]))

                    cond_arms_out[arm_name] = {
                        "mean_drift_rec16": self_ex_mean_16,
                        "final_drift_rec16": self_ex_final_16,
                        "mean_drift_rec32": self_ex_mean_32,
                        "final_drift_rec32": self_ex_final_32,
                        "self_hit_rate_ef16": self_hit_16,
                        "committed_mean_drift_rec16": rerun_comm,
                        "committed_final_drift_rec16": arm_res["final_drift_rec16"],
                        "committed_mean_drift_rec32": arm_res["mean_drift_rec32"],
                        "committed_final_drift_rec32": arm_res["final_drift_rec32"],
                        "mean_static_rec16": arm_res.get("mean_static_rec16", 0.0),
                        "total_cost": arm_res["total_cost"],
                        "total_reinsertions": arm_res["total_reinsertions"],
                    }

                seed_entry["conditions"][c_name] = {"arms": cond_arms_out}

            raw_seed_records.append(seed_entry)
            print(f"  Seed {seed} complete and all 20 committed checks matched exactly!")

    finally:
        # Restore module functions
        sweep_mod.evaluate_recall = orig_evaluate_recall
        interp_mod.evaluate_recall = orig_evaluate_recall
        sweep_mod.MaintenancePolicy.step = orig_policy_step

    # Aggregate condition-level statistics
    agg_conditions: dict[str, Any] = {}
    pair_defs = [
        ("periodic_k5_vs_no_repair", "periodic_k5", "no_repair"),
        ("round_robin_vs_no_repair", "round_robin", "no_repair"),
        ("round_robin_vs_periodic_k5", "round_robin", "periodic_k5"),
    ]

    for cond in conditions:
        c_name = cond["name"]
        f_frac = cond["f_frac"]
        jitter = cond["jitter"]

        arm_summary: dict[str, Any] = {}
        for arm in arms:
            m16 = [
                s["conditions"][c_name]["arms"][arm]["mean_drift_rec16"] for s in raw_seed_records
            ]
            f16 = [
                s["conditions"][c_name]["arms"][arm]["final_drift_rec16"] for s in raw_seed_records
            ]
            m32 = [
                s["conditions"][c_name]["arms"][arm]["mean_drift_rec32"] for s in raw_seed_records
            ]
            f32 = [
                s["conditions"][c_name]["arms"][arm]["final_drift_rec32"] for s in raw_seed_records
            ]
            st16 = [
                s["conditions"][c_name]["arms"][arm]["mean_static_rec16"] for s in raw_seed_records
            ]
            sh16 = [
                s["conditions"][c_name]["arms"][arm]["self_hit_rate_ef16"] for s in raw_seed_records
            ]
            cst = [s["conditions"][c_name]["arms"][arm]["total_cost"] for s in raw_seed_records]
            reins = [
                s["conditions"][c_name]["arms"][arm]["total_reinsertions"] for s in raw_seed_records
            ]

            arm_summary[arm] = {
                "mean_drift_rec16": {
                    "mean": float(np.mean(m16)),
                    "std": float(np.std(m16, ddof=1)),
                    "per_seed": m16,
                },
                "final_drift_rec16": {
                    "mean": float(np.mean(f16)),
                    "std": float(np.std(f16, ddof=1)),
                    "per_seed": f16,
                },
                "mean_drift_rec32": {
                    "mean": float(np.mean(m32)),
                    "std": float(np.std(m32, ddof=1)),
                    "per_seed": m32,
                },
                "final_drift_rec32": {
                    "mean": float(np.mean(f32)),
                    "std": float(np.std(f32, ddof=1)),
                    "per_seed": f32,
                },
                "mean_static_rec16": {
                    "mean": float(np.mean(st16)),
                    "std": float(np.std(st16, ddof=1)),
                    "per_seed": st16,
                },
                "self_hit_rate_ef16": {
                    "mean": float(np.mean(sh16)),
                    "std": float(np.std(sh16, ddof=1)),
                    "per_seed": sh16,
                },
                "mean_cost": {
                    "mean": float(np.mean(cst)),
                    "std": float(np.std(cst, ddof=1)),
                    "per_seed": cst,
                },
                "mean_reinsertions": {
                    "mean": float(np.mean(reins)),
                    "std": float(np.std(reins, ddof=1)),
                    "per_seed": reins,
                },
            }

        traj_paired: dict[str, dict[str, Any]] = {"rec16": {}, "rec32": {}}
        final_paired: dict[str, dict[str, Any]] = {"rec16": {}, "rec32": {}}

        for p_key, a1, a2 in pair_defs:
            # Trajectory-mean rec16
            diffs_m16 = [
                s["conditions"][c_name]["arms"][a1]["mean_drift_rec16"]
                - s["conditions"][c_name]["arms"][a2]["mean_drift_rec16"]
                for s in raw_seed_records
            ]
            traj_paired["rec16"][p_key] = compute_paired_stats(diffs_m16)

            # Trajectory-mean rec32
            diffs_m32 = [
                s["conditions"][c_name]["arms"][a1]["mean_drift_rec32"]
                - s["conditions"][c_name]["arms"][a2]["mean_drift_rec32"]
                for s in raw_seed_records
            ]
            traj_paired["rec32"][p_key] = compute_paired_stats(diffs_m32)

            # Final-step rec16
            diffs_f16 = [
                s["conditions"][c_name]["arms"][a1]["final_drift_rec16"]
                - s["conditions"][c_name]["arms"][a2]["final_drift_rec16"]
                for s in raw_seed_records
            ]
            final_paired["rec16"][p_key] = compute_paired_stats(diffs_f16)

            # Final-step rec32
            diffs_f32 = [
                s["conditions"][c_name]["arms"][a1]["final_drift_rec32"]
                - s["conditions"][c_name]["arms"][a2]["final_drift_rec32"]
                for s in raw_seed_records
            ]
            final_paired["rec32"][p_key] = compute_paired_stats(diffs_f32)

        agg_conditions[c_name] = {
            "f_frac": f_frac,
            "jitter": jitter,
            "arm_summary": arm_summary,
            "trajectory_mean_paired_differences": traj_paired,
            "final_step_paired_differences": final_paired,
        }

    elapsed = round(time.time() - start_time, 2)

    output_data: dict[str, Any] = {
        "metadata": {
            "title": "K-HNSW Interpolation Drift Self-Excluded Evaluation",
            "description": (
                "Comprehensive self-excluded drift recall evaluation alongside "
                "committed metric reproduction"
            ),
            "git_commit": git_commit,
            "dataset": str(data_path),
            "file_sha256": file_sha256,
            "args": {
                "n": n,
                "d": d,
                "ef_construction": ef_construction,
                "seeds": seeds,
                "k": k_eval,
                "ef_values": ef_values,
                "steps": steps,
                "t_final": t_final,
            },
            "conditions": [c["name"] for c in conditions],
            "arms": arms,
            "elapsed_seconds": elapsed,
        },
        "conditions": agg_conditions,
        "seed_records": raw_seed_records,
        "diagnostics": {
            "colocation_counts": diagnostics_colocation,
            "jitter_norm_ratios": diagnostics_jitter,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 110)
    print(f"BENCHMARK EXECUTION COMPLETED SUCCESSFULLY IN {elapsed}s")
    print(f"Output saved to: {output_path}")
    print("=" * 110)
    return output_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run self-excluded interpolation drift benchmark.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("docs/data/siftsmall/siftsmall_base.fvecs"),
        help="Path to real SIFT siftsmall_base.fvecs file",
    )
    parser.add_argument(
        "--committed-json",
        type=Path,
        default=Path("docs/data/interpolation_benchmark.json"),
        help="Path to committed interpolation_benchmark.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("docs/data/interpolation_self_excluded.json"),
        help="Output path for self-excluded benchmark results",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    data_path = repo_root / args.data_path if not args.data_path.is_absolute() else args.data_path
    comm_path = (
        repo_root / args.committed_json
        if not args.committed_json.is_absolute()
        else args.committed_json
    )
    out_path = (
        repo_root / args.output_path if not args.output_path.is_absolute() else args.output_path
    )

    run_benchmark(
        repo_root=repo_root,
        output_path=out_path,
        committed_json_path=comm_path,
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
