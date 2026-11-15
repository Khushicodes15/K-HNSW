"""Synthetic coordinate drift benchmark sweep driver (Member 4 — Benchmarks Lead).

Evaluates HNSW index degradation and maintenance policies under continuous
isotropic coordinate drift across multiple graph sizes, dimensions, seeds,
and displacement scales.

Policies evaluated:
--------------------
- "no_repair": Unmanaged passive drift (vectors update, graph edges frozen).
- "periodic_k1": Periodic batch reinsertion of all drifting nodes every step (K=1).
- "periodic_k5": Periodic batch reinsertion of all drifting nodes every 5 steps (K=5).
- "round_robin": Temporally staggered cyclic reinsertion with budget B matched to periodic_k5.
- "lazy_theta_1.0": Staleness-bounded lazy reinsertion (theta = 1.0).
- "lazy_theta_0.5": Staleness-bounded lazy reinsertion (theta = 0.5).
- "budgeted_priority": Priority queue reinsertion selecting top B nodes by departure ratio.

Key Constraints & Guidelines:
-----------------------------
- Strict PRNG stream decoupling (indep_dir protocol) via benchmarks.common.
- All headline arms have pre_repair=False (Claim 14).
- Distance accounting partitions into stale_check and reinsert.
- Setup / construction distance computations reported separately.
- Pre-registered decision rule applied: round_robin vs periodic_k5 across seeds.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmarks.common import (
    brute_force_knn,
    check_decision_rule,
    evaluate_recall,
    generate_isotropic_drift_data,
    save_results,
)
from khnsw.core.detection.cost import CostMeter
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.scheduling.policy import MaintenancePolicy


def build_base_index(
    n: int,
    d: int,
    seed: int,
    X0: NDArray[np.float32],
    m: int = 8,
    m0: int = 16,
    ef_construction: int = 16,
) -> tuple[HNSWIndex, int]:
    """Construct an initial HNSW index and record construction distance computations.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Dimensionality.
    seed : int
        Random seed for level generator.
    X0 : ndarray of shape (n, d)
        Initial positions.
    m : int, default=8
        Out-degree bound for upper layers.
    m0 : int, default=16
        Out-degree bound for layer 0.
    ef_construction : int, default=16
        Construction beam width.

    Returns
    -------
    index : HNSWIndex
        Populated graph index.
    setup_cost : int
        Distance computations evaluated during construction.
    """
    index = HNSWIndex(dim=d, m=m, m0=m0, ef_construction=ef_construction, seed=seed)
    ids = np.arange(n, dtype=np.int64)
    index.add(ids, X0)
    setup_cost = index.distance_computations
    return index, setup_cost


def simulate_arm(
    arm_name: str,
    base_index: HNSWIndex,
    X0: NDArray[np.float32],
    vels: NDArray[np.float32],
    drift_nodes: list[int],
    static_nodes: list[int],
    eval_drift: list[int],
    eval_static: list[int],
    steps: int,
    dt: float,
    policy_cfg: dict[str, Any],
    ef_values: list[int] | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Execute a single maintenance arm over the simulation steps.

    Parameters
    ----------
    arm_name : str
        Human-readable arm identifier.
    base_index : HNSWIndex
        Clean initial index clone.
    X0 : ndarray of shape (n, d)
        Initial coordinates.
    vels : ndarray of shape (n, d)
        Per-node velocity vectors.
    drift_nodes : list[int]
        Node IDs that are drifting.
    static_nodes : list[int]
        Node IDs that are stationary.
    eval_drift : list[int]
        Subset of drift node IDs used for recall evaluation.
    eval_static : list[int]
        Subset of static node IDs used for recall evaluation.
    steps : int
        Number of simulation time-steps.
    dt : float
        Time delta per step.
    policy_cfg : dict
        Parameters passed to MaintenancePolicy.
    ef_values : list[int], optional
        Search beam widths to evaluate (default [16, 32]).
    k : int, default=10
        Top-k recall parameter.

    Returns
    -------
    dict
        Structured metrics for this arm.
    """
    if ef_values is None:
        ef_values = [16, 32]

    index = copy.deepcopy(base_index)
    meter = CostMeter()
    policy = MaintenancePolicy(
        policy=policy_cfg.get("policy", "none"),
        K=policy_cfg.get("K", 1),
        B=policy_cfg.get("B", 1),
        theta=policy_cfg.get("theta", 1.0),
        pre_repair=False,
        seed=policy_cfg.get("seed", 42),
        meter=meter,
    )
    policy.register(index, X0)

    categories = {"stale_check": 0, "pre_repair": 0, "reinsert": 0}
    total_reinsert_count = 0
    step_records: list[dict[str, Any]] = []

    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels.astype(np.float64) * t_curr).astype(np.float32)

        dist_before = index.distance_computations
        meter_before = meter.count

        reinserted = policy.step(index, pos_step, t=t_curr, moving_nodes=drift_nodes)

        reinsert_cost_step = index.distance_computations - dist_before
        stale_cost_step = meter.count - meter_before

        categories["reinsert"] += reinsert_cost_step
        categories["stale_check"] += stale_cost_step
        total_reinsert_count += len(reinserted)

        # Recall queries at current positions
        q_d = pos_step[eval_drift]
        gt_d = brute_force_knn(pos_step, q_d, k=k)

        step_rec: dict[str, Any] = {
            "step": step,
            "t": t_curr,
            "reinserted_count": len(reinserted),
            "step_reinsert_cost": reinsert_cost_step,
            "step_stale_cost": stale_cost_step,
        }

        for ef in ef_values:
            rec_d = evaluate_recall(index, q_d, gt_d, k=k, ef=ef)
            step_rec[f"drift_rec{ef}"] = rec_d

        if eval_static:
            q_s = pos_step[eval_static]
            gt_s = brute_force_knn(pos_step, q_s, k=k)
            for ef in ef_values:
                rec_s = evaluate_recall(index, q_s, gt_s, k=k, ef=ef)
                step_rec[f"static_rec{ef}"] = rec_s

        step_records.append(step_rec)

    total_cost = sum(categories.values())

    result: dict[str, Any] = {
        "arm": arm_name,
        "total_cost": total_cost,
        "categories": categories,
        "total_reinsertions": total_reinsert_count,
        "step_records": step_records,
    }

    for ef in ef_values:
        d_vals = [r[f"drift_rec{ef}"] for r in step_records]
        result[f"mean_drift_rec{ef}"] = float(np.mean(d_vals))
        result[f"final_drift_rec{ef}"] = float(d_vals[-1])

        if eval_static:
            s_vals = [r[f"static_rec{ef}"] for r in step_records]
            result[f"mean_static_rec{ef}"] = float(np.mean(s_vals))
            result[f"final_static_rec{ef}"] = float(s_vals[-1])

    return result


def run_seed_configuration(
    n: int,
    d: int,
    seed: int,
    displacement_mult: float,
    drift_frac: float = 0.3,
    steps: int = 10,
    t_final: float = 0.30,
    arms: list[str] | None = None,
    ef_values: list[int] | None = None,
    k: int = 10,
    max_eval_queries: int = 100,
) -> dict[str, Any]:
    """Run all specified arms on one (n, d, displacement, seed) combination.

    Parameters
    ----------
    n : int
        Graph node count.
    d : int
        Dimensionality.
    seed : int
        Random seed.
    displacement_mult : float
        Displacement multiplier.
    drift_frac : float, default=0.3
        Fraction of moving nodes.
    steps : int, default=10
        Number of steps.
    t_final : float, default=0.30
        End time.
    arms : list[str], optional
        List of arm identifiers to run.
    ef_values : list[int], optional
        ef values for search evaluation.
    k : int, default=10
        k for recall evaluation.
    max_eval_queries : int, default=100
        Maximum query count for recall evaluation to bound test runtime.

    Returns
    -------
    dict
        Results keyed by arm name, plus configuration metadata.
    """
    if ef_values is None:
        ef_values = [16, 32]

    # Generate synthetic drift point cloud with decoupled streams
    X0, vels, drift_nodes = generate_isotropic_drift_data(
        n=n,
        d=d,
        seed=seed,
        drift_frac=drift_frac,
        displacement_mult=displacement_mult,
        t_final=t_final,
        steps=steps,
    )

    drift_set = set(drift_nodes)
    static_nodes = [i for i in range(n) if i not in drift_set]

    # Subsample queries for evaluation if node sets exceed max_eval_queries
    eval_drift = drift_nodes[: min(max_eval_queries, len(drift_nodes))]
    eval_static = static_nodes[: min(max_eval_queries, len(static_nodes))]

    # Build clean baseline index
    base_index, setup_cost = build_base_index(
        n=n,
        d=d,
        seed=seed,
        X0=X0,
        m=8,
        m0=16,
        ef_construction=16,
    )

    # Budget matching for round_robin vs periodic_k5
    # periodic_k5 triggers every 5 steps. Over 'steps' steps, triggers = steps // 5.
    # To match periodic_k5 total reinsertions, B = ceil(len(drift_nodes) / 5).
    b_matched = max(1, int(math.ceil(len(drift_nodes) / 5)))
    dt = t_final / steps

    all_arm_configs: dict[str, dict[str, Any]] = {
        "no_repair": {"policy": "none"},
        "periodic_k1": {"policy": "periodic", "K": 1},
        "periodic_k5": {"policy": "periodic", "K": 5},
        "round_robin": {"policy": "round_robin", "B": b_matched, "seed": seed},
        "lazy_theta_1.0": {"policy": "lazy", "theta": 1.0, "seed": seed},
        "lazy_theta_0.5": {"policy": "lazy", "theta": 0.5, "seed": seed},
        "budgeted_priority": {"policy": "budgeted_priority", "B": b_matched, "seed": seed},
    }

    if arms is None:
        target_arms = list(all_arm_configs.keys())
    else:
        target_arms = [a for a in arms if a in all_arm_configs]

    arm_results: dict[str, Any] = {}
    for arm_name in target_arms:
        cfg = all_arm_configs[arm_name]
        arm_res = simulate_arm(
            arm_name=arm_name,
            base_index=base_index,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            static_nodes=static_nodes,
            eval_drift=eval_drift,
            eval_static=eval_static,
            steps=steps,
            dt=dt,
            policy_cfg=cfg,
            ef_values=ef_values,
            k=k,
        )
        arm_results[arm_name] = arm_res

    return {
        "n": n,
        "d": d,
        "seed": seed,
        "displacement_mult": displacement_mult,
        "drift_frac": drift_frac,
        "steps": steps,
        "t_final": t_final,
        "n_drift": len(drift_nodes),
        "setup_cost": setup_cost,
        "b_matched": b_matched,
        "arms": arm_results,
    }


def aggregate_sweep_results(
    raw_runs: list[dict[str, Any]],
    ef_values: list[int] | None = None,
) -> dict[str, Any]:
    """Aggregate individual seed runs into cross-seed summary tables and check decision rule.

    Parameters
    ----------
    raw_runs : list[dict]
        List of configuration run dictionaries from run_seed_configuration.
    ef_values : list[int], optional
        List of ef search beam widths.

    Returns
    -------
    dict
        Aggregated benchmark summary with per-condition statistics and decision rule results.
    """
    if ef_values is None:
        ef_values = [16, 32]

    # Group runs by (n, d, displacement_mult)
    groups: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    for run in raw_runs:
        key = (run["n"], run["d"], float(run["displacement_mult"]))
        groups.setdefault(key, []).append(run)

    conditions_summary: list[dict[str, Any]] = []

    for (n, d, disp), runs in sorted(groups.items()):
        seeds = [r["seed"] for r in runs]
        arms_present = list(runs[0]["arms"].keys())

        arm_stats: dict[str, Any] = {}
        for arm in arms_present:
            mean_d16_list = [r["arms"][arm]["mean_drift_rec16"] for r in runs if "mean_drift_rec16" in r["arms"][arm]]
            final_d16_list = [r["arms"][arm]["final_drift_rec16"] for r in runs if "final_drift_rec16" in r["arms"][arm]]
            cost_list = [r["arms"][arm]["total_cost"] for r in runs]
            reinsert_count_list = [r["arms"][arm]["total_reinsertions"] for r in runs]

            arm_stats[arm] = {
                "mean_drift_rec16": float(np.mean(mean_d16_list)) if mean_d16_list else 0.0,
                "std_drift_rec16": float(np.std(mean_d16_list)) if mean_d16_list else 0.0,
                "final_drift_rec16": float(np.mean(final_d16_list)) if final_d16_list else 0.0,
                "std_final_drift_rec16": float(np.std(final_d16_list)) if final_d16_list else 0.0,
                "mean_cost": float(np.mean(cost_list)),
                "std_cost": float(np.std(cost_list)),
                "mean_reinsertions": float(np.mean(reinsert_count_list)),
            }

            if 32 in ef_values:
                mean_d32_list = [r["arms"][arm]["mean_drift_rec32"] for r in runs if "mean_drift_rec32" in r["arms"][arm]]
                arm_stats[arm]["mean_drift_rec32"] = float(np.mean(mean_d32_list)) if mean_d32_list else 0.0
                arm_stats[arm]["std_drift_rec32"] = float(np.std(mean_d32_list)) if mean_d32_list else 0.0

        # Pre-registered decision rule evaluation between round_robin and periodic_k5
        decision_info: dict[str, Any] = {}
        if "round_robin" in arm_stats and "periodic_k5" in arm_stats:
            rr_rec = arm_stats["round_robin"]["mean_drift_rec16"]
            per_rec = arm_stats["periodic_k5"]["mean_drift_rec16"]
            rr_cost = arm_stats["round_robin"]["mean_cost"]
            per_cost = arm_stats["periodic_k5"]["mean_cost"]

            arm_wins = check_decision_rule(
                recall_a=rr_rec,
                recall_b=per_rec,
                cost_a=rr_cost,
                cost_b=per_cost,
            )
            margin = rr_rec - per_rec
            cost_ratio = (rr_cost / per_cost) if per_cost > 0 else 1.0

            if margin >= 0.02:
                rule_trigger = "margin_ge_0.02"
            elif rr_rec >= per_rec - 0.01 and cost_ratio <= 0.5:
                rule_trigger = "cost_le_0.5_and_near_parity"
            else:
                rule_trigger = "inconclusive_or_not_supported"

            decision_info = {
                "arm_a": "round_robin",
                "arm_b": "periodic_k5",
                "recall_a": rr_rec,
                "recall_b": per_rec,
                "recall_margin": margin,
                "cost_a": rr_cost,
                "cost_b": per_cost,
                "cost_ratio": cost_ratio,
                "arm_a_wins": arm_wins,
                "rule_trigger": rule_trigger,
                "n_seeds": len(seeds),
            }

        conditions_summary.append({
            "n": n,
            "d": d,
            "displacement_mult": disp,
            "seeds": seeds,
            "setup_cost_mean": float(np.mean([r["setup_cost"] for r in runs])),
            "arm_stats": arm_stats,
            "decision_rule": decision_info,
        })

    return {
        "conditions": conditions_summary,
        "raw_runs": raw_runs,
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run synthetic coordinate drift benchmarks across sizes, dimensions, and policies."
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[500, 2000],
        help="List of graph node counts (default: 500 2000).",
    )
    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        default=[8, 64],
        help="List of dimensions (default: 8 64).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="List of integer random seeds (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--displacement",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 8.0, 27.0],
        help="List of displacement multipliers (default: 1.0 3.0 8.0 27.0).",
    )
    parser.add_argument(
        "--drift-frac",
        type=float,
        default=0.3,
        help="Fraction of nodes that move (default: 0.3).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Simulation time steps (default: 10).",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=0.30,
        help="Simulation final timestamp (default: 0.30).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="k nearest neighbors for recall (default: 10).",
    )
    parser.add_argument(
        "--ef",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Search beam widths for evaluation (default: 16 32).",
    )
    parser.add_argument(
        "--max-eval-queries",
        type=int,
        default=100,
        help="Max query count evaluated per step (default: 100).",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of arms to evaluate (default: all supported arms).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/data/synthetic_sweep_benchmark.json",
        help="Output JSON destination path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel evaluation (default: 1).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke-test run with minimal parameters (n=100, d=8, 1 seed, 3 steps).",
    )
    return parser.parse_args(args)


def run_benchmark_suite(parsed_args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full benchmark sweep described by parsed arguments."""
    if parsed_args.smoke:
        sizes = [100]
        dims = [8]
        seeds = [42]
        displacements = [8.0]
        steps = 3
        max_eval = 20
    else:
        sizes = parsed_args.n
        dims = parsed_args.d
        seeds = parsed_args.seeds
        displacements = parsed_args.displacement
        steps = parsed_args.steps
        max_eval = parsed_args.max_eval_queries

    work_items: list[dict[str, Any]] = []
    for n in sizes:
        for d in dims:
            for disp in displacements:
                for seed in seeds:
                    work_items.append({
                        "n": n,
                        "d": d,
                        "seed": seed,
                        "displacement_mult": disp,
                        "drift_frac": parsed_args.drift_frac,
                        "steps": steps,
                        "t_final": parsed_args.t_final,
                        "arms": parsed_args.arms,
                        "ef_values": parsed_args.ef,
                        "k": parsed_args.k,
                        "max_eval_queries": max_eval,
                    })

    print(
        f"=== K-HNSW Synthetic Drift Benchmark Sweep ==="
        f"\nTotal runs: {len(work_items)} | Sizes: {sizes} | Dims: {dims} | Seeds: {len(seeds)}"
        f"\nWorkers: {parsed_args.workers} | Output: {parsed_args.out}"
    )

    t0 = time.time()
    raw_runs: list[dict[str, Any]] = []

    if parsed_args.workers > 1 and len(work_items) > 1:
        with ProcessPoolExecutor(max_workers=parsed_args.workers) as executor:
            futures = [executor.submit(run_seed_configuration, **item) for item in work_items]
            for fut in as_completed(futures):
                res = fut.result()
                raw_runs.append(res)
                print(
                    f"  Done: n={res['n']} d={res['d']} disp={res['displacement_mult']} "
                    f"seed={res['seed']}"
                )
    else:
        for i, item in enumerate(work_items, 1):
            res = run_seed_configuration(**item)
            raw_runs.append(res)
            print(
                f"  [{i}/{len(work_items)}] Done: n={res['n']} d={res['d']} "
                f"disp={res['displacement_mult']} seed={res['seed']}"
            )

    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.2f}s. Aggregating results...")

    summary = aggregate_sweep_results(raw_runs, ef_values=parsed_args.ef)
    summary["metadata"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "parameters": {
            "sizes": sizes,
            "dims": dims,
            "seeds": seeds,
            "displacements": displacements,
            "drift_frac": parsed_args.drift_frac,
            "steps": steps,
            "t_final": parsed_args.t_final,
        },
    }

    save_results(summary, parsed_args.out)
    print(f"Results successfully saved to {parsed_args.out}")

    # Print summary table of decision rule outcomes
    print("\n=== Pre-Registered Decision Rule Summary (round_robin vs periodic_k5) ===")
    print(f"{'n':<6} {'d':<4} {'disp':<6} {'RR Rec16':<10} {'Per Rec16':<10} {'Margin':<8} {'Cost Ratio':<12} {'Arm A Wins?':<12} {'Trigger'}")
    print("-" * 80)
    for cond in summary["conditions"]:
        dec = cond.get("decision_rule", {})
        if dec:
            print(
                f"{cond['n']:<6} {cond['d']:<4} {cond['displacement_mult']:<6.1f} "
                f"{dec.get('recall_a', 0.0):<10.4f} {dec.get('recall_b', 0.0):<10.4f} "
                f"{dec.get('recall_margin', 0.0):<+8.4f} {dec.get('cost_ratio', 1.0):<12.3f} "
                f"{str(dec.get('arm_a_wins', False)):<12} {dec.get('rule_trigger', 'none')}"
            )

    return summary


def main() -> None:
    """CLI entry point."""
    args = parse_args(sys.argv[1:])
    run_benchmark_suite(args)


if __name__ == "__main__":
    main()
