"""Real-world embedding benchmark driver (Member 4 — Benchmarks Lead).

Evaluates HNSW maintenance policies on realistic high-dimensional datasets:
1. SIFT visual descriptors (d=128, non-negative, clustered).
2. GloVe semantic word embeddings (d=50 or d=100, cosine/normalized Euclidean).
3. Clustered synthetic fallback (generate_sift_like) when raw datasets are absent.

Fairness and accounting:
------------------------
- Initial graph uses M=8, M0=16, efConstruction=ef_construction.
- Drift calibrated to empirical 1-NN scale: mean_nn_distance(X0).
- Strict PRNG stream decoupling (indep_dir protocol).
- Pre-registered decision rule applied between round_robin and periodic_k5.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmarks.common import (
    check_decision_rule,
    save_results,
)
from benchmarks.datasets import (
    generate_sift_like,
    load_glove,
    load_sift,
    synthetic_gaussian,
)
from benchmarks.run_synthetic_sweep import (
    build_base_index,
    simulate_arm,
)
from khnsw.core.movement.base import mean_nn_distance


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_benchmark_dataset(
    dataset_name: str,
    data_path: str | None = None,
    n: int = 2000,
    d: int = 128,
    seed: int = 42,
    allow_fallback: bool = False,
) -> tuple[NDArray[np.float32], str]:
    """Load real embeddings or generate a realistic clustered fallback.

    Parameters
    ----------
    dataset_name : str
        One of {"sift", "glove", "sift_synthetic", "gaussian"}.
    data_path : str, optional
        Path to dataset file (.fvecs for SIFT, .txt for GloVe).
    n : int, default=2000
        Number of points to sample or generate.
    d : int, default=128
        Dimensionality.
    seed : int, default=42
        Random seed for sampling or synthetic generation.
    allow_fallback : bool, default=False
        Whether to allow synthetic fallback if the real data file is missing.

    Returns
    -------
    X0 : ndarray of shape (n, d), float32
        Dataset coordinates.
    source_info : str
        Human-readable description of how the data was acquired.
    """
    rng = np.random.default_rng(seed)

    if dataset_name == "sift":
        resolved_path = data_path
        if resolved_path is None:
            _repo_root = Path(__file__).resolve().parent.parent
            _default = _repo_root / "docs" / "data" / "siftsmall" / "siftsmall_base.fvecs"
            if _default.exists():
                resolved_path = str(_default)

        if resolved_path and Path(resolved_path).exists():
            raw = load_sift(resolved_path)
            if len(raw) > n:
                indices = rng.choice(len(raw), size=n, replace=False)
                X = raw[indices]
            else:
                X = raw[:n]
            return X.astype(np.float32), f"real_sift_texmex({resolved_path})"

        if not allow_fallback:
            target_p = resolved_path or data_path or "docs/data/siftsmall/siftsmall_base.fvecs"
            raise FileNotFoundError(
                f"Real SIFT file not found at '{target_p}'. "
                "Download Texmex siftsmall from IRISA "
                "(ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz) "
                "and extract to docs/data/siftsmall/siftsmall_base.fvecs, "
                "or pass --allow-fallback to allow SIFT-like synthetic generation."
            )

        sys.stderr.write(
            "[WARNING] Real SIFT file not found. "
            "Falling back to SIFT-like synthetic clustered data (--allow-fallback passed).\n"
        )
        fallback = generate_sift_like(n=n, d=d, seed=seed)
        return fallback, "SIFT-like synthetic"

    if dataset_name == "glove":
        if data_path and Path(data_path).exists():
            raw = load_glove(data_path, d=d, max_vectors=max(n * 2, 5000))
            if len(raw) > n:
                indices = rng.choice(len(raw), size=n, replace=False)
                X = raw[indices]
            else:
                X = raw[:n]
            return X.astype(np.float32), f"real_glove({data_path})"

        gauss = synthetic_gaussian(n=n, d=d, seed=seed)
        norms = np.linalg.norm(gauss, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (gauss / norms).astype(np.float32), "fallback_glove_spherical_gaussian"

    if dataset_name == "sift_synthetic":
        return generate_sift_like(n=n, d=d, seed=seed), "SIFT-like synthetic"

    return synthetic_gaussian(n=n, d=d, seed=seed), "synthetic_gaussian"


def generate_drift_field_from_data(
    X0: NDArray[np.float32],
    seed: int,
    drift_frac: float = 0.3,
    displacement_mult: float = 8.0,
    t_final: float = 0.30,
) -> tuple[NDArray[np.float32], list[int], float]:
    """Generate velocity vectors for an existing dataset with decoupled PRNG streams."""
    n, d = X0.shape
    rng_ids = np.random.default_rng(seed + 1)
    rng_dirs = np.random.default_rng(seed + 2)

    n_drift = max(1, int(round(n * drift_frac)))
    drift_nodes = sorted(rng_ids.choice(n, size=n_drift, replace=False).tolist())

    scale = mean_nn_distance(X0, sample=min(n, 500), seed=seed)
    if scale <= 0.0:
        scale = 1.0

    speed = (displacement_mult * scale) / max(t_final, 1e-9)

    normals = rng_dirs.standard_normal((n_drift, d)).astype(np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit_dirs = (normals / norms).astype(np.float32)

    vels = np.zeros((n, d), dtype=np.float32)
    for local_i, nid in enumerate(drift_nodes):
        vels[nid] = unit_dirs[local_i] * np.float32(speed)

    return vels, drift_nodes, scale


def run_single_seed_task(
    seed: int,
    dataset_name: str,
    data_path: str | None,
    n: int,
    d: int,
    displacement_mult: float,
    drift_frac: float,
    steps: int,
    t_final: float,
    ef_values: list[int],
    k: int,
    max_eval_queries: int,
    arms: list[str] | None,
    ef_construction: int,
    allow_fallback: bool,
) -> dict[str, Any]:
    """Execute maintenance policies for a single random seed."""
    X0, source_info = load_benchmark_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        n=n,
        d=d,
        seed=seed,
        allow_fallback=allow_fallback,
    )
    actual_d = X0.shape[1]
    vels, drift_nodes, scale = generate_drift_field_from_data(
        X0=X0,
        seed=seed,
        drift_frac=drift_frac,
        displacement_mult=displacement_mult,
        t_final=t_final,
    )

    drift_set = set(drift_nodes)
    static_nodes = [i for i in range(len(X0)) if i not in drift_set]
    eval_drift = drift_nodes[: min(max_eval_queries, len(drift_nodes))]
    eval_static = static_nodes[: min(max_eval_queries, len(static_nodes))]

    base_index, setup_cost = build_base_index(
        n=len(X0),
        d=actual_d,
        seed=seed,
        X0=X0,
        m=8,
        m0=16,
        ef_construction=ef_construction,
    )

    b_matched = max(1, int(math.ceil(len(drift_nodes) / 5)))
    dt = t_final / steps

    arm_configs: dict[str, dict[str, Any]] = {
        "no_repair": {"policy": "none"},
        "periodic_k1": {"policy": "periodic", "K": 1},
        "periodic_k5": {"policy": "periodic", "K": 5},
        "round_robin": {"policy": "round_robin", "B": b_matched, "seed": seed},
        "lazy_theta_1.0": {"policy": "lazy", "theta": 1.0, "seed": seed},
        "lazy_theta_0.5": {"policy": "lazy", "theta": 0.5, "seed": seed},
        "budgeted_priority": {"policy": "budgeted_priority", "B": b_matched, "seed": seed},
    }

    active_arms = (
        list(arm_configs.keys()) if arms is None else [a for a in arms if a in arm_configs]
    )
    seed_arm_results: dict[str, Any] = {}

    for arm_name in active_arms:
        cfg = arm_configs[arm_name]
        res = simulate_arm(
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
        seed_arm_results[arm_name] = res

    return {
        "seed": seed,
        "setup_cost": setup_cost,
        "scale_1nn": scale,
        "source_info": source_info,
        "arms": seed_arm_results,
    }


def run_embedding_benchmark(
    dataset_name: str,
    data_path: str | None = None,
    n: int = 2000,
    d: int = 128,
    seeds: list[int] | None = None,
    displacement_mult: float = 8.0,
    drift_frac: float = 0.3,
    steps: int = 10,
    t_final: float = 0.30,
    ef_values: list[int] | None = None,
    k: int = 10,
    max_eval_queries: int = 100,
    arms: list[str] | None = None,
    ef_construction: int = 16,
    allow_fallback: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Execute maintenance policy benchmark on the given embedding dataset."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if ef_values is None:
        ef_values = [16, 32]

    # Resolve real dataset file path and checksum if applicable
    file_sha256_val: str | None = None
    if dataset_name == "sift":
        resolved_path = data_path
        if resolved_path is None:
            _repo_root = Path(__file__).resolve().parent.parent
            _default = _repo_root / "docs" / "data" / "siftsmall" / "siftsmall_base.fvecs"
            if _default.exists():
                resolved_path = str(_default)
        if resolved_path and Path(resolved_path).exists():
            file_sha256_val = compute_file_sha256(resolved_path)

    all_seed_results: list[dict[str, Any]] = []

    if workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as executor:
            future_to_seed = {
                executor.submit(
                    run_single_seed_task,
                    seed=s,
                    dataset_name=dataset_name,
                    data_path=data_path,
                    n=n,
                    d=d,
                    displacement_mult=displacement_mult,
                    drift_frac=drift_frac,
                    steps=steps,
                    t_final=t_final,
                    ef_values=ef_values,
                    k=k,
                    max_eval_queries=max_eval_queries,
                    arms=arms,
                    ef_construction=ef_construction,
                    allow_fallback=allow_fallback,
                ): s
                for s in seeds
            }
            for fut in as_completed(future_to_seed):
                seed_rec = fut.result()
                all_seed_results.append(seed_rec)
                s_num = seed_rec["seed"]
                sc_val = seed_rec["scale_1nn"]
                print(f"  [Seed {s_num}] Finished n={n} (scale={sc_val:.2f})")
    else:
        for idx, seed in enumerate(seeds, 1):
            seed_rec = run_single_seed_task(
                seed=seed,
                dataset_name=dataset_name,
                data_path=data_path,
                n=n,
                d=d,
                displacement_mult=displacement_mult,
                drift_frac=drift_frac,
                steps=steps,
                t_final=t_final,
                ef_values=ef_values,
                k=k,
                max_eval_queries=max_eval_queries,
                arms=arms,
                ef_construction=ef_construction,
                allow_fallback=allow_fallback,
            )
            all_seed_results.append(seed_rec)
            sc_val = seed_rec["scale_1nn"]
            print(f"  [{idx}/{len(seeds)}] Finished seed={seed} n={n} (scale={sc_val:.2f})")

    # Sort results by seed order
    all_seed_results.sort(key=lambda r: r["seed"])

    # Summary statistics across seeds
    arms_present = list(all_seed_results[0]["arms"].keys())
    arm_stats: dict[str, Any] = {}
    for arm in arms_present:
        rec16_list = [r["arms"][arm]["mean_drift_rec16"] for r in all_seed_results]
        final16_list = [r["arms"][arm]["final_drift_rec16"] for r in all_seed_results]
        cost_list = [r["arms"][arm]["total_cost"] for r in all_seed_results]
        reinsert_list = [r["arms"][arm]["total_reinsertions"] for r in all_seed_results]

        rec32_list = [r["arms"][arm].get("mean_drift_rec32", 0.0) for r in all_seed_results]
        final32_list = [r["arms"][arm].get("final_drift_rec32", 0.0) for r in all_seed_results]

        arm_stats[arm] = {
            "mean_drift_rec16": float(np.mean(rec16_list)),
            "std_drift_rec16": float(np.std(rec16_list)),
            "final_drift_rec16": float(np.mean(final16_list)),
            "std_final_drift_rec16": float(np.std(final16_list)),
            "mean_drift_rec32": float(np.mean(rec32_list)),
            "std_drift_rec32": float(np.std(rec32_list)),
            "final_drift_rec32": float(np.mean(final32_list)),
            "std_final_drift_rec32": float(np.std(final32_list)),
            "mean_cost": float(np.mean(cost_list)),
            "std_cost": float(np.std(cost_list)),
            "mean_reinsertions": float(np.mean(reinsert_list)),
        }

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
            trigger = "margin_ge_0.02"
        elif rr_rec >= per_rec - 0.01 and cost_ratio <= 0.5:
            trigger = "cost_le_0.5_and_near_parity"
        else:
            trigger = "inconclusive_or_not_supported"

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
            "rule_trigger": trigger,
            "n_seeds": len(seeds),
        }

    return {
        "dataset": dataset_name,
        "n": n,
        "d": d,
        "displacement_mult": displacement_mult,
        "drift_frac": drift_frac,
        "steps": steps,
        "seeds": seeds,
        "ef_construction": ef_construction,
        "file_sha256": file_sha256_val,
        "source_info": all_seed_results[0]["source_info"],
        "mean_scale_1nn": float(np.mean([r["scale_1nn"] for r in all_seed_results])),
        "setup_cost_mean": float(np.mean([r["setup_cost"] for r in all_seed_results])),
        "arm_stats": arm_stats,
        "decision_rule": decision_info,
        "seed_results": all_seed_results,
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for embedding benchmarks."""
    parser = argparse.ArgumentParser(
        description="Evaluate drift maintenance on real embedding datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sift",
        choices=["sift", "glove", "sift_synthetic", "gaussian"],
        help="Dataset to benchmark (default: sift).",
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="File path to raw dataset file."
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[2000], help="Node count(s) (default: 2000)."
    )
    parser.add_argument("--d", type=int, default=128, help="Dimensionality (default: 128).")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[110, 111, 112, 113, 114],
        help="Random seeds (default: 110 111 112 113 114).",
    )
    parser.add_argument(
        "--displacement", type=float, default=8.0, help="Displacement multiplier (default: 8.0)."
    )
    parser.add_argument(
        "--drift-frac", type=float, default=0.3, help="Fraction drifting (default: 0.3)."
    )
    parser.add_argument("--steps", type=int, default=10, help="Time steps (default: 10).")
    parser.add_argument("--t-final", type=float, default=0.30, help="End time (default: 0.30).")
    parser.add_argument(
        "--ef-construction", type=int, default=200, help="Construction beam width (default: 200)."
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback to synthetic data if real data file is missing.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel worker processes (default: 1)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/data/real_embeddings_benchmark.json",
        help="Output JSON path.",
    )
    parser.add_argument("--smoke", action="store_true", help="Minimal smoke test run.")
    return parser.parse_args(args)


def main() -> None:
    """CLI entry point."""
    args = parse_args(sys.argv[1:])
    if args.smoke:
        n_list = [100]
        d = 32
        seeds = [110]
        steps = 3
        max_eval = 20
        workers = 1
    else:
        n_list = args.n
        d = args.d
        seeds = args.seeds
        steps = args.steps
        max_eval = 100
        workers = args.workers

    t0 = time.time()
    print("=== K-HNSW Real Embedding Drift Benchmark ===")
    print(
        f"Dataset: {args.dataset} | N: {n_list} | D: {d} | Seeds: {seeds} | "
        f"Disp: {args.displacement} | ef_c: {args.ef_construction} | Workers: {workers}"
    )

    all_runs_by_n: dict[str, Any] = {}

    for n in n_list:
        print(f"\n--- Running evaluation for n={n} ---")
        t_n0 = time.time()
        res = run_embedding_benchmark(
            dataset_name=args.dataset,
            data_path=args.data_path,
            n=n,
            d=d,
            seeds=seeds,
            displacement_mult=args.displacement,
            drift_frac=args.drift_frac,
            steps=steps,
            t_final=args.t_final,
            max_eval_queries=max_eval,
            ef_construction=args.ef_construction,
            allow_fallback=args.allow_fallback,
            workers=workers,
        )
        elapsed_n = time.time() - t_n0
        res["elapsed_seconds"] = elapsed_n
        all_runs_by_n[f"n{n}"] = res

        dec = res.get("decision_rule", {})
        if dec:
            rec_a = dec.get("recall_a", 0.0)
            rec_b = dec.get("recall_b", 0.0)
            rec_m = dec.get("recall_margin", 0.0)
            c_ratio = dec.get("cost_ratio", 1.0)
            a_wins = dec.get("arm_a_wins", False)
            trig = dec.get("rule_trigger")
            print(f"Finished n={n} in {elapsed_n:.1f}s.")
            print(f"  RR Recall@16: {rec_a:.4f} | Periodic Recall@16: {rec_b:.4f}")
            print(f"  Margin: {rec_m:+.4f} | Cost Ratio: {c_ratio:.3f}")
            print(f"  Arm A Wins?: {a_wins} ({trig})")

    total_elapsed = time.time() - t0
    final_output = {
        "metadata": {
            "title": "Real Texmex SIFT-128 Benchmark Evaluation",
            "ef_construction": args.ef_construction,
            "seeds": seeds,
            "displacement_mult": args.displacement,
            "drift_frac": args.drift_frac,
            "steps": steps,
            "elapsed_seconds": total_elapsed,
        },
        "runs": all_runs_by_n if len(n_list) > 1 else all_runs_by_n[f"n{n_list[0]}"],
    }

    save_results(final_output, args.out)
    print(f"\nAll evaluations completed in {total_elapsed:.1f}s. Saved to {args.out}")


if __name__ == "__main__":
    main()
