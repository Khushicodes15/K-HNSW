"""External hnswlib baseline comparison driver (Member 4 — Benchmarks Lead).

Evaluates the standard production C++ hnswlib library under continuous
coordinate drift using its native replacement protocol:
    `index.mark_deleted(label)` followed by `index.add_items(new_vector, label, replace_deleted=True)`

Accounting Notice (§6 in member4.md):
------------------------------------
Because C++ hnswlib uses compiled AVX/SSE kernels without exposing internal
distance counters, comparison against K-HNSW is conducted using:
1. Normalized reinsertion operation counts (fair operational work parity).
2. Pure wall-clock query and reinsertion throughput (seconds / QPS).

Graceful Environment Handling:
------------------------------
If `hnswlib` is not installed in the local environment, the runner provides
graceful diagnostics and synthetic emulation mode for verification.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmarks.common import (
    brute_force_knn,
    evaluate_recall,
    generate_isotropic_drift_data,
    save_results,
)

try:
    import hnswlib

    HNSWLIB_AVAILABLE = True
except ImportError:
    hnswlib = None
    HNSWLIB_AVAILABLE = False


class HnswlibBenchmarkEngine:
    """Benchmark wrapper around hnswlib or emulation fallback."""

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        seed: int = 42,
        max_elements: int = 10000,
    ) -> None:
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.seed = seed
        self.max_elements = max_elements
        self.is_native = HNSWLIB_AVAILABLE

        if self.is_native:
            self._index = hnswlib.Index(space="l2", dim=dim)
            self._index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=m,
                random_seed=seed,
                allow_replace_deleted=True,
            )
        else:
            # Emulation mode using numpy vectors when hnswlib binary is absent
            self._vectors: dict[int, NDArray[np.float32]] = {}

    def add(self, ids: NDArray[np.int64], vectors: NDArray[np.float32]) -> float:
        """Add vectors into the index and return wall-clock seconds spent."""
        t0 = time.perf_counter()
        if self.is_native:
            ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
            vecs_arr = np.ascontiguousarray(vectors, dtype=np.float32)
            self._index.add_items(vecs_arr, ids_arr)
        else:
            for i, nid in enumerate(ids):
                self._vectors[int(nid)] = np.asarray(vectors[i], dtype=np.float32).copy()
        return time.perf_counter() - t0

    def move(self, node_id: int, new_vector: NDArray[np.float32]) -> float:
        """Execute mark_deleted + replace_deleted update. Returns wall-clock seconds."""
        t0 = time.perf_counter()
        nid = int(node_id)
        vec = np.ascontiguousarray(new_vector.reshape(1, -1), dtype=np.float32)
        if self.is_native:
            self._index.mark_deleted(nid)
            self._index.add_items(vec, np.array([nid], dtype=np.int64), replace_deleted=True)
        else:
            self._vectors[nid] = vec[0].copy()
        return time.perf_counter() - t0

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        ef: int = 50,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Query top-k nearest neighbors."""
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if self.is_native:
            self._index.set_ef(ef)
            labels, dists = self._index.knn_query(q, k=k)
            return labels[0].astype(np.int64), dists[0].astype(np.float32)
        else:
            # Brute force emulation
            all_ids = list(self._vectors.keys())
            mat = np.stack([self._vectors[i] for i in all_ids], axis=0)
            diffs = mat - q[0]
            sq_dists = np.sum(diffs * diffs, axis=1)
            top_k_indices = np.argsort(sq_dists)[:k]
            res_ids = np.array([all_ids[i] for i in top_k_indices], dtype=np.int64)
            res_dists = sq_dists[top_k_indices].astype(np.float32)
            return res_ids, res_dists


def simulate_hnswlib_policy(
    policy_name: str,
    n: int,
    d: int,
    seed: int,
    X0: NDArray[np.float32],
    vels: NDArray[np.float32],
    drift_nodes: list[int],
    steps: int = 10,
    dt: float = 0.03,
    ef_values: list[int] | None = None,
    k: int = 10,
    max_eval_queries: int = 100,
) -> dict[str, Any]:
    """Execute a drift maintenance schedule on hnswlib.

    Supported policies:
    - "no_repair": passive coordinate drift without calling move()
    - "periodic_k5": all moving nodes updated every 5 steps
    - "round_robin": B = ceil(n_drift / 5) moving nodes updated each step
    - "periodic_k1": all moving nodes updated every step
    """
    if ef_values is None:
        ef_values = [16, 32]

    engine = HnswlibBenchmarkEngine(
        dim=d,
        m=16,
        ef_construction=16,
        seed=seed,
        max_elements=max(n * 2, 2000),
    )
    ids = np.arange(n, dtype=np.int64)
    setup_wall_clock = engine.add(ids, X0)

    n_drift = len(drift_nodes)
    b_matched = max(1, int(math.ceil(n_drift / 5)))
    eval_drift = drift_nodes[: min(max_eval_queries, n_drift)]

    step_records: list[dict[str, Any]] = []
    total_reinsert_count = 0
    total_update_time = 0.0
    rr_cursor = 0

    for step in range(1, steps + 1):
        t_curr = round(step * dt, 5)
        pos_step = (X0.astype(np.float64) + vels.astype(np.float64) * t_curr).astype(np.float32)

        to_update: list[int] = []
        if policy_name == "no_repair":
            to_update = []
        elif policy_name == "periodic_k1":
            to_update = list(drift_nodes)
        elif policy_name == "periodic_k5":
            if step % 5 == 0:
                to_update = list(drift_nodes)
        elif policy_name == "round_robin":
            for _ in range(b_matched):
                to_update.append(drift_nodes[rr_cursor % n_drift])
                rr_cursor += 1

        step_update_time = 0.0
        for nid in to_update:
            step_update_time += engine.move(nid, pos_step[nid])

        total_reinsert_count += len(to_update)
        total_update_time += step_update_time

        # Evaluate recall
        q_d = pos_step[eval_drift]
        gt_d = brute_force_knn(pos_step, q_d, k=k)

        rec_entry: dict[str, Any] = {
            "step": step,
            "t": t_curr,
            "reinserted_count": len(to_update),
            "update_wall_clock_s": step_update_time,
        }
        for ef in ef_values:
            rec_val = evaluate_recall(engine, q_d, gt_d, k=k, ef=ef)
            rec_entry[f"drift_rec{ef}"] = rec_val

        step_records.append(rec_entry)

    d16_vals = [r["drift_rec16"] for r in step_records if "drift_rec16" in r]
    d32_vals = [r["drift_rec32"] for r in step_records if "drift_rec32" in r]

    return {
        "policy": policy_name,
        "is_native_hnswlib": engine.is_native,
        "setup_wall_clock_s": setup_wall_clock,
        "total_reinsertions": total_reinsert_count,
        "total_update_time_s": total_update_time,
        "mean_drift_rec16": float(np.mean(d16_vals)) if d16_vals else 0.0,
        "final_drift_rec16": float(d16_vals[-1]) if d16_vals else 0.0,
        "mean_drift_rec32": float(np.mean(d32_vals)) if d32_vals else 0.0,
        "final_drift_rec32": float(d32_vals[-1]) if d32_vals else 0.0,
        "step_records": step_records,
    }


def run_hnswlib_sweep(
    n: int = 2000,
    d: int = 128,
    seeds: list[int] | None = None,
    displacement_mult: float = 8.0,
    drift_frac: float = 0.3,
    steps: int = 10,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Execute hnswlib comparative sweep across seeds."""
    if seeds is None:
        seeds = [42, 43, 44]

    policies = ["no_repair", "periodic_k5", "round_robin", "periodic_k1"]
    all_runs: list[dict[str, Any]] = []

    for seed in seeds:
        X0, vels, drift_nodes = generate_isotropic_drift_data(
            n=n,
            d=d,
            seed=seed,
            drift_frac=drift_frac,
            displacement_mult=displacement_mult,
            t_final=t_final,
            steps=steps,
        )
        dt = t_final / steps

        seed_res: dict[str, Any] = {"seed": seed, "policies": {}}
        for pol in policies:
            res = simulate_hnswlib_policy(
                policy_name=pol,
                n=n,
                d=d,
                seed=seed,
                X0=X0,
                vels=vels,
                drift_nodes=drift_nodes,
                steps=steps,
                dt=dt,
            )
            seed_res["policies"][pol] = res
        all_runs.append(seed_res)

    # Summary
    summary_stats: dict[str, Any] = {}
    for pol in policies:
        rec16_vals = [r["policies"][pol]["mean_drift_rec16"] for r in all_runs]
        reinsert_vals = [r["policies"][pol]["total_reinsertions"] for r in all_runs]
        wall_vals = [r["policies"][pol]["total_update_time_s"] for r in all_runs]
        summary_stats[pol] = {
            "mean_drift_rec16": float(np.mean(rec16_vals)),
            "std_drift_rec16": float(np.std(rec16_vals)),
            "mean_reinsertions": float(np.mean(reinsert_vals)),
            "mean_update_time_s": float(np.mean(wall_vals)),
        }

    return {
        "engine": "hnswlib",
        "is_native_hnswlib": HNSWLIB_AVAILABLE,
        "n": n,
        "d": d,
        "seeds": seeds,
        "displacement_mult": displacement_mult,
        "drift_frac": drift_frac,
        "steps": steps,
        "summary": summary_stats,
        "seed_runs": all_runs,
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for hnswlib baseline."""
    parser = argparse.ArgumentParser(description="Run hnswlib baseline comparison under coordinate drift.")
    parser.add_argument("--n", type=int, default=2000, help="Node count (default: 2000).")
    parser.add_argument("--d", type=int, default=128, help="Dimensionality (default: 128).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds (default: 42 43 44).")
    parser.add_argument("--displacement", type=float, default=8.0, help="Displacement multiplier (default: 8.0).")
    parser.add_argument("--drift-frac", type=float, default=0.3, help="Fraction drifting (default: 0.3).")
    parser.add_argument("--steps", type=int, default=10, help="Time steps (default: 10).")
    parser.add_argument("--t-final", type=float, default=0.30, help="End time (default: 0.30).")
    parser.add_argument(
        "--out",
        type=str,
        default="docs/data/hnswlib_baseline_benchmark.json",
        help="Output JSON path.",
    )
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test run.")
    return parser.parse_args(args)


def main() -> None:
    """CLI entry point."""
    args = parse_args(sys.argv[1:])
    if args.smoke:
        n, d, seeds, steps = 100, 16, [42], 3
    else:
        n, d, seeds, steps = args.n, args.d, args.seeds, args.steps

    print(f"=== C++ hnswlib Baseline Benchmark Driver ===")
    print(f"Native hnswlib installed: {HNSWLIB_AVAILABLE}")
    print(f"N: {n} | D: {d} | Seeds: {seeds} | Steps: {steps}")

    t0 = time.time()
    res = run_hnswlib_sweep(
        n=n,
        d=d,
        seeds=seeds,
        displacement_mult=args.displacement,
        drift_frac=args.drift_frac,
        steps=steps,
        t_final=args.t_final,
    )
    elapsed = time.time() - t0
    res["elapsed_seconds"] = elapsed

    if not HNSWLIB_AVAILABLE:
        print(
            "[WARNING] hnswlib is not installed — results are from NumPy brute-force "
            "emulation (is_native_hnswlib=False). Output JSON will NOT be written. "
            "Run on Linux/CI with native hnswlib to produce valid baseline numbers.",
            file=sys.stderr,
        )
    else:
        save_results(res, args.out)
        print(f"Completed in {elapsed:.2f}s. Saved to {args.out}")

    print("\n=== hnswlib Summary ===")
    print(f"{'Policy':<15} {'Mean Rec@16':<12} {'Reinsertions':<15} {'Update Time (s)'}")
    print("-" * 55)
    for pol, stats in res["summary"].items():
        print(f"{pol:<15} {stats['mean_drift_rec16']:<12.4f} {stats['mean_reinsertions']:<15.1f} {stats['mean_update_time_s']:.4f}")


if __name__ == "__main__":
    main()
