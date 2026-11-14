"""Benchmark script for Phase 5j: baseline confound, valid headroom, and displacement sweep.

Measures:
Item A: Baseline confound at t=0 (Vanilla HNSWIndex vs Settled base_index via settle_hnsw).
Item B: Valid pair comparison on Vanilla index (no_repair vs full rebuild ceiling) at 10x and 30x
        drift (10% drift).
Item C: Headroom sweep on Vanilla pair across displacement multiples {0.5, 2, 8, 32}
        x drift fractions {10%, 30%} x ef {16, 32} plus n=2000 run at displacements {8, 32}
        for seed 42.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from khnsw.core.detection.certificate_store import CertificateStore
from khnsw.core.detection.cost import CostMeter
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.movement.base import mean_nn_distance
from khnsw.core.movement.linear import LinearDrift
from khnsw.core.repair.loop import repair_loop


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


def count_edges(index: HNSWIndex) -> tuple[int, int, float]:
    """Return (level0_edges, total_edges, mean_level0_out_degree)."""
    n = len(index.ids())
    deg0 = [len(index.neighbors(i, 0)) for i in index.ids()]
    total_edges = sum(
        len(index.neighbors(i, lc)) for i in index.ids() for lc in range(index.node_level(i) + 1)
    )
    return sum(deg0), total_edges, float(np.mean(deg0)) if n > 0 else 0.0


def run_item_a(
    seeds: list[int] | None = None, n: int = 500, d: int = 8, m: int = 8
) -> dict[str, Any]:
    """Item A: Baseline confound at t=0."""
    if seeds is None:
        seeds = [42, 43, 44]
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        X0 = rng.randn(n, d).astype(np.float32)

        # 100 queries from N(0, I)
        rng_q = np.random.RandomState(seed + 1000)
        queries = rng_q.randn(100, d).astype(np.float32)
        gt = brute_force_knn(X0, queries, k=10)

        # 1. Vanilla HNSWIndex built by add() only (no settle)
        idx_vanilla = HNSWIndex(dim=d, m=m, seed=seed)
        idx_vanilla.add(np.arange(n, dtype=np.int64), X0)
        l0_v, total_v, mean_deg0_v = count_edges(idx_vanilla)
        rec16_v = evaluate_recall(idx_vanilla, queries, gt, k=10, ef=16)
        rec32_v = evaluate_recall(idx_vanilla, queries, gt, k=10, ef=32)

        # 2. Settled base_index (built by add() then settle_hnsw)
        idx_settled = HNSWIndex(dim=d, m=m, seed=seed)
        idx_settled.add(np.arange(n, dtype=np.int64), X0)
        store = CertificateStore()
        store.build(idx_settled, CostMeter())
        repair_loop(idx_settled, store)
        l0_s, total_s, mean_deg0_s = count_edges(idx_settled)
        rec16_s = evaluate_recall(idx_settled, queries, gt, k=10, ef=16)
        rec32_s = evaluate_recall(idx_settled, queries, gt, k=10, ef=32)

        per_seed.append(
            {
                "seed": seed,
                "vanilla": {
                    "mean_deg0": mean_deg0_v,
                    "l0_edges": l0_v,
                    "total_edges": total_v,
                    "recall16": rec16_v,
                    "recall32": rec32_v,
                },
                "settled": {
                    "mean_deg0": mean_deg0_s,
                    "l0_edges": l0_s,
                    "total_edges": total_s,
                    "recall16": rec16_s,
                    "recall32": rec32_s,
                },
            }
        )

    mean_summary = {
        "vanilla": {
            "mean_deg0": float(np.mean([p["vanilla"]["mean_deg0"] for p in per_seed])),
            "l0_edges": float(np.mean([p["vanilla"]["l0_edges"] for p in per_seed])),
            "total_edges": float(np.mean([p["vanilla"]["total_edges"] for p in per_seed])),
            "recall16": float(np.mean([p["vanilla"]["recall16"] for p in per_seed])),
            "recall32": float(np.mean([p["vanilla"]["recall32"] for p in per_seed])),
        },
        "settled": {
            "mean_deg0": float(np.mean([p["settled"]["mean_deg0"] for p in per_seed])),
            "l0_edges": float(np.mean([p["settled"]["l0_edges"] for p in per_seed])),
            "total_edges": float(np.mean([p["settled"]["total_edges"] for p in per_seed])),
            "recall16": float(np.mean([p["settled"]["recall16"] for p in per_seed])),
            "recall32": float(np.mean([p["settled"]["recall32"] for p in per_seed])),
        },
    }

    return {"per_seed": per_seed, "mean": mean_summary}


def run_item_b(
    seeds: list[int] | None = None,
    n: int = 500,
    d: int = 8,
    m: int = 8,
    speeds: list[float] | None = None,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Item B: Valid pair comparison on VANILLA index under identical drift protocol (10% drift)."""
    if seeds is None:
        seeds = [42, 43, 44]
    if speeds is None:
        speeds = [10.0, 30.0]
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        X0 = rng.randn(n, d).astype(np.float32)

        drifting_nodes = list(rng.choice(n, size=50, replace=False))
        drifting_set = set(drifting_nodes)
        static_nodes = [i for i in range(n) if i not in drifting_set]
        eval_static_nodes = static_nodes[:100]

        # Build vanilla index at t=0
        vanilla_index = HNSWIndex(dim=d, m=m, seed=seed)
        vanilla_index.add(np.arange(n, dtype=np.int64), X0)

        seed_results: dict[str, Any] = {"seed": seed, "speeds": {}}

        for speed in speeds:
            drift = LinearDrift(X0, seed=seed, speed=speed)
            drift._velocities[static_nodes] = 0.0
            pos_t_final = drift.positions(t_final).astype(np.float32)

            q_drift = pos_t_final[drifting_nodes]
            gt_drift = brute_force_knn(pos_t_final, q_drift, k=10)

            q_static = pos_t_final[eval_static_nodes]
            gt_static = brute_force_knn(pos_t_final, q_static, k=10)

            # 1. no_repair on VANILLA index: sync vectors to t_final
            idx_no_rep = copy.deepcopy(vanilla_index)
            for nid in range(n):
                idx_no_rep._store.set_vector(nid, pos_t_final[nid])

            rec16_drift_no_rep = evaluate_recall(idx_no_rep, q_drift, gt_drift, k=10, ef=16)
            rec32_drift_no_rep = evaluate_recall(idx_no_rep, q_drift, gt_drift, k=10, ef=32)
            rec16_static_no_rep = evaluate_recall(idx_no_rep, q_static, gt_static, k=10, ef=16)
            rec32_static_no_rep = evaluate_recall(idx_no_rep, q_static, gt_static, k=10, ef=32)
            cost_no_rep = 0

            # 2. Ceiling: full rebuild of all 500 nodes at final positions
            idx_ceiling = HNSWIndex(dim=d, m=m, seed=seed)
            idx_ceiling.add(np.arange(n, dtype=np.int64), pos_t_final)
            cost_ceiling = idx_ceiling.distance_computations

            rec16_drift_ceil = evaluate_recall(idx_ceiling, q_drift, gt_drift, k=10, ef=16)
            rec32_drift_ceil = evaluate_recall(idx_ceiling, q_drift, gt_drift, k=10, ef=32)
            rec16_static_ceil = evaluate_recall(idx_ceiling, q_static, gt_static, k=10, ef=16)
            rec32_static_ceil = evaluate_recall(idx_ceiling, q_static, gt_static, k=10, ef=32)

            spd_key = f"{int(speed)}x"
            seed_results["speeds"][spd_key] = {
                "no_repair": {
                    "drift_rec16": rec16_drift_no_rep,
                    "drift_rec32": rec32_drift_no_rep,
                    "static_rec16": rec16_static_no_rep,
                    "static_rec32": rec32_static_no_rep,
                    "cost": cost_no_rep,
                },
                "ceiling": {
                    "drift_rec16": rec16_drift_ceil,
                    "drift_rec32": rec32_drift_ceil,
                    "static_rec16": rec16_static_ceil,
                    "static_rec32": rec32_static_ceil,
                    "cost": cost_ceiling,
                },
                "headroom": {
                    "drift_rec16": rec16_drift_ceil - rec16_drift_no_rep,
                    "drift_rec32": rec32_drift_ceil - rec32_drift_no_rep,
                    "static_rec16": rec16_static_ceil - rec16_static_no_rep,
                    "static_rec32": rec32_static_ceil - rec32_static_no_rep,
                },
            }

        per_seed.append(seed_results)

    # Mean aggregation
    mean_summary: dict[str, Any] = {}
    for spd in ["10x", "30x"]:
        mean_summary[spd] = {
            "no_repair": {
                "drift_rec16": float(
                    np.mean([p["speeds"][spd]["no_repair"]["drift_rec16"] for p in per_seed])
                ),
                "drift_rec32": float(
                    np.mean([p["speeds"][spd]["no_repair"]["drift_rec32"] for p in per_seed])
                ),
                "static_rec16": float(
                    np.mean([p["speeds"][spd]["no_repair"]["static_rec16"] for p in per_seed])
                ),
                "static_rec32": float(
                    np.mean([p["speeds"][spd]["no_repair"]["static_rec32"] for p in per_seed])
                ),
                "cost": 0,
            },
            "ceiling": {
                "drift_rec16": float(
                    np.mean([p["speeds"][spd]["ceiling"]["drift_rec16"] for p in per_seed])
                ),
                "drift_rec32": float(
                    np.mean([p["speeds"][spd]["ceiling"]["drift_rec32"] for p in per_seed])
                ),
                "static_rec16": float(
                    np.mean([p["speeds"][spd]["ceiling"]["static_rec16"] for p in per_seed])
                ),
                "static_rec32": float(
                    np.mean([p["speeds"][spd]["ceiling"]["static_rec32"] for p in per_seed])
                ),
                "cost": float(np.mean([p["speeds"][spd]["ceiling"]["cost"] for p in per_seed])),
            },
            "headroom": {
                "drift_rec16": float(
                    np.mean([p["speeds"][spd]["headroom"]["drift_rec16"] for p in per_seed])
                ),
                "drift_rec32": float(
                    np.mean([p["speeds"][spd]["headroom"]["drift_rec32"] for p in per_seed])
                ),
                "static_rec16": float(
                    np.mean([p["speeds"][spd]["headroom"]["static_rec16"] for p in per_seed])
                ),
                "static_rec32": float(
                    np.mean([p["speeds"][spd]["headroom"]["static_rec32"] for p in per_seed])
                ),
            },
        }

    return {"per_seed": per_seed, "mean": mean_summary}


def run_item_c(
    seeds: list[int] | None = None,
    displacements: list[float] | None = None,
    drift_fractions: list[float] | None = None,
    n: int = 500,
    d: int = 8,
    m: int = 8,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Item C: Headroom sweep on vanilla pair across displacements x drift fractions x ef."""
    if seeds is None:
        seeds = [42, 43, 44]
    if displacements is None:
        displacements = [0.5, 2.0, 8.0, 32.0]
    if drift_fractions is None:
        drift_fractions = [0.10, 0.30]
    all_runs: list[dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        X0 = rng.randn(n, d).astype(np.float32)
        scale = mean_nn_distance(X0, seed=seed)

        # Base vanilla index
        vanilla_index = HNSWIndex(dim=d, m=m, seed=seed)
        vanilla_index.add(np.arange(n, dtype=np.int64), X0)

        for frac in drift_fractions:
            n_drift = int(round(n * frac))
            drifting_nodes = list(rng.choice(n, size=n_drift, replace=False))
            drifting_set = set(drifting_nodes)
            static_nodes = [i for i in range(n) if i not in drifting_set]
            eval_static = static_nodes[: min(100, len(static_nodes))]

            # Unit directions for drifting nodes
            rng_vel = np.random.default_rng(np.random.SeedSequence([seed, 0]))
            normals = rng_vel.standard_normal((n, d), dtype=np.float64)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            unit_dirs = normals / norms

            for mult in displacements:
                # Target displacement at t_final is mult * scale
                # velocity = unit_dirs * (mult * scale / t_final)
                vels = np.zeros((n, d), dtype=np.float64)
                vels[drifting_nodes] = unit_dirs[drifting_nodes] * (mult * scale / t_final)

                pos_t_final = (X0.astype(np.float64) + vels * t_final).astype(np.float32)

                q_drift = pos_t_final[drifting_nodes]
                gt_drift = brute_force_knn(pos_t_final, q_drift, k=10)

                q_static = pos_t_final[eval_static]
                gt_static = brute_force_knn(pos_t_final, q_static, k=10)

                # no_repair on vanilla
                idx_no_rep = copy.deepcopy(vanilla_index)
                for nid in range(n):
                    idx_no_rep._store.set_vector(nid, pos_t_final[nid])

                rec16_d_nr = evaluate_recall(idx_no_rep, q_drift, gt_drift, k=10, ef=16)
                rec32_d_nr = evaluate_recall(idx_no_rep, q_drift, gt_drift, k=10, ef=32)
                rec16_s_nr = evaluate_recall(idx_no_rep, q_static, gt_static, k=10, ef=16)
                rec32_s_nr = evaluate_recall(idx_no_rep, q_static, gt_static, k=10, ef=32)

                # ceiling: rebuild all
                idx_ceil = HNSWIndex(dim=d, m=m, seed=seed)
                idx_ceil.add(np.arange(n, dtype=np.int64), pos_t_final)
                cost_ceil = idx_ceil.distance_computations

                rec16_d_cl = evaluate_recall(idx_ceil, q_drift, gt_drift, k=10, ef=16)
                rec32_d_cl = evaluate_recall(idx_ceil, q_drift, gt_drift, k=10, ef=32)
                rec16_s_cl = evaluate_recall(idx_ceil, q_static, gt_static, k=10, ef=16)
                rec32_s_cl = evaluate_recall(idx_ceil, q_static, gt_static, k=10, ef=32)

                all_runs.append(
                    {
                        "n": n,
                        "seed": seed,
                        "drift_frac": frac,
                        "displacement_mult": mult,
                        "no_repair": {
                            "drift_rec16": rec16_d_nr,
                            "drift_rec32": rec32_d_nr,
                            "static_rec16": rec16_s_nr,
                            "static_rec32": rec32_s_nr,
                        },
                        "ceiling": {
                            "drift_rec16": rec16_d_cl,
                            "drift_rec32": rec32_d_cl,
                            "static_rec16": rec16_s_cl,
                            "static_rec32": rec32_s_cl,
                            "cost": cost_ceil,
                        },
                        "headroom": {
                            "drift_rec16": rec16_d_cl - rec16_d_nr,
                            "drift_rec32": rec32_d_cl - rec32_d_nr,
                            "static_rec16": rec16_s_cl - rec16_s_nr,
                            "static_rec32": rec32_s_cl - rec32_s_nr,
                        },
                    }
                )

    # Summary by (frac, mult)
    mean_by_condition: dict[str, Any] = {}
    for frac in drift_fractions:
        for mult in displacements:
            cond_key = f"frac_{int(frac * 100)}%_mult_{mult}x"
            matching = [
                r for r in all_runs if r["drift_frac"] == frac and r["displacement_mult"] == mult
            ]
            mean_by_condition[cond_key] = {
                "drift_frac": frac,
                "displacement_mult": mult,
                "no_repair": {
                    "drift_rec16": float(
                        np.mean([r["no_repair"]["drift_rec16"] for r in matching])
                    ),
                    "drift_rec32": float(
                        np.mean([r["no_repair"]["drift_rec32"] for r in matching])
                    ),
                    "static_rec16": float(
                        np.mean([r["no_repair"]["static_rec16"] for r in matching])
                    ),
                    "static_rec32": float(
                        np.mean([r["no_repair"]["static_rec32"] for r in matching])
                    ),
                },
                "ceiling": {
                    "drift_rec16": float(np.mean([r["ceiling"]["drift_rec16"] for r in matching])),
                    "drift_rec32": float(np.mean([r["ceiling"]["drift_rec32"] for r in matching])),
                    "static_rec16": float(
                        np.mean([r["ceiling"]["static_rec16"] for r in matching])
                    ),
                    "static_rec32": float(
                        np.mean([r["ceiling"]["static_rec32"] for r in matching])
                    ),
                    "cost": float(np.mean([r["ceiling"]["cost"] for r in matching])),
                },
                "headroom": {
                    "drift_rec16": float(np.mean([r["headroom"]["drift_rec16"] for r in matching])),
                    "drift_rec32": float(np.mean([r["headroom"]["drift_rec32"] for r in matching])),
                    "static_rec16": float(
                        np.mean([r["headroom"]["static_rec16"] for r in matching])
                    ),
                    "static_rec32": float(
                        np.mean([r["headroom"]["static_rec32"] for r in matching])
                    ),
                },
            }

    return {"all_runs": all_runs, "mean_by_condition": mean_by_condition}


def run_item_c_n2000(
    seed: int = 42,
    displacements: list[float] | None = None,
    drift_frac: float = 0.10,
    n: int = 2000,
    d: int = 8,
    m: int = 8,
    t_final: float = 0.30,
) -> dict[str, Any]:
    """Item C n=2000 run for seed 42."""
    if displacements is None:
        displacements = [8.0, 32.0]
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n, d).astype(np.float32)
    scale = mean_nn_distance(X0, seed=seed)

    vanilla_index = HNSWIndex(dim=d, m=m, seed=seed)
    vanilla_index.add(np.arange(n, dtype=np.int64), X0)

    n_drift = int(round(n * drift_frac))
    drifting_nodes = list(rng.choice(n, size=n_drift, replace=False))
    drifting_set = set(drifting_nodes)
    static_nodes = [i for i in range(n) if i not in drifting_set]
    eval_static = static_nodes[:100]

    rng_vel = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    normals = rng_vel.standard_normal((n, d), dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit_dirs = normals / norms

    results: list[dict[str, Any]] = []

    for mult in displacements:
        vels = np.zeros((n, d), dtype=np.float64)
        vels[drifting_nodes] = unit_dirs[drifting_nodes] * (mult * scale / t_final)
        pos_t_final = (X0.astype(np.float64) + vels * t_final).astype(np.float32)

        q_drift = pos_t_final[drifting_nodes]
        gt_drift = brute_force_knn(pos_t_final, q_drift, k=10)

        q_static = pos_t_final[eval_static]
        gt_static = brute_force_knn(pos_t_final, q_static, k=10)

        # no_repair
        idx_no_rep = copy.deepcopy(vanilla_index)
        for nid in range(n):
            idx_no_rep._store.set_vector(nid, pos_t_final[nid])

        rec16_d_nr = evaluate_recall(idx_no_rep, q_drift, gt_drift, k=10, ef=16)
        rec32_d_nr = evaluate_recall(idx_no_rep, q_drift, gt_drift, k=10, ef=32)
        rec16_s_nr = evaluate_recall(idx_no_rep, q_static, gt_static, k=10, ef=16)
        rec32_s_nr = evaluate_recall(idx_no_rep, q_static, gt_static, k=10, ef=32)

        # ceiling
        idx_ceil = HNSWIndex(dim=d, m=m, seed=seed)
        idx_ceil.add(np.arange(n, dtype=np.int64), pos_t_final)
        cost_ceil = idx_ceil.distance_computations

        rec16_d_cl = evaluate_recall(idx_ceil, q_drift, gt_drift, k=10, ef=16)
        rec32_d_cl = evaluate_recall(idx_ceil, q_drift, gt_drift, k=10, ef=32)
        rec16_s_cl = evaluate_recall(idx_ceil, q_static, gt_static, k=10, ef=16)
        rec32_s_cl = evaluate_recall(idx_ceil, q_static, gt_static, k=10, ef=32)

        results.append(
            {
                "n": n,
                "seed": seed,
                "drift_frac": drift_frac,
                "displacement_mult": mult,
                "no_repair": {
                    "drift_rec16": rec16_d_nr,
                    "drift_rec32": rec32_d_nr,
                    "static_rec16": rec16_s_nr,
                    "static_rec32": rec32_s_nr,
                },
                "ceiling": {
                    "drift_rec16": rec16_d_cl,
                    "drift_rec32": rec32_d_cl,
                    "static_rec16": rec16_s_cl,
                    "static_rec32": rec32_s_cl,
                    "cost": cost_ceil,
                },
                "headroom": {
                    "drift_rec16": rec16_d_cl - rec16_d_nr,
                    "drift_rec32": rec32_d_cl - rec32_d_nr,
                    "static_rec16": rec16_s_cl - rec16_s_nr,
                    "static_rec32": rec32_s_cl - rec32_s_nr,
                },
            }
        )

    return {"runs": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla headroom benchmark")
    parser.add_argument("--out", type=str, default="docs/data/vanilla_headroom.json")
    args = parser.parse_args()

    print("=== Running Item A: Baseline Confound ===")
    res_a = run_item_a(seeds=[42, 43, 44])
    print("Item A done.")

    print("=== Running Item B: Valid Pair Comparison ===")
    res_b = run_item_b(seeds=[42, 43, 44])
    print("Item B done.")

    print("=== Running Item C: Headroom Sweep (n=500) ===")
    res_c_500 = run_item_c(seeds=[42, 43, 44])
    print("Item C (n=500) done.")

    print("=== Running Item C: Headroom Sweep (n=2000, seed 42) ===")
    res_c_2000 = run_item_c_n2000(seed=42)
    print("Item C (n=2000) done.")

    full_output = {
        "item_a_baseline_confound": res_a,
        "item_b_valid_pair": res_b,
        "item_c_headroom_500": res_c_500,
        "item_c_headroom_2000": res_c_2000,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
