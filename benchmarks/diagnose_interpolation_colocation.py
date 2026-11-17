"""Diagnose exclusive co-location categories for interpolation drift at f=1.00 no jitter.

Imports the benchmark's own target-selection and drift generation functions.
Computes per-seed counts for the 100 evaluated drift queries across seeds 110-114:
  (i) sole static target: co-located with a static node that is the query's target,
      and no other drifting node shares that target.
  (ii) shared target: the query's target is shared by at least one other drifting node
       (colliding with that drifting node and, if static, the target).
  (iii) other: any other co-location event.
Writes docs/data/interpolation_colocation_categories.json with full run metadata.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from benchmarks.run_interpolation_benchmark import generate_interpolation_drift  # noqa: E402
from benchmarks.run_real_embeddings import load_benchmark_dataset  # noqa: E402


def get_git_head(repo_root: Path) -> str:
    """Get current git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def run_colocation_diagnostics(
    data_path: Path,
    seeds: list[int] | None = None,
    f_frac: float = 1.00,
    drift_frac: float = 0.30,
    t_final: float = 0.30,
    threshold: float = 1e-3,
    eval_query_count: int = 100,
) -> dict[str, Any]:
    """Compute exclusive co-location categories across evaluated drift queries."""
    if seeds is None:
        seeds = [110, 111, 112, 113, 114]

    per_seed: dict[str, dict[str, int]] = {}

    for seed in seeds:
        X0, _ = load_benchmark_dataset("sift", data_path=str(data_path), n=2000, d=128, seed=seed)
        vels, drift_nodes, targets, _ = generate_interpolation_drift(
            X0=X0, seed=seed, f_frac=f_frac, drift_frac=drift_frac, t_final=t_final, jitter=False
        )
        drift_set = set(drift_nodes)
        static_set = set(range(len(X0))) - drift_set
        eval_queries = drift_nodes[:eval_query_count]

        # Final positions at t = t_final
        # Note: No dedicated function exists in benchmarks/run_interpolation_benchmark.py;
        # pos_final is computed inline at benchmarks/run_interpolation_benchmark.py:299:
        # pos_final = (X0.astype(np.float64) + vels.astype(np.float64) * t_final).astype(np.float32)
        pos_final = (X0.astype(np.float64) + vels.astype(np.float64) * t_final).astype(np.float32)

        # Count occurrences of each target node across all drifting nodes
        target_counts: dict[int, int] = {}
        for d in drift_nodes:
            t = int(targets[d])
            target_counts[t] = target_counts.get(t, 0) + 1

        cat_i = 0
        cat_ii = 0
        cat_iii = 0
        coloc_total = 0

        for u in eval_queries:
            q_pos = pos_final[u]
            dists = np.linalg.norm(pos_final - q_pos, axis=1)
            dists[u] = np.inf
            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])

            if min_dist < threshold:
                coloc_total += 1
                tgt = int(targets[u])
                is_static_tgt = tgt in static_set
                shared = target_counts[tgt] > 1

                if is_static_tgt and not shared:
                    cat_i += 1
                elif shared:
                    cat_ii += 1
                else:
                    cat_iii += 1

        assert cat_i + cat_ii + cat_iii == coloc_total, "Categories must sum to total co-located"
        per_seed[str(seed)] = {
            "total_colocated": coloc_total,
            "sole_static_target": cat_i,
            "shared_target": cat_ii,
            "other": cat_iii,
        }

    # Aggregate percentage ranges across seeds
    totals = [per_seed[str(s)]["total_colocated"] for s in seeds]
    sole_static = [per_seed[str(s)]["sole_static_target"] for s in seeds]
    shared = [per_seed[str(s)]["shared_target"] for s in seeds]
    other = [per_seed[str(s)]["other"] for s in seeds]

    summary = {
        "metadata": {
            "git_head": get_git_head(_repo_root),
            "seeds": seeds,
            "f_frac": f_frac,
            "drift_frac": drift_frac,
            "t_final": t_final,
            "distance_threshold": threshold,
            "eval_query_count": eval_query_count,
            "n_queries": eval_query_count,
        },
        "per_seed": per_seed,
        "ranges": {
            "total_colocated_pct_range": f"{min(totals)}-{max(totals)}%",
            "sole_static_pct_range": f"{min(sole_static)}-{max(sole_static)}%",
            "shared_target_pct_range": f"{min(shared)}-{max(shared)}%",
            "other_pct_range": f"{min(other)}%",
        },
    }
    return summary


def main() -> None:
    data_path = _repo_root / "docs" / "data" / "siftsmall" / "siftsmall_base.fvecs"
    out_path = _repo_root / "docs" / "data" / "interpolation_colocation_categories.json"

    print("=== INTERPOLATION CO-LOCATION EXCLUSIVE CATEGORIES DIAGNOSTICS ===")
    print(f"Data file: {data_path}")
    print(f"Output:    {out_path}")

    res = run_colocation_diagnostics(data_path)

    print("\nPer-Seed Counts (out of 100 evaluated drift queries):")
    for s_str, rec in res["per_seed"].items():
        print(
            f"  Seed {s_str}: total={rec['total_colocated']} | "
            f"(i) sole_static={rec['sole_static_target']}, "
            f"(ii) shared_target={rec['shared_target']}, "
            f"(iii) other={rec['other']}"
        )

    print("\nAggregated Percentage Ranges across Seeds:")
    for k, v in res["ranges"].items():
        print(f"  {k}: {v}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
