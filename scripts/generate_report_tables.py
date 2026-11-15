"""Generate report tables directly from docs/data/external_benchmarks.json.

Guarantees 100% numerical fidelity between the raw committed JSON results
and the tables presented in Chapter 8 and Chapter 9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_benchmarks() -> dict[str, Any]:
    json_path = Path("docs/data/external_benchmarks.json")
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def format_arm_row(arm_name: str, params: str, stats: dict[str, Any]) -> str:
    mean_rec = stats.get("mean_drift_rec16", 0.0)
    std_rec = stats.get("std_drift_rec16", 0.0)
    final_rec = stats.get("final_drift_rec16", 0.0)
    std_final = stats.get("std_final_drift_rec16", 0.0)
    cost = stats.get("mean_cost", 0.0)
    std_cost = stats.get("std_cost", 0.0)
    reinserts = stats.get("mean_reinsertions", 0.0)

    cost_str = f"{cost:,.0f} ± {std_cost:,.0f}" if cost > 0 else "0"
    reinsert_str = f"{reinserts:,.0f}" if reinserts > 0 else "0"

    return (
        f"| `{arm_name}` | {params} "
        f"| {mean_rec:.4f} ± {std_rec:.3f} "
        f"| {final_rec:.4f} ± {std_final:.3f} "
        f"| {cost_str} "
        f"| {reinsert_str} |"
    )


def compute_paired_stats(seeds: list[int], rr_list: list[float], per_list: list[float]) -> dict[str, Any]:
    diffs = np.array(rr_list) - np.array(per_list)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    s_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    # t critical for 95% with df=4 is 2.776
    t_crit = 2.776 if n == 5 else 2.0
    se = s_d / np.sqrt(n) if n > 0 else 0.0
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se
    pos_count = int(np.sum(diffs > 0))

    per_seed = [{"seed": s, "rr": r, "per": p, "diff": d} for s, r, p, d in zip(seeds, rr_list, per_list, diffs)]

    return {
        "per_seed": per_seed,
        "mean_diff": mean_d,
        "std_diff": s_d,
        "ci_95": (ci_low, ci_high),
        "positive_count": pos_count,
        "total_seeds": n,
    }


def generate_paired_table_md(stats: dict[str, Any], title: str) -> str:
    lines = [
        f"#### Paired Differences: {title}",
        "",
        "| Seed | `round_robin` Recall@16 | `periodic_k5` Recall@16 | Paired Difference (RR − Per) | Sign |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in stats["per_seed"]:
        sign = "+" if row["diff"] > 0 else "-"
        lines.append(f"| **{row['seed']}** | {row['rr']:.4f} | {row['per']:.4f} | **{row['diff']:+.4f}** | {sign} |")

    ci = stats["ci_95"]
    lines.append("")
    lines.append(
        f"- **Mean Difference (Δ)**: **{stats['mean_diff']:+.4f}** ({stats['mean_diff']*100:+.2f} pp)"
    )
    lines.append(f"- **Sample Std ($s$)**: {stats['std_diff']:.4f}")
    lines.append(f"- **95% Confidence Interval**: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    lines.append(f"- **Positive Seeds**: **{stats['positive_count']} / {stats['total_seeds']}**")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = load_benchmarks()

    print("=" * 80)
    print("TABLES FOR CHAPTER 8: Dynamic Drift Maintenance on Synthetic Motion")
    print("=" * 80)

    # 1. Table 8.1A: Synthetic ef_construction=16 (seeds 42-46)
    conds_16 = data["synthetic_drift_ef16"]["conditions"]
    print("\n### Table 8.1A: Synthetic Gaussian Drift (efConstruction = 16, seeds 42–46)")
    print("Parameters: M=8, M0=16, efConstruction=16, delta=0.30, lambda=8.0x, k=10, ef=16, 10 steps\n")
    print("| Size | Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |")
    print("|:---|:---|:---|:---:|:---:|:---:|:---:|")

    for cond in conds_16:
        n = cond["n"]
        arm_configs = [
            ("no_repair", "—"),
            ("periodic_k1", "K=1"),
            ("periodic_k5", "K=5"),
            ("round_robin", f"B={cond['arm_stats']['round_robin']['mean_reinsertions']/10:.0f}"),
            ("lazy_theta_1.0", "θ=1.0"),
            ("lazy_theta_0.5", "θ=0.5"),
            ("budgeted_priority", f"B={cond['arm_stats']['budgeted_priority']['mean_reinsertions']/10:.0f}"),
        ]
        for idx, (arm, params) in enumerate(arm_configs):
            prefix = f"| **n={n}** " if idx == 0 else "| "
            row_str = format_arm_row(arm, params, cond["arm_stats"][arm])
            print(prefix + row_str[2:])

    # Paired differences for ef16
    for cond in conds_16:
        n = cond["n"]
        runs = [r for r in data["synthetic_drift_ef16"]["raw_runs"] if r["n"] == n]
        seeds = [r["seed"] for r in runs]
        rr = [r["arms"]["round_robin"]["mean_drift_rec16"] for r in runs]
        per = [r["arms"]["periodic_k5"]["mean_drift_rec16"] for r in runs]
        paired = compute_paired_stats(seeds, rr, per)
        print("\n" + generate_paired_table_md(paired, f"Synthetic n={n}, d=8, efConstruction=16"))

    # 2. Table 8.1B: Synthetic ef_construction=200 (seeds 100-104)
    conds_200 = data["synthetic_drift_ef200"]["conditions"]
    print("\n### Table 8.1B: Synthetic Gaussian Drift (efConstruction = 200, seeds 100–104)")
    print("Parameters: M=8, M0=16, efConstruction=200, delta=0.30, lambda=8.0x, k=10, ef=16, 10 steps\n")
    print("| Size | Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |")
    print("|:---|:---|:---|:---:|:---:|:---:|:---:|")

    for cond in conds_200:
        n = cond["n"]
        arm_configs = [
            ("no_repair", "—"),
            ("periodic_k1", "K=1"),
            ("periodic_k5", "K=5"),
            ("round_robin", f"B={cond['arm_stats']['round_robin']['mean_reinsertions']/10:.0f}"),
            ("lazy_theta_1.0", "θ=1.0"),
            ("lazy_theta_0.5", "θ=0.5"),
            ("budgeted_priority", f"B={cond['arm_stats']['budgeted_priority']['mean_reinsertions']/10:.0f}"),
        ]
        for idx, (arm, params) in enumerate(arm_configs):
            prefix = f"| **n={n}** " if idx == 0 else "| "
            row_str = format_arm_row(arm, params, cond["arm_stats"][arm])
            print(prefix + row_str[2:])

    # Paired differences for ef200
    for cond in conds_200:
        n = cond["n"]
        runs = [r for r in data["synthetic_drift_ef200"]["raw_runs"] if r["n"] == n]
        seeds = [r["seed"] for r in runs]
        rr = [r["arms"]["round_robin"]["mean_drift_rec16"] for r in runs]
        per = [r["arms"]["periodic_k5"]["mean_drift_rec16"] for r in runs]
        paired = compute_paired_stats(seeds, rr, per)
        print("\n" + generate_paired_table_md(paired, f"Synthetic n={n}, d=8, efConstruction=200"))

    print("=" * 80)
    print("TABLES FOR CHAPTER 9: Real Embeddings & External Baselines")
    print("=" * 80)

    # 3. Table 9.1: Real SIFT-128 Texmex (n=2000, seeds 42-46)
    real_sift = data["real_sift_evaluation"]
    print("\n### Table 9.1: Real Texmex SIFT-128 Embeddings (n=2000, seeds 42–46)")
    print(f"Data source: {real_sift.get('source_info', 'siftsmall_base.fvecs')}")
    print("Parameters: M=8, M0=16, efConstruction=200, delta=0.30, lambda=8.0x, k=10, ef=16, 10 steps\n")
    print("| Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |")
    print("|:---|:---|:---:|:---:|:---:|:---:|")

    arm_configs = [
        ("no_repair", "—"),
        ("periodic_k1", "K=1"),
        ("periodic_k5", "K=5"),
        ("round_robin", f"B={real_sift['arm_stats']['round_robin']['mean_reinsertions']/10:.0f}"),
        ("lazy_theta_1.0", "θ=1.0"),
        ("lazy_theta_0.5", "θ=0.5"),
        ("budgeted_priority", f"B={real_sift['arm_stats']['budgeted_priority']['mean_reinsertions']/10:.0f}"),
    ]
    for arm, params in arm_configs:
        print(format_arm_row(arm, params, real_sift["arm_stats"][arm]))

    # Paired differences for real SIFT
    seeds = real_sift["seeds"]
    rr = [r["arms"]["round_robin"]["mean_drift_rec16"] for r in real_sift["seed_results"]]
    per = [r["arms"]["periodic_k5"]["mean_drift_rec16"] for r in real_sift["seed_results"]]
    paired_sift = compute_paired_stats(seeds, rr, per)
    print("\n" + generate_paired_table_md(paired_sift, "Real Texmex SIFT-128 (n=2000, seeds 42–46)"))

    # 4. Table 9.2: SIFT-like synthetic clustered fallback (n=500, seeds 42-46)
    synth_sift = data["sift_like_synthetic_fallback"]
    print("\n### Table 9.2: SIFT-like Synthetic Clustered Fallback (n=500, seeds 42–46)")
    print("Parameters: M=8, M0=16, efConstruction=16, delta=0.30, lambda=8.0x, k=10, ef=16, 10 steps\n")
    print("| Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |")
    print("|:---|:---|:---:|:---:|:---:|:---:|")

    for arm, params in arm_configs:
        print(format_arm_row(arm, params, synth_sift["arm_stats"][arm]))

    # Paired differences for synthetic SIFT
    seeds = synth_sift["seeds"]
    rr = [r["arms"]["round_robin"]["mean_drift_rec16"] for r in synth_sift["seed_results"]]
    per = [r["arms"]["periodic_k5"]["mean_drift_rec16"] for r in synth_sift["seed_results"]]
    paired_synth_sift = compute_paired_stats(seeds, rr, per)
    print("\n" + generate_paired_table_md(paired_synth_sift, "SIFT-like Synthetic Clustered Fallback (n=500, seeds 42–46)"))


if __name__ == "__main__":
    main()
