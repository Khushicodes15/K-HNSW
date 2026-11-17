"""Generate report tables directly from docs/data/external_benchmarks.json.

Guarantees 100% numerical fidelity between the raw committed JSON results
and the tables presented in Chapter 8 and Chapter 9.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def load_benchmarks() -> dict[str, Any]:
    json_path = Path("docs/data/external_benchmarks.json")
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def format_arm_row(
    arm_name: str, params: str, stats: dict[str, Any], include_rec32: bool = False
) -> str:
    mean_rec16 = stats.get("mean_drift_rec16", 0.0)
    std_rec16 = stats.get("std_drift_rec16", 0.0)
    final_rec16 = stats.get("final_drift_rec16", 0.0)
    std_final16 = stats.get("std_final_drift_rec16", 0.0)
    cost = stats.get("mean_cost", 0.0)
    std_cost = stats.get("std_cost", 0.0)
    reinserts = stats.get("mean_reinsertions", 0.0)

    cost_str = f"{cost:,.0f} ± {std_cost:,.0f}" if cost > 0 else "0"
    reinsert_str = f"{reinserts:,.0f}" if reinserts > 0 else "0"

    if include_rec32 and "mean_drift_rec32" in stats:
        mean_rec32 = stats.get("mean_drift_rec32", 0.0)
        std_rec32 = stats.get("std_drift_rec32", 0.0)
        return (
            f"| `{arm_name}` | {params} "
            f"| {mean_rec16:.4f} ± {std_rec16:.3f} "
            f"| {mean_rec32:.4f} ± {std_rec32:.3f} "
            f"| {final_rec16:.4f} ± {std_final16:.3f} "
            f"| {cost_str} "
            f"| {reinsert_str} |"
        )

    return (
        f"| `{arm_name}` | {params} "
        f"| {mean_rec16:.4f} ± {std_rec16:.3f} "
        f"| {final_rec16:.4f} ± {std_final16:.3f} "
        f"| {cost_str} "
        f"| {reinsert_str} |"
    )


def compute_paired_stats(
    seeds: list[int], rr_list: list[float], per_list: list[float], seed_boot: int = 42
) -> dict[str, Any]:
    diffs = np.array(rr_list) - np.array(per_list)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    s_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    t_crit = 2.776 if n == 5 else 2.0
    se = s_d / np.sqrt(n) if n > 0 else 0.0
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se
    pos_count = int(np.sum(diffs > 0))

    rng = np.random.default_rng(seed_boot)
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(10000)]
    boot_ci = tuple(float(x) for x in np.percentile(boot_means, [2.5, 97.5]))

    per_seed = [
        {"seed": s, "rr": r, "per": p, "diff": d}
        for s, r, p, d in zip(seeds, rr_list, per_list, diffs, strict=False)
    ]

    return {
        "per_seed": per_seed,
        "mean_diff": mean_d,
        "std_diff": s_d,
        "ci_95": (ci_low, ci_high),
        "boot_ci_95": boot_ci,
        "positive_count": pos_count,
        "total_seeds": n,
    }


def generate_paired_table_md(
    stats: dict[str, Any], title: str, metric_label: str = "Recall@16"
) -> str:
    lines = [
        f"#### Paired Differences: {title} ({metric_label})",
        "",
        (
            f"| Seed | `round_robin` {metric_label} | `periodic_k5` {metric_label} | "
            "Paired Difference (RR − Per) | Sign |"
        ),
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in stats["per_seed"]:
        sign = "+" if row["diff"] > 0 else "-"
        s_val = row["seed"]
        rr_v = row["rr"]
        per_v = row["per"]
        d_v = row["diff"]
        lines.append(f"| **{s_val}** | {rr_v:.4f} | {per_v:.4f} | **{d_v:+.4f}** | {sign} |")

    ci = stats["ci_95"]
    boot_ci = stats.get("boot_ci_95")
    md = stats["mean_diff"]
    lines.append("")
    lines.append(f"- **Mean Difference (Δ)**: **{md:+.4f}** ({md * 100:+.2f} pp)")
    lines.append(f"- **Sample Std ($s$)**: {stats['std_diff']:.4f}")
    lines.append(f"- **95% Confidence Interval (t-stat)**: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    if boot_ci:
        lines.append(f"- **95% Bootstrap CI**: [{boot_ci[0]:+.4f}, {boot_ci[1]:+.4f}]")
    lines.append(f"- **Positive Seeds**: **{stats['positive_count']} / {stats['total_seeds']}**")
    lines.append("")
    return "\n".join(lines)


def update_report(report_path: Path | None = None) -> None:
    if report_path is None:
        report_path = Path("docs/report/chapter9_external_baselines.md")

    data = load_benchmarks()
    content = report_path.read_text(encoding="utf-8")

    # Generate Table 9.1A rows from condition 0 of real_sift_evaluation
    c2000 = data["real_sift_evaluation"]["conditions"][0]
    arm_configs = [
        ("no_repair", "—"),
        ("periodic_k1", "K=1"),
        ("periodic_k5", "K=5"),
        ("round_robin", f"B={c2000['arm_stats']['round_robin']['mean_reinsertions'] / 10:.0f}"),
        ("lazy_theta_1.0", "θ=1.0"),
        ("lazy_theta_0.5", "θ=0.5"),
        (
            "budgeted_priority",
            f"B={c2000['arm_stats']['budgeted_priority']['mean_reinsertions'] / 10:.0f}",
        ),
    ]
    rows_9_1a = [
        format_arm_row(arm, params, c2000["arm_stats"][arm], include_rec32=True)
        for arm, params in arm_configs
    ]

    pattern_9_1a = re.compile(
        r"(### Table 9\.1A: Real Texmex SIFT-128 Embeddings.*?\n"
        r"\|:---\|:---\|:---:\|:---:\|:---:\|:---:\|:---:\|\n)"
        r"(?:\| `.*?` \|.*?\n)+"
        r"(\n#### Paired Differences: Real Texmex SIFT-128 \(\$n=2000\$)",
        re.DOTALL,
    )

    replacement_9_1a = r"\g<1>" + "\n".join(rows_9_1a) + "\n" + r"\g<2>"
    new_content, count = pattern_9_1a.subn(replacement_9_1a, content)
    if count == 0:
        raise RuntimeError(f"Failed to locate Table 9.1A in {report_path}")

    report_path.write_text(new_content, encoding="utf-8")
    print(f"Successfully updated Table 9.1A in {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or update report tables.")
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="Update report markdown files in-place with the latest benchmark tables.",
    )
    args = parser.parse_args()

    if args.update_report:
        update_report()
        return

    data = load_benchmarks()

    print("=" * 80)
    print("TABLES FOR CHAPTER 8: Dynamic Drift Maintenance on Synthetic Motion")
    print("=" * 80)

    # 1. Table 8.1A: Synthetic ef_construction=16 (seeds 42-46)
    conds_16 = data["synthetic_drift_ef16"]["conditions"]
    print("\n### Table 8.1A: Synthetic Gaussian Drift (efConstruction = 16, seeds 42–46)")
    print(
        "Parameters: M=8, M0=16, efConstruction=16, delta=0.30, lambda=8.0x, "
        "k=10, ef=16, 10 steps\n"
    )
    t_hdr = (
        "| Size | Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 "
        "| Mean Cost (Dists) | Reinsertions |"
    )
    print(t_hdr)
    print("|:---|:---|:---|:---:|:---:|:---:|:---:|")

    for cond in conds_16:
        n = cond["n"]
        arm_configs = [
            ("no_repair", "—"),
            ("periodic_k1", "K=1"),
            ("periodic_k5", "K=5"),
            ("round_robin", f"B={cond['arm_stats']['round_robin']['mean_reinsertions'] / 10:.0f}"),
            ("lazy_theta_1.0", "θ=1.0"),
            ("lazy_theta_0.5", "θ=0.5"),
            (
                "budgeted_priority",
                f"B={cond['arm_stats']['budgeted_priority']['mean_reinsertions'] / 10:.0f}",
            ),
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
    print(
        "Parameters: M=8, M0=16, efConstruction=200, delta=0.30, lambda=8.0x, "
        "k=10, ef=16, 10 steps\n"
    )
    print(t_hdr)
    print("|:---|:---|:---|:---:|:---:|:---:|:---:|")

    for cond in conds_200:
        n = cond["n"]
        arm_configs = [
            ("no_repair", "—"),
            ("periodic_k1", "K=1"),
            ("periodic_k5", "K=5"),
            ("round_robin", f"B={cond['arm_stats']['round_robin']['mean_reinsertions'] / 10:.0f}"),
            ("lazy_theta_1.0", "θ=1.0"),
            ("lazy_theta_0.5", "θ=0.5"),
            (
                "budgeted_priority",
                f"B={cond['arm_stats']['budgeted_priority']['mean_reinsertions'] / 10:.0f}",
            ),
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

    # 3. Table 9.1: Real Texmex SIFT-128 Embeddings (seeds 110-114, ef_construction=200)
    real_sift_data = data["real_sift_evaluation"]
    sift_conditions = real_sift_data.get("conditions", [real_sift_data])

    for cond in sift_conditions:
        n = cond["n"]
        sha = cond.get("file_sha256", real_sift_data.get("file_sha256", "unknown"))
        scale = cond.get("mean_scale_1nn", 0.0)
        seeds = cond.get("seeds", [110, 111, 112, 113, 114])
        s_start, s_end = seeds[0], seeds[-1]
        print(f"\n### Table 9.1: Real Texmex SIFT-128 Embeddings (n={n}, seeds {s_start}–{s_end})")
        print(f"Data Source: {cond.get('source_info', 'siftsmall_base.fvecs')} (SHA-256: `{sha}`)")
        print(f"Scale Calibration: empirical 1-NN mean distance = {scale:.2f}")
        print(
            "Parameters: M=8, M0=16, efConstruction=200, delta=0.30, lambda=8.0x, k=10, 10 steps\n"
        )
        sift_hdr = (
            "| Policy Arm | Parameters | Mean Drift Rec@16 | Mean Drift Rec@32 "
            "| Final Drift Rec@16 | Mean Cost (Dists) | Reinsertions |"
        )
        print(sift_hdr)
        print("|:---|:---|:---:|:---:|:---:|:---:|:---:|")

        arm_configs = [
            ("no_repair", "—"),
            ("periodic_k1", "K=1"),
            ("periodic_k5", "K=5"),
            ("round_robin", f"B={cond['arm_stats']['round_robin']['mean_reinsertions'] / 10:.0f}"),
            ("lazy_theta_1.0", "θ=1.0"),
            ("lazy_theta_0.5", "θ=0.5"),
            (
                "budgeted_priority",
                f"B={cond['arm_stats']['budgeted_priority']['mean_reinsertions'] / 10:.0f}",
            ),
        ]
        for arm, params in arm_configs:
            print(format_arm_row(arm, params, cond["arm_stats"][arm], include_rec32=True))

        # Paired differences for Rec@16
        rr_16 = [r["arms"]["round_robin"]["mean_drift_rec16"] for r in cond["seed_results"]]
        per_16 = [r["arms"]["periodic_k5"]["mean_drift_rec16"] for r in cond["seed_results"]]
        paired_16 = compute_paired_stats(seeds, rr_16, per_16)
        print(
            "\n"
            + generate_paired_table_md(
                paired_16,
                f"Real Texmex SIFT-128 (n={n}, seeds {seeds[0]}–{seeds[-1]})",
                "Recall@16",
            )
        )

        # Paired differences for Rec@32
        rr_32 = [r["arms"]["round_robin"]["mean_drift_rec32"] for r in cond["seed_results"]]
        per_32 = [r["arms"]["periodic_k5"]["mean_drift_rec32"] for r in cond["seed_results"]]
        paired_32 = compute_paired_stats(seeds, rr_32, per_32)
        print(
            "\n"
            + generate_paired_table_md(
                paired_32,
                f"Real Texmex SIFT-128 (n={n}, seeds {seeds[0]}–{seeds[-1]})",
                "Recall@32",
            )
        )

    # 4. Table 9.2: SIFT-like synthetic clustered fallback (n=500, seeds 42-46)
    synth_sift = data["sift_like_synthetic_fallback"]
    print("\n### Table 9.2: SIFT-like Synthetic Clustered (n=500, seeds 42–46)")
    print(f"Data source: {synth_sift.get('source_info', 'SIFT-like synthetic')}")
    print(
        "Parameters: M=8, M0=16, efConstruction=16, delta=0.30, lambda=8.0x, "
        "k=10, ef=16, 10 steps\n"
    )
    fallback_hdr = (
        "| Policy Arm | Parameters | Mean Drift Rec@16 | Final Drift Rec@16 "
        "| Mean Cost (Dists) | Reinsertions |"
    )
    print(fallback_hdr)
    print("|:---|:---|:---:|:---:|:---:|:---:|")

    arm_configs_synth = [
        ("no_repair", "—"),
        ("periodic_k1", "K=1"),
        ("periodic_k5", "K=5"),
        (
            "round_robin",
            f"B={synth_sift['arm_stats']['round_robin']['mean_reinsertions'] / 10:.0f}",
        ),
        ("lazy_theta_1.0", "θ=1.0"),
        ("lazy_theta_0.5", "θ=0.5"),
        (
            "budgeted_priority",
            f"B={synth_sift['arm_stats']['budgeted_priority']['mean_reinsertions'] / 10:.0f}",
        ),
    ]
    for arm, params in arm_configs_synth:
        print(format_arm_row(arm, params, synth_sift["arm_stats"][arm]))

    # Paired differences for synthetic SIFT
    seeds_synth = synth_sift["seeds"]
    rr_synth = [r["arms"]["round_robin"]["mean_drift_rec16"] for r in synth_sift["seed_results"]]
    per_synth = [r["arms"]["periodic_k5"]["mean_drift_rec16"] for r in synth_sift["seed_results"]]
    paired_synth_sift = compute_paired_stats(seeds_synth, rr_synth, per_synth)
    print(
        "\n"
        + generate_paired_table_md(
            paired_synth_sift,
            "SIFT-like Synthetic Clustered (n=500, seeds 42–46)",
            "Recall@16",
        )
    )


if __name__ == "__main__":
    main()
