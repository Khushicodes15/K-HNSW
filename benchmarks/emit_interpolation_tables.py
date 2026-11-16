"""Generate markdown tables from interpolation benchmark JSON artifacts.

Pure generator functions that produce markdown tables for Chapter 9:
- T1: by-arm summary (5 conditions x 4 arms)
- T2_traj: paired differences (trajectory-mean, Rec@16 and Rec@32)
- T2_final: paired differences (final-step, Rec@16 and Rec@32)
- T3: self-excluded vs committed side-by-side with self-hit rates
- T4: co-location counts and jitter-norm ratios

Supports:
- --write: updates markdown chapter file between <!-- BEGIN GENERATED: <name> -->
  and <!-- END GENERATED: <name> -->
- --check: verifies chapter file matches exactly without modifying; exits 1 if any difference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repo root and src/ are in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from benchmarks.run_interpolation_benchmark import compute_paired_stats  # noqa: E402

CONDITION_DISPLAY_NAMES = {
    "f_0.25": "f=0.25 (no jitter)",
    "f_0.50": "f=0.50 (no jitter)",
    "f_1.00": "f=1.00 (no jitter)",
    "f_0.50_jitter": "f=0.50 (with jitter)",
    "f_1.00_jitter": "f=1.00 (with jitter)",
}

PAIR_DISPLAY_NAMES = {
    "periodic_k5_vs_no_repair": "periodic_k5 - no_repair",
    "round_robin_vs_no_repair": "round_robin - no_repair",
    "round_robin_vs_periodic_k5": "round_robin - periodic_k5",
}


def format_val_pm_sd(mean_val: float, sd_val: float, precision: int = 4) -> str:
    """Format mean +/- sd."""
    return f"{mean_val:.{precision}f} +/- {sd_val:.{precision}f}"


def format_cost_pm_sd(mean_val: float, sd_val: float) -> str:
    """Format cost as integer with commas +/- sd."""
    return f"{mean_val:,.0f} +/- {sd_val:,.0f}"


def generate_t1_by_arm_summary(data: dict[str, Any]) -> str:
    """Generate Table T1: by-arm summary table across 5 conditions and 4 arms."""
    h1 = (
        "| Condition | Policy Arm | Mean Drift Rec@16 | Final Drift Rec@16 "
        "| Mean Drift Rec@32 | Final Drift Rec@32 | Mean Static Rec@16 "
        "| Mean Cost (Dists) | Reinsertions |"
    )
    h2 = "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        cond_data = data["conditions"][cond_name]["arm_summary"]
        for arm_name in ["no_repair", "periodic_k5", "round_robin", "rebuild"]:
            arm = cond_data[arm_name]
            m16 = format_val_pm_sd(arm["mean_drift_rec16"]["mean"], arm["mean_drift_rec16"]["std"])
            f16 = format_val_pm_sd(
                arm["final_drift_rec16"]["mean"], arm["final_drift_rec16"]["std"]
            )
            m32 = format_val_pm_sd(arm["mean_drift_rec32"]["mean"], arm["mean_drift_rec32"]["std"])
            f32 = format_val_pm_sd(
                arm["final_drift_rec32"]["mean"], arm["final_drift_rec32"]["std"]
            )
            st16 = format_val_pm_sd(
                arm["mean_static_rec16"]["mean"], arm["mean_static_rec16"]["std"]
            )
            cst = format_cost_pm_sd(arm["mean_cost"]["mean"], arm["mean_cost"]["std"])
            reins = f"{arm['mean_reinsertions']['mean']:,.0f}"
            row = (
                f"| {display_name} | `{arm_name}` | {m16} | {f16} | {m32} | "
                f"{f32} | {st16} | {cst} | {reins} |"
            )
            lines.append(row)

    return "\n".join(lines)


def generate_t2_paired_differences(
    data: dict[str, Any], timing_type: str = "trajectory_mean"
) -> str:
    """Generate Table T2: paired differences table (trajectory_mean or final_step)."""
    is_traj = timing_type == "trajectory_mean"
    dict_key = "trajectory_mean_paired_differences" if is_traj else "final_step_paired_differences"
    metric_label = "Traj" if is_traj else "Final"

    h1 = (
        f"| Condition | Comparison | {metric_label} Rec@16 Diff | 95% CI | Pos "
        f"| {metric_label} Rec@32 Diff | 95% CI | Pos |"
    )
    h2 = "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        cond_pairs = data["conditions"][cond_name][dict_key]
        for p_key, p_display in PAIR_DISPLAY_NAMES.items():
            p16 = cond_pairs["rec16"][p_key]
            p32 = cond_pairs["rec32"][p_key]

            diff16 = f"{p16['mean']:+.4f}"
            ci16 = f"[{p16['t_interval_95'][0]:+.4f}, {p16['t_interval_95'][1]:+.4f}]"
            pos16 = f"{p16['positive_seeds']}/5"

            diff32 = f"{p32['mean']:+.4f}"
            ci32 = f"[{p32['t_interval_95'][0]:+.4f}, {p32['t_interval_95'][1]:+.4f}]"
            pos32 = f"{p32['positive_seeds']}/5"

            row = (
                f"| {display_name} | {p_display} | {diff16} | {ci16} | {pos16} "
                f"| {diff32} | {ci32} | {pos32} |"
            )
            lines.append(row)

    return "\n".join(lines)


def generate_t3_self_excluded_vs_committed(
    data: dict[str, Any], committed_data: dict[str, Any]
) -> str:
    """Generate Table T3: side-by-side comparison of self-excluded vs committed metrics."""
    h1 = (
        "| Condition | Arm | Comm Traj Rec@16 | Self-Ex Traj Rec@16 | Comm Final Rec@16 "
        "| Self-Ex Final Rec@16 | Self-Hit ef=16 | Comm Traj Rec@32 | Self-Ex Traj Rec@32 "
        "| Comm Final Rec@32 | Self-Ex Final Rec@32 |"
    )
    h2 = "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        se_arms = data["conditions"][cond_name]["arm_summary"]
        comm_arms = committed_data["conditions"][cond_name]["arm_summary"]

        for arm_name in ["no_repair", "periodic_k5", "round_robin", "rebuild"]:
            se_a = se_arms[arm_name]
            comm_a = comm_arms[arm_name]

            c_m16 = f"{comm_a['mean_drift_rec16']['mean']:.4f}"
            s_m16 = f"{se_a['mean_drift_rec16']['mean']:.4f}"
            c_f16 = f"{comm_a['final_drift_rec16']['mean']:.4f}"
            s_f16 = f"{se_a['final_drift_rec16']['mean']:.4f}"
            s_hit = f"{se_a['self_hit_rate_ef16']['mean']:.4f}"

            c_m32 = f"{comm_a['mean_drift_rec32']['mean']:.4f}"
            s_m32 = f"{se_a['mean_drift_rec32']['mean']:.4f}"
            c_f32 = f"{comm_a['final_drift_rec32']['mean']:.4f}"
            s_f32 = f"{se_a['final_drift_rec32']['mean']:.4f}"

            row = (
                f"| {display_name} | `{arm_name}` | {c_m16} | {s_m16} | {c_f16} | {s_f16} "
                f"| {s_hit} | {c_m32} | {s_m32} | {c_f32} | {s_f32} |"
            )
            lines.append(row)

    return "\n".join(lines)


def generate_t4_colocation_and_jitter(data: dict[str, Any]) -> str:
    """Generate Table T4: co-location counts and jitter-norm ratios."""
    h1 = (
        "| Condition | Seed 110 | Seed 111 | Seed 112 | Seed 113 | Seed 114 "
        "| Mean Near Count / 100 | Smallest Nonzero Dist / mean_1nn |"
    )
    h2 = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [
        "#### Duplicate Co-Location at Final Step ($t=0.30$)",
        h1,
        h2,
    ]

    coloc_data = data["diagnostics"]["colocation_counts"]
    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        cond_entry = coloc_data[cond_name]
        counts = [cond_entry[str(s)]["near_count"] for s in [110, 111, 112, 113, 114]]
        mean_c = sum(counts) / len(counts)
        ratios = [cond_entry[str(s)]["ratio_to_mean_1nn"] for s in [110, 111, 112, 113, 114]]
        min_ratio = min(ratios)

        cnt_cols = " | ".join(f"{c}/100" for c in counts)
        lines.append(f"| {display_name} | {cnt_cols} | {mean_c:.1f}/100 | {min_ratio:.6e} |")

    lines.append("")
    lines.append("#### Measured Jitter Displacement Norm Relative to 1-NN Distance")
    j_h1 = (
        "| Seed | Mean ||jitter|| / mean_1nn | Min ||jitter|| / mean_1nn "
        "| Max ||jitter|| / mean_1nn | Mean Ratio / 0.05 |"
    )
    lines.append(j_h1)
    lines.append("|:---:|:---:|:---:|:---:|:---:|")

    jitter_data = data["diagnostics"]["jitter_norm_ratios"]
    all_means = []
    for s in [110, 111, 112, 113, 114]:
        j = jitter_data[str(s)]
        all_means.append(j["mean_ratio"])
        row = (
            f"| {s} | {j['mean_ratio']:.6f} | {j['min_ratio']:.6f} | "
            f"{j['max_ratio']:.6f} | {j['ratio_div_005']:.6f} |"
        )
        lines.append(row)

    pooled_mean = sum(all_means) / len(all_means)
    pooled_div = pooled_mean / 0.05
    min_pooled = min(jitter_data[str(s)]["min_ratio"] for s in [110, 111, 112, 113, 114])
    max_pooled = max(jitter_data[str(s)]["max_ratio"] for s in [110, 111, 112, 113, 114])
    p_row = (
        f"| **Pooled** | **{pooled_mean:.6f}** | **{min_pooled:.6f}** | "
        f"**{max_pooled:.6f}** | **{pooled_div:.6f}** |"
    )
    lines.append(p_row)

    return "\n".join(lines)


def generate_all_tables(self_ex_json_path: Path, committed_json_path: Path) -> dict[str, str]:
    """Pure function generating all markdown tables from JSON artifacts."""
    with open(self_ex_json_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(committed_json_path, encoding="utf-8") as f:
        committed_data = json.load(f)

    # Verification of recomputation vs stored values
    pair_defs = [
        ("periodic_k5_vs_no_repair", "periodic_k5", "no_repair"),
        ("round_robin_vs_no_repair", "round_robin", "no_repair"),
        ("round_robin_vs_periodic_k5", "round_robin", "periodic_k5"),
    ]
    raw_seed_records = data["seed_records"]

    for cond_name in CONDITION_DISPLAY_NAMES:
        stored_traj = data["conditions"][cond_name]["trajectory_mean_paired_differences"]
        stored_final = data["conditions"][cond_name]["final_step_paired_differences"]

        for p_key, a1, a2 in pair_defs:
            # Traj rec16
            diffs_m16 = [
                s["conditions"][cond_name]["arms"][a1]["mean_drift_rec16"]
                - s["conditions"][cond_name]["arms"][a2]["mean_drift_rec16"]
                for s in raw_seed_records
            ]
            recomp_traj_16 = compute_paired_stats(diffs_m16)
            assert abs(recomp_traj_16["mean"] - stored_traj["rec16"][p_key]["mean"]) < 1e-9, (
                f"Stored trajectory mean diff for {p_key} rec16 does not match recomputed data: "
                f"{stored_traj['rec16'][p_key]['mean']} != {recomp_traj_16['mean']}"
            )

            # Final rec16
            diffs_f16 = [
                s["conditions"][cond_name]["arms"][a1]["final_drift_rec16"]
                - s["conditions"][cond_name]["arms"][a2]["final_drift_rec16"]
                for s in raw_seed_records
            ]
            recomp_final_16 = compute_paired_stats(diffs_f16)
            assert abs(recomp_final_16["mean"] - stored_final["rec16"][p_key]["mean"]) < 1e-9, (
                f"Stored final-step diff for {p_key} rec16 does not match recomputed data: "
                f"{stored_final['rec16'][p_key]['mean']} != {recomp_final_16['mean']}"
            )

    print(
        "[VERIFIED] Generator recomputed paired differences match stored JSON stats "
        "exactly across all conditions and metrics."
    )

    t2_traj = generate_t2_paired_differences(data, "trajectory_mean")
    t2_final = generate_t2_paired_differences(data, "final_step")
    return {
        "interpolation_by_arm_summary": generate_t1_by_arm_summary(data),
        "interpolation_paired_differences_trajectory_mean": t2_traj,
        "interpolation_paired_differences_final_step": t2_final,
        "interpolation_self_excluded_vs_committed": generate_t3_self_excluded_vs_committed(
            data, committed_data
        ),
        "interpolation_colocation_and_jitter": generate_t4_colocation_and_jitter(data),
    }


def update_markdown_file(file_path: Path, tables: dict[str, str], check_only: bool = False) -> bool:
    """Update or check markdown file containing generated table blocks."""
    if not file_path.exists():
        raise FileNotFoundError(f"Chapter file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    original_content = content
    has_diff = False

    for name, table_md in tables.items():
        pattern = rf"(<!-- BEGIN GENERATED: {name} -->\r?\n)(.*?)(<!-- END GENERATED: {name} -->)"
        match = re.search(pattern, content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Marker <!-- BEGIN GENERATED: {name} --> not found in {file_path}")

        current_block = match.group(2)
        if current_block.strip() != table_md.strip():
            has_diff = True
            if check_only:
                print(f"[CHECK FAILED] Block '{name}' differs in {file_path.name}")

        replacement = rf"\g<1>{table_md}\n\g<3>"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if not check_only and content != original_content:
        file_path.write_text(content, encoding="utf-8")
        print(f"[UPDATED] Written generated blocks to {file_path}")

    return not has_diff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit markdown tables for interpolation benchmark."
    )
    parser.add_argument(
        "--self-ex-json",
        type=Path,
        default=Path("docs/data/interpolation_self_excluded.json"),
        help="Path to interpolation_self_excluded.json",
    )
    parser.add_argument(
        "--committed-json",
        type=Path,
        default=Path("docs/data/interpolation_benchmark.json"),
        help="Path to committed interpolation_benchmark.json",
    )
    parser.add_argument(
        "--chapter-file",
        type=Path,
        default=Path("docs/report/chapter9_external_baselines.md"),
        help="Path to chapter 9 markdown file",
    )
    parser.add_argument(
        "--write", action="store_true", help="Write tables between markers in chapter file"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that chapter file matches exactly without modifying",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    self_ex_path = (
        repo_root / args.self_ex_json if not args.self_ex_json.is_absolute() else args.self_ex_json
    )
    comm_path = (
        repo_root / args.committed_json
        if not args.committed_json.is_absolute()
        else args.committed_json
    )
    chap_path = (
        repo_root / args.chapter_file if not args.chapter_file.is_absolute() else args.chapter_file
    )

    tables = generate_all_tables(self_ex_path, comm_path)

    if args.write:
        update_markdown_file(chap_path, tables, check_only=False)
    elif args.check:
        clean = update_markdown_file(chap_path, tables, check_only=True)
        if not clean:
            print("[CHECK FAILED] One or more generated table blocks differ.")
            sys.exit(1)
        else:
            print("[CHECK PASSED] All generated table blocks match chapter file exactly.")
    else:
        for name, md in tables.items():
            print(f"\n<!-- BEGIN GENERATED: {name} -->\n{md}\n<!-- END GENERATED: {name} -->\n")


if __name__ == "__main__":
    main()
