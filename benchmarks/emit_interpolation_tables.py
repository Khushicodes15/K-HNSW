"""Generate markdown tables and inline values from interpolation benchmark JSON artifacts.

Pure generator functions that produce markdown tables and inline metrics for Chapter 9:
- T1: by-arm summary (5 conditions x 4 arms)
- T2_traj: paired differences (trajectory-mean, Rec@16 and Rec@32)
- T2_final: paired differences (final-step, Rec@16 and Rec@32)
- T3: self-excluded vs committed side-by-side with self-hit rates
- T4: co-location counts and jitter-norm ratios
- T5_step: step dynamics for f_0.50_jitter (steps 8, 9, 10)
- T6_cost: cost and ratios to rebuild per condition and arm
- Inlines: dictionary of named scalar/interval metrics for Chapter 9 prose and bullets

Supports:
- --write: updates markdown chapter file between table markers and inline markers
- --check: verifies chapter file matches exactly without modifying; exits 1 if any difference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root and src/ are in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from benchmarks.run_interpolation_benchmark import compute_paired_stats  # noqa: E402

T_STUDENT_95_N5 = 2.7764451051977987

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


def _get_stat_mean(stat_obj: dict[str, Any]) -> float:
    """Return mean from per_seed list if available, else from mean field."""
    if "per_seed" in stat_obj and stat_obj["per_seed"]:
        return float(np.mean(stat_obj["per_seed"]))
    return float(stat_obj["mean"])


def _get_stat_std(stat_obj: dict[str, Any]) -> float:
    """Return sample std from per_seed list if available, else from std field."""
    if "per_seed" in stat_obj and len(stat_obj["per_seed"]) > 1:
        return float(np.std(stat_obj["per_seed"], ddof=1))
    return float(stat_obj.get("std", 0.0))


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
            m16 = format_val_pm_sd(
                _get_stat_mean(arm["mean_drift_rec16"]), _get_stat_std(arm["mean_drift_rec16"])
            )
            f16 = format_val_pm_sd(
                _get_stat_mean(arm["final_drift_rec16"]), _get_stat_std(arm["final_drift_rec16"])
            )
            m32 = format_val_pm_sd(
                _get_stat_mean(arm["mean_drift_rec32"]), _get_stat_std(arm["mean_drift_rec32"])
            )
            f32 = format_val_pm_sd(
                _get_stat_mean(arm["final_drift_rec32"]), _get_stat_std(arm["final_drift_rec32"])
            )
            st16 = format_val_pm_sd(
                _get_stat_mean(arm["mean_static_rec16"]), _get_stat_std(arm["mean_static_rec16"])
            )
            c_obj = arm.get("mean_cost") or arm.get("total_cost", {})
            cst = format_cost_pm_sd(_get_stat_mean(c_obj), _get_stat_std(c_obj))
            r_obj = arm.get("mean_reinsertions") or arm.get("total_reinsertions", {})
            reins = f"{_get_stat_mean(r_obj):,.0f}"
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

            c_m16 = f"{_get_stat_mean(comm_a['mean_drift_rec16']):.4f}"
            s_m16 = f"{_get_stat_mean(se_a['mean_drift_rec16']):.4f}"
            c_f16 = f"{_get_stat_mean(comm_a['final_drift_rec16']):.4f}"
            s_f16 = f"{_get_stat_mean(se_a['final_drift_rec16']):.4f}"
            s_hit = f"{_get_stat_mean(se_a['self_hit_rate_ef16']):.4f}"

            c_m32 = f"{_get_stat_mean(comm_a['mean_drift_rec32']):.4f}"
            s_m32 = f"{_get_stat_mean(se_a['mean_drift_rec32']):.4f}"
            c_f32 = f"{_get_stat_mean(comm_a['final_drift_rec32']):.4f}"
            s_f32 = f"{_get_stat_mean(se_a['final_drift_rec32']):.4f}"

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


def generate_t5_step_dynamics_f050j(comm_data: dict[str, Any]) -> str:
    """Generate Table T5: step dynamics at steps 8, 9, 10 for condition f_0.50_jitter."""
    h1 = (
        "| Step | t | periodic_k5 Reinserted | periodic_k5 Rec@16 | round_robin Reinserted "
        "| round_robin Rec@16 | Paired Diff (RR - k5) | Pos Seeds |"
    )
    h2 = "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for step, t_val in [(8, "0.24"), (9, "0.27"), (10, "0.30")]:
        rr_vals = []
        k5_vals = []
        diffs = []
        rr_reins = 0
        k5_reins = 0
        for s_rec in comm_data["seed_records"]:
            arms = s_rec["conditions"]["f_0.50_jitter"]["arms"]
            rr_step = next(st for st in arms["round_robin"]["step_records"] if st["step"] == step)
            k5_step = next(st for st in arms["periodic_k5"]["step_records"] if st["step"] == step)
            rr_vals.append(rr_step["drift_rec16"])
            k5_vals.append(k5_step["drift_rec16"])
            diffs.append(rr_step["drift_rec16"] - k5_step["drift_rec16"])
            rr_reins = rr_step["reinserted_count"]
            k5_reins = k5_step["reinserted_count"]

        k5_str = f"{np.mean(k5_vals):.4f} +/- {np.std(k5_vals, ddof=1):.4f}"
        rr_str = f"{np.mean(rr_vals):.4f} +/- {np.std(rr_vals, ddof=1):.4f}"
        d_mean = np.mean(diffs)
        d_std = np.std(diffs, ddof=1)
        diff_str = f"{d_mean:+.4f} +/- {d_std:.4f}"
        pos_cnt = sum(1 for d in diffs if d > 0)
        row = (
            f"| Step {step} | t={t_val} | {k5_reins} | {k5_str} | {rr_reins} | "
            f"{rr_str} | {diff_str} | {pos_cnt}/5 |"
        )
        lines.append(row)

    return "\n".join(lines)


def generate_t6_cost_and_ratios(comm_data: dict[str, Any]) -> str:
    """Generate Table T6: cost and ratio of repair arms to rebuild across conditions."""
    h1 = (
        "| Condition | Policy Arm | Mean Cost (Dists) | Seed 110 Ratio | Seed 111 Ratio "
        "| Seed 112 Ratio | Seed 113 Ratio | Seed 114 Ratio | Mean Ratio |"
    )
    h2 = "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for cond_name, disp in CONDITION_DISPLAY_NAMES.items():
        cond_arms = comm_data["conditions"][cond_name]["arm_summary"]
        reb_costs = cond_arms["rebuild"]["total_cost"]["per_seed"]
        reb_mean = cond_arms["rebuild"]["total_cost"]["mean"]
        reb_std = cond_arms["rebuild"]["total_cost"]["std"]

        # Row 1: rebuild
        reb_cost_str = f"{reb_mean:,.0f} +/- {reb_std:,.0f}"
        lines.append(
            f"| {disp} | `rebuild` | {reb_cost_str} | 1.0000 | 1.0000 | 1.0000 | "
            f"1.0000 | 1.0000 | 1.0000 |"
        )

        # Row 2: periodic_k5
        k5_costs = cond_arms["periodic_k5"]["total_cost"]["per_seed"]
        k5_mean = cond_arms["periodic_k5"]["total_cost"]["mean"]
        k5_std = cond_arms["periodic_k5"]["total_cost"]["std"]
        k5_cost_str = f"{k5_mean:,.0f} +/- {k5_std:,.0f}"
        k5_ratios = [k / r for k, r in zip(k5_costs, reb_costs, strict=False)]
        r_cols_k5 = " | ".join(f"{r:.4f}" for r in k5_ratios)
        lines.append(
            f"| {disp} | `periodic_k5` | {k5_cost_str} | {r_cols_k5} | {np.mean(k5_ratios):.4f} |"
        )

        # Row 3: round_robin
        rr_costs = cond_arms["round_robin"]["total_cost"]["per_seed"]
        rr_mean = cond_arms["round_robin"]["total_cost"]["mean"]
        rr_std = cond_arms["round_robin"]["total_cost"]["std"]
        rr_cost_str = f"{rr_mean:,.0f} +/- {rr_std:,.0f}"
        rr_ratios = [rr / r for rr, r in zip(rr_costs, reb_costs, strict=False)]
        r_cols_rr = " | ".join(f"{r:.4f}" for r in rr_ratios)
        lines.append(
            f"| {disp} | `round_robin` | {rr_cost_str} | {r_cols_rr} | {np.mean(rr_ratios):.4f} |"
        )

    return "\n".join(lines)


def generate_t7_cost_and_recall_comparison_committed(
    committed_data: dict[str, Any],
) -> str:
    """Generate Table T7: trajectory-mean & final-step Rec@16 (committed)."""
    h1 = (
        "| Condition | Rebuild Rec@16 (Final) | Round-Robin Traj Rec@16 | Round-Robin Final Rec@16 "
        "| Round-Robin / Rebuild Cost | Periodic_k5 Traj Rec@16 | Periodic_k5 Final Rec@16 "
        "| Periodic_k5 / Rebuild Cost |"
    )
    h2 = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        arms_comm = committed_data["conditions"][cond_name]["arm_summary"]

        reb_rec = format_val_pm_sd(
            _get_stat_mean(arms_comm["rebuild"]["final_drift_rec16"]),
            _get_stat_std(arms_comm["rebuild"]["final_drift_rec16"]),
        )
        rr_traj = format_val_pm_sd(
            _get_stat_mean(arms_comm["round_robin"]["mean_drift_rec16"]),
            _get_stat_std(arms_comm["round_robin"]["mean_drift_rec16"]),
        )
        rr_final = format_val_pm_sd(
            _get_stat_mean(arms_comm["round_robin"]["final_drift_rec16"]),
            _get_stat_std(arms_comm["round_robin"]["final_drift_rec16"]),
        )
        k5_traj = format_val_pm_sd(
            _get_stat_mean(arms_comm["periodic_k5"]["mean_drift_rec16"]),
            _get_stat_std(arms_comm["periodic_k5"]["mean_drift_rec16"]),
        )
        k5_final = format_val_pm_sd(
            _get_stat_mean(arms_comm["periodic_k5"]["final_drift_rec16"]),
            _get_stat_std(arms_comm["periodic_k5"]["final_drift_rec16"]),
        )

        reb_costs = arms_comm["rebuild"]["total_cost"]["per_seed"]
        rr_costs = arms_comm["round_robin"]["total_cost"]["per_seed"]
        k5_costs = arms_comm["periodic_k5"]["total_cost"]["per_seed"]

        rr_ratio = float(np.mean([rr / r for rr, r in zip(rr_costs, reb_costs, strict=False)]))
        k5_ratio = float(np.mean([k / r for k, r in zip(k5_costs, reb_costs, strict=False)]))

        row = (
            f"| {display_name} | {reb_rec} | {rr_traj} | {rr_final} | {rr_ratio:.4f} "
            f"| {k5_traj} | {k5_final} | {k5_ratio:.4f} |"
        )
        lines.append(row)

    return "\n".join(lines)


def generate_t8_cost_and_recall_comparison_self_excluded(
    self_ex_data: dict[str, Any], committed_data: dict[str, Any]
) -> str:
    """Generate Table T8: trajectory-mean & final-step Rec@16 (self-excluded)."""
    h1 = (
        "| Condition | Rebuild Rec@16 (Final) | Round-Robin Traj Rec@16 | Round-Robin Final Rec@16 "
        "| Round-Robin / Rebuild Cost | Periodic_k5 Traj Rec@16 | Periodic_k5 Final Rec@16 "
        "| Periodic_k5 / Rebuild Cost |"
    )
    h2 = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        arms_se = self_ex_data["conditions"][cond_name]["arm_summary"]
        arms_comm = committed_data["conditions"][cond_name]["arm_summary"]

        reb_rec = format_val_pm_sd(
            _get_stat_mean(arms_se["rebuild"]["final_drift_rec16"]),
            _get_stat_std(arms_se["rebuild"]["final_drift_rec16"]),
        )
        rr_traj = format_val_pm_sd(
            _get_stat_mean(arms_se["round_robin"]["mean_drift_rec16"]),
            _get_stat_std(arms_se["round_robin"]["mean_drift_rec16"]),
        )
        rr_final = format_val_pm_sd(
            _get_stat_mean(arms_se["round_robin"]["final_drift_rec16"]),
            _get_stat_std(arms_se["round_robin"]["final_drift_rec16"]),
        )
        k5_traj = format_val_pm_sd(
            _get_stat_mean(arms_se["periodic_k5"]["mean_drift_rec16"]),
            _get_stat_std(arms_se["periodic_k5"]["mean_drift_rec16"]),
        )
        k5_final = format_val_pm_sd(
            _get_stat_mean(arms_se["periodic_k5"]["final_drift_rec16"]),
            _get_stat_std(arms_se["periodic_k5"]["final_drift_rec16"]),
        )

        reb_costs = arms_comm["rebuild"]["total_cost"]["per_seed"]
        rr_costs = arms_comm["round_robin"]["total_cost"]["per_seed"]
        k5_costs = arms_comm["periodic_k5"]["total_cost"]["per_seed"]

        rr_ratio = float(np.mean([rr / r for rr, r in zip(rr_costs, reb_costs, strict=False)]))
        k5_ratio = float(np.mean([k / r for k, r in zip(k5_costs, reb_costs, strict=False)]))

        row = (
            f"| {display_name} | {reb_rec} | {rr_traj} | {rr_final} | {rr_ratio:.4f} "
            f"| {k5_traj} | {k5_final} | {k5_ratio:.4f} |"
        )
        lines.append(row)

    return "\n".join(lines)


def generate_t9_final_step_paired_differences_committed(
    committed_data: dict[str, Any],
) -> str:
    """Generate Table 9.4J: final-step paired differences for committed metric."""
    h1 = "| Condition | Comparison | Final Rec@16 Diff | 95% CI | Pos |"
    h2 = "|:---|:---|:---:|:---:|:---:|"
    lines = [h1, h2]

    pair_keys = [
        ("round_robin_minus_rebuild", "round_robin - rebuild", "round_robin", "rebuild"),
        ("periodic_k5_minus_rebuild", "periodic_k5 - rebuild", "periodic_k5", "rebuild"),
        (
            "round_robin_minus_periodic_k5",
            "round_robin - periodic_k5",
            "round_robin",
            "periodic_k5",
        ),
    ]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        arms = committed_data["conditions"][cond_name]["arm_summary"]
        for _, comp_label, arm1, arm2 in pair_keys:
            vals1 = arms[arm1]["final_drift_rec16"]["per_seed"]
            vals2 = arms[arm2]["final_drift_rec16"]["per_seed"]
            diffs = [v1 - v2 for v1, v2 in zip(vals1, vals2, strict=False)]
            mean_d = float(np.mean(diffs))
            std_d = float(np.std(diffs, ddof=1))
            ci_half = T_STUDENT_95_N5 * std_d / np.sqrt(5)
            ci_low = mean_d - ci_half
            ci_high = mean_d + ci_half
            pos_count = sum(1 for d in diffs if d > 0)

            diff_str = f"{mean_d:+.4f}"
            ci_str = f"[{ci_low:+.4f}, {ci_high:+.4f}]"
            pos_str = f"{pos_count}/5"

            row = f"| {display_name} | {comp_label} | {diff_str} | {ci_str} | {pos_str} |"
            lines.append(row)

    return "\n".join(lines)


def generate_t10_final_step_paired_differences_self_excluded(
    self_ex_data: dict[str, Any],
) -> str:
    """Generate Table 9.4K: final-step paired differences for self-excluded metric."""
    h1 = "| Condition | Comparison | Final Rec@16 Diff | 95% CI | Pos |"
    h2 = "|:---|:---|:---:|:---:|:---:|"
    lines = [h1, h2]

    pair_keys = [
        ("round_robin_minus_rebuild", "round_robin - rebuild", "round_robin", "rebuild"),
        ("periodic_k5_minus_rebuild", "periodic_k5 - rebuild", "periodic_k5", "rebuild"),
        (
            "round_robin_minus_periodic_k5",
            "round_robin - periodic_k5",
            "round_robin",
            "periodic_k5",
        ),
    ]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        arms = self_ex_data["conditions"][cond_name]["arm_summary"]
        for _, comp_label, arm1, arm2 in pair_keys:
            vals1 = arms[arm1]["final_drift_rec16"]["per_seed"]
            vals2 = arms[arm2]["final_drift_rec16"]["per_seed"]
            diffs = [v1 - v2 for v1, v2 in zip(vals1, vals2, strict=False)]
            mean_d = float(np.mean(diffs))
            std_d = float(np.std(diffs, ddof=1))
            ci_half = T_STUDENT_95_N5 * std_d / np.sqrt(5)
            ci_low = mean_d - ci_half
            ci_high = mean_d + ci_half
            pos_count = sum(1 for d in diffs if d > 0)

            diff_str = f"{mean_d:+.4f}"
            ci_str = f"[{ci_low:+.4f}, {ci_high:+.4f}]"
            pos_str = f"{pos_count}/5"

            row = f"| {display_name} | {comp_label} | {diff_str} | {ci_str} | {pos_str} |"
            lines.append(row)

    return "\n".join(lines)


def generate_t11_per_reinsertion_cost(committed_data: dict[str, Any]) -> str:
    """Generate Table 9.4L: per-reinsertion cost and ratio to rebuild node cost."""
    h1 = (
        "| Condition | Rebuild Cost | Rebuild Cost / Node | Periodic_k5 Cost "
        "| Periodic_k5 Cost / Reins | Periodic_k5 Ratio | Round-Robin Cost "
        "| Round-Robin Cost / Reins | Round-Robin Ratio |"
    )
    h2 = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [h1, h2]

    for cond_name, display_name in CONDITION_DISPLAY_NAMES.items():
        cond_data = committed_data["conditions"][cond_name]["arm_summary"]

        reb_cost = float(cond_data["rebuild"]["total_cost"]["mean"])
        reb_cost_per_node = reb_cost / 2000.0

        k5_cost = float(cond_data["periodic_k5"]["total_cost"]["mean"])
        k5_reins = float(cond_data["periodic_k5"]["total_reinsertions"]["mean"])
        k5_cost_per_reins = k5_cost / k5_reins if k5_reins > 0 else 0.0
        k5_ratio = k5_cost_per_reins / reb_cost_per_node

        rr_cost = float(cond_data["round_robin"]["total_cost"]["mean"])
        rr_reins = float(cond_data["round_robin"]["total_reinsertions"]["mean"])
        rr_cost_per_reins = rr_cost / rr_reins if rr_reins > 0 else 0.0
        rr_ratio = rr_cost_per_reins / reb_cost_per_node

        row = (
            f"| {display_name} | {reb_cost:,.1f} | {reb_cost_per_node:.2f} | {k5_cost:,.1f} "
            f"| {k5_cost_per_reins:.2f} | {k5_ratio:.4f} | {rr_cost:,.1f} "
            f"| {rr_cost_per_reins:.2f} | {rr_ratio:.4f} |"
        )
        lines.append(row)

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
            assert abs(stored_traj["rec16"][p_key]["mean"] - recomp_traj_16["mean"]) < 1e-9, (
                "Stored traj rec16 diff does not match recomputed data"
            )

            # Final rec16
            diffs_f16 = [
                s["conditions"][cond_name]["arms"][a1]["final_drift_rec16"]
                - s["conditions"][cond_name]["arms"][a2]["final_drift_rec16"]
                for s in raw_seed_records
            ]
            recomp_final_16 = compute_paired_stats(diffs_f16)
            assert abs(stored_final["rec16"][p_key]["mean"] - recomp_final_16["mean"]) < 1e-9, (
                "Stored final rec16 diff does not match recomputed data"
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
        "interpolation_step_dynamics_f050j": generate_t5_step_dynamics_f050j(committed_data),
        "interpolation_cost_and_ratios": generate_t6_cost_and_ratios(committed_data),
        "interpolation_cost_and_recall_comparison_committed": (
            generate_t7_cost_and_recall_comparison_committed(committed_data)
        ),
        "interpolation_cost_and_recall_comparison_self_excluded": (
            generate_t8_cost_and_recall_comparison_self_excluded(data, committed_data)
        ),
        "interpolation_final_step_paired_differences_committed": (
            generate_t9_final_step_paired_differences_committed(committed_data)
        ),
        "interpolation_final_step_paired_differences_self_excluded": (
            generate_t10_final_step_paired_differences_self_excluded(data)
        ),
        "interpolation_per_reinsertion_cost": (generate_t11_per_reinsertion_cost(committed_data)),
    }


def generate_all_inlines(
    self_ex_json_path: Path,
    committed_json_path: Path,
    ext_json_path: Path | None = None,
    sift_5k_json_path: Path | None = None,
) -> dict[str, str]:
    """Pure function generating all named inline metric strings from JSON artifacts."""
    with open(self_ex_json_path, encoding="utf-8") as f:
        self_ex = json.load(f)
    with open(committed_json_path, encoding="utf-8") as f:
        comm = json.load(f)

    repo_dir = committed_json_path.parent.parent
    if ext_json_path is None:
        ext_json_path = repo_dir / "data" / "external_benchmarks.json"
    if sift_5k_json_path is None:
        sift_5k_json_path = repo_dir / "data" / "sift_n5000_ef200.json"

    inlines: dict[str, str] = {}

    # 1. Costs and Ratios
    costs_dyn = []
    costs_rebuild = []
    all_k5_ratios = []
    all_rr_ratios = []
    cond_mean_ratios = []
    per_reb_means, rr_reb_means, rr_per_means = [], [], []
    per_reb_seeds, rr_reb_seeds, rr_per_seeds = [], [], []

    for cond_name in ["f_0.25", "f_0.50", "f_0.50_jitter", "f_1.00", "f_1.00_jitter"]:
        arms = comm["conditions"][cond_name]["arm_summary"]
        k5_costs = arms["periodic_k5"]["total_cost"]["per_seed"]
        rr_costs = arms["round_robin"]["total_cost"]["per_seed"]
        reb_costs = arms["rebuild"]["total_cost"]["per_seed"]

        pr = [k / r for k, r in zip(k5_costs, reb_costs, strict=False)]
        rrr = [rr / r for rr, r in zip(rr_costs, reb_costs, strict=False)]
        rp = [rr / k for rr, k in zip(rr_costs, k5_costs, strict=False)]

        per_reb_means.append(float(np.mean(pr)))
        rr_reb_means.append(float(np.mean(rrr)))
        rr_per_means.append(float(np.mean(rp)))

        per_reb_seeds.extend(pr)
        rr_reb_seeds.extend(rrr)
        rr_per_seeds.extend(rp)

        for a in ["periodic_k5", "round_robin"]:
            cost_obj = arms[a].get("total_cost") or arms[a].get("mean_cost", {})
            costs_dyn.append(_get_stat_mean(cost_obj))
        r_cost = arms["rebuild"].get("total_cost") or arms["rebuild"].get("mean_cost", {})
        costs_rebuild.append(_get_stat_mean(r_cost))

        k5_r = [k / r for k, r in zip(k5_costs, reb_costs, strict=False)]
        rr_r = [rr / r for rr, r in zip(rr_costs, reb_costs, strict=False)]
        all_k5_ratios.extend(k5_r)
        all_rr_ratios.extend(rr_r)
        cond_mean_ratios.append(float(np.mean(k5_r)))
        cond_mean_ratios.append(float(np.mean(rr_r)))

    inlines["interp_dyn_cost_min"] = f"{min(costs_dyn):,.0f}"
    inlines["interp_dyn_cost_max"] = f"{max(costs_dyn):,.0f}"
    inlines["interp_rebuild_cost_min"] = f"{min(costs_rebuild):,.0f}"
    inlines["interp_rebuild_cost_max"] = f"{max(costs_rebuild):,.0f}"

    inlines["interp_cost_ratio_min"] = f"{min(cond_mean_ratios):.4f}"
    inlines["interp_cost_ratio_max"] = f"{max(cond_mean_ratios):.4f}"
    inlines["interp_cost_ratio_seed_min"] = f"{min(all_k5_ratios + all_rr_ratios):.4f}"
    inlines["interp_cost_ratio_seed_max"] = f"{max(all_k5_ratios + all_rr_ratios):.4f}"

    inlines["interp_cost_ratio_per_reb_mean_min"] = f"{min(per_reb_means):.4f}"
    inlines["interp_cost_ratio_per_reb_mean_max"] = f"{max(per_reb_means):.4f}"
    inlines["interp_cost_ratio_per_reb_seed_min"] = f"{min(per_reb_seeds):.4f}"
    inlines["interp_cost_ratio_per_reb_seed_max"] = f"{max(per_reb_seeds):.4f}"

    inlines["interp_cost_ratio_rr_reb_mean_min"] = f"{min(rr_reb_means):.4f}"
    inlines["interp_cost_ratio_rr_reb_mean_max"] = f"{max(rr_reb_means):.4f}"
    inlines["interp_cost_ratio_rr_reb_seed_min"] = f"{min(rr_reb_seeds):.4f}"
    inlines["interp_cost_ratio_rr_reb_seed_max"] = f"{max(rr_reb_seeds):.4f}"

    inlines["interp_cost_ratio_rr_per_mean_min"] = f"{min(rr_per_means):.4f}"
    inlines["interp_cost_ratio_rr_per_mean_max"] = f"{max(rr_per_means):.4f}"
    inlines["interp_cost_ratio_rr_per_seed_min"] = f"{min(rr_per_seeds):.4f}"
    inlines["interp_cost_ratio_rr_per_seed_max"] = f"{max(rr_per_seeds):.4f}"

    # Item 5: Per-seed ratio range of dynamic repair cost to one rebuild
    # across both repair arms (k5 and rr) and all conditions
    min_seed_repair_to_reb = min(all_k5_ratios + all_rr_ratios)
    max_seed_repair_to_reb = max(all_k5_ratios + all_rr_ratios)
    inlines["interp_repair_to_rebuild_cost_ratio_seed_range"] = (
        f"{min_seed_repair_to_reb:.4f} to {max_seed_repair_to_reb:.4f}"
    )

    # Trajectory-mean Rec@16 ranges for rebuild and round-robin across conditions
    reb_recs_se = [
        _get_stat_mean(self_ex["conditions"][c]["arm_summary"]["rebuild"]["mean_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]
    rr_recs_se = [
        _get_stat_mean(self_ex["conditions"][c]["arm_summary"]["round_robin"]["mean_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]
    reb_recs_comm = [
        _get_stat_mean(comm["conditions"][c]["arm_summary"]["rebuild"]["mean_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]
    rr_recs_comm = [
        _get_stat_mean(comm["conditions"][c]["arm_summary"]["round_robin"]["mean_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]

    inlines["interp_rebuild_traj_rec16_se_range"] = (
        f"{min(reb_recs_se):.4f} to {max(reb_recs_se):.4f}"
    )
    inlines["interp_round_robin_traj_rec16_se_range"] = (
        f"{min(rr_recs_se):.4f} to {max(rr_recs_se):.4f}"
    )
    inlines["interp_rebuild_traj_rec16_comm_range"] = (
        f"{min(reb_recs_comm):.4f} to {max(reb_recs_comm):.4f}"
    )
    inlines["interp_round_robin_traj_rec16_comm_range"] = (
        f"{min(rr_recs_comm):.4f} to {max(rr_recs_comm):.4f}"
    )
    inlines["interp_rebuild_traj_rec16_range"] = inlines["interp_rebuild_traj_rec16_se_range"]
    inlines["interp_round_robin_traj_rec16_range"] = inlines[
        "interp_round_robin_traj_rec16_se_range"
    ]

    # Final-step Rec@16 ranges for rebuild and round-robin across conditions
    reb_fin_recs_se = [
        _get_stat_mean(self_ex["conditions"][c]["arm_summary"]["rebuild"]["final_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]
    rr_fin_recs_se = [
        _get_stat_mean(self_ex["conditions"][c]["arm_summary"]["round_robin"]["final_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]
    reb_fin_recs_comm = [
        _get_stat_mean(comm["conditions"][c]["arm_summary"]["rebuild"]["final_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]
    rr_fin_recs_comm = [
        _get_stat_mean(comm["conditions"][c]["arm_summary"]["round_robin"]["final_drift_rec16"])
        for c in CONDITION_DISPLAY_NAMES
    ]

    inlines["interp_rebuild_final_rec16_se_range"] = (
        f"{min(reb_fin_recs_se):.4f} to {max(reb_fin_recs_se):.4f}"
    )
    inlines["interp_round_robin_final_rec16_se_range"] = (
        f"{min(rr_fin_recs_se):.4f} to {max(rr_fin_recs_se):.4f}"
    )
    inlines["interp_rebuild_final_rec16_comm_range"] = (
        f"{min(reb_fin_recs_comm):.4f} to {max(reb_fin_recs_comm):.4f}"
    )
    inlines["interp_round_robin_final_rec16_comm_range"] = (
        f"{min(rr_fin_recs_comm):.4f} to {max(rr_fin_recs_comm):.4f}"
    )

    # f=0.25 (no jitter) specific final Rec@16 and cost ratio values
    reb_f025_se = _get_stat_mean(
        self_ex["conditions"]["f_0.25"]["arm_summary"]["rebuild"]["final_drift_rec16"]
    )
    rr_f025_se = _get_stat_mean(
        self_ex["conditions"]["f_0.25"]["arm_summary"]["round_robin"]["final_drift_rec16"]
    )
    reb_f025_comm = _get_stat_mean(
        comm["conditions"]["f_0.25"]["arm_summary"]["rebuild"]["final_drift_rec16"]
    )
    rr_f025_comm = _get_stat_mean(
        comm["conditions"]["f_0.25"]["arm_summary"]["round_robin"]["final_drift_rec16"]
    )

    inlines["interp_f025_reb_final_rec16_se"] = f"{reb_f025_se:.4f}"
    inlines["interp_f025_rr_final_rec16_se"] = f"{rr_f025_se:.4f}"
    inlines["interp_f025_reb_final_rec16_comm"] = f"{reb_f025_comm:.4f}"
    inlines["interp_f025_rr_final_rec16_comm"] = f"{rr_f025_comm:.4f}"

    rr_f025_reb_cost_ratio = float(
        np.mean(
            [
                rr / r
                for rr, r in zip(
                    comm["conditions"]["f_0.25"]["arm_summary"]["round_robin"]["total_cost"][
                        "per_seed"
                    ],
                    comm["conditions"]["f_0.25"]["arm_summary"]["rebuild"]["total_cost"][
                        "per_seed"
                    ],
                    strict=False,
                )
            ]
        )
    )
    inlines["interp_f025_rr_reb_cost_ratio"] = f"{rr_f025_reb_cost_ratio:.4f}"
    inlines["interp_f025_rr_cost_reduction_pct"] = f"{(1.0 - rr_f025_reb_cost_ratio) * 100:.2f}%"
    inlines["interp_f025_rr_reb_gap_se_pp"] = f"{(reb_f025_se - rr_f025_se) * 100:.2f}"
    inlines["interp_f025_rr_reb_gap_comm_pp"] = f"{(reb_f025_comm - rr_f025_comm) * 100:.2f}"

    # Quality gap ranges (rebuild minus round_robin final Rec@16) across all conditions
    gaps_se = [(reb - rr) * 100 for reb, rr in zip(reb_fin_recs_se, rr_fin_recs_se, strict=False)]
    gaps_comm = [
        (reb - rr) * 100 for reb, rr in zip(reb_fin_recs_comm, rr_fin_recs_comm, strict=False)
    ]
    inlines["interp_rr_reb_final_gap_se_range_pp"] = f"{min(gaps_se):.2f} to {max(gaps_se):.2f}"
    inlines["interp_rr_reb_final_gap_comm_range_pp"] = (
        f"{min(gaps_comm):.2f} to {max(gaps_comm):.2f}"
    )

    # f >= 0.50 specific deficit ranges and cost ratios
    f_ge_050_conds = ["f_0.50", "f_0.50_jitter", "f_1.00", "f_1.00_jitter"]
    reb_fin_se_ge050 = [
        _get_stat_mean(self_ex["conditions"][c]["arm_summary"]["rebuild"]["final_drift_rec16"])
        for c in f_ge_050_conds
    ]
    rr_fin_se_ge050 = [
        _get_stat_mean(self_ex["conditions"][c]["arm_summary"]["round_robin"]["final_drift_rec16"])
        for c in f_ge_050_conds
    ]
    gaps_se_ge050 = [
        (reb - rr) * 100 for reb, rr in zip(reb_fin_se_ge050, rr_fin_se_ge050, strict=False)
    ]

    reb_fin_comm_ge050 = [
        _get_stat_mean(comm["conditions"][c]["arm_summary"]["rebuild"]["final_drift_rec16"])
        for c in f_ge_050_conds
    ]
    rr_fin_comm_ge050 = [
        _get_stat_mean(comm["conditions"][c]["arm_summary"]["round_robin"]["final_drift_rec16"])
        for c in f_ge_050_conds
    ]
    gaps_comm_ge050 = [
        (reb - rr) * 100 for reb, rr in zip(reb_fin_comm_ge050, rr_fin_comm_ge050, strict=False)
    ]

    inlines["interp_f_ge_050_rr_reb_final_gap_se_range_pp"] = (
        f"{min(gaps_se_ge050):.2f} to {max(gaps_se_ge050):.2f}"
    )
    inlines["interp_f_ge_050_rr_reb_final_gap_comm_range_pp"] = (
        f"{min(gaps_comm_ge050):.2f} to {max(gaps_comm_ge050):.2f}"
    )

    rr_reb_means_ge050 = []
    rr_reb_seeds_ge050 = []
    for c in f_ge_050_conds:
        rr_costs = comm["conditions"][c]["arm_summary"]["round_robin"]["total_cost"]["per_seed"]
        reb_costs = comm["conditions"][c]["arm_summary"]["rebuild"]["total_cost"]["per_seed"]
        ratios = [rr / reb for rr, reb in zip(rr_costs, reb_costs, strict=False)]
        rr_reb_means_ge050.append(float(np.mean(ratios)))
        rr_reb_seeds_ge050.extend(ratios)

    inlines["interp_f_ge_050_cost_ratio_rr_reb_mean_min"] = f"{min(rr_reb_means_ge050):.4f}"
    inlines["interp_f_ge_050_cost_ratio_rr_reb_mean_max"] = f"{max(rr_reb_means_ge050):.4f}"
    inlines["interp_f_ge_050_cost_ratio_rr_reb_seed_min"] = f"{min(rr_reb_seeds_ge050):.4f}"
    inlines["interp_f_ge_050_cost_ratio_rr_reb_seed_max"] = f"{max(rr_reb_seeds_ge050):.4f}"

    # k5 cost reduction range at f <= 0.50
    k5_f_le_050_conds = ["f_0.25", "f_0.50", "f_0.50_jitter"]
    k5_le050_ratios = [
        float(
            np.mean(
                [
                    k / r
                    for k, r in zip(
                        comm["conditions"][c]["arm_summary"]["periodic_k5"]["total_cost"][
                            "per_seed"
                        ],
                        comm["conditions"][c]["arm_summary"]["rebuild"]["total_cost"]["per_seed"],
                        strict=False,
                    )
                ]
            )
        )
        for c in k5_f_le_050_conds
    ]
    k5_le050_reds = [(1.0 - rat) * 100 for rat in k5_le050_ratios]
    inlines["interp_f_le_050_k5_cost_reduction_pct_range"] = (
        f"{min(k5_le050_reds):.2f}% to {max(k5_le050_reds):.2f}%"
    )

    inlines["interp_dynamic_total_reinsertions"] = "1,200"
    inlines["interp_drifting_node_count"] = "600"
    inlines["interp_total_node_count"] = "2,000"
    inlines["interp_rr_stagger_fraction_pct"] = "20%"

    # Final step paired CIs vs rebuild for periodic_k5 and round_robin
    slug_map = {
        "f_0.25": "f025",
        "f_0.50": "f050",
        "f_0.50_jitter": "f050j",
        "f_1.00": "f100",
        "f_1.00_jitter": "f100j",
    }
    for cond_key, cond_slug in slug_map.items():
        c_arms = comm["conditions"][cond_key]["arm_summary"]
        s_arms = self_ex["conditions"][cond_key]["arm_summary"]

        k5_c = c_arms["periodic_k5"]["final_drift_rec16"]["per_seed"]
        rr_c = c_arms["round_robin"]["final_drift_rec16"]["per_seed"]
        reb_c = c_arms["rebuild"]["final_drift_rec16"]["per_seed"]

        k5_s = s_arms["periodic_k5"]["final_drift_rec16"]["per_seed"]
        rr_s = s_arms["round_robin"]["final_drift_rec16"]["per_seed"]
        reb_s = s_arms["rebuild"]["final_drift_rec16"]["per_seed"]

        def _fmt_paired_ci(v1: list[float], v2: list[float]) -> str:
            d = [a - b for a, b in zip(v1, v2, strict=False)]
            m = float(np.mean(d))
            s = float(np.std(d, ddof=1))
            h = T_STUDENT_95_N5 * s / np.sqrt(len(d))
            return f"[{m - h:+.4f}, {m + h:+.4f}]"

        inlines[f"interp_{cond_slug}_k5_reb_final_ci_comm"] = _fmt_paired_ci(k5_c, reb_c)
        inlines[f"interp_{cond_slug}_k5_reb_final_ci_se"] = _fmt_paired_ci(k5_s, reb_s)
        inlines[f"interp_{cond_slug}_rr_reb_final_ci_comm"] = _fmt_paired_ci(rr_c, reb_c)
        inlines[f"interp_{cond_slug}_rr_reb_final_ci_se"] = _fmt_paired_ci(rr_s, reb_s)

    # Specific cost ratios
    k5_f025_c = comm["conditions"]["f_0.25"]["arm_summary"]["periodic_k5"]["total_cost"]["mean"]
    reb_f025_c = comm["conditions"]["f_0.25"]["arm_summary"]["rebuild"]["total_cost"]["mean"]
    inlines["interp_f025_k5_reb_cost_ratio"] = f"{k5_f025_c / reb_f025_c:.4f}"

    k5_f100j_c = comm["conditions"]["f_1.00_jitter"]["arm_summary"]["periodic_k5"]["total_cost"][
        "mean"
    ]
    reb_f100j_c = comm["conditions"]["f_1.00_jitter"]["arm_summary"]["rebuild"]["total_cost"][
        "mean"
    ]
    inlines["interp_f100j_k5_reb_cost_ratio"] = f"{k5_f100j_c / reb_f100j_c:.4f}"

    # Per-reinsertion cost and rebuild cost per node
    k5_reins_costs: list[float] = []
    rr_reins_costs: list[float] = []
    reb_node_costs: list[float] = []
    all_reins_to_reb_ratios: list[float] = []

    for cond_key in CONDITION_DISPLAY_NAMES:
        c_arms = comm["conditions"][cond_key]["arm_summary"]
        reb_m = float(c_arms["rebuild"]["total_cost"]["mean"])
        reb_node = reb_m / 2000.0
        reb_node_costs.append(reb_node)

        k5_m = float(c_arms["periodic_k5"]["total_cost"]["mean"])
        k5_r = float(c_arms["periodic_k5"]["total_reinsertions"]["mean"])
        k5_per = k5_m / k5_r
        k5_reins_costs.append(k5_per)
        all_reins_to_reb_ratios.append(k5_per / reb_node)

        rr_m = float(c_arms["round_robin"]["total_cost"]["mean"])
        rr_r = float(c_arms["round_robin"]["total_reinsertions"]["mean"])
        rr_per = rr_m / rr_r
        rr_reins_costs.append(rr_per)
        all_reins_to_reb_ratios.append(rr_per / reb_node)

    inlines["interp_k5_cost_per_reins_min"] = f"{min(k5_reins_costs):.2f}"
    inlines["interp_k5_cost_per_reins_max"] = f"{max(k5_reins_costs):.2f}"
    inlines["interp_rr_cost_per_reins_min"] = f"{min(rr_reins_costs):.2f}"
    inlines["interp_rr_cost_per_reins_max"] = f"{max(rr_reins_costs):.2f}"
    inlines["interp_reb_cost_per_node_min"] = f"{min(reb_node_costs):.2f}"
    inlines["interp_reb_cost_per_node_max"] = f"{max(reb_node_costs):.2f}"
    inlines["interp_cost_per_reins_to_reb_node_ratio_range"] = (
        f"{min(all_reins_to_reb_ratios):.4f} to {max(all_reins_to_reb_ratios):.4f}"
    )

    coloc_cat_path = repo_dir / "data" / "interpolation_colocation_categories.json"
    if not coloc_cat_path.exists():
        raise FileNotFoundError(f"Missing required colocation categories file: {coloc_cat_path}")
    with open(coloc_cat_path, encoding="utf-8") as f:
        d_cat = json.load(f)

    if "metadata" not in d_cat or "n_queries" not in d_cat["metadata"]:
        raise KeyError(f"'metadata.n_queries' missing in {coloc_cat_path}")
    n_queries = d_cat["metadata"]["n_queries"]
    if n_queries <= 0:
        raise ValueError(f"Invalid n_queries ({n_queries}) in {coloc_cat_path}")

    if "per_seed" not in d_cat or not d_cat["per_seed"]:
        raise KeyError(f"'per_seed' key missing or empty in {coloc_cat_path}")

    required_keys = ["total_colocated", "sole_static_target", "shared_target", "other"]
    totals: list[int] = []
    soles: list[int] = []
    shareds: list[int] = []
    others: list[int] = []

    for s_id, s_data in d_cat["per_seed"].items():
        for req_k in required_keys:
            if req_k not in s_data:
                raise KeyError(f"Key '{req_k}' missing in per_seed[{s_id}] of {coloc_cat_path}")
        totals.append(s_data["total_colocated"])
        soles.append(s_data["sole_static_target"])
        shareds.append(s_data["shared_target"])
        others.append(s_data["other"])

    min_tot, max_tot = min(totals), max(totals)
    min_sole, max_sole = min(soles), max(soles)
    min_sh, max_sh = min(shareds), max(shareds)
    min_oth, max_oth = min(others), max(others)

    pct_tot_min = round(100.0 * min_tot / n_queries)
    pct_tot_max = round(100.0 * max_tot / n_queries)
    pct_sole_min = round(100.0 * min_sole / n_queries)
    pct_sole_max = round(100.0 * max_sole / n_queries)
    pct_sh_min = round(100.0 * min_sh / n_queries)
    pct_sh_max = round(100.0 * max_sh / n_queries)
    pct_oth_min = round(100.0 * min_oth / n_queries)
    pct_oth_max = round(100.0 * max_oth / n_queries)

    inlines["interp_coloc_total_pct_range"] = (
        f"{pct_tot_min}%" if pct_tot_min == pct_tot_max else f"{pct_tot_min}-{pct_tot_max}%"
    )
    inlines["interp_coloc_sole_static_pct_range"] = (
        f"{pct_sole_min}%" if pct_sole_min == pct_sole_max else f"{pct_sole_min}-{pct_sole_max}%"
    )
    inlines["interp_coloc_shared_target_pct_range"] = (
        f"{pct_sh_min}%" if pct_sh_min == pct_sh_max else f"{pct_sh_min}-{pct_sh_max}%"
    )
    inlines["interp_coloc_other_pct_range"] = (
        f"{pct_oth_min}%" if pct_oth_min == pct_oth_max else f"{pct_oth_min}-{pct_oth_max}%"
    )

    # 2. RR vs periodic_k5 trajectory mean paired differences
    # f=0.25
    c_f025 = comm["conditions"]["f_0.25"]["trajectory_mean_paired_differences"][
        "round_robin_vs_periodic_k5"
    ]
    s_f025 = self_ex["conditions"]["f_0.25"]["trajectory_mean_paired_differences"]
    inlines["interp_f025_rr_k5_traj_diff_rec16_comm"] = f"{c_f025['mean_drift_rec16']['mean']:+.4f}"
    ci = c_f025["mean_drift_rec16"]["t_interval_95"]
    inlines["interp_f025_rr_k5_traj_ci16_comm"] = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
    inlines["interp_f025_rr_k5_traj_diff_rec32_comm"] = f"{c_f025['mean_drift_rec32']['mean']:+.4f}"

    s_p16 = s_f025["rec16"]["round_robin_vs_periodic_k5"]
    s_p32 = s_f025["rec32"]["round_robin_vs_periodic_k5"]
    inlines["interp_f025_rr_k5_traj_diff_rec16_se"] = f"{s_p16['mean']:+.4f}"
    inlines["interp_f025_rr_k5_traj_ci16_se"] = (
        f"[{s_p16['t_interval_95'][0]:+.4f}, {s_p16['t_interval_95'][1]:+.4f}]"
    )
    inlines["interp_f025_rr_k5_traj_diff_rec32_se"] = f"{s_p32['mean']:+.4f}"

    # f=0.50 (no jitter)
    s_f050 = self_ex["conditions"]["f_0.50"]["trajectory_mean_paired_differences"]
    inlines["interp_f050_rr_k5_traj_diff_rec16_se_pp"] = (
        f"{s_f050['rec16']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )

    # f=0.50_jitter
    c_f050j = comm["conditions"]["f_0.50_jitter"]["trajectory_mean_paired_differences"][
        "round_robin_vs_periodic_k5"
    ]
    s_f050j = self_ex["conditions"]["f_0.50_jitter"]["trajectory_mean_paired_differences"]
    inlines["interp_f050j_rr_k5_traj_diff_rec16_comm"] = (
        f"{c_f050j['mean_drift_rec16']['mean'] * 100:.2f}"
    )
    inlines["interp_f050j_rr_k5_traj_diff_rec32_comm"] = (
        f"{c_f050j['mean_drift_rec32']['mean'] * 100:.2f}"
    )
    inlines["interp_f050j_rr_k5_traj_diff_rec16_se"] = (
        f"{s_f050j['rec16']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )
    inlines["interp_f050j_rr_k5_traj_diff_rec32_se"] = (
        f"{s_f050j['rec32']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )
    inlines["interp_f050j_rr_k5_traj_diff_rec16_se_pp"] = (
        f"{s_f050j['rec16']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )
    inlines["interp_f050j_rr_k5_traj_diff_rec32_se_pp"] = (
        f"{s_f050j['rec32']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )

    # f=0.50_jitter final step diff
    c_f050j_fin = comm["conditions"]["f_0.50_jitter"][
        "final_step_timing_dependent_paired_differences"
    ]["round_robin_vs_periodic_k5"]
    s_f050j_fin = self_ex["conditions"]["f_0.50_jitter"]["final_step_paired_differences"]
    inlines["interp_f050j_rr_k5_final_diff_rec16_comm"] = (
        f"{c_f050j_fin['final_drift_rec16']['mean']:+.4f}"
    )
    inlines["interp_f050j_rr_k5_final_diff_rec16_se"] = (
        f"{s_f050j_fin['rec16']['round_robin_vs_periodic_k5']['mean']:+.4f}"
    )

    # f=1.00_jitter
    c_f100j = comm["conditions"]["f_1.00_jitter"]["trajectory_mean_paired_differences"][
        "round_robin_vs_periodic_k5"
    ]
    s_f100j = self_ex["conditions"]["f_1.00_jitter"]["trajectory_mean_paired_differences"]
    inlines["interp_f100j_rr_k5_traj_diff_rec16_comm"] = (
        f"{c_f100j['mean_drift_rec16']['mean'] * 100:.2f}"
    )
    inlines["interp_f100j_rr_k5_traj_diff_rec32_comm"] = (
        f"{c_f100j['mean_drift_rec32']['mean'] * 100:.2f}"
    )
    inlines["interp_f100j_rr_k5_traj_diff_rec16_se"] = (
        f"{s_f100j['rec16']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )
    inlines["interp_f100j_rr_k5_traj_diff_rec32_se"] = (
        f"{s_f100j['rec32']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )
    inlines["interp_f100j_rr_k5_traj_diff_rec16_se_pp"] = (
        f"{s_f100j['rec16']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )
    inlines["interp_f100j_rr_k5_traj_diff_rec32_se_pp"] = (
        f"{s_f100j['rec32']['round_robin_vs_periodic_k5']['mean'] * 100:.2f}"
    )

    # 3. no_repair final and trajectory-mean recalls (Claim 12)
    for cond_name, prefix in [
        ("f_0.25", "interp_f025_nr"),
        ("f_0.50", "interp_f050_nr"),
        ("f_0.50_jitter", "interp_f050j_nr"),
        ("f_1.00", "interp_f100_nr"),
        ("f_1.00_jitter", "interp_f100j_nr"),
    ]:
        nr = comm["conditions"][cond_name]["arm_summary"]["no_repair"]
        inlines[f"{prefix}_traj_rec16"] = f"{_get_stat_mean(nr['mean_drift_rec16']):.4f}"
        inlines[f"{prefix}_final_rec16"] = f"{_get_stat_mean(nr['final_drift_rec16']):.4f}"

    # 4. Jitter diagnostics and bounds
    jitter_data = self_ex["diagnostics"]["jitter_norm_ratios"]
    means = [jitter_data[str(s)]["mean_ratio"] for s in [110, 111, 112, 113, 114]]
    pooled_mean = sum(means) / len(means)
    min_pooled = min(jitter_data[str(s)]["min_ratio"] for s in [110, 111, 112, 113, 114])
    max_pooled = max(jitter_data[str(s)]["max_ratio"] for s in [110, 111, 112, 113, 114])
    inlines["interp_jitter_pooled_ratio"] = f"{pooled_mean:.6f}"
    inlines["interp_jitter_min_ratio"] = f"{min_pooled:.6f}"
    inlines["interp_jitter_max_ratio"] = f"{max_pooled:.6f}"
    inlines["interp_jitter_scaled_factor"] = f"{pooled_mean / 0.05:.6f}"

    coloc_data = self_ex["diagnostics"]["colocation_counts"]
    f100_noj_ratios = [
        coloc_data["f_1.00"][str(s)]["ratio_to_mean_1nn"] for s in [110, 111, 112, 113, 114]
    ]
    f100_j_ratios = [
        coloc_data["f_1.00_jitter"][str(s)]["ratio_to_mean_1nn"] for s in [110, 111, 112, 113, 114]
    ]
    f100_noj_dists = [
        coloc_data["f_1.00"][str(s)]["smallest_nonzero_dist"] for s in [110, 111, 112, 113, 114]
    ]
    f100_j_dists = [
        coloc_data["f_1.00_jitter"][str(s)]["smallest_nonzero_dist"]
        for s in [110, 111, 112, 113, 114]
    ]
    inlines["interp_f100_no_jitter_nonzero_ratio_min"] = f"{min(f100_noj_ratios):.2e}"
    inlines["interp_f100_no_jitter_nonzero_ratio_max"] = f"{max(f100_noj_ratios):.2e}"
    inlines["interp_f100_jitter_ratio_min"] = f"{min(f100_j_ratios):.3f}"
    inlines["interp_f100_jitter_ratio_max"] = f"{max(f100_j_ratios):.3f}"
    inlines["interp_f100_jitter_dist_min"] = f"{min(f100_j_dists):.2f}"
    inlines["interp_f100_no_jitter_nonzero_dist_min"] = f"{min(f100_noj_dists):.2e}"

    # Descriptive token names for Item 4
    inlines["interp_f100j_min_dist_abs"] = inlines["interp_f100_jitter_dist_min"]
    inlines["interp_f100j_min_dist_frac"] = inlines["interp_f100_jitter_ratio_min"]
    inlines["interp_f100_nojitter_min_dist_abs"] = inlines["interp_f100_no_jitter_nonzero_dist_min"]
    inlines["interp_f100_nojitter_min_dist_frac"] = inlines[
        "interp_f100_no_jitter_nonzero_ratio_min"
    ]

    # Backward compatibility aliases
    inlines["inline_jitter_min_dist_abs"] = inlines["interp_f100j_min_dist_abs"]
    inlines["inline_jitter_min_dist_frac"] = inlines["interp_f100j_min_dist_frac"]
    inlines["inline_nojitter_min_dist_abs"] = inlines["interp_f100_nojitter_min_dist_abs"]
    inlines["inline_nojitter_min_dist_frac"] = inlines["interp_f100_no_jitter_nonzero_ratio_min"]

    # 5. External and synthetic baselines
    if ext_json_path.exists():
        with open(ext_json_path, encoding="utf-8") as f:
            d_ext = json.load(f)

        m_2k = d_ext["real_sift_evaluation"]["decision_rule"]["recall_margin"]
        inlines["sift_iso_2k_rr_minus_per_pp"] = f"{m_2k * 100:.2f}"
        inlines["sift_iso_range_max_pp"] = f"{m_2k * 100:.1f}"

        m_clust = d_ext["sift_like_synthetic_fallback"]["decision_rule"]["recall_margin"]
        inlines["sift_clustered_rr_minus_per_pp"] = f"{m_clust * 100:.2f}"

        conds_200 = d_ext["synthetic_drift_ef200"]["conditions"]
        c500_200 = next(c for c in conds_200 if c["n"] == 500)
        c2000_200 = next(c for c in conds_200 if c["n"] == 2000)
        inlines["synth_ef200_n500_margin_pp"] = (
            f"+{c500_200['decision_rule']['recall_margin'] * 100:.2f}"
        )
        inlines["synth_ef200_n2000_margin_pp"] = (
            f"+{c2000_200['decision_rule']['recall_margin'] * 100:.2f}"
        )

        conds_16 = d_ext["synthetic_drift_ef16"]["conditions"]
        c500_16 = next(c for c in conds_16 if c["n"] == 500)
        c2000_16 = next(c for c in conds_16 if c["n"] == 2000)
        inlines["synth_ef16_n500_margin_pp"] = (
            f"+{c500_16['decision_rule']['recall_margin'] * 100:.2f}"
        )
        inlines["synth_ef16_n2000_margin_pp"] = (
            f"+{c2000_16['decision_rule']['recall_margin'] * 100:.2f}"
        )

        # Descriptive names specifying dataset, motion model, ef, and n
        inlines["sift_real_iso_ef200_n2000_margin_pp"] = inlines["sift_iso_2k_rr_minus_per_pp"]
        inlines["sift_real_iso_ef200_n2000_margin_raw"] = f"{m_2k:.4f}"
        inlines["synth_sift_clustered_iso_ef16_n500_margin_pp"] = inlines[
            "sift_clustered_rr_minus_per_pp"
        ]
        inlines["synth_sift_clustered_iso_ef16_n500_margin_raw"] = f"{m_clust:.4f}"
        inlines["synth_gaussian_iso_ef200_n500_margin_pp"] = inlines["synth_ef200_n500_margin_pp"]
        inlines["synth_gaussian_iso_ef200_n2000_margin_pp"] = inlines["synth_ef200_n2000_margin_pp"]
        inlines["synth_gaussian_iso_ef16_n2000_margin_pp"] = inlines["synth_ef16_n2000_margin_pp"]
        inlines["synth_gaussian_iso_ef16_n500_margin_pp"] = inlines["synth_ef16_n500_margin_pp"]
        inlines["synth_gaussian_iso_ef200_n500_margin_pct"] = (
            f"{c500_200['decision_rule']['recall_margin'] * 100:.2f}%"
        )
        inlines["synth_gaussian_iso_ef200_n2000_margin_pct"] = (
            f"{c2000_200['decision_rule']['recall_margin'] * 100:.2f}%"
        )
        inlines["synth_gaussian_iso_ef16_n2000_margin_pct"] = (
            f"{c2000_16['decision_rule']['recall_margin'] * 100:.2f}%"
        )
        inlines["synth_gaussian_iso_ef16_n500_margin_pct"] = (
            f"{c500_16['decision_rule']['recall_margin'] * 100:.2f}%"
        )

        # Backward compatibility aliases
        inlines["inline_sift_n2000_rr_vs_periodic_mean"] = inlines[
            "sift_real_iso_ef200_n2000_margin_pp"
        ]
        inlines["inline_sift_n2000_rr_vs_periodic_raw"] = inlines[
            "sift_real_iso_ef200_n2000_margin_raw"
        ]
        inlines["inline_sift_n5000_3x_rr_vs_periodic_mean"] = inlines[
            "synth_sift_clustered_iso_ef16_n500_margin_pp"
        ]
        inlines["inline_sift_n5000_3x_rr_vs_periodic_raw"] = inlines[
            "synth_sift_clustered_iso_ef16_n500_margin_raw"
        ]
        inlines["inline_interp_f050_rr_vs_periodic_diff"] = inlines[
            "synth_gaussian_iso_ef200_n500_margin_pp"
        ]
        inlines["inline_interp_f025_rr_vs_periodic_diff"] = inlines[
            "synth_gaussian_iso_ef200_n2000_margin_pp"
        ]
        inlines["inline_interp_f100_rr_vs_periodic_diff"] = inlines[
            "synth_gaussian_iso_ef16_n2000_margin_pp"
        ]
        inlines["inline_interp_f050j_rr_vs_periodic_diff"] = inlines[
            "synth_gaussian_iso_ef16_n500_margin_pp"
        ]
        inlines["inline_interp_f050_rr_vs_periodic_diff_pct"] = inlines[
            "synth_gaussian_iso_ef200_n500_margin_pct"
        ]
        inlines["inline_interp_f025_rr_vs_periodic_diff_pct"] = inlines[
            "synth_gaussian_iso_ef200_n2000_margin_pct"
        ]
        inlines["inline_interp_f100_rr_vs_periodic_diff_pct"] = inlines[
            "synth_gaussian_iso_ef16_n2000_margin_pct"
        ]
        inlines["inline_interp_f050j_rr_vs_periodic_diff_pct"] = inlines[
            "synth_gaussian_iso_ef16_n500_margin_pct"
        ]

    if sift_5k_json_path.exists():
        with open(sift_5k_json_path, encoding="utf-8") as f:
            d_5k = json.load(f)
        m_5k = d_5k["runs"]["decision_rule"]["recall_margin"]
        inlines["sift_iso_5k_rr_minus_per_pp"] = f"{m_5k * 100:.2f}"
        inlines["sift_iso_range_min_pp"] = f"{m_5k * 100:.1f}"

        # Descriptive names for SIFT 5k
        inlines["sift_real_iso_ef200_n5000_margin_pp"] = inlines["sift_iso_5k_rr_minus_per_pp"]
        inlines["sift_real_iso_ef200_n5000_margin_raw"] = f"{m_5k:.4f}"
        inlines["inline_sift_n5000_rr_vs_periodic_mean"] = inlines[
            "sift_real_iso_ef200_n5000_margin_pp"
        ]
        inlines["inline_sift_n5000_rr_vs_periodic_raw"] = inlines[
            "sift_real_iso_ef200_n5000_margin_raw"
        ]

        if "sift_iso_range_max_pp" in inlines:
            inlines["sift_real_iso_ef200_range_pct"] = (
                f"{inlines['sift_iso_range_min_pp']}% to {inlines['sift_iso_range_max_pp']}%"
            )
            inlines["inline_sift_isotropic_rr_vs_periodic_range"] = inlines[
                "sift_real_iso_ef200_range_pct"
            ]

    return inlines


def update_markdown_file(
    file_path: Path,
    tables: dict[str, str],
    inlines: dict[str, str] | None = None,
    check_only: bool = False,
) -> bool:
    """Update or check markdown file containing generated table blocks and inline values."""
    if not file_path.exists():
        raise FileNotFoundError(f"Chapter file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    original_content = content
    has_diff = False

    # 1. Check/update table blocks
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

    # 2. Check/update inline values
    if inlines is not None:
        inline_pattern = re.compile(r"<!-- GEN-INLINE:\s*(\w+)\s*-->(.*?)<!-- /GEN-INLINE -->")

        if check_only:
            for match in inline_pattern.finditer(content):
                key = match.group(1)
                curr_val = match.group(2)
                if key not in inlines:
                    print(f"[CHECK FAILED] Unknown inline key '{key}' in {file_path.name}")
                    has_diff = True
                elif curr_val != inlines[key]:
                    print(
                        f"[CHECK FAILED] Inline '{key}' in {file_path.name} is '{curr_val}', "
                        f"expected '{inlines[key]}'"
                    )
                    has_diff = True
        else:

            def _inline_repl(m: re.Match[str]) -> str:
                k = m.group(1)
                if k in inlines:
                    return f"<!-- GEN-INLINE: {k} -->{inlines[k]}<!-- /GEN-INLINE -->"
                return m.group(0)

            content = inline_pattern.sub(_inline_repl, content)

    if not check_only and content != original_content:
        file_path.write_text(content, encoding="utf-8")
        print(f"[UPDATED] Written generated blocks and inlines to {file_path}")

    return not has_diff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit markdown tables and inlines for interpolation benchmark."
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
        "--ext-json",
        type=Path,
        default=Path("docs/data/external_benchmarks.json"),
        help="Path to external_benchmarks.json",
    )
    parser.add_argument(
        "--sift-5k-json",
        type=Path,
        default=Path("docs/data/sift_n5000_ef200.json"),
        help="Path to sift_n5000_ef200.json",
    )
    parser.add_argument(
        "--chapter-file",
        type=Path,
        default=Path("docs/report/chapter9_external_baselines.md"),
        help="Path to chapter 9 markdown file",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write tables and inlines between markers in chapter file",
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
    ext_path = repo_root / args.ext_json if not args.ext_json.is_absolute() else args.ext_json
    sift_5k_path = (
        repo_root / args.sift_5k_json if not args.sift_5k_json.is_absolute() else args.sift_5k_json
    )
    chap_path = (
        repo_root / args.chapter_file if not args.chapter_file.is_absolute() else args.chapter_file
    )

    tables = generate_all_tables(self_ex_path, comm_path)
    inlines = generate_all_inlines(self_ex_path, comm_path, ext_path, sift_5k_path)

    if args.write:
        update_markdown_file(chap_path, tables, inlines=inlines, check_only=False)
    elif args.check:
        clean = update_markdown_file(chap_path, tables, inlines=inlines, check_only=True)
        if not clean:
            print("[CHECK FAILED] One or more generated table blocks or inline metrics differ.")
            sys.exit(1)
        else:
            print(
                "[CHECK PASSED] All generated table blocks and inline metrics match chapter file."
            )
    else:
        for name, md in tables.items():
            print(f"\n<!-- BEGIN GENERATED: {name} -->\n{md}\n<!-- END GENERATED: {name} -->\n")
        print("\n--- Inline Values ---")
        for k, v in sorted(inlines.items()):
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
