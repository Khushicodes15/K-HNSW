"""Unit tests for benchmarks/run_interpolation_benchmark.py (Member 4 interpolation benchmark).

Tests:
1. with f=0 nodes do not move (velocities are all zero, coordinates unchanged).
2. for f > 0, the final position lies on the segment between the start and the target.
3. with jitter=True, the final position lies on the segment between start and jittered target.
4. parse_args handles smoke mode and custom arguments.
5. run_interpolation_benchmark minimal smoke execution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.run_interpolation_benchmark import (
    generate_interpolation_drift,
    parse_args,
    run_interpolation_benchmark,
)


class TestInterpolationProtocol:
    def test_f_zero_nodes_do_not_move(self) -> None:
        """With f=0, velocities must be identically zero and nodes do not move."""
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((100, 16)).astype(np.float32)

        vels, drift_nodes, targets, target_pts = generate_interpolation_drift(
            X0=X0,
            seed=42,
            f_frac=0.0,
            drift_frac=0.3,
            t_final=0.30,
            jitter=False,
        )

        assert len(drift_nodes) == 30
        assert vels.shape == (100, 16)
        # All velocities must be zero
        np.testing.assert_allclose(vels, np.zeros_like(vels), atol=1e-7)

        # Final positions equal initial positions exactly
        t_final = 0.30
        pos_final = (X0.astype(np.float64) + vels.astype(np.float64) * t_final).astype(np.float32)
        np.testing.assert_allclose(pos_final, X0, atol=1e-7)

    def test_final_position_on_segment_between_start_and_target(self) -> None:
        """For f in (0, 1], final position lies on the line segment between X0 and target."""
        rng = np.random.default_rng(100)
        X0 = rng.standard_normal((100, 32)).astype(np.float32)

        for f_frac in [0.25, 0.50, 0.75, 1.00]:
            t_final = 0.30
            vels, drift_nodes, targets, target_pts = generate_interpolation_drift(
                X0=X0,
                seed=100,
                f_frac=f_frac,
                drift_frac=0.4,
                t_final=t_final,
                jitter=False,
            )

            assert len(drift_nodes) == 40
            pos_final = (X0.astype(np.float64) + vels.astype(np.float64) * t_final).astype(
                np.float32
            )

            for i, u in enumerate(drift_nodes):
                start = X0[u]
                target = target_pts[i]
                final = pos_final[u]

                # Segment property: dist(start, final) + dist(final, target) == dist(start, target)
                d_total = float(np.linalg.norm(target - start))
                d_start_final = float(np.linalg.norm(final - start))
                d_final_target = float(np.linalg.norm(target - final))

                assert d_total > 0.0
                np.testing.assert_allclose(
                    d_start_final + d_final_target, d_total, rtol=1e-5, atol=1e-5
                )

                # Fractional distance matches f_frac exactly
                np.testing.assert_allclose(d_start_final / d_total, f_frac, rtol=1e-5, atol=1e-5)

                # Convex combination formula: final == (1 - f) * start + f * target
                expected_final = (1.0 - f_frac) * start + f_frac * target
                np.testing.assert_allclose(final, expected_final, rtol=1e-5, atol=1e-5)

            # Static nodes do not move
            static_nodes = [j for j in range(100) if j not in set(drift_nodes)]
            np.testing.assert_allclose(pos_final[static_nodes], X0[static_nodes], atol=1e-7)

    def test_jitter_displacement_on_segment(self) -> None:
        """With jitter=True, final position lies on segment between start and jittered target."""
        rng = np.random.default_rng(200)
        X0 = rng.standard_normal((50, 16)).astype(np.float32)

        t_final = 0.20
        f_frac = 0.50
        vels, drift_nodes, targets, target_pts = generate_interpolation_drift(
            X0=X0,
            seed=200,
            f_frac=f_frac,
            drift_frac=0.3,
            t_final=t_final,
            jitter=True,
            jitter_scale=0.05,
        )

        pos_final = (X0.astype(np.float64) + vels.astype(np.float64) * t_final).astype(np.float32)

        for i, u in enumerate(drift_nodes):
            start = X0[u]
            target = target_pts[i]
            final = pos_final[u]

            expected_final = 0.5 * start + 0.5 * target
            np.testing.assert_allclose(final, expected_final, rtol=1e-5, atol=1e-5)


class TestInterpolationExecution:
    def test_parse_args(self) -> None:
        args = parse_args(["--smoke", "--n", "50", "--seeds", "110", "111"])
        assert args.smoke is True
        assert args.n == 50
        assert args.seeds == [110, 111]

    def test_smoke_execution(self) -> None:
        """Run minimal execution to verify end-to-end integration.

        Verifies jitter and final-step statistics.
        """
        sift_path = (
            Path(__file__).resolve().parent.parent.parent
            / "docs"
            / "data"
            / "siftsmall"
            / "siftsmall_base.fvecs"
        )
        res = run_interpolation_benchmark(
            data_path=str(sift_path),
            n=30,
            d=128,
            ef_construction=16,
            seeds=[110],
            conditions=[
                {"name": "f_0.50", "f_frac": 0.50, "jitter": False},
                {"name": "f_1.00_jitter", "f_frac": 1.00, "jitter": True},
            ],
            output_path=None,
            allow_fallback=True,
        )
        assert "metadata" in res
        assert "jitter_formulation" in res["metadata"]
        assert res["metadata"]["jitter_formulation"]["type"] == "coordinate_wise_gaussian"
        assert "0.05 * mean_1nn" in res["metadata"]["jitter_formulation"]["sigma_per_coordinate"]
        assert "motion_equation" in res["metadata"]["jitter_formulation"]

        assert "conditions" in res
        for cond_name in ["f_0.50", "f_1.00_jitter"]:
            assert cond_name in res["conditions"]
            cond = res["conditions"][cond_name]
            assert "arm_summary" in cond
            assert "round_robin" in cond["arm_summary"]
            assert "rebuild" in cond["arm_summary"]
            assert "no_repair" in cond["arm_summary"]
            assert "periodic_k5" in cond["arm_summary"]
            assert "trajectory_mean_paired_differences" in cond
            assert "final_step_timing_dependent_paired_differences" in cond
            assert "paired_differences" in cond

            # Verify paired differences structure
            final_p = cond["final_step_timing_dependent_paired_differences"]
            for pair in [
                "periodic_k5_vs_no_repair",
                "round_robin_vs_no_repair",
                "round_robin_vs_periodic_k5",
            ]:
                assert pair in final_p
                assert "final_drift_rec16" in final_p[pair]
                assert "mean" in final_p[pair]["final_drift_rec16"]
                assert "t_interval_95" in final_p[pair]["final_drift_rec16"]
