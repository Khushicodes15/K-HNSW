"""Unit tests for benchmarks/run_synthetic_sweep.py (Member 4 synthetic sweep).

Verifies:
1. build_base_index builds an index with correct dimensions and setup cost tracking.
2. simulate_arm executes single arms and returns complete metric dictionaries.
3. run_seed_configuration runs all arms and outputs expected structure.
4. aggregate_sweep_results aggregates metrics and computes the pre-registered decision rule.
5. parse_args handles smoke mode and custom arguments cleanly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from benchmarks.run_synthetic_sweep import (
    aggregate_sweep_results,
    build_base_index,
    parse_args,
    run_seed_configuration,
    simulate_arm,
)


class TestBuildBaseIndex:
    def test_builds_correctly(self) -> None:
        n, d = 40, 8
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((n, d)).astype(np.float32)
        index, setup_cost = build_base_index(n, d, seed=42, X0=X0, m=4, m0=8, ef_construction=10)
        assert index.dim == d
        assert len(index.ids()) == n
        assert setup_cost > 0


class TestSimulateArm:
    @pytest.fixture
    def setup_data(self) -> dict[str, Any]:
        n, d = 30, 4
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((n, d)).astype(np.float32)
        vels = np.zeros((n, d), dtype=np.float32)
        vels[:10] = 1.0  # first 10 nodes drift
        index, _ = build_base_index(n, d, seed=0, X0=X0, m=4, m0=8, ef_construction=10)
        drift_nodes = list(range(10))
        static_nodes = list(range(10, n))
        return {
            "n": n,
            "d": d,
            "X0": X0,
            "vels": vels,
            "base_index": index,
            "drift_nodes": drift_nodes,
            "static_nodes": static_nodes,
        }

    def test_no_repair(self, setup_data: dict[str, Any]) -> None:
        res = simulate_arm(
            arm_name="no_repair",
            base_index=setup_data["base_index"],
            X0=setup_data["X0"],
            vels=setup_data["vels"],
            drift_nodes=setup_data["drift_nodes"],
            static_nodes=setup_data["static_nodes"],
            eval_drift=setup_data["drift_nodes"][:5],
            eval_static=setup_data["static_nodes"][:5],
            steps=2,
            dt=0.05,
            policy_cfg={"policy": "none"},
            ef_values=[16],
            k=5,
        )
        assert res["arm"] == "no_repair"
        assert res["total_reinsertions"] == 0
        assert res["total_cost"] == 0
        assert "mean_drift_rec16" in res
        assert 0.0 <= res["mean_drift_rec16"] <= 1.0

    def test_round_robin(self, setup_data: dict[str, Any]) -> None:
        res = simulate_arm(
            arm_name="round_robin",
            base_index=setup_data["base_index"],
            X0=setup_data["X0"],
            vels=setup_data["vels"],
            drift_nodes=setup_data["drift_nodes"],
            static_nodes=setup_data["static_nodes"],
            eval_drift=setup_data["drift_nodes"][:5],
            eval_static=setup_data["static_nodes"][:5],
            steps=2,
            dt=0.05,
            policy_cfg={"policy": "round_robin", "B": 2, "seed": 42},
            ef_values=[16],
            k=5,
        )
        assert res["arm"] == "round_robin"
        assert res["total_reinsertions"] == 4  # 2 per step * 2 steps
        assert res["total_cost"] > 0
        assert len(res["step_records"]) == 2


class TestRunSeedConfiguration:
    def test_runs_minimal_config(self) -> None:
        res = run_seed_configuration(
            n=40,
            d=4,
            seed=42,
            displacement_mult=1.0,
            drift_frac=0.3,
            steps=2,
            t_final=0.1,
            arms=["no_repair", "round_robin"],
            ef_values=[16],
            k=5,
            max_eval_queries=10,
        )
        assert res["n"] == 40
        assert res["seed"] == 42
        assert "no_repair" in res["arms"]
        assert "round_robin" in res["arms"]
        assert res["b_matched"] >= 1


class TestAggregateAndDecisionRule:
    def test_aggregation(self) -> None:
        runs = [
            run_seed_configuration(
                n=30,
                d=4,
                seed=s,
                displacement_mult=1.0,
                drift_frac=0.3,
                steps=2,
                t_final=0.1,
                arms=["periodic_k5", "round_robin"],
                ef_values=[16],
                k=5,
                max_eval_queries=10,
            )
            for s in [42, 43]
        ]
        summary = aggregate_sweep_results(runs, ef_values=[16])
        assert len(summary["conditions"]) == 1
        cond = summary["conditions"][0]
        assert cond["n"] == 30
        assert "decision_rule" in cond
        dec = cond["decision_rule"]
        assert dec["arm_a"] == "round_robin"
        assert dec["arm_b"] == "periodic_k5"
        assert "arm_a_wins" in dec
        assert isinstance(dec["arm_a_wins"], bool)


class TestParseArgs:
    def test_smoke_flag(self) -> None:
        args = parse_args(["--smoke"])
        assert args.smoke is True

    def test_custom_values(self) -> None:
        args = parse_args(
            ["--n", "500", "1000", "--d", "32", "--seeds", "99", "--displacement", "2.0"]
        )
        assert args.n == [500, 1000]
        assert args.d == [32]
        assert args.seeds == [99]
        assert args.displacement == [2.0]
