"""Unit tests for benchmarks/run_hnswlib_baseline.py (Member 4 external baseline).

Tests:
1. HnswlibBenchmarkEngine add/move/search in native or emulation mode.
2. simulate_hnswlib_policy executes with no_repair and round_robin.
3. run_hnswlib_sweep executes across seeds and produces summary statistics.
4. parse_args handles smoke flag and arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.run_hnswlib_baseline import (
    HnswlibBenchmarkEngine,
    parse_args,
    run_hnswlib_sweep,
    simulate_hnswlib_policy,
)


class TestHnswlibBenchmarkEngine:
    def test_add_and_search(self) -> None:
        engine = HnswlibBenchmarkEngine(dim=4, m=4, ef_construction=10, seed=42)
        X = np.array([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
        ids = np.array([0, 1], dtype=np.int64)
        wall = engine.add(ids, X)
        assert wall >= 0.0

        q = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        labels, dists = engine.search(q, k=1, ef=10)
        assert len(labels) == 1
        assert labels[0] == 0

    def test_move(self) -> None:
        engine = HnswlibBenchmarkEngine(dim=2, m=4, ef_construction=10, seed=42)
        X = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
        engine.add(np.array([0, 1], dtype=np.int64), X)

        # Move node 0 far away to [20, 20]
        engine.move(0, np.array([20.0, 20.0], dtype=np.float32))

        # Query near [0, 0] — now node 1 ([5, 5]) should be closer than node 0 ([20, 20])
        q = np.array([0.0, 0.0], dtype=np.float32)
        labels, _ = engine.search(q, k=1, ef=10)
        assert labels[0] == 1


class TestSimulateHnswlibPolicy:
    def test_no_repair_and_round_robin(self) -> None:
        n, d = 30, 4
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((n, d)).astype(np.float32)
        vels = np.zeros((n, d), dtype=np.float32)
        vels[:10] = 1.0
        drift_nodes = list(range(10))

        res_none = simulate_hnswlib_policy(
            policy_name="no_repair",
            n=n,
            d=d,
            seed=0,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            steps=2,
            dt=0.05,
        )
        assert res_none["policy"] == "no_repair"
        assert res_none["total_reinsertions"] == 0

        res_rr = simulate_hnswlib_policy(
            policy_name="round_robin",
            n=n,
            d=d,
            seed=0,
            X0=X0,
            vels=vels,
            drift_nodes=drift_nodes,
            steps=2,
            dt=0.05,
        )
        assert res_rr["policy"] == "round_robin"
        assert res_rr["total_reinsertions"] > 0


class TestRunHnswlibSweep:
    def test_sweep_minimal(self) -> None:
        res = run_hnswlib_sweep(
            n=30,
            d=4,
            seeds=[42],
            displacement_mult=1.0,
            drift_frac=0.3,
            steps=2,
            t_final=0.1,
        )
        assert res["engine"] == "hnswlib"
        assert "summary" in res
        assert "round_robin" in res["summary"]
        assert "periodic_k5" in res["summary"]


class TestParseArgs:
    def test_smoke(self) -> None:
        args = parse_args(["--smoke", "--n", "50"])
        assert args.smoke is True
