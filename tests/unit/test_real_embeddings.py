"""Unit tests for benchmarks/run_real_embeddings.py (Member 4 real embedding benchmarks).

Tests:
1. load_benchmark_dataset loads synthetic clustered and normal fallbacks properly.
2. generate_drift_field_from_data computes decoupled velocities and scale.
3. run_embedding_benchmark executes with all arms on small dataset.
4. parse_args handles smoke mode and custom arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.run_real_embeddings import (
    generate_drift_field_from_data,
    load_benchmark_dataset,
    parse_args,
    run_embedding_benchmark,
)


class TestLoadBenchmarkDataset:
    def test_sift_synthetic(self) -> None:
        X, info = load_benchmark_dataset("sift_synthetic", n=50, d=32, seed=42)
        assert X.shape == (50, 32)
        assert X.dtype == np.float32
        assert float(X.min()) >= 0.0
        assert "sift" in info

    def test_glove_fallback(self) -> None:
        X, info = load_benchmark_dataset("glove", n=40, d=16, seed=7)
        assert X.shape == (40, 16)
        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(norms, np.ones(40), atol=1e-5)

    def test_gaussian(self) -> None:
        X, info = load_benchmark_dataset("gaussian", n=60, d=8, seed=1)
        assert X.shape == (60, 8)


class TestGenerateDriftFieldFromData:
    def test_drift_generation(self) -> None:
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((50, 8)).astype(np.float32)
        vels, drift_nodes, scale = generate_drift_field_from_data(
            X0=X0,
            seed=42,
            drift_frac=0.4,
            displacement_mult=2.0,
            t_final=0.2,
        )
        assert len(drift_nodes) == 20
        assert scale > 0.0
        assert vels.shape == (50, 8)
        # Static nodes have zero velocity
        for i in range(50):
            if i not in set(drift_nodes):
                np.testing.assert_allclose(vels[i], np.zeros(8))
            else:
                assert np.linalg.norm(vels[i]) > 0.0


class TestRunEmbeddingBenchmark:
    def test_runs_minimal_embedding_benchmark(self) -> None:
        res = run_embedding_benchmark(
            dataset_name="sift_synthetic",
            n=40,
            d=16,
            seeds=[42],
            displacement_mult=1.0,
            drift_frac=0.3,
            steps=2,
            t_final=0.1,
            arms=["no_repair", "round_robin"],
            ef_values=[16],
            k=5,
            max_eval_queries=10,
        )
        assert res["dataset"] == "sift_synthetic"
        assert "arm_stats" in res
        assert "no_repair" in res["arm_stats"]
        assert "round_robin" in res["arm_stats"]
        assert len(res["seed_results"]) == 1


class TestParseArgs:
    def test_smoke_flag(self) -> None:
        args = parse_args(["--smoke", "--dataset", "sift_synthetic"])
        assert args.smoke is True
        assert args.dataset == "sift_synthetic"
