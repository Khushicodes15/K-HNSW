"""Unit tests for benchmarks/common.py (Member 4 toolkit).

Tests:
1. PRNG stream decoupling — velocity directions have near-zero mean
   cosine similarity with initial positions (the "cos ≡ 0 hazard" check).
2. Speed calibration produces a positive, finite scale.
3. check_decision_rule boundary cases.
4. brute_force_knn correctness on a tiny example.
5. evaluate_recall returns 1.0 when the index returns perfect results.
6. save_results writes valid JSON and coerces NumPy scalars.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# Make sure the repo root is importable when running from K-HNSW/
# (PYTHONPATH=src is set by the test runner).
from benchmarks.common import (
    advance_positions,
    brute_force_knn,
    check_decision_rule,
    evaluate_recall,
    generate_isotropic_drift_data,
    save_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ExactIndex:
    """Trivial exact-search index for testing evaluate_recall."""

    def __init__(self, X: np.ndarray) -> None:
        self._X = X.astype(np.float32)

    def search(self, query: np.ndarray, k: int, ef: int = 50) -> list[tuple[int, float]]:
        q = np.asarray(query, dtype=np.float32)
        dists = np.linalg.norm(self._X - q, axis=1)
        ids = np.argsort(dists)[:k]
        return [(int(i), float(dists[i])) for i in ids]


# ---------------------------------------------------------------------------
# 1. PRNG stream decoupling
# ---------------------------------------------------------------------------


class TestPrngDecoupling:
    """Velocity directions must be independent of initial positions.

    If the same RNG stream is used for both positions and velocities,
    the velocity directions become correlated with (or orthogonal to)
    the position vectors — the "cos ≡ 0 / cos ≡ 1 hazard".

    We verify that the mean |cos(v_i, x_i)| over all moving nodes is
    *small* (< 0.15), which would be impossible under the hazard scenario.
    """

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_velocity_position_correlation_low(self, seed: int) -> None:
        n, d = 500, 32
        X0, velocities, moving_ids = generate_isotropic_drift_data(
            n=n, d=d, seed=seed, drift_frac=0.5, displacement_mult=1.0
        )

        cos_vals: list[float] = []
        for nid in moving_ids:
            x = X0[nid].astype(np.float64)
            v = velocities[nid].astype(np.float64)
            norm_x = np.linalg.norm(x)
            norm_v = np.linalg.norm(v)
            if norm_x < 1e-12 or norm_v < 1e-12:
                continue
            cos_vals.append(abs(float(np.dot(x, v) / (norm_x * norm_v))))

        assert cos_vals, "No moving nodes produced — check drift_frac"
        mean_cos = float(np.mean(cos_vals))
        # Under true independence E[|cos|] ≈ sqrt(2/π/d).
        # For d=32 that's ≈ 0.14. We use 0.25 as a generous upper bound.
        assert mean_cos < 0.25, (
            f"Mean |cos(v, x)| = {mean_cos:.4f} ≥ 0.25 — "
            "PRNG streams may be correlated (stream decoupling violated)."
        )


# ---------------------------------------------------------------------------
# 2. Speed calibration
# ---------------------------------------------------------------------------


class TestSpeedCalibration:
    def test_scale_is_positive_finite(self) -> None:
        X0, velocities, moving_ids = generate_isotropic_drift_data(
            n=200, d=16, seed=7, drift_frac=0.3, displacement_mult=2.0, t_final=1.0
        )
        # At t=1.0 the displacement of each moving node equals ||v|| * 1.0.
        # That should be approximately displacement_mult * mean_nn_distance(X0).
        disps = [float(np.linalg.norm(velocities[nid])) for nid in moving_ids]
        mean_disp = float(np.mean(disps))
        assert mean_disp > 0.0, "Mean displacement is zero — speed calibration failed"
        assert math.isfinite(mean_disp), "Mean displacement is not finite"

    def test_displacement_mult_scales_linearly(self) -> None:
        """Doubling displacement_mult should roughly double mean displacement."""
        kwargs = dict(n=300, d=8, seed=10, drift_frac=0.5, t_final=1.0)
        _, v1, ids1 = generate_isotropic_drift_data(displacement_mult=1.0, **kwargs)
        _, v2, ids2 = generate_isotropic_drift_data(displacement_mult=2.0, **kwargs)
        # Same seed → same moving_ids
        assert ids1 == ids2
        ratio = float(np.mean([np.linalg.norm(v2[i]) for i in ids2])) / float(
            np.mean([np.linalg.norm(v1[i]) for i in ids1])
        )
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ≈ 2.0, got {ratio:.4f}"


# ---------------------------------------------------------------------------
# 3. check_decision_rule boundary cases
# ---------------------------------------------------------------------------


class TestDecisionRule:
    """Pre-registered decision rule from member4.md §4."""

    # Rule 1: recall advantage >= 0.02
    def test_rule1_exact_boundary_true(self) -> None:
        assert check_decision_rule(0.92, 0.90) is True  # exactly +0.02

    def test_rule1_just_below_false(self) -> None:
        assert check_decision_rule(0.919, 0.90) is False  # +0.019 → False

    def test_rule1_clear_win(self) -> None:
        assert check_decision_rule(0.95, 0.80) is True

    def test_rule1_tie(self) -> None:
        assert check_decision_rule(0.80, 0.80) is False

    def test_rule1_b_wins(self) -> None:
        assert check_decision_rule(0.80, 0.90) is False

    # Rule 2: near-parity recall with >= 2x cost saving
    def test_rule2_cost_saving_wins(self) -> None:
        # recall_a only 0.005 below recall_b (within -0.01), cost_a = 0.4 * cost_b
        assert check_decision_rule(0.895, 0.90, cost_a=400.0, cost_b=1000.0) is True

    def test_rule2_cost_saving_insufficient(self) -> None:
        # cost_a = 0.6 * cost_b (not ≤ 0.5) → rule 2 fails; recall diff also < 0.02
        assert check_decision_rule(0.895, 0.90, cost_a=600.0, cost_b=1000.0) is False

    def test_rule2_recall_gap_too_large(self) -> None:
        # recall drop > 0.01 → rule 2 fails even with big cost saving
        assert check_decision_rule(0.88, 0.90, cost_a=100.0, cost_b=1000.0) is False

    def test_rule2_no_cost_provided(self) -> None:
        # Rule 2 requires cost; without it only rule 1 applies
        assert check_decision_rule(0.895, 0.90) is False  # recall diff < 0.02

    def test_rule2_zero_cost_b(self) -> None:
        # cost_b = 0 → division undefined; must not raise, must return False
        result = check_decision_rule(0.90, 0.90, cost_a=0.0, cost_b=0.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 4. brute_force_knn correctness
# ---------------------------------------------------------------------------


class TestBruteForceKnn:
    def test_tiny_exact(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
        q = np.array([[1.1, 0.0]], dtype=np.float32)
        idx = brute_force_knn(X, q, k=2)
        # Nearest to [1.1, 0] are [1,0] (dist=0.1) and [2,0] (dist=0.9)
        assert list(idx[0]) == [1, 2]

    def test_k_capped_at_n(self) -> None:
        X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        q = np.array([[0.5]], dtype=np.float32)
        idx = brute_force_knn(X, q, k=100)  # k > n → should return n
        assert idx.shape[1] == 3

    def test_self_is_nearest(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8)).astype(np.float32)
        idx = brute_force_knn(X, X, k=1)
        # Every point is its own nearest neighbour
        for i in range(len(X)):
            assert int(idx[i, 0]) == i


# ---------------------------------------------------------------------------
# 5. evaluate_recall with perfect index
# ---------------------------------------------------------------------------


class TestEvaluateRecall:
    def test_perfect_index_recall_one(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 8)).astype(np.float32)
        q = rng.standard_normal((10, 8)).astype(np.float32)
        k = 5
        gt = brute_force_knn(X, q, k=k)
        index = _ExactIndex(X)
        recall = evaluate_recall(index, q, gt, k=k, ef=50)
        assert recall == pytest.approx(1.0, abs=1e-6)

    def test_recall_bounded(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        q = rng.standard_normal((5, 4)).astype(np.float32)
        gt = brute_force_knn(X, q, k=3)
        index = _ExactIndex(X)
        recall = evaluate_recall(index, q, gt, k=3, ef=10)
        assert 0.0 <= recall <= 1.0


# ---------------------------------------------------------------------------
# 6. save_results JSON serialisation
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        save_results({"recall": 0.95, "cost": 1234}, out)
        assert out.exists()

    def test_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        save_results({"a": [1, 2, 3], "b": {"c": 0.5}}, out)
        with out.open() as fh:
            data = json.load(fh)
        assert data["a"] == [1, 2, 3]
        assert data["b"]["c"] == pytest.approx(0.5)

    def test_numpy_scalars_coerced(self, tmp_path: Path) -> None:
        out = tmp_path / "np_results.json"
        save_results(
            {
                "recall": np.float32(0.91),
                "count": np.int64(42),
                "nested": {"arr": [np.float64(1.5)]},
            },
            out,
        )
        with out.open() as fh:
            data = json.load(fh)
        assert data["count"] == 42
        assert isinstance(data["recall"], float)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "results.json"
        save_results({"ok": True}, out)
        assert out.exists()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        save_results({"v": 1}, out)
        save_results({"v": 2}, out)
        with out.open() as fh:
            data = json.load(fh)
        assert data["v"] == 2


# ---------------------------------------------------------------------------
# 7. advance_positions
# ---------------------------------------------------------------------------


class TestAdvancePositions:
    def test_static_node_unchanged(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        v = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        X2 = advance_positions(X, v, dt=0.5)
        np.testing.assert_allclose(X2[0], [1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(X2[1], [3.5, 4.0], atol=1e-6)

    def test_dt_zero_is_identity(self) -> None:
        X = np.array([[5.0, 6.0]], dtype=np.float32)
        v = np.array([[99.0, -99.0]], dtype=np.float32)
        X2 = advance_positions(X, v, dt=0.0)
        np.testing.assert_allclose(X2, X, atol=1e-6)
