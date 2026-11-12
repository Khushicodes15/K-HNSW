"""Unit and property tests for kinetic movement models."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from khnsw.core.interfaces import LinearTrajectory, MovementModel
from khnsw.core.movement import (
    ClusterDrift,
    DiscreteReplacement,
    LinearDrift,
    TeleportAdversary,
    mean_nn_distance,
    rng_for,
)


@pytest.fixture
def sample_positions() -> np.ndarray:
    """Deterministic sample point array (20 points in 4 dimensions)."""
    rng = np.random.default_rng(12345)
    return rng.standard_normal((20, 4), dtype=np.float32)


def test_mean_nn_distance_brute_force_agreement() -> None:
    """mean_nn_distance matches exact full pairwise distance on small data."""
    rng = np.random.default_rng(42)
    n, d = 50, 8
    X = rng.standard_normal((n, d), dtype=np.float64)

    # Brute-force all-pairs calculation
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    expected_mean_nn = float(np.mean(np.min(dists, axis=1)))

    computed_mean_nn = mean_nn_distance(X, sample=n, seed=0)
    np.testing.assert_allclose(computed_mean_nn, expected_mean_nn, rtol=1e-9)


def test_mean_nn_distance_edge_cases() -> None:
    """mean_nn_distance returns 0.0 for n < 2 and handles 1D arrays."""
    # n = 0
    assert mean_nn_distance(np.empty((0, 4)), sample=100) == 0.0
    # n = 1
    assert mean_nn_distance(np.array([[1.0, 2.0, 3.0]]), sample=100) == 0.0
    # 1D array of shape (n,) -> treated as (n, 1)
    arr_1d = np.array([0.0, 2.0, 5.0])
    # Distances to nearest: 0->2 (dist=2), 2->0 (dist=2), 5->2 (dist=3). Mean = 7/3
    expected_1d = (2.0 + 2.0 + 3.0) / 3.0
    np.testing.assert_allclose(mean_nn_distance(arr_1d), expected_1d, rtol=1e-9)

    # Subsampling determinism
    rng = np.random.default_rng(99)
    big_X = rng.standard_normal((100, 3))
    val1 = mean_nn_distance(big_X, sample=20, seed=42)
    val2 = mean_nn_distance(big_X, sample=20, seed=42)
    assert val1 == val2


def test_rng_for_determinism() -> None:
    """rng_for generates identical sequences for identical (seed, step) independent of calls."""
    gen1 = rng_for(seed=10, step=3)
    seq1 = gen1.uniform(size=5)

    # Call arbitrary other generators
    rng_for(seed=10, step=4).uniform(size=10)
    rng_for(seed=20, step=3).uniform(size=10)

    gen2 = rng_for(seed=10, step=3)
    seq2 = gen2.uniform(size=5)

    np.testing.assert_array_equal(seq1, seq2)


def test_model_protocol_conformance(sample_positions: np.ndarray) -> None:
    """Verify runtime protocol conformance for all models."""
    m_linear = LinearDrift(sample_positions, seed=1)
    m_cluster = ClusterDrift(sample_positions, seed=1)
    m_discrete = DiscreteReplacement(sample_positions, seed=1)
    m_teleport = TeleportAdversary(sample_positions, seed=1)

    for m in [m_linear, m_cluster, m_discrete, m_teleport]:
        assert isinstance(m, MovementModel)
        assert hasattr(m, "name")
        assert hasattr(m, "seed")
        assert hasattr(m, "positions")
        assert hasattr(m, "changed_nodes")

    # LinearTrajectory protocol check
    assert isinstance(m_linear, LinearTrajectory)
    assert isinstance(m_cluster, LinearTrajectory)
    assert not isinstance(m_discrete, LinearTrajectory)
    assert not isinstance(m_teleport, LinearTrajectory)


def test_model_determinism_and_shuffled_order(sample_positions: np.ndarray) -> None:
    """Verify models are deterministic and independent of query order."""
    models_cls = [LinearDrift, ClusterDrift, DiscreteReplacement, TeleportAdversary]

    for cls in models_cls:
        m1 = cls(sample_positions, seed=42)
        m2 = cls(sample_positions, seed=42)
        m_diff = cls(sample_positions, seed=99)

        pos1 = m1.positions(3.7)
        pos2 = m2.positions(3.7)
        pos_diff = m_diff.positions(3.7)

        np.testing.assert_array_equal(pos1, pos2)
        assert not np.array_equal(pos1, pos_diff)

    # Query order independence test (especially critical for stateful step caching)
    query_times = [0.0, 1.2, 5.5, 2.1, 7.8, 3.0, 6.4, 4.0]
    for cls in [DiscreteReplacement, TeleportAdversary]:
        m_asc = cls(sample_positions, seed=42)
        m_shuffled = cls(sample_positions, seed=42)

        res_asc = {t: m_asc.positions(t) for t in sorted(query_times)}
        res_shuffled = {t: m_shuffled.positions(t) for t in query_times}

        for t in query_times:
            np.testing.assert_array_equal(res_shuffled[t], res_asc[t])


def test_shapes_dtypes_finiteness_and_t0(sample_positions: np.ndarray) -> None:
    """Verify positions(t) shape, float32 dtype, finiteness, and t=0 matches base."""
    n, d = sample_positions.shape
    models = [
        LinearDrift(sample_positions, seed=42),
        ClusterDrift(sample_positions, seed=42),
        DiscreteReplacement(sample_positions, seed=42),
        TeleportAdversary(sample_positions, seed=42),
    ]

    for m in models:
        # t = 0 matches base
        pos0 = m.positions(0.0)
        assert pos0.shape == (n, d)
        assert pos0.dtype == np.float32
        np.testing.assert_array_equal(pos0, sample_positions.astype(np.float32))

        # t > 0
        pos_t = m.positions(2.8)
        assert pos_t.shape == (n, d)
        assert pos_t.dtype == np.float32
        assert np.all(np.isfinite(pos_t))


def test_linear_models_trajectory_and_precision(sample_positions: np.ndarray) -> None:
    """Linear models trajectory matches positions(t) within 1e-5 relative tolerance."""
    n, _ = sample_positions.shape
    linear_models = [
        LinearDrift(sample_positions, seed=42, speed=1.5),
        ClusterDrift(sample_positions, seed=42, speed=1.5, noise=0.2),
    ]

    query_times = [0.5, 1.0, 2.5, 10.0]
    for model in linear_models:
        for t in query_times:
            pos_batch = model.positions(t)
            for i in range(n):
                a, b = model.trajectory(i)
                assert a.dtype == np.float64
                assert b.dtype == np.float64
                expected = (a + b * t).astype(np.float32)
                np.testing.assert_allclose(pos_batch[i], expected, rtol=1e-5, atol=1e-6)

        # Out-of-bounds trajectory query raises IndexError
        with pytest.raises(IndexError):
            model.trajectory(n + 10)
        with pytest.raises(IndexError):
            model.trajectory(-1)


def test_changed_nodes_exact_agreement(sample_positions: np.ndarray) -> None:
    """changed_nodes matches exact row differences for all models."""
    models = [
        LinearDrift(sample_positions, seed=42),
        ClusterDrift(sample_positions, seed=42),
        DiscreteReplacement(sample_positions, seed=42, fraction=0.3),
        TeleportAdversary(sample_positions, seed=42, fraction=0.3),
    ]

    pairs = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.2, 2.8), (2.0, 5.0)]
    for m in models:
        for t_prev, t_curr in pairs:
            changed = m.changed_nodes(t_prev, t_curr)
            if t_prev == t_curr:
                assert changed == []
            else:
                p_prev = m.positions(t_prev)
                p_curr = m.positions(t_curr)
                expected_changed = [
                    int(idx) for idx in np.where(np.any(p_prev != p_curr, axis=1))[0]
                ]
                assert changed == expected_changed


def test_discrete_replacement_mechanics(sample_positions: np.ndarray) -> None:
    """DiscreteReplacement moves round(fraction*n) nodes by exactly displacement*scale."""
    n = sample_positions.shape[0]
    fraction = 0.25
    expected_moved_count = int(round(fraction * n))

    model = DiscreteReplacement(
        sample_positions,
        seed=42,
        fraction=fraction,
        displacement=2.0,
    )
    scale = model.scale

    p0 = model.positions(0.0)
    p1 = model.positions(1.0)

    diff = np.any(p0 != p1, axis=1)
    moved_nodes = np.where(diff)[0]
    assert len(moved_nodes) == expected_moved_count

    # Moved nodes displacement magnitude == displacement * scale
    for idx in moved_nodes:
        disp_norm = float(np.linalg.norm(p1[idx] - p0[idx]))
        expected_disp = model.displacement * scale
        np.testing.assert_allclose(disp_norm, expected_disp, rtol=1e-4)

    # Unmoved nodes are bit-identical
    unmoved_nodes = np.where(~diff)[0]
    for idx in unmoved_nodes:
        np.testing.assert_array_equal(p0[idx], p1[idx])


def test_teleport_adversary_mechanics(sample_positions: np.ndarray) -> None:
    """TeleportAdversary relocates round(fraction*n) nodes within jitter_radius of another node."""
    n = sample_positions.shape[0]
    fraction = 0.2
    expected_moved_count = int(round(fraction * n))

    model = TeleportAdversary(sample_positions, seed=42, fraction=fraction)
    p0 = model.positions(0.0)
    p1 = model.positions(1.0)

    diff = np.any(p0 != p1, axis=1)
    moved_nodes = np.where(diff)[0]
    assert len(moved_nodes) == expected_moved_count

    # Each relocated node must land within jitter radius of some other node's base coordinate
    jitter_radius = model.jitter_radius
    base_pos = sample_positions.astype(np.float32)

    for idx in moved_nodes:
        # Distance to all other base positions
        other_bases = np.delete(base_pos, idx, axis=0)
        dists = np.linalg.norm(other_bases - p1[idx], axis=1)
        min_dist = float(np.min(dists))
        assert min_dist <= jitter_radius + 1e-5


def test_edge_cases() -> None:
    """Verify behavior on extreme edge cases: n=1, n=2, fraction=0, 1, dim=1, large t."""
    # n = 1, dim = 1
    pt_1 = np.array([[5.0]], dtype=np.float32)
    m1 = LinearDrift(pt_1, seed=1)
    assert m1.positions(0.0).shape == (1, 1)
    assert m1.positions(10.0).shape == (1, 1)
    assert m1.scale == 0.0
    np.testing.assert_array_equal(m1.positions(10.0), pt_1)

    m1_disc = DiscreteReplacement(pt_1, seed=1, fraction=1.0)
    assert m1_disc.positions(5.0).shape == (1, 1)

    m1_tele = TeleportAdversary(pt_1, seed=1, fraction=1.0)
    assert m1_tele.positions(5.0).shape == (1, 1)

    # n = 2, dim = 3
    pt_2 = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    m2 = ClusterDrift(pt_2, seed=1, n_clusters=5)
    assert m2.positions(2.0).shape == (2, 3)

    # fraction = 0.0 -> no nodes move
    m_f0 = DiscreteReplacement(pt_2, seed=1, fraction=0.0)
    np.testing.assert_array_equal(m_f0.positions(5.0), pt_2)
    assert m_f0.changed_nodes(0.0, 5.0) == []

    # fraction = 1.0 -> all nodes move
    m_f1 = DiscreteReplacement(pt_2, seed=1, fraction=1.0)
    assert len(m_f1.changed_nodes(0.0, 1.0)) == 2

    # Large t (t = 10000.0) produces finite float32 positions
    m_large = LinearDrift(pt_2, seed=1)
    pos_large = m_large.positions(10000.0)
    assert np.all(np.isfinite(pos_large))


@settings(max_examples=25, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=25),
    d=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=1, max_value=1000),
    t=st.floats(min_value=0.1, max_value=20.0),
)
def test_hypothesis_linear_drift_property(n: int, d: int, seed: int, t: float) -> None:
    """Property test: LinearDrift trajectories consistently reproduce batch positions."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d), dtype=np.float32)
    model = LinearDrift(base, seed=seed, speed=1.2)

    positions = model.positions(t)
    assert positions.shape == (n, d)
    assert positions.dtype == np.float32
    assert np.all(np.isfinite(positions))

    # Pick a random node and verify closed-form trajectory match
    node_id = int(seed % n)
    a, b = model.trajectory(node_id)
    expected = (a + b * t).astype(np.float32)
    np.testing.assert_allclose(positions[node_id], expected, rtol=1e-5, atol=1e-6)
