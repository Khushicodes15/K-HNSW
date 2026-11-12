"""Unit and property tests for kinetic geometric certificate algorithms."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from khnsw.core.detection.certificates import (
    dist_quadratic,
    edge_failure_time,
    first_violation_time,
    is_violated_static,
    quad_sub,
    real_roots,
)

# =============================================================================
# 1. dist_quadratic evaluations vs direct squared distances
# =============================================================================


@pytest.mark.parametrize("dim", [2, 3, 8, 16, 32, 64])
def test_dist_quadratic_eval_matches_direct(dim: int) -> None:
    """Verify dist_quadratic coefficients evaluate to direct squared distances."""
    rng = np.random.default_rng(42 + dim)
    for _ in range(20):
        a_i = rng.standard_normal(dim).astype(np.float32)
        b_i = rng.standard_normal(dim).astype(np.float32)
        a_j = rng.standard_normal(dim).astype(np.float32)
        b_j = rng.standard_normal(dim).astype(np.float32)

        alpha, beta, gamma = dist_quadratic(a_i, b_i, a_j, b_j)

        # Test at several random timestamps
        for t in rng.uniform(-10.0, 10.0, size=5):
            t_f = float(t)
            d_quad = alpha * t_f * t_f + beta * t_f + gamma

            # Direct computation in float64
            p_i = a_i.astype(np.float64) + b_i.astype(np.float64) * t_f
            p_j = a_j.astype(np.float64) + b_j.astype(np.float64) * t_f
            diff = p_j - p_i
            d_direct = float(np.dot(diff, diff))

            assert math.isclose(d_quad, d_direct, rel_tol=1e-9, abs_tol=1e-12)


def test_quad_sub() -> None:
    """Verify quad_sub correctly subtracts quadratic coefficients."""
    q1 = (3.0, -2.0, 5.0)
    q2 = (1.0, 4.0, 2.0)
    sub = quad_sub(q1, q2)
    assert sub == (2.0, -6.0, 3.0)


# =============================================================================
# 2. real_roots: known polynomials, near-zero alpha, large magnitudes
# =============================================================================


def test_real_roots_known_polynomials() -> None:
    """Test real_roots against standard algebraic roots and degenerate forms."""
    # (t - 2)(t - 5) = t^2 - 7t + 10 = 0 -> [2.0, 5.0]
    roots = real_roots(1.0, -7.0, 10.0)
    assert len(roots) == 2
    assert math.isclose(roots[0], 2.0, abs_tol=1e-9)
    assert math.isclose(roots[1], 5.0, abs_tol=1e-9)

    # (t + 3)(t - 1) = t^2 + 2t - 3 = 0 -> [-3.0, 1.0]
    roots = real_roots(1.0, 2.0, -3.0)
    assert len(roots) == 2
    assert math.isclose(roots[0], -3.0, abs_tol=1e-9)
    assert math.isclose(roots[1], 1.0, abs_tol=1e-9)

    # Tangent double root: (t - 4)^2 = t^2 - 8t + 16 = 0 -> [4.0]
    roots = real_roots(1.0, -8.0, 16.0)
    assert len(roots) == 1
    assert math.isclose(roots[0], 4.0, abs_tol=1e-9)

    # Negative discriminant: t^2 + 9 = 0 -> []
    assert real_roots(1.0, 0.0, 9.0) == []

    # Linear degenerate: 0*t^2 + 3t - 6 = 0 -> [2.0]
    roots = real_roots(0.0, 3.0, -6.0)
    assert len(roots) == 1
    assert math.isclose(roots[0], 2.0, abs_tol=1e-9)

    # Constant non-zero: 0*t^2 + 0*t + 5 = 0 -> []
    assert real_roots(0.0, 0.0, 5.0) == []

    # Zero polynomial: 0*t^2 + 0*t + 0 = 0 -> []
    assert real_roots(0.0, 0.0, 0.0) == []


def test_real_roots_near_zero_alpha() -> None:
    """Test real_roots when alpha is tiny relative to beta (linear transition)."""
    # 1e-16 * t^2 + 4t - 8 = 0 -> t ~= 2.0
    roots = real_roots(1e-16, 4.0, -8.0)
    assert len(roots) == 1
    assert math.isclose(roots[0], 2.0, abs_tol=1e-7)


def test_real_roots_large_magnitudes() -> None:
    """Test real_roots with very large polynomial coefficients."""
    # 1e9 * (t - 1)(t - 2) = 1e9 t^2 - 3e9 t + 2e9 = 0 -> [1.0, 2.0]
    roots = real_roots(1e9, -3e9, 2e9)
    assert len(roots) == 2
    assert math.isclose(roots[0], 1.0, abs_tol=1e-8)
    assert math.isclose(roots[1], 2.0, abs_tol=1e-8)


# =============================================================================
# 3. is_violated_static truth table including ties and ordering condition
# =============================================================================


def test_is_violated_static_truth_table() -> None:
    """Verify is_violated_static truth table including ties and D(u,w) <= D(u,v)."""
    # 1. Base violation: D(u,w) < D(u,v) and D(v,w) < D(u,v) -> True
    assert is_violated_static(d_uv=10.0, d_uw=5.0, d_vw=4.0) is True

    # 2. Ordering boundary: D(u,w) == D(u,v) and D(v,w) < D(u,v) -> True
    assert is_violated_static(d_uv=10.0, d_uw=10.0, d_vw=4.0) is True

    # 3. Ordering violated: D(u,w) > D(u,v) -> False
    assert is_violated_static(d_uv=10.0, d_uw=10.5, d_vw=4.0) is False

    # 4. Tie on (v, w): D(v,w) == D(u,v) -> ties are NOT violations -> False
    assert is_violated_static(d_uv=10.0, d_uw=5.0, d_vw=10.0) is False

    # 5. Non-violating (v, w) distance: D(v,w) > D(u,v) -> False
    assert is_violated_static(d_uv=10.0, d_uw=5.0, d_vw=12.0) is False

    # 6. Complete equality / tie: D(u,v) == D(u,w) == D(v,w) -> False
    assert is_violated_static(d_uv=10.0, d_uw=10.0, d_vw=10.0) is False

    # 7. Zero distances (colocated nodes)
    assert is_violated_static(d_uv=0.0, d_uw=0.0, d_vw=0.0) is False


# =============================================================================
# 4. Explicit Degenerate Cases for first_violation_time
# =============================================================================


def test_first_violation_time_explicit_degeneracies() -> None:
    """Test first_violation_time on explicit degenerate configurations."""
    t0 = 0.0
    horizon = 10.0

    # A. Identical points: u = v = w
    zero = np.zeros(2, dtype=np.float32)
    assert first_violation_time((zero, zero), (zero, zero), (zero, zero), t0, horizon) is None

    # B. Zero velocities: static points
    # Case B1: static points violate condition at t0 -> returns t0
    # u=(0,0), v=(0,10) => D(uv)=100; w=(0,3) => D(uw)=9, D(vw)=49 => violated!
    u_pos = np.array([0.0, 0.0], dtype=np.float32)
    v_pos = np.array([0.0, 10.0], dtype=np.float32)
    w_pos = np.array([0.0, 3.0], dtype=np.float32)
    vel_zero = np.zeros(2, dtype=np.float32)

    res = first_violation_time((u_pos, vel_zero), (v_pos, vel_zero), (w_pos, vel_zero), t0, horizon)
    assert res == t0

    # Case B2: static points do NOT violate condition -> returns None
    w_far = np.array([0.0, 20.0], dtype=np.float32)
    res_none = first_violation_time(
        (u_pos, vel_zero), (v_pos, vel_zero), (w_far, vel_zero), t0, horizon
    )
    assert res_none is None

    # C. Parallel velocities (same relative velocity, distances constant)
    vel_parallel = np.array([1.5, -2.0], dtype=np.float32)
    res_par = first_violation_time(
        (u_pos, vel_parallel), (v_pos, vel_parallel), (w_pos, vel_parallel), t0, horizon
    )
    assert res_par == t0

    # D. Violation already true at t0
    t_start = 2.0
    res_tstart = first_violation_time(
        (u_pos, vel_zero), (v_pos, vel_zero), (w_pos, vel_zero), t_start, horizon
    )
    assert res_tstart == t_start

    # E. Violation occurs only AFTER horizon
    # u at (0,0) stationary
    # v at (10, 0) stationary => D(uv) = 100
    # w starts at (50, 0), moving towards u and v with velocity (-1, 0)
    # w reaches (0, 0) at t=50. At horizon=10, w is at (40, 0), D(uw)=1600 > 100.
    w_delayed = np.array([50.0, 0.0], dtype=np.float32)
    w_vel = np.array([-1.0, 0.0], dtype=np.float32)
    res_delayed = first_violation_time(
        (u_pos, vel_zero), (v_pos, vel_zero), (w_delayed, w_vel), t0=0.0, horizon=10.0
    )
    assert res_delayed is None

    # Horizon < t0 returns None
    assert (
        first_violation_time((u_pos, vel_zero), (v_pos, vel_zero), (w_pos, vel_zero), 5.0, 2.0)
        is None
    )


def test_first_violation_time_tangent_root() -> None:
    """Test tangent root where a difference polynomial touches zero."""
    # Construct 1D positions where D(u, w)(t) touches D(u, v)(t) from above at t = 2.0
    # u at 0, stationary. D(u, v) = 4 (v at 2).
    # w(t) position = 2 + (t - 2)^2 ... but w has linear motion:
    # In 2D: u=(0,0), v=(2,0) => D(uv)=4.
    # w(t) = (0, t - 2): D(uw)(t) = (t - 2)^2.
    # At t=2, D(uw)(2) = 0 <= 4.
    # For all t, D(uw)(t) >= 0.
    # D(vw)(t) = 4 + (t - 2)^2. At t=2, D(vw)=4 == D(uv)=4 (tie on D(vw) < D(uv) is not violated!).
    # So at t=2, D(vw) = D(uv), which is a tie, not a violation.
    u = (np.array([0.0, 0.0], dtype=np.float32), np.zeros(2, dtype=np.float32))
    v = (np.array([2.0, 0.0], dtype=np.float32), np.zeros(2, dtype=np.float32))
    w = (np.array([0.0, -2.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32))
    res = first_violation_time(u, v, w, t0=0.0, horizon=5.0)
    assert res is None


# =============================================================================
# 5. edge_failure_time: minimum over witnesses and correct witness index
# =============================================================================


def test_edge_failure_time_min_witness() -> None:
    """Verify edge_failure_time returns minimum failure time and correct index."""
    t0 = 0.0
    horizon = 10.0

    # Source u at (0, 0), v at (10, 0) => D(u, v) = 100
    u_traj = (np.array([0.0, 0.0], dtype=np.float32), np.zeros(2, dtype=np.float32))
    v_traj = (np.array([10.0, 0.0], dtype=np.float32), np.zeros(2, dtype=np.float32))

    # Witness 0: arrives late (t ~= 6.0)
    w0_traj = (np.array([30.0, 0.0], dtype=np.float32), np.array([-4.0, 0.0], dtype=np.float32))
    # Witness 1: arrives early (t ~= 2.0)
    w1_traj = (np.array([14.0, 0.0], dtype=np.float32), np.array([-4.0, 0.0], dtype=np.float32))
    # Witness 2: never enters violation region
    w2_traj = (np.array([100.0, 100.0], dtype=np.float32), np.zeros(2, dtype=np.float32))

    witnesses = [w0_traj, w1_traj, w2_traj]
    result = edge_failure_time(u_traj, v_traj, witnesses, t0=t0, horizon=horizon)

    assert result is not None
    t_fail, w_idx = result

    # Witness 1 must be the earliest
    assert w_idx == 1

    t1 = first_violation_time(u_traj, v_traj, w1_traj, t0, horizon)
    assert t1 is not None
    assert math.isclose(t_fail, t1, abs_tol=1e-9)

    # When all witnesses never violate: returns None
    result_none = edge_failure_time(u_traj, v_traj, [w2_traj], t0=t0, horizon=horizon)
    assert result_none is None


# =============================================================================
# 6. Property-based test with Hypothesis: Brute-Force Grid Comparison (>=20,000 pts)
# =============================================================================


@settings(max_examples=40, deadline=1500)
@given(
    coords=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=12,
        max_size=12,
    )
)
def test_first_violation_time_agrees_with_dense_grid_scan(coords: list[float]) -> None:
    """Property test: first_violation_time agrees with dense-grid brute-force scan.

    Tests that:
    1. If a violation is sampled by the dense grid at t_sample, then T <= t_sample + eps.
    2. If T is returned, a violation exists within a small tolerance after T.
    3. If T is None, the grid scan finds no violation.
    """
    t0 = 0.0
    horizon = 4.0
    n_points = 25000  # >= 20,000 points required

    au = np.array(coords[0:2], dtype=np.float32)
    bu = np.array(coords[2:4], dtype=np.float32)
    av = np.array(coords[4:6], dtype=np.float32)
    bv = np.array(coords[6:8], dtype=np.float32)
    aw = np.array(coords[8:10], dtype=np.float32)
    bw = np.array(coords[10:12], dtype=np.float32)

    # Skip measure-zero degenerate coincident trajectories
    assume(float(np.linalg.norm(au - av)) > 0.05)
    assume(float(np.linalg.norm(au - aw)) > 0.05)
    assume(float(np.linalg.norm(av - aw)) > 0.05)

    u_traj = (au, bu)
    v_traj = (av, bv)
    w_traj = (aw, bw)

    # Dense grid brute-force evaluation
    t_grid = np.linspace(t0, horizon, n_points, dtype=np.float64)
    dt = float(t_grid[1] - t_grid[0])

    pu = au.astype(np.float64) + bu.astype(np.float64) * t_grid[:, None]
    pv = av.astype(np.float64) + bv.astype(np.float64) * t_grid[:, None]
    pw = aw.astype(np.float64) + bw.astype(np.float64) * t_grid[:, None]

    d_uv = np.sum((pv - pu) ** 2, axis=1)
    d_uw = np.sum((pw - pu) ** 2, axis=1)
    d_vw = np.sum((pw - pv) ** 2, axis=1)

    # Vectorized violation check using strict condition with numerical tolerance
    tol = 1e-10
    violated_grid = (d_uw <= d_uv + tol) & (d_vw < d_uv - tol)

    first_grid_idx = None
    if np.any(violated_grid):
        first_grid_idx = int(np.argmax(violated_grid))

    # Analytic certificate solver
    t_analytic = first_violation_time(u_traj, v_traj, w_traj, t0=t0, horizon=horizon, tol=tol)

    if first_grid_idx is not None:
        t_sample = float(t_grid[first_grid_idx])

        # 1. Analytic failure time must NOT be later than the first sampled violation
        assert t_analytic is not None, (
            f"Grid detected violation at t={t_sample}, but analytic solver returned None"
        )
        assert t_analytic <= t_sample + 1e-9, (
            f"Analytic T={t_analytic} is later than grid sample {t_sample}"
        )

        # 2. A violation exists within a small tolerance window [T, T + 2*dt]
        assert t_sample - t_analytic <= 2.0 * dt + 1e-8, (
            f"Gap between T={t_analytic} and grid {t_sample} exceeds tolerance: "
            f"{t_sample - t_analytic}"
        )
    else:
        # If the grid found no violation, then either analytic solver returned None,
        # or the violation interval was narrower than grid step dt (measure-zero boundary spike).
        if t_analytic is not None:
            # Check if there is indeed a valid violation very close to T
            # Evaluate at T + 0.5*dt
            t_probe = min(horizon, t_analytic + 0.5 * dt)
            p_u = au.astype(np.float64) + bu.astype(np.float64) * t_probe
            p_v = av.astype(np.float64) + bv.astype(np.float64) * t_probe
            p_w = aw.astype(np.float64) + bw.astype(np.float64) * t_probe
            d1 = float(np.dot(p_v - p_u, p_v - p_u))
            d2 = float(np.dot(p_w - p_u, p_w - p_u))
            d3 = float(np.dot(p_w - p_v, p_w - p_v))
            assert is_violated_static(d1, d2, d3, tol=tol), (
                f"Analytic solver returned T={t_analytic} but probe at {t_probe} was not violated"
            )
