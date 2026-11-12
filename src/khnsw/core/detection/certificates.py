"""Kinetic geometric certificate math for HNSW edge validity.

This module is strictly pure Python standard library and NumPy only.
All certificate computations are executed in float64 precision for numerical stability.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "dist_quadratic",
    "edge_failure_time",
    "first_violation_time",
    "is_violated_static",
    "quad_sub",
    "real_roots",
]


def dist_quadratic(
    a_i: NDArray[np.floating[Any]],
    b_i: NDArray[np.floating[Any]],
    a_j: NDArray[np.floating[Any]],
    b_j: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """Compute quadratic polynomial coefficients of squared distance D(i, j)(t).

    For linear trajectories p_i(t) = a_i + b_i*t and p_j(t) = a_j + b_j*t,
    the squared Euclidean distance is:
        D(i, j)(t) = ||p_j(t) - p_i(t)||^2 = alpha * t^2 + beta * t + gamma

    Where Delta_a = a_j - a_i, Delta_b = b_j - b_i:
        alpha = ||Delta_b||^2 = Delta_b . Delta_b
        beta  = 2 * (Delta_a . Delta_b)
        gamma = ||Delta_a||^2 = Delta_a . Delta_a

    All operations are evaluated in float64 precision.

    Parameters
    ----------
    a_i : NDArray
        Spatial coordinates of node i at t=0, shape (d,).
    b_i : NDArray
        Velocity vector of node i, shape (d,).
    a_j : NDArray
        Spatial coordinates of node j at t=0, shape (d,).
    b_j : NDArray
        Velocity vector of node j, shape (d,).

    Returns
    -------
    tuple[float, float, float]
        Coefficients (alpha, beta, gamma) as float64 scalars.
    """
    ai = np.asarray(a_i, dtype=np.float64)
    bi = np.asarray(b_i, dtype=np.float64)
    aj = np.asarray(a_j, dtype=np.float64)
    bj = np.asarray(b_j, dtype=np.float64)

    delta_a = aj - ai
    delta_b = bj - bi

    alpha = float(np.dot(delta_b, delta_b))
    beta = float(2.0 * np.dot(delta_a, delta_b))
    gamma = float(np.dot(delta_a, delta_a))

    return alpha, beta, gamma


def quad_sub(
    q1: tuple[float, float, float],
    q2: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute the quadratic difference polynomial q1(t) - q2(t).

    Parameters
    ----------
    q1 : tuple[float, float, float]
        Coefficients (alpha1, beta1, gamma1).
    q2 : tuple[float, float, float]
        Coefficients (alpha2, beta2, gamma2).

    Returns
    -------
    tuple[float, float, float]
        Difference coefficients (alpha1 - alpha2, beta1 - beta2, gamma1 - gamma2).
    """
    return (
        float(q1[0] - q2[0]),
        float(q1[1] - q2[1]),
        float(q1[2] - q2[2]),
    )


def real_roots(
    alpha: float,
    beta: float,
    gamma: float,
    tol: float = 1e-12,
) -> list[float]:
    """Compute the sorted real roots of alpha * t^2 + beta * t + gamma = 0.

    Handles degenerate and singular regimes:
    - Constant degenerate cases (alpha ~ 0, beta ~ 0): returns []
    - Linear degenerate cases (alpha ~ 0, |beta| > tol): returns [-gamma / beta]
    - Negative discriminant (Delta < -tol_discr): returns []
    - Tangent double roots (|Delta| <= tol_discr): returns [-beta / (2*alpha)]
    - Distinct real roots (Delta > tol_discr): returns sorted [t1, t2] computed
      via numerically stable quadratic formula (Citardauq formulation).

    Parameters
    ----------
    alpha : float
        Quadratic coefficient.
    beta : float
        Linear coefficient.
    gamma : float
        Constant coefficient.
    tol : float, default=1e-12
        Tolerance for near-zero thresholding.

    Returns
    -------
    list[float]
        Sorted list of real roots.
    """
    a = float(alpha)
    b = float(beta)
    c = float(gamma)

    scale = max(abs(a), abs(b), abs(c), 1.0)
    abs_tol = tol * scale

    # Check if quadratic term is negligible (linear or constant equation)
    if abs(a) <= abs_tol:
        if abs(b) <= abs_tol:
            # Constant polynomial: no isolated discrete roots
            return []
        return [float(-c / b)]

    # Compute discriminant
    discr = b * b - 4.0 * a * c
    discr_tol = tol * (b * b + 4.0 * abs(a * c) + 1.0)

    if discr < -discr_tol:
        return []

    if abs(discr) <= discr_tol:
        root = -b / (2.0 * a)
        return [float(root)]

    # Two distinct real roots: Citardauq stable formulation
    sqrt_discr = math.sqrt(max(0.0, discr))
    sgn = 1.0 if b >= 0.0 else -1.0
    q = -0.5 * (b + sgn * sqrt_discr)

    t1 = q / a
    t2 = c / q if abs(q) > 0.0 else -b / a

    roots = sorted([float(t1), float(t2)])
    if len(roots) == 2 and abs(roots[1] - roots[0]) <= tol * max(abs(roots[0]), 1.0):
        return [roots[0]]
    return roots


def is_violated_static(
    d_uv: float,
    d_uw: float,
    d_vw: float,
    tol: float = 1e-12,
) -> bool:
    """Check if directed edge u -> v is violated by witness w.

    Condition:
        D(u, w) <= D(u, v) AND D(v, w) < D(u, v)

    Ties are NOT violations:
        - Strict inequality D(v, w) < D(u, v) means ties on (v, w) are valid.
        - Ordering condition requires D(u, w) <= D(u, v).

    Parameters
    ----------
    d_uv : float
        Squared distance D(u, v).
    d_uw : float
        Squared distance D(u, w).
    d_vw : float
        Squared distance D(v, w).
    tol : float, default=1e-12
        Tolerance for floating-point comparisons.

    Returns
    -------
    bool
        True if edge u -> v is violated by witness w, False otherwise.
    """
    uv = float(d_uv)
    uw = float(d_uw)
    vw = float(d_vw)

    scale = max(abs(uv), abs(uw), abs(vw), 1.0)
    abs_tol = tol * scale

    # D(v, w) < D(u, v) strictly
    if vw >= uv - abs_tol:
        return False

    # D(u, w) <= D(u, v)
    if uw > uv + abs_tol:
        return False

    return True


def first_violation_time(
    traj_u: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    traj_v: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    traj_w: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    t0: float,
    horizon: float,
    tol: float = 1e-12,
) -> float | None:
    """Compute the earliest time t in [t0, horizon] when witness w violates edge u -> v.

    Formulates polynomials:
        g1(t) = D(u, w)(t) - D(u, v)(t)  (relevant iff <= 0)
        g2(t) = D(v, w)(t) - D(u, v)(t)  (violating iff < 0)

    Solves for roots of g1 and g2, tests the violation predicate at boundary roots
    and interval midpoints, and returns the infimum time where violation begins.

    Parameters
    ----------
    traj_u : tuple[NDArray, NDArray]
        Trajectory (a_u, b_u) for node u.
    traj_v : tuple[NDArray, NDArray]
        Trajectory (a_v, b_v) for node v.
    traj_w : tuple[NDArray, NDArray]
        Trajectory (a_w, b_w) for node w.
    t0 : float
        Start timestamp (t0 >= 0.0).
    horizon : float
        Evaluation horizon (horizon >= t0).
    tol : float, default=1e-12
        Numerical tolerance.

    Returns
    -------
    float | None
        Earliest violation timestamp t in [t0, horizon], or None if no violation occurs.
    """
    if horizon < t0:
        return None

    au, bu = traj_u
    av, bv = traj_v
    aw, bw = traj_w

    q_uv = dist_quadratic(au, bu, av, bv)
    q_uw = dist_quadratic(au, bu, aw, bw)
    q_vw = dist_quadratic(av, bv, aw, bw)

    g1 = quad_sub(q_uw, q_uv)
    g2 = quad_sub(q_vw, q_uv)

    def eval_at(t: float) -> bool:
        t_f = float(t)
        t2 = t_f * t_f
        d_uv = q_uv[0] * t2 + q_uv[1] * t_f + q_uv[2]
        d_uw = q_uw[0] * t2 + q_uw[1] * t_f + q_uw[2]
        d_vw = q_vw[0] * t2 + q_vw[1] * t_f + q_vw[2]
        return is_violated_static(d_uv, d_uw, d_vw, tol=tol)

    # 1. Check if already violated at t0
    if eval_at(t0):
        return float(t0)

    # 2. Find real roots of g1 and g2
    roots_g1 = real_roots(g1[0], g1[1], g1[2], tol=tol)
    roots_g2 = real_roots(g2[0], g2[1], g2[2], tol=tol)

    # Filter roots strictly inside (t0, horizon]
    interior_roots = sorted(set(r for r in (roots_g1 + roots_g2) if t0 < r <= horizon))

    # Construct boundary partition: [t0, r_1, r_2, ..., horizon]
    raw_points = [float(t0)] + interior_roots
    if not interior_roots or interior_roots[-1] < horizon:
        raw_points.append(float(horizon))

    # Deduplicate adjacent points closer than eps
    eps = tol * max(abs(horizon), 1.0)
    partition: list[float] = [raw_points[0]]
    for pt in raw_points[1:]:
        if pt - partition[-1] > eps:
            partition.append(pt)

    # 3. Test open intervals and endpoints in chronological order
    for i in range(len(partition) - 1):
        left = partition[i]
        right = partition[i + 1]
        mid = (left + right) * 0.5

        # If the open interval is violated, violation begins at 'left'
        if eval_at(mid):
            return float(left)

        # Check right boundary endpoint
        if eval_at(right):
            return float(right)

    return None


def edge_failure_time(
    u_traj: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    v_traj: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    witness_trajs: Sequence[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]],
    t0: float,
    horizon: float,
    tol: float = 1e-12,
) -> tuple[float, int] | None:
    """Compute the earliest failure time and witness index for edge u -> v.

    Evaluates first_violation_time across all witness trajectories and returns
    the minimum failure time and the index of the responsible witness.

    Parameters
    ----------
    u_traj : tuple[NDArray, NDArray]
        Trajectory (a_u, b_u) of source node u.
    v_traj : tuple[NDArray, NDArray]
        Trajectory (a_v, b_v) of target node v.
    witness_trajs : Sequence[tuple[NDArray, NDArray]]
        Collection of witness candidate trajectories (a_w, b_w).
    t0 : float
        Start timestamp (t0 >= 0.0).
    horizon : float
        Horizon timestamp (horizon >= t0).
    tol : float, default=1e-12
        Tolerance for numerical checks.

    Returns
    -------
    tuple[float, int] | None
        Tuple of (earliest_failure_time, witness_index), or None if no witness
        violates the edge in [t0, horizon].
    """
    earliest_time: float | None = None
    best_witness: int | None = None

    for idx, w_traj in enumerate(witness_trajs):
        t_fail = first_violation_time(u_traj, v_traj, w_traj, t0=t0, horizon=horizon, tol=tol)
        if t_fail is not None:
            if earliest_time is None or t_fail < earliest_time:
                earliest_time = t_fail
                best_witness = idx

    if earliest_time is None or best_witness is None:
        return None
    return float(earliest_time), best_witness
