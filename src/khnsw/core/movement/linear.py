"""Linear kinematic drift movement model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from khnsw.core.movement.base import mean_nn_distance


class LinearDrift:
    """Linear kinetic trajectory model where points move with constant velocities.

    Each point i moves along a linear trajectory p_i(t) = a_i + b_i * t, where:
    - a_i is the initial coordinate base_positions[i] in float64.
    - b_i is a deterministic random unit direction scaled by speed * scale.
    - scale is the mean nearest-neighbor Euclidean distance of base_positions.

    Implements the LinearTrajectory protocol for closed-form failure time detection.
    """

    def __init__(
        self,
        base_positions: NDArray[Any],
        seed: int = 42,
        speed: float = 1.0,
    ) -> None:
        """Initialize linear drift model.

        Parameters
        ----------
        base_positions : NDArray[Any]
            Initial coordinates of n points in d dimensions, shape (n, d).
        seed : int, default=42
            Deterministic seed for velocity direction sampling.
        speed : float, default=1.0
            Velocity multiplier relative to nearest-neighbor scale per unit time.
        """
        arr = np.asarray(base_positions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim != 2:
            msg = f"base_positions must be 1D or 2D, got shape {arr.shape}"
            raise ValueError(msg)

        self._base: NDArray[np.float64] = arr
        self._seed = int(seed)
        self._speed = float(speed)
        self._n, self._dim = self._base.shape
        self._scale: float = mean_nn_distance(self._base, seed=self._seed)

        # Deterministically sample velocity unit vectors
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, 0]))
        if self._n > 0 and self._dim > 0:
            normals = rng.standard_normal((self._n, self._dim), dtype=np.float64)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            unit_dirs = normals / norms
            self._velocities: NDArray[np.float64] = unit_dirs * (self._speed * self._scale)
        else:
            self._velocities = np.zeros((self._n, self._dim), dtype=np.float64)

    @property
    def name(self) -> str:
        """Descriptive model identifier."""
        return "linear_drift"

    @property
    def seed(self) -> int:
        """Random seed for reproducible kinematics."""
        return self._seed

    @property
    def speed(self) -> float:
        """Speed multiplier."""
        return self._speed

    @property
    def scale(self) -> float:
        """Nearest-neighbor spatial scale."""
        return self._scale

    def positions(self, t: float) -> NDArray[np.float32]:
        """Compute the spatial positions of all indexed nodes at simulation timestamp t.

        Parameters
        ----------
        t : float
            Simulation time parameter (t >= 0.0).

        Returns
        -------
        NDArray[np.float32]
            2D array of positions for all n points, shape (n, dim).
        """
        pos64 = self._base + self._velocities * float(t)
        return pos64.astype(np.float32)

    def trajectory(self, node_id: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the linear trajectory coefficients (a, b) for a given node.

        Parameters
        ----------
        node_id : int
            Identifier of target node.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple of (initial_offset_a, velocity_vector_b), each of shape (dim,).
        """
        if not (0 <= node_id < self._n):
            msg = f"node_id {node_id} out of bounds for index of size {self._n}"
            raise IndexError(msg)
        return self._base[node_id].copy(), self._velocities[node_id].copy()

    def changed_nodes(self, t_prev: float, t_curr: float) -> list[int]:
        """Identify which node IDs have changed coordinates between two timestamps.

        Parameters
        ----------
        t_prev : float
            Previous timestamp.
        t_curr : float
            Current timestamp.

        Returns
        -------
        list[int]
            Sorted list of node IDs whose spatial coordinates differ.
        """
        if t_prev == t_curr:
            return []
        p_prev = self.positions(t_prev)
        p_curr = self.positions(t_curr)
        diff = np.any(p_prev != p_curr, axis=1)
        return [int(idx) for idx in np.where(diff)[0]]
