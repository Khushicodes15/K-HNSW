"""Discrete replacement kinetic movement model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from khnsw.core.movement.base import mean_nn_distance, rng_for


class DiscreteReplacement:
    """Discrete kinetic movement model with periodic random step displacements.

    At each integer step s = 1, 2, ..., a deterministic subset of round(fraction * n)
    nodes (chosen via rng_for(seed, s)) is displaced by exactly displacement * scale
    in a randomly sampled unit direction, where scale is the mean nearest-neighbor
    Euclidean distance.

    Position at continuous time t is given by cumulative displacements through step floor(t):
        positions(t) = base + sum_{s=1}^{floor(t)} delta_s

    Cumulative state is computed deterministically in strictly increasing step sequence
    and cached, ensuring that results are bit-identical regardless of query order.

    Does NOT implement the LinearTrajectory protocol.
    """

    def __init__(
        self,
        base_positions: NDArray[Any],
        seed: int = 42,
        fraction: float = 0.05,
        displacement: float = 1.0,
    ) -> None:
        """Initialize discrete replacement model.

        Parameters
        ----------
        base_positions : NDArray[Any]
            Initial coordinates of n points in d dimensions, shape (n, d).
        seed : int, default=42
            Deterministic simulation seed.
        fraction : float, default=0.05
            Fraction of nodes to move at each integer step.
        displacement : float, default=1.0
            Magnitude multiplier of displacement relative to nearest-neighbor scale.
        """
        arr = np.asarray(base_positions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim != 2:
            msg = f"base_positions must be 1D or 2D, got shape {arr.shape}"
            raise ValueError(msg)

        self._base: NDArray[np.float64] = arr
        self._seed = int(seed)
        self._fraction = float(fraction)
        self._displacement = float(displacement)
        self._n, self._dim = self._base.shape
        self._scale: float = mean_nn_distance(self._base, seed=self._seed)

        # Cache of cumulative positions indexed by integer step: 0 -> base
        self._cumulative_cache: dict[int, NDArray[np.float64]] = {0: self._base.copy()}
        self._max_cached_step = 0

    @property
    def name(self) -> str:
        """Descriptive model identifier."""
        return "discrete_replacement"

    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed

    @property
    def fraction(self) -> float:
        """Fraction of moving nodes per step."""
        return self._fraction

    @property
    def displacement(self) -> float:
        """Displacement magnitude multiplier."""
        return self._displacement

    @property
    def scale(self) -> float:
        """Nearest-neighbor spatial scale."""
        return self._scale

    def _ensure_step(self, step: int) -> None:
        """Sequentially compute and cache cumulative positions up to target step."""
        if step <= self._max_cached_step:
            return

        k = min(self._n, max(0, int(round(self._fraction * self._n))))
        current = self._cumulative_cache[self._max_cached_step]

        for s in range(self._max_cached_step + 1, step + 1):
            next_pos = current.copy()
            if k > 0 and self._dim > 0:
                rng = rng_for(self._seed, s)
                chosen = rng.choice(self._n, size=k, replace=False)
                normals = rng.standard_normal((k, self._dim), dtype=np.float64)
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                unit_dirs = normals / norms
                disp_vectors = unit_dirs * (self._displacement * self._scale)
                next_pos[chosen] += disp_vectors
            self._cumulative_cache[s] = next_pos
            current = next_pos

        self._max_cached_step = step

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
        step = int(np.floor(float(t)))
        if step <= 0:
            return self._base.astype(np.float32)

        self._ensure_step(step)
        return self._cumulative_cache[step].astype(np.float32)

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
        s_prev = int(np.floor(float(t_prev)))
        s_curr = int(np.floor(float(t_curr)))
        if s_prev == s_curr:
            return []
        p_prev = self.positions(t_prev)
        p_curr = self.positions(t_curr)
        diff = np.any(p_prev != p_curr, axis=1)
        return [int(idx) for idx in np.where(diff)[0]]
