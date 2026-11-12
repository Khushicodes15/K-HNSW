"""Clustered drift kinetic movement model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from khnsw.core.movement.base import mean_nn_distance


class ClusterDrift:
    """Kinetic trajectory model with coherent semantic cluster velocities and local noise.

    Each node belongs to a spatial cluster determined via seeded k-means-lite.
    All nodes in a cluster share a common cluster velocity vector, super-positioned with
    an idiosyncratic per-node noise velocity:
        b_i = V_{cluster(i)} + v_{noise, i}
    where:
    - V_k = u_k * speed * scale (u_k is a random unit vector for cluster k).
    - v_{noise, i} = eta_i * (noise * speed * scale) (eta_i is a random unit vector for node i).
    - scale is the mean nearest-neighbor Euclidean distance.

    Cluster assignment mechanism:
    Seeded k-means-lite: If n <= n_clusters, each node forms its own cluster. Otherwise,
    n_clusters centroids are initialized by sampling without replacement from base_positions
    using the model's seeded generator. 10 iterations of Lloyd's algorithm (Voronoi partition
    assignment and centroid mean update) are executed deterministically. Each node is assigned
    to its nearest final centroid.

    Implements the LinearTrajectory protocol since motion remains linear in time:
        p_i(t) = a_i + b_i * t
    """

    def __init__(
        self,
        base_positions: NDArray[Any],
        seed: int = 42,
        n_clusters: int = 5,
        speed: float = 1.0,
        noise: float = 0.1,
    ) -> None:
        """Initialize clustered drift model.

        Parameters
        ----------
        base_positions : NDArray[Any]
            Initial coordinates of n points in d dimensions, shape (n, d).
        seed : int, default=42
            Deterministic seed for clustering and velocity directions.
        n_clusters : int, default=5
            Number of kinematic clusters.
        speed : float, default=1.0
            Cluster velocity multiplier relative to nearest-neighbor scale.
        noise : float, default=0.1
            Idiosyncratic node velocity noise as a fraction of cluster speed.
        """
        arr = np.asarray(base_positions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim != 2:
            msg = f"base_positions must be 1D or 2D, got shape {arr.shape}"
            raise ValueError(msg)

        self._base: NDArray[np.float64] = arr
        self._seed = int(seed)
        self._n_clusters = max(1, int(n_clusters))
        self._speed = float(speed)
        self._noise = float(noise)
        self._n, self._dim = self._base.shape
        self._scale: float = mean_nn_distance(self._base, seed=self._seed)

        # Deterministic clustering via seeded k-means-lite
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, 0]))
        cluster_labels = self._compute_clusters(rng)

        # Generate cluster velocities and per-node noise
        if self._n > 0 and self._dim > 0:
            k = min(self._n_clusters, max(1, self._n))
            c_normals = rng.standard_normal((k, self._dim), dtype=np.float64)
            c_norms = np.linalg.norm(c_normals, axis=1, keepdims=True)
            c_norms[c_norms == 0.0] = 1.0
            c_dirs = c_normals / c_norms
            c_velocities = c_dirs * (self._speed * self._scale)

            # Node velocities from cluster velocity
            node_c_vel = c_velocities[cluster_labels]

            # Per-node noise velocity
            n_normals = rng.standard_normal((self._n, self._dim), dtype=np.float64)
            n_norms = np.linalg.norm(n_normals, axis=1, keepdims=True)
            n_norms[n_norms == 0.0] = 1.0
            n_dirs = n_normals / n_norms
            node_noise_vel = n_dirs * (self._noise * self._speed * self._scale)

            self._velocities: NDArray[np.float64] = node_c_vel + node_noise_vel
        else:
            self._velocities = np.zeros((self._n, self._dim), dtype=np.float64)

    def _compute_clusters(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Perform deterministic k-means-lite clustering."""
        if self._n == 0:
            return np.empty((0,), dtype=np.int64)
        if self._n <= self._n_clusters:
            return np.arange(self._n, dtype=np.int64)

        k = self._n_clusters
        # Sample initial centroids deterministically
        init_indices = rng.choice(self._n, size=k, replace=False)
        centroids = self._base[init_indices].copy()

        # Fixed 10 Lloyd iterations for deterministic convergence
        labels = np.zeros(self._n, dtype=np.int64)
        for _ in range(10):
            # Compute distances: (n, k)
            dists = np.linalg.norm(
                self._base[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
            )
            labels = np.argmin(dists, axis=1)

            # Update centroids
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centroids[j] = np.mean(self._base[mask], axis=0)

        return labels

    @property
    def name(self) -> str:
        """Descriptive model identifier."""
        return "cluster_drift"

    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed

    @property
    def speed(self) -> float:
        """Speed multiplier."""
        return self._speed

    @property
    def noise(self) -> float:
        """Noise ratio."""
        return self._noise

    @property
    def n_clusters(self) -> int:
        """Configured cluster count."""
        return self._n_clusters

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
            Target node identifier.

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
