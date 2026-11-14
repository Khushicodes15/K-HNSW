"""Unified public maintenance policy API for K-HNSW kinetic graph management.

Pure Python standard library and NumPy only (browser/Pyodide safe).
Provides a uniform, pluggable interface used by client modules, benchmarks,
and downstream agents (Members 4 and 5).

Supported policies:
- "none": Passive drift without intervention.
- "periodic": Reinserts all moving nodes every K steps.
- "round_robin": Reinserts the next B nodes in deterministic cyclic order each step.
- "lazy": Reinserts moving nodes whose displacement exceeds theta * r_i.
- "budgeted_priority": Reinserts the top B moving nodes with highest score ||x-p|| / r_i.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import EventEmitter
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.scheduling.tracker import (
    in_neighbour_pre_repair,
    reinsert_node_fixed_level,
)

VALID_POLICIES = frozenset({"none", "periodic", "round_robin", "lazy", "budgeted_priority"})


class MaintenancePolicy:
    """Configurable graph maintenance policy for kinetic drift scenarios.

    Parameters
    ----------
    policy : str
        Policy identifier in {"none", "periodic", "round_robin", "lazy", "budgeted_priority"}.
    K : int, default=1
        Step period multiplier for "periodic" policy (triggers when step % K == 0).
    B : int, default=1
        Maximum reinsertion budget per step for "round_robin" and "budgeted_priority".
    theta : float, default=1.0
        Departure threshold multiplier for "lazy" policy (triggers when ||x-p|| > theta * r).
    pre_repair : bool, default=False
        Whether to execute heuristic in-neighbor pre-repair on static nodes before reinsertion.
    seed : int, default=42
        Deterministic random seed for tie-breaking or reproducibility.
    meter : CostMeter, optional
        Cost meter tracking distance computations across categories.
    emitter : EventEmitter, optional
        Kinetic event emitter. Note: Because docs/contracts/events.md defines no
        'node_reinsert' or 'maintenance' event type, reinsertions do not emit
        uncontracted events to maintain schema conformity for downstream consumers.
    """

    def __init__(
        self,
        policy: str = "lazy",
        K: int = 1,
        B: int = 1,
        theta: float = 1.0,
        pre_repair: bool = False,
        seed: int = 42,
        meter: CostMeter | None = None,
        emitter: EventEmitter | None = None,
    ) -> None:
        if policy not in VALID_POLICIES:
            raise ValueError(f"Unknown policy '{policy}'. Must be one of {sorted(VALID_POLICIES)}")

        self._policy = policy
        self._K = max(1, int(K))
        self._B = max(0, int(B))
        self._theta = float(theta)
        self._pre_repair = bool(pre_repair)
        self._seed = int(seed)
        self._meter = meter if meter is not None else CostMeter()
        self._emitter = emitter

        self._anchors: dict[int, NDArray[np.float32]] = {}
        self._radii: dict[int, float] = {}
        self._step_count = 0
        self._rr_cursor = 0
        self._registered = False

    @property
    def policy(self) -> str:
        """Active policy identifier."""
        return self._policy

    @property
    def K(self) -> int:
        """Periodic interval K."""
        return self._K

    @property
    def B(self) -> int:
        """Budget per step B."""
        return self._B

    @property
    def theta(self) -> float:
        """Staleness threshold theta."""
        return self._theta

    @property
    def pre_repair(self) -> bool:
        """Whether in-neighbor pre-repair is enabled."""
        return self._pre_repair

    @property
    def meter(self) -> CostMeter:
        """Cost meter."""
        return self._meter

    @property
    def emitter(self) -> EventEmitter | None:
        """Event emitter."""
        return self._emitter

    def register(self, index: HNSWIndex, positions: NDArray[Any] | None = None) -> None:
        """Initialize anchors and neighborhood radii from graph state."""
        self._anchors.clear()
        self._radii.clear()
        for nid in index.ids():
            p_vec = (
                np.asarray(positions[nid], dtype=np.float32).copy()
                if positions is not None
                else np.asarray(index.vector(nid), dtype=np.float32).copy()
            )
            self._anchors[nid] = p_vec

            nbrs = list(index.neighbors(nid, level=0))
            if nbrs:
                max_r = 0.0
                for nb in nbrs:
                    nb_vec = np.asarray(index.vector(nb), dtype=np.float32)
                    d = float(np.linalg.norm(p_vec - nb_vec))
                    if d > max_r:
                        max_r = d
                self._radii[nid] = max_r
            else:
                self._radii[nid] = 0.0
        self._registered = True

    def _update_node_anchor(self, index: HNSWIndex, nid: int, pos: NDArray[Any]) -> None:
        """Refresh anchor and radius after reinsertion."""
        p_vec = np.asarray(pos, dtype=np.float32).copy()
        self._anchors[nid] = p_vec
        nbrs = list(index.neighbors(nid, level=0))
        if nbrs:
            max_r = max(float(np.linalg.norm(p_vec - index.vector(nb))) for nb in nbrs)
            self._radii[nid] = max_r
        else:
            self._radii[nid] = 0.0

    def step(
        self,
        index: HNSWIndex,
        positions: NDArray[Any],
        t: float = 0.0,
        moving_nodes: Sequence[int] | None = None,
    ) -> list[int]:
        """Advance maintenance policy by one simulation step.

        Parameters
        ----------
        index : HNSWIndex
            Underlying hierarchical graph index.
        positions : NDArray[Any]
            Current spatial coordinates of all nodes at timestamp t, shape (N, D).
        t : float, default=0.0
            Current simulation timestamp.
        moving_nodes : Sequence[int], optional
            Node IDs with non-zero velocities. If None, inferred by checking displacement
            from initial anchors (> 1e-6).

        Returns
        -------
        list[int]
            List of node IDs that were reinserted during this step.
        """
        pos_mat = np.asarray(positions, dtype=np.float32)
        n = len(pos_mat)

        if not self._registered:
            self.register(index, pos_mat)

        self._step_count += 1

        # Synchronize updated coordinates into index vector store
        for nid in range(n):
            index._store.set_vector(nid, pos_mat[nid])

        if moving_nodes is not None:
            moving_list = [int(x) for x in moving_nodes]
        else:
            moving_list = [
                nid
                for nid in range(n)
                if float(np.linalg.norm(pos_mat[nid] - self._anchors.get(nid, pos_mat[nid]))) > 1e-6
            ]

        moving_set = set(moving_list)
        static_set = {i for i in range(n) if i not in moving_set}

        reinserted: list[int] = []

        if self._policy == "none":
            return []

        elif self._policy == "periodic":
            if self._step_count % self._K == 0:
                reinserted = list(moving_list)

        elif self._policy == "round_robin":
            if self._B > 0 and moving_list:
                for _ in range(min(self._B, len(moving_list))):
                    chosen = moving_list[self._rr_cursor % len(moving_list)]
                    self._rr_cursor += 1
                    reinserted.append(chosen)

        elif self._policy == "lazy":
            stale_nodes = []
            for nid in moving_list:
                p_anch = self._anchors[nid]
                r_node = self._radii[nid]
                _ = self._meter.dist(p_anch, pos_mat[nid])
                disp = float(np.linalg.norm(pos_mat[nid] - p_anch))
                if disp > (self._theta * r_node):
                    stale_nodes.append(nid)
            reinserted = stale_nodes

        elif self._policy == "budgeted_priority":
            scores: list[tuple[float, int]] = []
            for nid in moving_list:
                p_anch = self._anchors[nid]
                r_node = self._radii[nid]
                _ = self._meter.dist(p_anch, pos_mat[nid])
                disp = float(np.linalg.norm(pos_mat[nid] - p_anch))
                s = disp / r_node if r_node > 0.0 else 0.0
                scores.append((s, nid))

            scores.sort(key=lambda item: item[0], reverse=True)
            eligible = [nid for s, nid in scores if s > 0.0]
            reinserted = eligible[: self._B]

        # Execute reinsertion maintenance
        if reinserted:
            if self._pre_repair:
                _ = in_neighbour_pre_repair(index, reinserted, static_set)

            for nid in reinserted:
                _ = reinsert_node_fixed_level(index, nid, pos_mat[nid])
                self._update_node_anchor(index, nid, pos_mat[nid])

        return reinserted
