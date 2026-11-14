"""Node-level displacement tracking and lazy reinsertion scheduling.

Pure Python standard library and NumPy only (browser/Pyodide safe).
Tracks node drift relative to an anchor position and neighborhood radius.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from khnsw.core.detection.cost import CostMeter
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.search.heuristic import select_neighbors_heuristic


def in_neighbour_pre_repair(
    index: HNSWIndex, drifting_nodes: Sequence[int], static_set: set[int]
) -> int:
    """Re-select static in-neighbors' lists using standard heuristic.

    Returns the number of distance computations incurred.
    """
    heuristic_dists = 0
    for d_id in drifting_nodes:
        top_lvl = index.node_level(d_id)
        for lc in range(top_lvl + 1):
            static_in = [u for u in index.in_neighbors(d_id, lc) if u in static_set]
            d_nbrs = set(index.neighbors(d_id, lc))
            for u in static_in:
                u_nbrs = set(index.neighbors(u, lc))
                cands = (u_nbrs | d_nbrs) - {u, d_id}
                max_m = index.max_degree(lc)
                cand_list = list(cands)
                if cand_list:
                    heuristic_dists += len(cand_list)
                    pruned = select_neighbors_heuristic(index._store, u, cand_list, max_m, level=lc)
                    heuristic_dists += (len(pruned) * (len(pruned) - 1)) // 2
                    index._store.set_neighbors(u, lc, pruned)
    return heuristic_dists


def reinsert_node_fixed_level(index: HNSWIndex, node_id: int, pos: NDArray[Any]) -> int:
    """Delete and reinsert a single node with fixed level at pos.

    Returns distance computations incurred.
    """
    before_cost = index.distance_computations
    orig_lvl = index.node_level(node_id)
    v_node = np.asarray(pos, dtype=np.float32).copy()
    index.delete(node_id)

    orig_sample = index._level_gen.sample
    index._level_gen.sample = lambda lvl=orig_lvl: lvl  # type: ignore[method-assign]
    try:
        index._insert_one(node_id, v_node)
    finally:
        index._level_gen.sample = orig_sample  # type: ignore[method-assign]

    return index.distance_computations - before_cost


class NodeDisplacementTracker:
    """Tracks node movement and schedules lazy reinsertion based on displacement bounds.

    For each node i:
    - Anchor p_i: spatial coordinate vector at the time of initial registration or reinsertion.
    - Radius r_i: Euclidean distance from p_i to node i's farthest level-0 neighbor:
        r_i = max_{j in N_0(i)} ||p_i - x_j||_2
    - Staleness condition:
        stale(i, t) = ||x_i(t) - p_i||_2 > theta * r_i

    When stale(i, t) triggers:
    1. Node i is reinserted via index._insert_one with fixed layer level (optionally with
       static in-neighbor pre-repair).
    2. Anchor p_i is updated to x_i(t).
    3. Radius r_i is recomputed from the freshly established level-0 neighbors.
    """

    def __init__(
        self,
        index: HNSWIndex,
        theta: float = 1.0,
        pre_repair: bool = False,
        meter: CostMeter | None = None,
    ) -> None:
        """Initialize displacement tracker.

        Parameters
        ----------
        index : HNSWIndex
            Underlying hierarchical graph index.
        theta : float, default=1.0
            Staleness threshold multiplier relative to node radius.
        pre_repair : bool, default=False
            Whether to execute in-neighbor pre-repair on static nodes before reinserting.
        meter : CostMeter, optional
            Cost meter for tracking distance computations.
        """
        self._index = index
        self._theta = float(theta)
        self._pre_repair = bool(pre_repair)
        self._meter = meter if meter is not None else CostMeter()

        self._anchors: dict[int, NDArray[np.float32]] = {}
        self._radii: dict[int, float] = {}

    @property
    def index(self) -> HNSWIndex:
        """Underlying HNSW index."""
        return self._index

    @property
    def theta(self) -> float:
        """Staleness multiplier threshold theta."""
        return self._theta

    @property
    def pre_repair(self) -> bool:
        """Whether in-neighbor pre-repair is enabled."""
        return self._pre_repair

    @property
    def meter(self) -> CostMeter:
        """Cost meter tracking distance computations."""
        return self._meter

    def anchor(self, node_id: int) -> NDArray[np.float32]:
        """Return the current anchor position of node_id."""
        return self._anchors[int(node_id)]

    def radius(self, node_id: int) -> float:
        """Return the current radius of node_id."""
        return self._radii[int(node_id)]

    def register_node(self, node_id: int, pos: NDArray[Any] | None = None) -> None:
        """Register or reset the anchor and radius for node_id.

        Parameters
        ----------
        node_id : int
            Node identifier to register.
        pos : NDArray[Any], optional
            Anchor coordinate vector. If None, retrieves from index.
        """
        nid = int(node_id)
        if pos is None:
            p_vec = np.asarray(self._index.vector(nid), dtype=np.float32).copy()
        else:
            p_vec = np.asarray(pos, dtype=np.float32).copy()

        self._anchors[nid] = p_vec

        # Radius = distance from p_vec to farthest level-0 neighbor
        nbrs = list(self._index.neighbors(nid, level=0))
        if nbrs:
            max_r = 0.0
            for nb in nbrs:
                nb_vec = np.asarray(self._index.vector(nb), dtype=np.float32)
                d = float(np.linalg.norm(p_vec - nb_vec))
                if d > max_r:
                    max_r = d
            self._radii[nid] = max_r
        else:
            self._radii[nid] = 0.0

    def register_all(self, positions: NDArray[Any] | None = None) -> None:
        """Register anchors and radii for all nodes currently in the index."""
        for nid in self._index.ids():
            pos = positions[nid] if positions is not None else None
            self.register_node(nid, pos=pos)

    def is_stale(self, node_id: int, pos_t: NDArray[Any]) -> bool:
        """Check if node_id has drifted beyond its staleness threshold theta * radius.

        Incurs exactly 1 distance computation metered in CostMeter.

        Parameters
        ----------
        node_id : int
            Node identifier to check.
        pos_t : NDArray[Any]
            Current coordinates x_i(t).

        Returns
        -------
        bool
            True if ||pos_t - p_i|| > theta * r_i, False otherwise.
        """
        nid = int(node_id)
        p_anchor = self._anchors[nid]
        r_node = self._radii[nid]

        curr = np.asarray(pos_t, dtype=np.float32)
        # Metered distance calculation
        _ = self._meter.dist(p_anchor, curr)

        disp = float(np.linalg.norm(curr - p_anchor))
        return disp > (self._theta * r_node)

    def check_and_repair(
        self,
        current_positions: NDArray[Any],
        moving_nodes: Sequence[int],
        static_nodes: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate staleness only for moving nodes, and reinsert stale nodes.

        Parameters
        ----------
        current_positions : NDArray[Any]
            Coordinates of all nodes at current simulation timestamp, shape (N, D).
        moving_nodes : Sequence[int]
            List of node IDs that have non-zero velocity.
        static_nodes : Sequence[int], optional
            List of static node IDs for in-neighbor pre-repair. If None, inferred as
            complement of moving_nodes.

        Returns
        -------
        dict[str, Any]
            Execution summary containing:
            - stale_nodes: list of reinserted node IDs
            - reinsert_count: int
            - stale_check_cost: int
            - reinsert_cost: int
            - pre_repair_cost: int
            - total_cost: int
        """
        pos_mat = np.asarray(current_positions, dtype=np.float32)
        moving_list = [int(x) for x in moving_nodes]

        # Only nodes with non-zero velocity are evaluated
        stale_check_before = self._meter.count
        stale_nodes: list[int] = []
        for nid in moving_list:
            if self.is_stale(nid, pos_mat[nid]):
                stale_nodes.append(nid)

        stale_check_cost = self._meter.count - stale_check_before

        pre_repair_cost = 0
        reinsert_cost = 0

        if stale_nodes:
            if self._pre_repair:
                if static_nodes is not None:
                    static_set = set(int(x) for x in static_nodes)
                else:
                    moving_set = set(moving_list)
                    static_set = {i for i in self._index.ids() if i not in moving_set}
                pre_repair_cost = in_neighbour_pre_repair(self._index, stale_nodes, static_set)

            # Reinsert each stale node at current position with fixed level
            for nid in stale_nodes:
                c = reinsert_node_fixed_level(self._index, nid, pos_mat[nid])
                reinsert_cost += c
                # Reset anchor and radius after reinsertion
                self.register_node(nid, pos=pos_mat[nid])

        total_cost = stale_check_cost + pre_repair_cost + reinsert_cost

        return {
            "stale_nodes": stale_nodes,
            "reinsert_count": len(stale_nodes),
            "stale_check_cost": stale_check_cost,
            "pre_repair_cost": pre_repair_cost,
            "reinsert_cost": reinsert_cost,
            "total_cost": total_cost,
        }


class BudgetedPriorityTracker:
    """Tracks node displacement and reinserts top-B highest-scoring nodes each step.

    For each moving node i:
    - Anchor p_i: spatial coordinates at registration or last reinsertion.
    - Radius r_i: max distance to level-0 neighbors at registration or last reinsertion.
    - Score s_i = ||x_i(t) - p_i||_2 / r_i.

    At each step:
    - Checks displacement for moving nodes (1 distance check per moving node).
    - Sorts moving nodes by s_i descending (highest score first).
    - Reinserts at most budget B nodes (with score > min_score).
    - Updates anchor p_i <- x_i(t) and recomputes r_i for reinserted nodes.
    """

    def __init__(
        self,
        index: HNSWIndex,
        budget: int = 1,
        meter: CostMeter | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Initialize budgeted priority tracker.

        Parameters
        ----------
        index : HNSWIndex
            Underlying hierarchical graph index.
        budget : int, default=1
            Maximum number of nodes to reinsert per step (B).
        meter : CostMeter, optional
            Cost meter for tracking distance computations.
        min_score : float, default=0.0
            Minimum score threshold required for a node to be eligible for reinsertion.
        """
        self._index = index
        self._budget = max(0, int(budget))
        self._meter = meter if meter is not None else CostMeter()
        self._min_score = float(min_score)

        self._anchors: dict[int, NDArray[np.float32]] = {}
        self._radii: dict[int, float] = {}

    @property
    def index(self) -> HNSWIndex:
        """Underlying HNSW index."""
        return self._index

    @property
    def budget(self) -> int:
        """Reinsertion budget per step B."""
        return self._budget

    @property
    def meter(self) -> CostMeter:
        """Cost meter."""
        return self._meter

    def anchor(self, node_id: int) -> NDArray[np.float32]:
        """Return the current anchor position of node_id."""
        return self._anchors[int(node_id)]

    def radius(self, node_id: int) -> float:
        """Return the current radius of node_id."""
        return self._radii[int(node_id)]

    def register_node(self, node_id: int, pos: NDArray[Any] | None = None) -> None:
        """Register anchor and radius for node_id."""
        nid = int(node_id)
        if pos is None:
            p_vec = np.asarray(self._index.vector(nid), dtype=np.float32).copy()
        else:
            p_vec = np.asarray(pos, dtype=np.float32).copy()

        self._anchors[nid] = p_vec

        nbrs = list(self._index.neighbors(nid, level=0))
        if nbrs:
            max_r = 0.0
            for nb in nbrs:
                nb_vec = np.asarray(self._index.vector(nb), dtype=np.float32)
                d = float(np.linalg.norm(p_vec - nb_vec))
                if d > max_r:
                    max_r = d
            self._radii[nid] = max_r
        else:
            self._radii[nid] = 0.0

    def register_all(self, positions: NDArray[Any] | None = None) -> None:
        """Register anchors and radii for all nodes currently in the index."""
        for nid in self._index.ids():
            pos = positions[nid] if positions is not None else None
            self.register_node(nid, pos=pos)

    def compute_score(self, node_id: int, pos_t: NDArray[Any]) -> float:
        """Compute score s_i = ||x_i(t) - p_i||_2 / r_i for node_id.

        Incurs exactly 1 distance computation metered in CostMeter.
        """
        nid = int(node_id)
        p_anchor = self._anchors[nid]
        r_node = self._radii[nid]

        curr = np.asarray(pos_t, dtype=np.float32)
        _ = self._meter.dist(p_anchor, curr)

        disp = float(np.linalg.norm(curr - p_anchor))
        return disp / r_node if r_node > 0.0 else 0.0

    def step(
        self,
        current_positions: NDArray[Any],
        moving_nodes: Sequence[int],
    ) -> dict[str, Any]:
        """Execute one budgeted priority step.

        Parameters
        ----------
        current_positions : NDArray[Any]
            Positions at current timestamp, shape (N, D).
        moving_nodes : Sequence[int]
            List of node IDs that have non-zero velocity.

        Returns
        -------
        dict[str, Any]
            Execution summary containing:
            - reinserted_nodes: list of reinserted node IDs (ordered highest score first)
            - reinsert_count: int (at most budget B)
            - stale_check_cost: int
            - reinsert_cost: int
            - total_cost: int
        """
        pos_mat = np.asarray(current_positions, dtype=np.float32)
        moving_list = [int(x) for x in moving_nodes]

        stale_check_before = self._meter.count
        scores: list[tuple[float, int]] = []
        for nid in moving_list:
            s = self.compute_score(nid, pos_mat[nid])
            scores.append((s, nid))

        stale_check_cost = self._meter.count - stale_check_before

        # Sort descending by score (highest score first)
        scores.sort(key=lambda item: item[0], reverse=True)

        # Select at most budget B nodes with score > min_score
        eligible = [nid for s, nid in scores if s > self._min_score]
        selected = eligible[: self._budget]

        reinsert_cost = 0
        for nid in selected:
            c = reinsert_node_fixed_level(self._index, nid, pos_mat[nid])
            reinsert_cost += c
            # Anchor and radius reset after reinsertion -> score resets to 0
            self.register_node(nid, pos=pos_mat[nid])

        total_cost = stale_check_cost + reinsert_cost

        return {
            "reinserted_nodes": selected,
            "reinsert_count": len(selected),
            "stale_check_cost": stale_check_cost,
            "reinsert_cost": reinsert_cost,
            "total_cost": total_cost,
        }
