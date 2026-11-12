"""Kinetic certificate store tracking topology validity and witness certificates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from khnsw.core.detection.affected_scan import affected_after_move
from khnsw.core.detection.certificates import is_violated_static

if TYPE_CHECKING:
    from khnsw.core.detection.cost import CostMeter
    from khnsw.core.interfaces import GraphIndex


class EdgeStatus(str, Enum):
    """Static kinetic certificate validity status."""

    VALID = "VALID"
    VIOLATED = "VIOLATED"


@dataclass(frozen=True)
class Transition:
    """Status transition event observed on a directed edge.

    Attributes
    ----------
    edge : tuple[int, int, int]
        Directed edge identifier (u, v, level).
    old : str | None
        Previous validity status ("VALID", "VIOLATED", or None for newly appeared edges).
    new : str
        New validity status ("VALID" or "VIOLATED").
    witness : int | None
        Violating witness node ID, or None if the edge is VALID.
    kind : str
        Semantic classification ("failure", "healed", "appeared", "unchanged").
    """

    edge: tuple[int, int, int]
    old: str | None
    new: str
    witness: int | None
    kind: str = "unchanged"

    def __post_init__(self) -> None:
        """Infer transition kind if default unchanged was specified."""
        if self.kind == "unchanged":
            inferred = "unchanged"
            if self.old is None:
                inferred = "appeared"
            elif self.old == EdgeStatus.VALID.value and self.new == EdgeStatus.VIOLATED.value:
                inferred = "failure"
            elif self.old == EdgeStatus.VIOLATED.value and self.new == EdgeStatus.VALID.value:
                inferred = "healed"
            object.__setattr__(self, "kind", inferred)


class CertificateStore:
    """In-memory cache and tracker for directed edge certificates and witnesses.

    Maintains the status (VALID or VIOLATED) and canonical witness of every active edge
    in a GraphIndex. Provides incremental update routines following node movements,
    neighbor rewiring, and deletions.
    """

    def __init__(self) -> None:
        """Initialize empty certificate store."""
        # edge (u, v, level) -> "VALID" or "VIOLATED"
        self._edge_status: dict[tuple[int, int, int], str] = {}
        # edge (u, v, level) -> witness ID or None
        self._edge_witness: dict[tuple[int, int, int], int | None] = {}
        self._baseline_total: int = 0
        self._baseline_violated: int = 0

    def build(self, index: GraphIndex, meter: CostMeter) -> list[Transition]:
        """Certify every edge at every level of the graph to establish the initial baseline.

        Baseline violations resulting from standard asymmetric reverse-link appending
        are recorded as legacy state and generate NO failure events.

        Parameters
        ----------
        index : GraphIndex
            Initial graph index.
        meter : CostMeter
            Cost meter for distance computation accounting.

        Returns
        -------
        list[Transition]
            Empty list, as legacy baseline violations produce no failure transitions.
        """
        self._edge_status.clear()
        self._edge_witness.clear()

        top_lvl = index.max_level()
        if top_lvl < 0:
            self._baseline_total = 0
            self._baseline_violated = 0
            return []

        sorted_ids = sorted(index.ids())
        for lvl in range(top_lvl + 1):
            for u in sorted_ids:
                if index.node_level(u) >= lvl:
                    self.certify_source(index, u, lvl, meter)

        self._baseline_total = len(self._edge_status)
        self._baseline_violated = sum(
            1 for s in self._edge_status.values() if s == EdgeStatus.VIOLATED.value
        )
        return []

    def certify_source(
        self,
        index: GraphIndex,
        u: int,
        level: int,
        meter: CostMeter,
    ) -> dict[tuple[int, int, int], tuple[str, int | None]]:
        """Certify all outgoing edges originating from node u at layer level.

        Distance Accounting:
        --------------------
        Let k = len(N_l(u)) be the out-degree of u at layer level.
        1. Source-to-neighbor distances: D(u, w) is computed for each w in N_l(u) -> k distances.
        2. Pairwise neighbor distances: D(v, w) is computed for each distinct pair {v, w} in N_l(u)
           -> k * (k - 1) / 2 distances.
        Total distances computed: D(k) = k + k*(k-1)/2 = k*(k+1)/2.

        Parameters
        ----------
        index : GraphIndex
            Graph index providing vectors and neighbor lists.
        u : int
            Source node ID.
        level : int
            Hierarchy layer level.
        meter : CostMeter
            Cost meter for distance computation accounting.

        Returns
        -------
        dict[tuple[int, int, int], tuple[str, int | None]]
            Dictionary mapping each edge (u, v, level) to (status, witness).
        """
        u_int = int(u)
        lvl_int = int(level)
        nbrs = list(index.neighbors(u_int, lvl_int))
        k = len(nbrs)

        if k == 0:
            return {}

        # 1. Compute D(u, w) for each w in N_l(u)
        vec_u = index.vector(u_int)
        d_u: dict[int, float] = {}
        for w in nbrs:
            d_u[w] = meter.dist(vec_u, index.vector(w))

        # 2. Compute D(v, w) for all unordered pairs {v, w} in N_l(u)
        d_pair: dict[tuple[int, int], float] = {}
        for i in range(k):
            v_node = nbrs[i]
            vec_v = index.vector(v_node)
            for j in range(i + 1, k):
                w_node = nbrs[j]
                pair_key = (min(v_node, w_node), max(v_node, w_node))
                d_pair[pair_key] = meter.dist(vec_v, index.vector(w_node))

        # 3. Evaluate certificate condition for each directed edge (u, v, level)
        results: dict[tuple[int, int, int], tuple[str, int | None]] = {}
        for v in nbrs:
            edge_key = (u_int, v, lvl_int)
            d_uv = d_u[v]

            violating_witnesses: list[int] = []
            for w in nbrs:
                if w == v:
                    continue
                d_uw = d_u[w]
                pair_key = (min(v, w), max(v, w))
                d_vw = d_pair[pair_key]

                if is_violated_static(d_uv, d_uw, d_vw):
                    violating_witnesses.append(w)

            if violating_witnesses:
                status = EdgeStatus.VIOLATED.value
                witness = min(violating_witnesses)  # deterministic canonical witness
            else:
                status = EdgeStatus.VALID.value
                witness = None

            self._edge_status[edge_key] = status
            self._edge_witness[edge_key] = witness
            results[edge_key] = (status, witness)

        return results

    def on_move(
        self,
        index: GraphIndex,
        node: int,
        meter: CostMeter,
    ) -> list[Transition]:
        """Incremental update triggered after index.move(node, ...) has occurred.

        Recertifies exactly the sources identified by affected_after_move and classifies
        status transitions.

        Parameters
        ----------
        index : GraphIndex
            Graph index with updated node coordinates.
        node : int
            Node ID that was moved.
        meter : CostMeter
            Cost meter for distance computation accounting.

        Returns
        -------
        list[Transition]
            Sorted list of failure and healed transition events.
        """
        node_id = int(node)
        affected = affected_after_move(index, node_id)
        if not affected:
            return []

        # Snapshot pre-move status of affected edges
        old_state: dict[tuple[int, int, int], tuple[str | None, int | None]] = {
            edge: (self._edge_status.get(edge), self._edge_witness.get(edge)) for edge in affected
        }

        # Identify unique source pairs (u, level) to recertify in deterministic order
        sources = sorted({(u, lvl) for (u, _, lvl) in affected})
        for u, lvl in sources:
            self.certify_source(index, u, lvl, meter)

        transitions: list[Transition] = []
        for edge in sorted(affected):
            old_s, _ = old_state[edge]
            new_s = self._edge_status.get(edge)
            new_w = self._edge_witness.get(edge)

            if new_s is not None and old_s != new_s:
                if old_s == EdgeStatus.VALID.value and new_s == EdgeStatus.VIOLATED.value:
                    transitions.append(
                        Transition(
                            edge=edge,
                            old=old_s,
                            new=new_s,
                            witness=new_w,
                            kind="failure",
                        )
                    )
                elif old_s == EdgeStatus.VIOLATED.value and new_s == EdgeStatus.VALID.value:
                    transitions.append(
                        Transition(
                            edge=edge,
                            old=old_s,
                            new=new_s,
                            witness=new_w,
                            kind="healed",
                        )
                    )

        return transitions

    def on_neighbor_change(
        self,
        index: GraphIndex,
        u: int,
        level: int,
        meter: CostMeter,
    ) -> list[Transition]:
        """Incremental update after set_neighbors, add_edge, or remove_edge on (u, level).

        Parameters
        ----------
        index : GraphIndex
            Graph index containing modified topology.
        u : int
            Source node ID whose neighbor list was changed.
        level : int
            Hierarchy layer level.
        meter : CostMeter
            Cost meter for distance computation accounting.

        Returns
        -------
        list[Transition]
            List of transition events (new edges reported as appeared, status changes reported).
        """
        u_int = int(u)
        lvl_int = int(level)

        curr_nbrs = set(index.neighbors(u_int, lvl_int))
        curr_edges = {(u_int, int(v), lvl_int) for v in curr_nbrs}

        prev_edges = {e for e in self._edge_status if e[0] == u_int and e[2] == lvl_int}

        # 1. Drop removed edges
        removed = prev_edges - curr_edges
        for e in removed:
            self._edge_status.pop(e, None)
            self._edge_witness.pop(e, None)

        # 2. Snapshot old state for persisting edges
        old_state: dict[tuple[int, int, int], str] = {
            e: self._edge_status[e] for e in prev_edges & curr_edges
        }

        # 3. Recertify source (u, level)
        self.certify_source(index, u_int, lvl_int, meter)

        transitions: list[Transition] = []
        for e in sorted(curr_edges):
            if e not in prev_edges:
                # Newly added edge: reported as appeared with initial status
                transitions.append(
                    Transition(
                        edge=e,
                        old=None,
                        new=self._edge_status[e],
                        witness=self._edge_witness[e],
                        kind="appeared",
                    )
                )
            else:
                old_s = old_state[e]
                new_s = self._edge_status[e]
                new_w = self._edge_witness[e]
                if old_s != new_s:
                    if old_s == EdgeStatus.VALID.value and new_s == EdgeStatus.VIOLATED.value:
                        transitions.append(
                            Transition(
                                edge=e,
                                old=old_s,
                                new=new_s,
                                witness=new_w,
                                kind="failure",
                            )
                        )
                    elif old_s == EdgeStatus.VIOLATED.value and new_s == EdgeStatus.VALID.value:
                        transitions.append(
                            Transition(
                                edge=e,
                                old=old_s,
                                new=new_s,
                                witness=new_w,
                                kind="healed",
                            )
                        )

        return transitions

    def on_remove(
        self,
        index: GraphIndex,
        node: int,
        meter: CostMeter,
    ) -> list[Transition]:
        """Incremental update triggered AFTER node has been removed from index.

        Design Choice:
        --------------
        Assumes `index.remove(node)` has already executed on the graph index. The store
        uses its cached topology to locate all in-neighbors that had `node` in their
        neighbor lists, purges all edges incident to `node`, and recertifies surviving
        sources.

        Parameters
        ----------
        index : GraphIndex
            Graph index with node already removed.
        node : int
            Node ID that was removed.
        meter : CostMeter
            Cost meter for distance computation accounting.

        Returns
        -------
        list[Transition]
            List of status transitions among surviving affected edges.
        """
        node_id = int(node)

        # Sources that had directed edge (u, node, lvl) in the store
        affected_sources = sorted({(e[0], e[2]) for e in self._edge_status if e[1] == node_id})

        # Purge all incident edges
        incident = [e for e in self._edge_status if e[0] == node_id or e[1] == node_id]
        for e in incident:
            self._edge_status.pop(e, None)
            self._edge_witness.pop(e, None)

        transitions: list[Transition] = []
        for u, lvl in affected_sources:
            if not index.contains(u):
                continue

            prev_out = {e for e in self._edge_status if e[0] == u and e[2] == lvl}
            old_state = {e: self._edge_status[e] for e in prev_out}

            self.certify_source(index, u, lvl, meter)

            for e in sorted(prev_out):
                if e not in self._edge_status:
                    continue
                old_s = old_state[e]
                new_s = self._edge_status[e]
                new_w = self._edge_witness[e]
                if old_s != new_s:
                    if old_s == EdgeStatus.VALID.value and new_s == EdgeStatus.VIOLATED.value:
                        transitions.append(
                            Transition(
                                edge=e,
                                old=old_s,
                                new=new_s,
                                witness=new_w,
                                kind="failure",
                            )
                        )
                    elif old_s == EdgeStatus.VIOLATED.value and new_s == EdgeStatus.VALID.value:
                        transitions.append(
                            Transition(
                                edge=e,
                                old=old_s,
                                new=new_s,
                                witness=new_w,
                                kind="healed",
                            )
                        )

        return transitions

    def status(self, edge: tuple[int, int, int]) -> str:
        """Return the current certificate status ("VALID" or "VIOLATED") of an edge."""
        return self._edge_status[edge]

    def witness(self, edge: tuple[int, int, int]) -> int | None:
        """Return the canonical violating witness ID, or None if the edge is VALID."""
        return self._edge_witness[edge]

    def counts(self) -> dict[str, int]:
        """Return counts of total, valid, and violated edges currently in the store."""
        total = len(self._edge_status)
        violated = sum(1 for s in self._edge_status.values() if s == EdgeStatus.VIOLATED.value)
        return {
            "total": total,
            "valid": total - violated,
            "violated": violated,
        }

    def baseline_violation_rate(self) -> float:
        """Return the fraction of baseline edges that were violated at initial build time."""
        if self._baseline_total == 0:
            return 0.0
        return float(self._baseline_violated) / float(self._baseline_total)
