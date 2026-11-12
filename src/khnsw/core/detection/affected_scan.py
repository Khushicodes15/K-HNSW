"""Pure structural identification of graph edges affected by node motion and topology updates.

This module contains zero distance computations. It operates strictly on graph index topology
to compute the theoretical closed neighborhood of edges whose certificate validity can alter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from khnsw.core.interfaces import GraphIndex


def affected_after_move(index: GraphIndex, node: int) -> set[tuple[int, int, int]]:
    """Identify all directed edges whose certificate status can change after a node moves.

    According to the dependency claim, an edge (x, y, l) depends only on vec(x), vec(y),
    and vec(w) for w in N_l(x). After `index.move(node, ...)`, only vec(node) changes.
    The edges whose status can change are:
    (a) Every out-edge (node, x, l) for all levels l <= node_level(node).
    (b) Every in-edge (u, node, l) for u in in_neighbors(node, l).
    (c) Every other out-edge (u, v, l) of in-neighbors u in in_neighbors(node, l),
        because `node` is in N_l(u) and acts as a candidate witness.

    Parameters
    ----------
    index : GraphIndex
        Graph index containing current topology and reverse links.
    node : int
        Node ID of the point that was moved.

    Returns
    -------
    set[tuple[int, int, int]]
        Set of affected directed edge keys (u, v, level).
    """
    affected: set[tuple[int, int, int]] = set()
    node_id = int(node)
    top_lvl = index.node_level(node_id)

    for lvl in range(top_lvl + 1):
        # (a) All outgoing edges from the moved node
        for v in index.neighbors(node_id, lvl):
            affected.add((node_id, int(v), lvl))

        # (b) and (c) All outgoing edges of all in-neighbors of the moved node
        # (This naturally includes the in-edges (u, node, lvl) as well as sibling edges (u, v, lvl))
        for u in index.in_neighbors(node_id, lvl):
            u_int = int(u)
            # Explicitly ensure (b) in-edge is added
            affected.add((u_int, node_id, lvl))
            # (c) All out-edges of in-neighbor u
            for v in index.neighbors(u_int, lvl):
                affected.add((u_int, int(v), lvl))

    return affected


def affected_after_neighbor_change(
    index: GraphIndex,
    u: int,
    level: int,
) -> set[tuple[int, int, int]]:
    """Identify all directed edges whose certificate status can change after rewiring N_l(u).

    When N_l(u) is mutated via set_neighbors, add_edge, or remove_edge, no spatial
    coordinates change. The only edges whose status can change are the outgoing edges
    originating from u at the modified level.

    Parameters
    ----------
    index : GraphIndex
        Graph index containing current topology.
    u : int
        Source node ID whose neighbor list was modified.
    level : int
        Hierarchy layer level where the modification occurred.

    Returns
    -------
    set[tuple[int, int, int]]
        Set of affected directed edge keys (u, v, level).
    """
    u_int = int(u)
    lvl_int = int(level)
    return {(u_int, int(v), lvl_int) for v in index.neighbors(u_int, lvl_int)}
