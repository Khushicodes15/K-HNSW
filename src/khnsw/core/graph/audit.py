"""Graph audit and structural invariant verification.

Pure Python standard library and NumPy only (browser/Pyodide safe).
Inspects GraphIndex implementations using strictly public query interfaces.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable


def _resolve_ids(index: Any, ids: Iterable[int] | None = None) -> list[int]:
    """Extract a list of unique integer node IDs from explicit argument or index.ids()."""
    if ids is not None:
        return [int(x) for x in ids]
    if hasattr(index, "ids") and callable(index.ids):
        return [int(x) for x in index.ids()]
    raise ValueError(
        "Node IDs must be explicitly provided when the index does not expose an ids() method."
    )


def degree_stats(
    index: Any,
    ids: Iterable[int] | None = None,
) -> dict[int, dict[str, float | int]]:
    """Compute per-level out-degree statistics (min, max, mean, count).

    Parameters
    ----------
    index : GraphIndex
        Target graph index exposing ``max_level``, ``node_level``, and ``neighbors``.
    ids : Iterable[int] | None, optional
        Iterable of node IDs to analyze. If None, queries ``index.ids()``.

    Returns
    -------
    dict[int, dict[str, float | int]]
        Mapping of layer level to summary stats:
        ``{"count": int, "min": int, "max": int, "mean": float}``.
    """
    id_list = _resolve_ids(index, ids)
    max_lvl = index.max_level()
    stats: dict[int, dict[str, float | int]] = {}

    if max_lvl < 0 or not id_list:
        return stats

    for lvl in range(max_lvl + 1):
        nodes_at_lvl = [nid for nid in id_list if index.node_level(nid) >= lvl]
        count = len(nodes_at_lvl)
        if count == 0:
            stats[lvl] = {"count": 0, "min": 0, "max": 0, "mean": 0.0}
            continue

        degs = [len(index.neighbors(nid, lvl)) for nid in nodes_at_lvl]
        stats[lvl] = {
            "count": count,
            "min": min(degs),
            "max": max(degs),
            "mean": float(sum(degs) / count),
        }

    return stats


def check_degree_bounds(
    index: Any,
    ids: Iterable[int] | None = None,
    m: int | None = None,
    m0: int | None = None,
) -> list[str]:
    """Check out-degree upper bounds for each node across all applicable levels.

    Parameters
    ----------
    index : GraphIndex
        Target graph index exposing ``max_level``, ``node_level``, and ``neighbors``.
    ids : Iterable[int] | None, optional
        Iterable of node IDs to evaluate. If None, queries ``index.ids()``.
    m : int | None, optional
        Maximum allowable out-degree for upper layers (level > 0). If None, attempts
        to read from ``index.m`` or ``index.max_degree(level)``.
    m0 : int | None, optional
        Maximum allowable out-degree for base layer (level 0). If None, attempts
        to read from ``index.m0`` or ``index.max_degree(0)``.

    Returns
    -------
    list[str]
        List of violation descriptions (empty if all degrees are within bounds).
    """
    id_list = _resolve_ids(index, ids)
    max_lvl = index.max_level()
    violations: list[str] = []

    if max_lvl < 0 or not id_list:
        return violations

    resolved_m0 = m0 if m0 is not None else getattr(index, "m0", None)
    resolved_m = m if m is not None else getattr(index, "m", None)

    for lvl in range(max_lvl + 1):
        if lvl == 0:
            bound = (
                resolved_m0
                if resolved_m0 is not None
                else (index.max_degree(0) if hasattr(index, "max_degree") else None)
            )
        else:
            bound = (
                resolved_m
                if resolved_m is not None
                else (index.max_degree(lvl) if hasattr(index, "max_degree") else None)
            )

        if bound is None:
            raise ValueError(
                f"Degree bound for level {lvl} could not be determined. Pass m and m0 explicitly."
            )

        for nid in id_list:
            if index.node_level(nid) >= lvl:
                deg = len(index.neighbors(nid, lvl))
                if deg > bound:
                    violations.append(
                        f"Degree bound violation: node {nid} at level {lvl} "
                        f"has out-degree {deg} > max allowed {bound}"
                    )

    return violations


def zero_in_degree_count(
    index: Any,
    ids: Iterable[int] | None = None,
    level: int = 0,
) -> int:
    """Count the number of nodes at a specific level that have zero incoming edges.

    Parameters
    ----------
    index : GraphIndex
        Target graph index exposing ``node_level`` and ``in_neighbors``.
    ids : Iterable[int] | None, optional
        Iterable of node IDs to evaluate. If None, queries ``index.ids()``.
    level : int, optional
        Target hierarchy layer level (default 0).

    Returns
    -------
    int
        Count of nodes active at ``level`` with ``len(in_neighbors(nid, level)) == 0``.
    """
    id_list = _resolve_ids(index, ids)
    count = 0
    for nid in id_list:
        if index.node_level(nid) >= level:
            if len(index.in_neighbors(nid, level)) == 0:
                count += 1
    return count


def reachable_from_entry(
    index: Any,
    ids: Iterable[int] | None = None,
    level: int = 0,
) -> set[int]:
    """Compute the set of nodes reachable via directed forward edges from the entry point.

    Entry Point Treatment at Lower Levels:
        In HNSW, ``entry_point()`` designates the root entry node at the top layer
        (``max_level``). By the layer-hierarchy invariant, any node present at layer
        :math:`L` is guaranteed to exist at all layers :math:`0 \\le l \\le L`.
        Therefore, for any search level :math:`\\le \\text{max\\_level}`, the top-layer
        entry point itself serves as the natural root node at that level.

    Parameters
    ----------
    index : GraphIndex
        Target graph index exposing ``entry_point``, ``max_level``, ``node_level``,
        and ``neighbors``.
    ids : Iterable[int] | None, optional
        Iterable of node IDs in the graph. Unused during forward traversal but
        validated if provided.
    level : int, optional
        Hierarchy layer level to traverse (default 0).

    Returns
    -------
    set[int]
        Set of node identifiers reachable by following forward edges at ``level``.
        Returns an empty set if the graph is empty or if ``level > max_level``.
    """
    ep = index.entry_point()
    max_lvl = index.max_level()

    if ep is None or max_lvl < 0 or level > max_lvl:
        return set()

    # The entry point must exist at `level`
    if index.node_level(ep) < level:
        return set()

    visited: set[int] = {ep}
    queue: deque[int] = deque([ep])

    while queue:
        curr = queue.popleft()
        for nbr in index.neighbors(curr, level):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    return visited


def connected_components(
    index: Any,
    ids: Iterable[int] | None = None,
    level: int = 0,
) -> list[set[int]]:
    """Compute weakly connected components of the graph at a specified level.

    Weakly Connected Components:
        Treats the directed graph as undirected: an edge :math:`u \\to v` connects
        :math:`u` and :math:`v` in both directions. In a valid HNSW graph where
        bidirectional edge consistency holds (:math:`u \\in \\text{in\\_neighbors}(v)
        \\iff v \\in \\text{neighbors}(u)`), weak connectivity coincides with standard
        undirected graph connectivity.

    Stack Safety:
        Implemented iteratively using a FIFO queue (breadth-first search) to avoid
        recursion depth limits and guarantee safe execution in WebAssembly / Pyodide.

    Parameters
    ----------
    index : GraphIndex
        Target graph index exposing ``node_level``, ``neighbors``, and ``in_neighbors``.
    ids : Iterable[int] | None, optional
        Iterable of node IDs to analyze. If None, queries ``index.ids()``.
    level : int, optional
        Hierarchy layer level to analyze (default 0).

    Returns
    -------
    list[set[int]]
        List of disjoint sets of node IDs, each representing a connected component
        among nodes present at ``level``.
    """
    id_list = _resolve_ids(index, ids)
    nodes_at_level = {nid for nid in id_list if index.node_level(nid) >= level}

    visited: set[int] = set()
    components: list[set[int]] = []

    for nid in sorted(nodes_at_level):
        if nid in visited:
            continue

        comp: set[int] = {nid}
        visited.add(nid)
        queue: deque[int] = deque([nid])

        while queue:
            curr = queue.popleft()
            # In an undirected sense, traverse both forward and incoming edges
            fwd_nbrs = index.neighbors(curr, level)
            in_nbrs = index.in_neighbors(curr, level)
            adjacent = set(fwd_nbrs) | set(in_nbrs)

            for nbr in adjacent:
                if nbr in nodes_at_level and nbr not in visited:
                    visited.add(nbr)
                    comp.add(nbr)
                    queue.append(nbr)

        components.append(comp)

    return components


def validate_invariants(
    index: Any,
    ids: Iterable[int] | None = None,
    m: int | None = None,
    m0: int | None = None,
) -> list[str]:
    """Validate all formal structural invariants of the HNSW graph index.

    Checks:
        1. Entry-point validity:
           - Empty graph: ``entry_point is None`` and ``max_level == -1``.
           - Non-empty graph: ``entry_point`` in node set and has ``node_level == max_level``.
           - ``max_level`` equals the highest ``node_level`` across all indexed nodes.
        2. Layer-hierarchy invariant:
           - Node level in bounds: :math:`0 \\le \\text{node\\_level}(u) \\le \\text{max\\_level}`.
           - Every node is defined across all layers :math:`0 \\le l \\le \\text{node\\_level}(u)`.
        3. Degree bounds & self-loop prevention:
           - Out-degree :math:`\\le M_0` at level 0, and :math:`\\le M` at levels :math:`l > 0`.
           - No self-loops: :math:`u \\notin \\text{neighbors}(u, l)`.
        4. Forward/reverse edge consistency (bidirectional pairing):
           - :math:`u \\in \\text{in\\_neighbors}(v, l) \\iff v \\in \\text{neighbors}(u, l)`.
           - Target and source nodes exist and have maximum level :math:`\\ge l`.

    Parameters
    ----------
    index : GraphIndex
        Target graph index implementing the ``GraphIndex`` contract.
    ids : Iterable[int] | None, optional
        Iterable of active node IDs. If None, queries ``index.ids()``.
    m : int | None, optional
        Maximum upper-layer out-degree bound. Inferred from index if None.
    m0 : int | None, optional
        Maximum base-layer out-degree bound. Inferred from index if None.

    Returns
    -------
    list[str]
        List of formatted violation strings. An empty list signifies full compliance.
    """
    id_list = _resolve_ids(index, ids)
    id_set = set(id_list)
    max_lvl = index.max_level()
    ep = index.entry_point()
    violations: list[str] = []

    # 1. Empty graph invariants
    if not id_list:
        if max_lvl != -1:
            violations.append(
                f"Entry point invariant violation: empty index must have max_level == -1, "
                f"got {max_lvl}"
            )
        if ep is not None:
            violations.append(
                f"Entry point invariant violation: empty index must have entry_point == None, "
                f"got {ep}"
            )
        return violations

    # 2. Non-empty entry point validity
    if ep is None:
        violations.append(
            "Entry point invariant violation: non-empty index has entry_point == None"
        )
    else:
        if ep not in id_set:
            violations.append(
                f"Entry point invariant violation: entry_point {ep} not found in index node set"
            )
        else:
            ep_level = index.node_level(ep)
            if ep_level != max_lvl:
                violations.append(
                    f"Entry point invariant violation: entry_point {ep} level {ep_level} "
                    f"!= max_level {max_lvl}"
                )

    if max_lvl < 0:
        violations.append(
            f"Entry point invariant violation: non-empty index has negative max_level {max_lvl}"
        )
    else:
        # Check max_level matches highest node level
        actual_highest_level = max(index.node_level(nid) for nid in id_list)
        if max_lvl != actual_highest_level:
            violations.append(
                f"Entry point invariant violation: max_level {max_lvl} "
                f"!= highest node level {actual_highest_level}"
            )

    # 3. Layer hierarchy & node levels
    for nid in id_list:
        nl = index.node_level(nid)
        if nl < 0 or nl > max_lvl:
            violations.append(
                f"Layer hierarchy violation: node {nid} level {nl} out of bounds [0, {max_lvl}]"
            )

    # 4. Degree bounds
    degree_violations = check_degree_bounds(index, id_list, m=m, m0=m0)
    violations.extend(degree_violations)

    # 5. Self-loops, target validity, and bidirectional edge consistency
    for nid in id_list:
        nl = index.node_level(nid)
        for lvl in range(nl + 1):
            try:
                nbrs = index.neighbors(nid, lvl)
            except Exception as e:
                violations.append(
                    f"Layer hierarchy violation: failed to query neighbors({nid}, {lvl}): {e}"
                )
                continue

            try:
                in_nbrs = index.in_neighbors(nid, lvl)
            except Exception as e:
                violations.append(
                    f"Layer hierarchy violation: failed to query in_neighbors({nid}, {lvl}): {e}"
                )
                continue

            # Check for self-loops
            if nid in nbrs:
                violations.append(
                    f"Topology violation: self-loop detected on node {nid} at level {lvl}"
                )

            # Check forward edges: for each v in neighbors(nid, lvl), nid in in_neighbors(v, lvl)
            for v in nbrs:
                if v not in id_set:
                    violations.append(
                        f"Topology violation: node {nid} has neighbor {v} at level {lvl} "
                        "which does not exist in graph"
                    )
                else:
                    v_nl = index.node_level(v)
                    if v_nl < lvl:
                        violations.append(
                            f"Layer hierarchy violation: edge {nid}->{v} at level {lvl} "
                            f"but target {v} max level is {v_nl} < {lvl}"
                        )
                    else:
                        v_in = index.in_neighbors(v, lvl)
                        if nid not in v_in:
                            violations.append(
                                f"Bidirectional consistency violation: edge {nid}->{v} at level "
                                f"{lvl} missing reverse link in in_neighbors({v}, {lvl})"
                            )

            # Check reverse edges: for each u in in_neighbors(nid, lvl), nid in neighbors(u, lvl)
            for u in in_nbrs:
                if u not in id_set:
                    violations.append(
                        f"Topology violation: node {nid} has in_neighbor {u} at level {lvl} "
                        "which does not exist in graph"
                    )
                else:
                    u_nl = index.node_level(u)
                    if u_nl < lvl:
                        violations.append(
                            f"Layer hierarchy violation: reverse edge {u}->{nid} at level {lvl} "
                            f"but source {u} max level is {u_nl} < {lvl}"
                        )
                    else:
                        u_fwd = index.neighbors(u, lvl)
                        if nid not in u_fwd:
                            violations.append(
                                f"Bidirectional consistency violation: reverse link {u}->{nid} "
                                f"at level {lvl} missing forward link in neighbors({u}, {lvl})"
                            )

    return violations
