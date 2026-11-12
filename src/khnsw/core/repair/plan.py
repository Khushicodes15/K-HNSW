"""Repair planning primitives for K-HNSW kinetic graph maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from khnsw.core.detection.cost import CostMeter
from khnsw.core.interfaces import MutableGraphIndex
from khnsw.core.repair.prune import prune, rank_pool


@dataclass(frozen=True)
class Rewrite:
    """Descriptor for a planned neighbor list rewrite on a single node and level.

    Attributes
    ----------
    node : int
        Source node identifier.
    level : int
        Hierarchy layer level.
    old : tuple[int, ...]
        Existing neighbor list prior to rewrite.
    new : tuple[int, ...]
        Planned replacement neighbor list after pruning.
    """

    node: int
    level: int
    old: tuple[int, ...]
    new: tuple[int, ...]


@dataclass(frozen=True)
class RepairPlan:
    """Immutable repair plan specifying rewrites and computational cost.

    Attributes
    ----------
    rule : str
        Heuristic repair rule identifier: "native", "mnru_beta", "mnru_gamma",
        "mnru_thn", or "khnsw".
    moved : int | None
        Identifier of the moved node triggering the repair, or None for purely
        violation-driven planning.
    rewrites : tuple[Rewrite, ...]
        Ordered sequence of distinct neighbor rewrites, sorted deterministically
        by (node, level).
    distance_computations : int
        Total number of distance metric evaluations incurred during plan formulation.
    """

    rule: str
    moved: int | None
    rewrites: tuple[Rewrite, ...]
    distance_computations: int


def native_plan(
    index: MutableGraphIndex,
    moved: int,
    *,
    ef: int | None = None,
    meter: CostMeter | None = None,
) -> RepairPlan:
    r"""Formulate a repair plan using hnswlib's native updatePoint neighbor repair logic.

    Algorithm Overview:
    -------------------
    For each layer level in 0..node_level(moved):
    1. L1 = N_l(moved) (out-neighbors of the moved node).
    2. Candidate pool = {moved} U L1 U \bigcup_{x \in L1} N_l(x).
    3. Repaired set = all v \in L1 in ascending node ID order.
    4. For each v \in Repaired set:
       - Candidate pool for v is (pool \ {v}).
       - Rank candidates via `rank_pool`.
       - Prune via `prune` with cap = max_degree(level), alpha = 1.0, skip_if_below_cap = True.
       - If set(new_neighbors) != set(old_neighbors), record a Rewrite.

    Deferred Out-Link Rebuild Note:
    -------------------------------
    In hnswlib, rebuilding the out-links of the moved node itself (N_l(moved)) is handled
    by `repairConnectionsForUpdate` via hierarchical beam search from the top layer.
    Since beam search and entry-point traversal belong to subsequent routing phases,
    out-link rebuilding for `moved` is deliberately deferred.

    Parameters
    ----------
    index : MutableGraphIndex
        Graph index instance (read-only snapshot during planning).
    moved : int
        Identifier of the moved node.
    ef : int | None, optional
        Exploration budget cap for candidate ranking.
    meter : CostMeter | None, optional
        Computational distance counter.

    Returns
    -------
    RepairPlan
        Immutable repair plan with rule="native".
    """
    effective_meter = meter if meter is not None else CostMeter()
    start_count = effective_meter.count

    moved_id = int(moved)
    max_lvl = index.node_level(moved_id)
    rewrites: list[Rewrite] = []

    for level in range(max_lvl + 1):
        l1_list = list(index.neighbors(moved_id, level))
        if not l1_list:
            continue

        pool: set[int] = {moved_id}
        pool.update(l1_list)
        for x in l1_list:
            pool.update(index.neighbors(x, level))

        repaired_nodes = sorted(set(l1_list))
        cap = index.max_degree(level)

        for v in repaired_nodes:
            cand_pool = pool - {v}
            ranked = rank_pool(
                base=v,
                pool=cand_pool,
                ef=ef,
                meter=effective_meter,
                vector_of=index.vector,
            )
            selected = prune(
                ranked=ranked,
                base=v,
                cap=cap,
                alpha=1.0,
                skip_if_below_cap=True,
                meter=effective_meter,
                vector_of=index.vector,
            )
            new_nbrs = tuple(cand_id for _, cand_id in selected)
            old_nbrs = tuple(index.neighbors(v, level))

            if set(new_nbrs) != set(old_nbrs):
                rewrites.append(Rewrite(node=v, level=level, old=old_nbrs, new=new_nbrs))

    rewrites.sort(key=lambda r: (r.node, r.level))
    dist_comps = effective_meter.count - start_count
    return RepairPlan(
        rule="native",
        moved=moved_id,
        rewrites=tuple(rewrites),
        distance_computations=dist_comps,
    )


def mnru_plan(
    index: MutableGraphIndex,
    moved: int,
    *,
    alpha: float = 1.0,
    scope: str = "one_hop",
    ef: int | None = None,
    meter: CostMeter | None = None,
) -> RepairPlan:
    r"""Formulate a repair plan using our re-implementation of MN-RU (Algorithm 2 & variants).

    Algorithm Overview:
    -------------------
    For each layer level in 0..node_level(moved):
    1. Scope candidates:
       - "one_hop": N_l(moved)
       - "two_hop": N_l(moved) U \bigcup_{x \in N_l(moved)} N_l(x)
    2. Mutual Neighbor Filtering:
       Repaired set = all v \in Scope such that moved \in N_l(v) (directed edge v -> moved exists).
       Assert moved \notin Repaired set.
       Repaired set sorted ascending by node ID.
    3. For each v \in Repaired set:
       - Candidate pool for v = ({moved} U N_l(moved) U N_l(v)) \ {v}.
       - Rank candidates via `rank_pool`.
       - Prune via `prune` with cap = max_degree(level), alpha = alpha, skip_if_below_cap = True.
       - If set(new_neighbors) != set(old_neighbors), record a Rewrite.

    Rule Classification:
    --------------------
    - alpha=1.0, scope="one_hop"  -> "mnru_beta"
    - alpha=1.1, scope="one_hop"  -> "mnru_gamma"
    - alpha=1.1, scope="two_hop"  -> "mnru_thn"

    Parameters
    ----------
    index : MutableGraphIndex
        Graph index instance (read-only snapshot during planning).
    moved : int
        Identifier of the moved node.
    alpha : float, default=1.0
        Heuristic pruning relaxation factor applied to squared Euclidean distances.
    scope : str, default="one_hop"
        Neighbor search scope ("one_hop" or "two_hop").
    ef : int | None, optional
        Exploration budget cap for candidate ranking.
    meter : CostMeter | None, optional
        Computational distance counter.

    Returns
    -------
    RepairPlan
        Immutable repair plan labeled with the appropriate MN-RU rule variant.
    """
    if scope not in ("one_hop", "two_hop"):
        raise ValueError(f"Invalid scope '{scope}', must be 'one_hop' or 'two_hop'.")

    # Determine rule name
    if alpha == 1.0 and scope == "one_hop":
        rule = "mnru_beta"
    elif alpha == 1.1 and scope == "one_hop":
        rule = "mnru_gamma"
    elif alpha == 1.1 and scope == "two_hop":
        rule = "mnru_thn"
    else:
        rule = f"mnru_{scope}_{alpha}"

    effective_meter = meter if meter is not None else CostMeter()
    start_count = effective_meter.count

    moved_id = int(moved)
    max_lvl = index.node_level(moved_id)
    rewrites: list[Rewrite] = []

    for level in range(max_lvl + 1):
        n_moved = set(index.neighbors(moved_id, level))
        if not n_moved:
            continue

        if scope == "one_hop":
            candidates_for_repair = set(n_moved)
        else:  # "two_hop"
            candidates_for_repair = set(n_moved)
            for x in n_moved:
                candidates_for_repair.update(index.neighbors(x, level))

        # Filter: only nodes v having a directed edge v -> moved
        repaired_nodes = sorted(
            v
            for v in candidates_for_repair
            if v != moved_id and moved_id in index.neighbors(v, level)
        )
        assert moved_id not in repaired_nodes

        cap = index.max_degree(level)

        for v in repaired_nodes:
            cand_pool = ({moved_id} | n_moved | set(index.neighbors(v, level))) - {v}
            ranked = rank_pool(
                base=v,
                pool=cand_pool,
                ef=ef,
                meter=effective_meter,
                vector_of=index.vector,
            )
            selected = prune(
                ranked=ranked,
                base=v,
                cap=cap,
                alpha=alpha,
                skip_if_below_cap=True,
                meter=effective_meter,
                vector_of=index.vector,
            )
            new_nbrs = tuple(cand_id for _, cand_id in selected)
            old_nbrs = tuple(index.neighbors(v, level))

            if set(new_nbrs) != set(old_nbrs):
                rewrites.append(Rewrite(node=v, level=level, old=old_nbrs, new=new_nbrs))

    rewrites.sort(key=lambda r: (r.node, r.level))
    dist_comps = effective_meter.count - start_count
    return RepairPlan(
        rule=rule,
        moved=moved_id,
        rewrites=tuple(rewrites),
        distance_computations=dist_comps,
    )


def khnsw_plan(
    index: MutableGraphIndex,
    violated: dict[tuple[int, int], Sequence[int]],
    *,
    ef: int | None = None,
    meter: CostMeter | None = None,
) -> RepairPlan:
    r"""Formulate a repair plan using K-HNSW certificate-driven localized rewiring.

    Algorithm Overview:
    -------------------
    For each (u, level) in sorted order of violated.keys():
    1. V = violated[(u, level)] (targets of violated outgoing edges from u).
    2. If V is empty, continue.
    3. Candidate pool = (N_l(u) U \bigcup_{v \in V} N_l(v)) \ {u}.
    4. Rank candidates via `rank_pool`.
    5. Prune via `prune` with cap = max_degree(level), alpha = 1.0, and
       CRITICALLY skip_if_below_cap = False.
       Disabling skip_if_below_cap guarantees that invalid/violated edges are actively pruned
       even if degree < cap.
    6. If set(new_neighbors) != set(old_neighbors), record a Rewrite.

    Parameters
    ----------
    index : MutableGraphIndex
        Graph index instance (read-only snapshot during planning).
    violated : dict[tuple[int, int], Sequence[int]]
        Mapping from (source_node, level) to the collection of violating target node IDs.
    ef : int | None, optional
        Exploration budget cap for candidate ranking.
    meter : CostMeter | None, optional
        Computational distance counter.

    Returns
    -------
    RepairPlan
        Immutable repair plan with rule="khnsw".
    """
    effective_meter = meter if meter is not None else CostMeter()
    start_count = effective_meter.count

    rewrites: list[Rewrite] = []

    for u, level in sorted(violated.keys()):
        targets = violated[(u, level)]
        if not targets:
            continue

        u_id = int(u)
        lvl = int(level)

        cand_pool = set(index.neighbors(u_id, lvl))
        for v in targets:
            cand_pool.update(index.neighbors(int(v), lvl))
        cand_pool.discard(u_id)

        cap = index.max_degree(lvl)
        ranked = rank_pool(
            base=u_id,
            pool=cand_pool,
            ef=ef,
            meter=effective_meter,
            vector_of=index.vector,
        )
        selected = prune(
            ranked=ranked,
            base=u_id,
            cap=cap,
            alpha=1.0,
            skip_if_below_cap=False,
            meter=effective_meter,
            vector_of=index.vector,
        )
        new_nbrs = tuple(cand_id for _, cand_id in selected)
        old_nbrs = tuple(index.neighbors(u_id, lvl))

        if set(new_nbrs) != set(old_nbrs):
            rewrites.append(Rewrite(node=u_id, level=lvl, old=old_nbrs, new=new_nbrs))

    rewrites.sort(key=lambda r: (r.node, r.level))
    dist_comps = effective_meter.count - start_count
    return RepairPlan(
        rule="khnsw",
        moved=None,
        rewrites=tuple(rewrites),
        distance_computations=dist_comps,
    )
