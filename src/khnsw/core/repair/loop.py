"""Iterative certificate-driven repair loop for K-HNSW graphs.

Executes localized topological rewiring for confirmed certificate failures and moved nodes,
preserving structural invariants, reverse-edge integrity, and version semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from khnsw.core.detection.certificate_store import EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.plan import RepairPlan, Rewrite, khnsw_plan
from khnsw.core.repair.prune import prune, rank_pool

if TYPE_CHECKING:
    from khnsw.core.detection.certificate_store import CertificateStore, Transition
    from khnsw.core.detection.kinetic import KineticDetector
    from khnsw.core.events.schema import EventEmitter
    from khnsw.core.interfaces import MutableGraphIndex


@dataclass(frozen=True)
class RepairResult:
    """Outcome summary of a repair loop execution.

    Attributes
    ----------
    rounds : int
        Total number of repair rounds executed.
    edges_repaired : int
        Total count of unique violated directed edges successfully repaired.
    still_violated : set[tuple[int, int, int]]
        Set of directed edges (u, v, level) that remain violated at termination.
    budget_used : dict[str, int]
        Distance computations partitioned by category:
        {"search": int, "planning": int, "store": int, "detector": int, "reverse_link": int}.
    stopped_reason : str
        Reason for termination: "converged", "budget", or "max_rounds".
    """

    rounds: int
    edges_repaired: int
    still_violated: set[tuple[int, int, int]]
    budget_used: dict[str, int]
    stopped_reason: str


class _IndexStoreAdapter:
    """Minimal adapter providing GraphStore duck-typing over GraphIndex for search_layer."""

    def __init__(self, index: MutableGraphIndex) -> None:
        self._index = index

    @property
    def dim(self) -> int:
        return self._index.dim

    def __len__(self) -> int:
        return self._index.size()

    def __contains__(self, item: Any) -> bool:
        try:
            return self._index.contains(int(item))
        except (ValueError, TypeError):
            return False

    def node_level(self, nid: int) -> int:
        return self._index.node_level(int(nid))

    def vector(self, nid: int) -> np.ndarray:
        return self._index.vector(int(nid))

    def neighbors(self, nid: int, level: int) -> list[int]:
        return self._index.neighbors(int(nid), int(level))


class _CostMeterCounterAdapter:
    """Adapter exposing DistanceCounter interface backed by a CostMeter instance."""

    def __init__(self, meter: CostMeter) -> None:
        self._meter = meter

    @property
    def value(self) -> int:
        return self._meter.count

    @value.setter
    def value(self, v: int) -> None:
        self._meter._count = int(v)


def _rebuild_moved_out_links(
    index: MutableGraphIndex,
    moved: int,
    search_layer_fn: Callable[..., Any],
    store_or_adapter: Any,
    meter_search: CostMeter,
    meter_planning: CostMeter,
    ef: int | None,
    meter_reverse_link: CostMeter | None = None,
    detector: KineticDetector | None = None,
    t: float = 0.0,
) -> list[Rewrite]:
    """Rebuild the out-links of the moved node across its hierarchy levels."""
    moved_id = int(moved)
    top_lvl = index.node_level(moved_id)
    max_lvl = index.max_level()

    # Find an entry point that is not the moved node itself
    ep: int | None = index.entry_point()
    if ep == moved_id:
        ep = None
        # Locate any other node present at highest possible level
        for lvl in range(max_lvl, -1, -1):
            candidates = [
                nid for nid in index.ids() if nid != moved_id and index.node_level(nid) >= lvl
            ]
            if candidates:
                ep = min(candidates)
                break

    if ep is None:
        return []

    counter = _CostMeterCounterAdapter(meter_search)
    effective_ef = ef if ef is not None else 16
    curr_ep = ep

    # Phase 1: Greedy search down from entry point level to top_lvl + 1
    ep_max_lvl = index.node_level(curr_ep)
    for lc in range(ep_max_lvl, top_lvl, -1):
        try:
            res = search_layer_fn(
                store_or_adapter,
                query=index.vector(moved_id),
                entry_points=[curr_ep],
                ef=1,
                level=lc,
                counter=counter,
            )
        except TypeError:
            res = search_layer_fn(
                store_or_adapter,
                index.vector(moved_id),
                [curr_ep],
                1,
                lc,
            )
        if res:
            first = res[0]
            if hasattr(first, "node_id"):
                curr_ep = int(first.node_id)
            elif isinstance(first, (tuple, list)) and len(first) >= 2:
                curr_ep = int(first[1])
            elif isinstance(first, (int, np.integer)):
                curr_ep = int(first)

    rewrites: list[Rewrite] = []

    # Phase 2: Beam search from min(top_lvl, ep_max_lvl) down to 0
    search_start_lvl = min(top_lvl, index.node_level(curr_ep))
    for lc in range(search_start_lvl, -1, -1):
        try:
            res = search_layer_fn(
                store_or_adapter,
                query=index.vector(moved_id),
                entry_points=[curr_ep],
                ef=effective_ef,
                level=lc,
                counter=counter,
            )
        except TypeError:
            res = search_layer_fn(
                store_or_adapter,
                index.vector(moved_id),
                [curr_ep],
                effective_ef,
                lc,
            )

        cand_ids: set[int] = set()
        for r in res:
            if hasattr(r, "node_id"):
                cand_ids.add(int(r.node_id))
            elif isinstance(r, (tuple, list)) and len(r) >= 2:
                cand_ids.add(int(r[1]))
            elif isinstance(r, (int, np.integer)):
                cand_ids.add(int(r))

        # Include existing neighbors of moved node at this level
        cand_ids.update(index.neighbors(moved_id, lc))
        cand_ids.discard(moved_id)

        cap = index.max_degree(lc)
        ranked = rank_pool(
            base=moved_id,
            pool=cand_ids,
            ef=ef,
            meter=meter_planning,
            vector_of=index.vector,
        )

        # K-HNSW rule: skip_if_below_cap = False.
        # When pool size is below cap, the prune still applies the diversity test.
        # This differs from the baselines on purpose.
        selected = prune(
            ranked=ranked,
            base=moved_id,
            cap=cap,
            alpha=1.0,
            skip_if_below_cap=False,
            meter=meter_planning,
            vector_of=index.vector,
        )

        new_nbrs = tuple(c for _, c in selected)
        old_nbrs = tuple(index.neighbors(moved_id, lc))

        if selected:
            curr_ep = selected[0][1]

        if set(new_nbrs) != set(old_nbrs):
            rewrites.append(Rewrite(node=moved_id, level=lc, old=old_nbrs, new=new_nbrs))

        # Reverse link additions: for each new out-neighbour w at level lc, attempt w -> moved
        if meter_reverse_link is not None:
            rev_candidates = [
                w for w in new_nbrs if w != moved_id and moved_id not in index.neighbors(w, lc)
            ]
            if rev_candidates:
                if detector is not None:
                    rev_sync_nodes: set[int] = set(rev_candidates)
                    for w in rev_candidates:
                        rev_sync_nodes.update(index.neighbors(w, lc))
                    rev_sync_nodes.add(moved_id)
                    detector.sync_nodes(rev_sync_nodes, t)

                for w in new_nbrs:
                    if w == moved_id:
                        continue
                    old_w_nbrs = tuple(index.neighbors(w, lc))
                    if moved_id in old_w_nbrs:
                        continue
                    cap_w = index.max_degree(lc)
                    if len(old_w_nbrs) < cap_w:
                        new_w_nbrs = tuple(list(old_w_nbrs) + [moved_id])
                    else:
                        cand_pool = set(old_w_nbrs) | {moved_id}
                        ranked_w = rank_pool(
                            base=w,
                            pool=cand_pool,
                            ef=ef,
                            meter=meter_reverse_link,
                            vector_of=index.vector,
                        )
                        selected_w = prune(
                            ranked=ranked_w,
                            base=w,
                            cap=cap_w,
                            alpha=1.0,
                            skip_if_below_cap=False,
                            meter=meter_reverse_link,
                            vector_of=index.vector,
                        )
                        new_w_nbrs = tuple(c for _, c in selected_w)

                    if set(new_w_nbrs) != set(old_w_nbrs):
                        rewrites.append(Rewrite(node=w, level=lc, old=old_w_nbrs, new=new_w_nbrs))

    return rewrites


def repair_loop(
    index: MutableGraphIndex,
    store: CertificateStore,
    detector: KineticDetector | None = None,
    *,
    failures: Sequence[tuple[int, int, int]] | Sequence[Transition] | None = None,
    moved_node: int | None = None,
    moved: int | None = None,
    t: float = 0.0,
    search_layer: Callable[..., Any] | None = None,
    budget: int | float | None = None,
    max_rounds: int = 10,
    ef: int | None = None,
    emitter: EventEmitter | None = None,
    reverse_links: bool = True,
) -> RepairResult:
    r"""Execute the iterative K-HNSW certificate-driven repair loop.

    Algorithm Overview:
    -------------------
    1. Extract confirmed failure edges from detector.advance(t), provided Transitions,
       or direct certificate store status.
    2. Formulate scope = source nodes and their reverse in-neighbors at affected levels.
    3. Call detector.sync_nodes(scope | candidate_pools, t) BEFORE planning.
    4. Plan rewrites with khnsw_plan, rebuilding the moved node's out-links exactly once
       in the same pass when moved_node is specified.
    5. Execute apply_plan to atomically mutate graph index topology and emit trace events.
    6. Synchronize CertificateStore via store.on_neighbor_change for every rewritten source.
    7. Synchronize KineticDetector via detector.on_topology_change(u, lvl, t=t) with explicit t.
    8. Repeat until no VIOLATED edges remain in scope, computational budget is exhausted,
       or max_rounds (default 10) is reached.

    Parameters
    ----------
    index : MutableGraphIndex
        Target mutable graph index instance.
    store : CertificateStore
        Certificate store maintaining edge statuses and witnesses.
    detector : KineticDetector | None, optional
        Kinetic event detector tracking spatial trajectory certificates.
    failures : Sequence[tuple[int, int, int]] | Sequence[Transition] | None, optional
        Pre-identified failure edges or transitions. If None and detector is supplied,
        failures are obtained via detector.advance(t).
    moved_node : int | None, optional
        Node ID of a point that underwent spatial movement.
    moved : int | None, optional
        Alias for moved_node.
    t : float, default=0.0
        Current simulation timestamp. Passed explicitly to detector methods.
    search_layer : Callable[..., Any] | None, optional
        Injected beam search implementation (Algorithm 2). If None, standard search_layer is used.
    budget : int | float | None, optional
        Computational budget cap on total pairwise distance evaluations across all categories.
        If exceeded, repair loop halts cleanly at safe round boundaries without dropping edges.
    max_rounds : int, default=10
        Maximum number of iterative repair rounds.
    ef : int | None, optional
        Exploration factor for candidate pool search and ranking.
    emitter : EventEmitter | None, optional
        Trace event emitter for simulation logging.

    Returns
    -------
    RepairResult
        Immutable result container with execution statistics and termination state.
    """
    effective_moved = moved if moved is not None else moved_node
    moved_id = int(effective_moved) if effective_moved is not None else None

    # Resolve search_layer implementation
    if search_layer is not None:
        search_layer_fn = search_layer
    else:
        from khnsw.core.search.search_layer import search_layer as default_search_layer

        search_layer_fn = default_search_layer

    store_adapter = _IndexStoreAdapter(index)

    # Cost meters for each category
    meter_search = CostMeter()
    meter_planning = CostMeter()
    meter_store = CostMeter()
    meter_detector = CostMeter()
    meter_reverse_link = CostMeter() if reverse_links else None

    def total_cost() -> int:
        return (
            meter_search.count
            + meter_planning.count
            + meter_store.count
            + meter_detector.count
            + (meter_reverse_link.count if meter_reverse_link is not None else 0)
        )

    # 1. Harvest initial failure edges
    initial_failures: list[tuple[int, int, int]] = []
    if failures is not None:
        for f in failures:
            if hasattr(f, "edge") and hasattr(f, "kind"):
                if f.kind == "failure" or f.new == EdgeStatus.VIOLATED.value:
                    initial_failures.append(f.edge)
            elif isinstance(f, tuple) and len(f) == 3:
                initial_failures.append(f)
    elif detector is not None:
        det_m = getattr(detector, "_meter", None)
        det_meter_before = det_m.count if det_m is not None else 0
        adv_edges = detector.advance(t)
        det_meter_after = det_m.count if det_m is not None else 0
        meter_detector._count += max(0, det_meter_after - det_meter_before)
        initial_failures.extend(adv_edges)
    else:
        # Fallback: scan store for existing VIOLATED edges
        for u in index.ids():
            for lvl in range(index.node_level(u) + 1):
                for v in index.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if store.status(edge) == EdgeStatus.VIOLATED.value:
                        initial_failures.append(edge)

    # 2. Formulate initial scope
    scope: set[int] = set()
    for u, _v, lvl in initial_failures:
        scope.add(u)
        scope.update(index.in_neighbors(u, lvl))

    if moved_id is not None and index.contains(moved_id):
        scope.add(moved_id)
        for lvl in range(index.node_level(moved_id) + 1):
            scope.update(index.in_neighbors(moved_id, lvl))

    rounds = 0
    repaired_edges: set[tuple[int, int, int]] = set()
    moved_rebuilt = False
    stopped_reason = ""

    # Initial budget exhaustion check
    if budget is not None and total_cost() >= budget:
        stopped_reason = "budget"

    while stopped_reason == "" and rounds < max_rounds:
        # Check budget before beginning next round
        if budget is not None and total_cost() >= budget:
            stopped_reason = "budget"
            break

        # Collect current VIOLATED edges within scope
        current_violated: dict[tuple[int, int], list[int]] = {}
        for u in sorted(scope):
            if not index.contains(u):
                continue
            for lvl in range(index.node_level(u) + 1):
                for v in index.neighbors(u, lvl):
                    edge = (u, v, lvl)
                    if store._edge_status.get(edge) == EdgeStatus.VIOLATED.value:
                        current_violated.setdefault((u, lvl), []).append(v)

        # Termination condition: no violated edges in scope and moved node already rebuilt
        if not current_violated:
            if moved_id is None or moved_rebuilt:
                stopped_reason = "converged"
                break

        # 3. Pre-planning node synchronization
        nodes_to_sync: set[int] = set(scope)
        for (u, lvl), targets in current_violated.items():
            cand_pool = set(index.neighbors(u, lvl))
            for v in targets:
                cand_pool.update(index.neighbors(int(v), lvl))
            cand_pool.discard(u)
            nodes_to_sync.update(cand_pool)

        if not moved_rebuilt and moved_id is not None and index.contains(moved_id):
            nodes_to_sync.add(moved_id)
            for lvl in range(index.node_level(moved_id) + 1):
                nodes_to_sync.update(index.neighbors(moved_id, lvl))

        if detector is not None:
            detector.sync_nodes(nodes_to_sync, t)

        # 4. Planning: plan with khnsw_plan, rebuilding moved node out-links in same pass
        rewrites: list[Rewrite] = []

        # Only rebuild moved node out-links and reverse-links once in the first round
        if not moved_rebuilt and moved_id is not None and index.contains(moved_id):
            moved_rewrites = _rebuild_moved_out_links(
                index=index,
                moved=moved_id,
                search_layer_fn=search_layer_fn,
                store_or_adapter=store_adapter,
                meter_search=meter_search,
                meter_planning=meter_planning,
                ef=ef,
                meter_reverse_link=meter_reverse_link,
                detector=detector,
                t=t,
            )
            rewrites.extend(moved_rewrites)
            moved_rebuilt = True

        # Plan rewrites for remaining violated sources (excluding nodes already rewritten)
        rewritten_nodes_levels = {(r.node, r.level) for r in rewrites}
        violated_for_khnsw: dict[tuple[int, int], list[int]] = {}
        for (u, lvl), targets in current_violated.items():
            if (u, lvl) in rewritten_nodes_levels:
                continue
            violated_for_khnsw[(u, lvl)] = targets

        if violated_for_khnsw:
            khnsw_subplan = khnsw_plan(
                index=index,
                violated=violated_for_khnsw,
                ef=ef,
                meter=meter_planning,
            )
            rewrites.extend(khnsw_subplan.rewrites)

        if not rewrites:
            # No topological changes formulated; check if violations persist
            stopped_reason = "converged"
            break

        # Check budget before atomic application
        # If budget exceeded, do NOT apply plan; stop at this safe point
        if budget is not None and total_cost() >= budget:
            stopped_reason = "budget"
            break

        # 5. Apply plan atomically
        combined_plan = RepairPlan(
            rule="khnsw",
            moved=moved_id,
            rewrites=tuple(sorted(rewrites, key=lambda r: (r.node, r.level))),
            distance_computations=meter_planning.count
            + (meter_reverse_link.count if meter_reverse_link is not None else 0),
        )

        apply_plan(
            index=index,
            plan=combined_plan,
            emitter=emitter,
            store=store,
            reason="heuristic_prune",
            meter=meter_store,
        )

        # 6 & 7. Store recertification and detector topology update
        for r in combined_plan.rewrites:
            # Expand scope to encompass modified sources and their reverse in-neighbors
            scope.add(r.node)
            scope.update(index.in_neighbors(r.node, r.level))

            if detector is not None:
                detector.on_topology_change(r.node, r.level, t=t)

        # Track successfully repaired edges
        for (u, lvl), targets in current_violated.items():
            for v in targets:
                edge = (u, v, lvl)
                # If edge was dropped or is now VALID, it has been successfully repaired
                if store._edge_status.get(edge) != EdgeStatus.VIOLATED.value:
                    repaired_edges.add(edge)

        rounds += 1

    if stopped_reason == "" and rounds >= max_rounds:
        stopped_reason = "max_rounds"

    # Compute set of directed edges that remain VIOLATED in the index
    still_violated: set[tuple[int, int, int]] = set()
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                edge = (u, v, lvl)
                if store._edge_status.get(edge) == EdgeStatus.VIOLATED.value:
                    still_violated.add(edge)

    if not still_violated and stopped_reason != "budget":
        stopped_reason = "converged"

    return RepairResult(
        rounds=rounds,
        edges_repaired=len(repaired_edges),
        still_violated=still_violated,
        budget_used={
            "search": meter_search.count,
            "planning": meter_planning.count,
            "store": meter_store.count,
            "detector": meter_detector.count,
            "reverse_link": meter_reverse_link.count if meter_reverse_link is not None else 0,
        },
        stopped_reason=stopped_reason,
    )
