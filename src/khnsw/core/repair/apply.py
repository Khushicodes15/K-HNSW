"""Execution and application of repair plans onto MutableGraphIndex instances."""

from __future__ import annotations

from khnsw.core.detection.certificate_store import CertificateStore
from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import EventEmitter, EventType
from khnsw.core.interfaces import MutableGraphIndex
from khnsw.core.repair.plan import RepairPlan


def apply_plan(
    index: MutableGraphIndex,
    plan: RepairPlan,
    emitter: EventEmitter | None = None,
    store: CertificateStore | None = None,
    reason: str = "heuristic_prune",
    meter: CostMeter | None = None,
) -> None:
    """Apply an immutable RepairPlan onto a mutable graph index.

    Operations and Invariants:
    --------------------------
    1. Validation (atomic pre-check):
       - Validates that every target and source node exists within [0, len(index)).
       - Validates that rewrite levels are within [0, index.node_level(node)].
       - Validates that no self-loops exist (node not in new).
       - Validates that no duplicate neighbors exist in new.
       - Validates that len(new) <= index.max_degree(level).
       If any check fails, raises ValueError immediately without mutating index.
    2. Graph Mutation:
       Calls index.set_neighbors(rewrite.node, rewrite.level, list(rewrite.new)) for
       each planned rewrite.
    3. Event Emission:
       If emitter is provided, emits deterministic events:
       - `edge_remove` for each dropped neighbor in ascending target ID order, with
         payload {"target_id": dropped_id, "reason": reason}.
       - `edge_add` for each added neighbor in ascending target ID order, with
         payload {"target_id": added_id}.
    4. Certificate Store Synchronization:
       If store is provided, calls
       `store.on_neighbor_change(index, rewrite.node, rewrite.level, meter)`
       to maintain invariant certificate states and witness caches.

    Parameters
    ----------
    index : MutableGraphIndex
        Mutable graph index instance to mutate.
    plan : RepairPlan
        Immutable repair plan to execute.
    emitter : EventEmitter | None, optional
        Trace event emitter for simulation logging.
    store : CertificateStore | None, optional
        Certificate store to synchronize after topological modifications.
    reason : str, default="heuristic_prune"
        Reason label included in edge_remove trace event payloads.
    meter : CostMeter | None, optional
        Computational distance counter passed to store.on_neighbor_change.

    Raises
    ------
    ValueError
        If any planned rewrite violates graph integrity or degree bounds.
    """
    # 1. Atomic Pre-Validation
    for rewrite in plan.rewrites:
        u = rewrite.node
        lvl = rewrite.level

        if not index.contains(u):
            raise ValueError(f"Source node {u} does not exist in graph index.")

        u_max_lvl = index.node_level(u)
        if not (0 <= lvl <= u_max_lvl):
            raise ValueError(f"Level {lvl} out of valid bounds [0, {u_max_lvl}] for node {u}.")

        if u in rewrite.new:
            raise ValueError(
                f"Self-loop detected: source node {u} present in replacement {rewrite.new}."
            )

        if len(rewrite.new) != len(set(rewrite.new)):
            raise ValueError(
                f"Duplicate neighbor IDs present in replacement {rewrite.new} for node {u}."
            )

        for target in rewrite.new:
            if not index.contains(target):
                raise ValueError(f"Target neighbor {target} does not exist in graph index.")

        cap = index.max_degree(lvl)
        if len(rewrite.new) > cap:
            raise ValueError(
                f"Replacement count {len(rewrite.new)} exceeds max_degree {cap} at level {lvl}."
            )

    # 2. Execution and Synchronization
    effective_meter = meter if meter is not None else CostMeter()

    for rewrite in plan.rewrites:
        u = rewrite.node
        lvl = rewrite.level
        old_set = set(index.neighbors(u, lvl))
        new_list = list(rewrite.new)
        new_set = set(new_list)

        index.set_neighbors(u, lvl, new_list)

        if emitter is not None:
            dropped = sorted(old_set - new_set)
            added = sorted(new_set - old_set)
            u_ver = index.version(u)

            for d in dropped:
                emitter.emit(
                    type=EventType.edge_remove,
                    node=u,
                    level=lvl,
                    version=u_ver,
                    payload={"target_id": d, "reason": reason},
                )

            for a in added:
                emitter.emit(
                    type=EventType.edge_add,
                    node=u,
                    level=lvl,
                    version=u_ver,
                    payload={"target_id": a},
                )

        if store is not None:
            store.on_neighbor_change(index, u, lvl, effective_meter)
