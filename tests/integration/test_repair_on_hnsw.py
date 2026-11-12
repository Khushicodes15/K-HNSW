"""Integration tests verifying repair primitives against Member 2's HNSWIndex.

Validates that localized topological repair policies (native, mnru_beta, mnru_gamma,
mnru_thn, and khnsw) correctly repair HNSWIndex topology across continuous movement
steps, preserving all structural invariants, degree bounds, reverse-edge consistency,
and version semantics, while emitting trace events matching graph mutations.
"""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.detection.certificate_store import CertificateStore, EdgeStatus
from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.graph.audit import check_degree_bounds, validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.repair.apply import apply_plan
from khnsw.core.repair.plan import khnsw_plan, mnru_plan, native_plan


def _collect_violated(
    index: HNSWIndex,
    store: CertificateStore,
) -> dict[tuple[int, int], list[int]]:
    """Helper to collect all currently violated edges from the store grouped by (u, level)."""
    violated: dict[tuple[int, int], list[int]] = {}
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            for v in index.neighbors(u, lvl):
                if store.status((u, v, lvl)) == EdgeStatus.VIOLATED:
                    violated.setdefault((u, lvl), []).append(v)
    return violated


@pytest.mark.parametrize("d", [2, 8])
@pytest.mark.parametrize(
    "rule",
    ["native", "mnru_beta", "mnru_gamma", "mnru_thn", "khnsw"],
)
def test_repair_on_hnsw_integration(rule: str, d: int) -> None:
    """Run 30 moves on HNSWIndex (n=200, M=8), verifying repair behavior and invariants."""
    n = 200
    m = 8
    m0 = 16
    seed = 42 + d

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)

    # 1. Build HNSWIndex
    idx = HNSWIndex(dim=d, m=m, seed=seed)
    ids = np.arange(n, dtype=np.int64)
    idx.add(ids, X)

    # 2. Build CertificateStore
    meter = CostMeter()
    store = CertificateStore()
    store.build(idx, meter)

    # 3. Non-vacuity: assert pre-settle construction violation spike > 0
    pre_settle_violated = _collect_violated(idx, store)
    pre_settle_count = sum(len(targets) for targets in pre_settle_violated.values())
    assert pre_settle_count > 0, "Non-vacuity guard: pre-settle violations must be > 0"

    # 4. Settle pass using khnsw_plan
    settle_plan = khnsw_plan(idx, violated=pre_settle_violated, meter=meter)
    assert len(settle_plan.rewrites) > 0, "Non-vacuity guard: settle plan must contain rewrites"
    apply_plan(idx, settle_plan, store=store, meter=meter)

    # Invariants after settle pass
    assert validate_invariants(idx, m=m, m0=m0) == []
    assert check_degree_bounds(idx, m=m, m0=m0) == []

    # 5. 30 small random moves
    move_rng = np.random.default_rng(seed + 100)
    total_rewrites_across_run = 0
    total_events_across_run = 0

    for step in range(30):
        # Pick a node with active neighbors at level 0
        moved = int(move_rng.choice(idx.ids()))
        delta = move_rng.normal(scale=0.1, size=d).astype(np.float32)
        v_before = {nid: idx.version(nid) for nid in idx.ids()}

        # Snapshot adjacency before move and repair
        adj_pre: dict[tuple[int, int], list[int]] = {
            (u, lvl): list(idx.neighbors(u, lvl))
            for u in idx.ids()
            for lvl in range(idx.node_level(u) + 1)
        }

        # Perform move
        idx.move(moved, idx.vector(moved) + delta)
        store.on_move(idx, moved, meter)

        # Version check immediately after move: only moved node increments
        assert idx.version(moved) == v_before[moved] + 1
        for nid in idx.ids():
            if nid != moved:
                assert idx.version(nid) == v_before[nid]

        # Formulate repair plan based on selected rule
        if rule == "native":
            plan = native_plan(idx, moved, meter=meter)
        elif rule == "mnru_beta":
            plan = mnru_plan(idx, moved, alpha=1.0, scope="one_hop", meter=meter)
        elif rule == "mnru_gamma":
            plan = mnru_plan(idx, moved, alpha=1.1, scope="one_hop", meter=meter)
        elif rule == "mnru_thn":
            plan = mnru_plan(idx, moved, alpha=1.1, scope="two_hop", meter=meter)
        elif rule == "khnsw":
            violated = _collect_violated(idx, store)
            plan = khnsw_plan(idx, violated=violated, meter=meter)
        else:
            raise ValueError(f"Unknown rule {rule}")

        # Execute apply_plan with event tracking
        sink = ListSink()
        emitter = EventEmitter("integration_run", sink)
        apply_plan(idx, plan, emitter=emitter, store=store, meter=meter)

        total_rewrites_across_run += len(plan.rewrites)
        total_events_across_run += len(sink.events)

        # Post-apply Invariant Checks
        assert validate_invariants(idx, m=m, m0=m0) == []
        assert check_degree_bounds(idx, m=m, m0=m0) == []

        # In-neighbors consistency
        for u in idx.ids():
            for lvl in range(idx.node_level(u) + 1):
                fwd = idx.neighbors(u, lvl)
                for v in fwd:
                    assert u in idx.in_neighbors(v, lvl)

        # Version invariance: edge mutations in apply_plan must not alter versions
        assert idx.version(moved) == v_before[moved] + 1
        for nid in idx.ids():
            if nid != moved:
                assert idx.version(nid) == v_before[nid]

        # Net effect of emitted events must match adjacency diff
        adj_post: dict[tuple[int, int], list[int]] = {
            (u, lvl): list(idx.neighbors(u, lvl))
            for u in idx.ids()
            for lvl in range(idx.node_level(u) + 1)
        }

        for u in idx.ids():
            for lvl in range(idx.node_level(u) + 1):
                old_set = set(adj_pre[(u, lvl)])
                new_set = set(adj_post[(u, lvl)])
                expected_removed = old_set - new_set
                expected_added = new_set - old_set

                emitted_rem = [
                    e.payload["target_id"]
                    for e in sink.events
                    if e.type == EventType.edge_remove and e.node == u and e.level == lvl
                ]
                emitted_add = [
                    e.payload["target_id"]
                    for e in sink.events
                    if e.type == EventType.edge_add and e.node == u and e.level == lvl
                ]

                assert set(emitted_rem) == expected_removed
                assert set(emitted_add) == expected_added
                assert len(emitted_rem) == len(expected_removed)
                assert len(emitted_add) == len(expected_added)

        # khnsw-specific assertions
        if rule == "khnsw":
            post_violated = _collect_violated(idx, store)
            assert len(post_violated) == 0, (
                f"Step {step}: expected 0 violations after khnsw repair, found {post_violated}"
            )

            # Incremental store equals fresh build
            fresh_meter = CostMeter()
            fresh_store = CertificateStore()
            fresh_store.build(idx, fresh_meter)

            for u in idx.ids():
                for lvl in range(idx.node_level(u) + 1):
                    for v in idx.neighbors(u, lvl):
                        edge = (u, v, lvl)
                        assert store.status(edge) == fresh_store.status(edge)
                        assert store.witness(edge) == fresh_store.witness(edge)

    # Non-vacuity assertion across the 30-step trajectory
    assert total_rewrites_across_run > 0, (
        f"Non-vacuity guard: expected rewrites during run for {rule} (d={d})"
    )
    assert total_events_across_run > 0, (
        f"Non-vacuity guard: expected emitted events during run for {rule} (d={d})"
    )
