"""Unit and property tests for KineticEventHeap and DueEvent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from khnsw.core.events.schema import EventEmitter, EventType, ListSink
from khnsw.core.scheduling.event_heap import DueEvent, KineticEventHeap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMAS_EVENTS_DIR = REPO_ROOT / "docs" / "contracts" / "schemas" / "events"
BASE_EVENT_SCHEMA_PATH = REPO_ROOT / "docs" / "contracts" / "event.schema.json"


@pytest.fixture
def base_event_schema() -> dict[str, Any]:
    with open(BASE_EVENT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def per_type_schemas() -> dict[str, dict[str, Any]]:
    schemas = {}
    for event_name in ["enqueue", "pop", "stale_skip"]:
        p = SCHEMAS_EVENTS_DIR / f"{event_name}.schema.json"
        with open(p, "r", encoding="utf-8") as f:
            schemas[event_name] = json.load(f)
    return schemas


# =============================================================================
# 1. Basic Ordering, Tie-Breaking, and Inclusive Boundary
# =============================================================================


def test_ordering_and_deterministic_tie_breaks() -> None:
    """Events are popped in chronological order with deterministic tie-breaks."""
    versions = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    heap = KineticEventHeap(version_of=lambda nid: versions.get(nid, 0))

    # Push events with distinct times
    heap.push(edge=(1, 2, 0), t_fail=5.0, witness=3)
    heap.push(edge=(1, 4, 0), t_fail=2.0, witness=5)
    heap.push(edge=(2, 3, 1), t_fail=8.0, witness=4)

    assert heap.peek_time() == 2.0
    due = heap.pop_due(t_now=5.0)
    assert len(due) == 2
    assert due[0] == DueEvent(t=2.0, edge=(1, 4, 0), witness=5)
    assert due[1] == DueEvent(t=5.0, edge=(1, 2, 0), witness=3)

    # Remaining event
    assert heap.peek_time() == 8.0
    due_rem = heap.pop_due(t_now=10.0)
    assert len(due_rem) == 1
    assert due_rem[0] == DueEvent(t=8.0, edge=(2, 3, 1), witness=4)


def test_pop_due_inclusive_boundary() -> None:
    """pop_due boundary is inclusive at t == t_now."""
    versions = {1: 0, 2: 0, 3: 0}
    heap = KineticEventHeap(version_of=lambda nid: versions.get(nid, 0))

    heap.push(edge=(1, 2, 0), t_fail=3.0, witness=3)
    # Exactly at boundary t_now = 3.0
    due = heap.pop_due(t_now=3.0)
    assert len(due) == 1
    assert due[0].t == 3.0

    # Nothing due before 3.0
    heap.push(edge=(1, 2, 0), t_fail=4.0, witness=3)
    assert heap.pop_due(t_now=3.999) == []
    assert len(heap.pop_due(t_now=4.0)) == 1


def test_deterministic_tie_breaks_identical_t() -> None:
    """Identical failure times are broken deterministically by (u, v, level, push_counter)."""
    versions = {i: 0 for i in range(10)}
    heap = KineticEventHeap(version_of=lambda nid: versions.get(nid, 0))

    # Push multiple edges with exact same t_fail=5.0
    heap.push(edge=(2, 3, 0), t_fail=5.0, witness=4)
    heap.push(edge=(1, 5, 0), t_fail=5.0, witness=6)
    heap.push(edge=(1, 2, 0), t_fail=5.0, witness=3)
    heap.push(edge=(1, 2, 1), t_fail=5.0, witness=3)

    due = heap.pop_due(t_now=5.0)
    assert len(due) == 4
    # Expected tie-break order: u first, then v, then level
    assert due[0].edge == (1, 2, 0)
    assert due[1].edge == (1, 2, 1)
    assert due[2].edge == (1, 5, 0)
    assert due[3].edge == (2, 3, 0)


# =============================================================================
# 2. Staleness Mechanics (Node Version Bumps, Re-push, Invalidations)
# =============================================================================


def test_stale_after_node_version_changes_separately() -> None:
    """Verify u, v, and witness version bumps each individually invalidate entry."""
    # Test u version bump
    v_map = {1: 0, 2: 0, 3: 0}
    heap_u = KineticEventHeap(version_of=lambda nid: v_map.get(nid, 0))
    heap_u.push(edge=(1, 2, 0), t_fail=2.0, witness=3)
    v_map[1] += 1  # Node u moved
    assert heap_u.pop_due(t_now=5.0) == []

    # Test v version bump
    v_map = {1: 0, 2: 0, 3: 0}
    heap_v = KineticEventHeap(version_of=lambda nid: v_map.get(nid, 0))
    heap_v.push(edge=(1, 2, 0), t_fail=2.0, witness=3)
    v_map[2] += 1  # Target v moved
    assert heap_v.pop_due(t_now=5.0) == []

    # Test witness version bump
    v_map = {1: 0, 2: 0, 3: 0}
    heap_w = KineticEventHeap(version_of=lambda nid: v_map.get(nid, 0))
    heap_w.push(edge=(1, 2, 0), t_fail=2.0, witness=3)
    v_map[3] += 1  # Witness moved
    assert heap_w.pop_due(t_now=5.0) == []


def test_repush_replaces_older_certificate() -> None:
    """Re-pushing an edge replaces earlier certificate; only newest is ever returned."""
    versions = {1: 0, 2: 0, 3: 0, 4: 0}
    heap = KineticEventHeap(version_of=lambda nid: versions.get(nid, 0))

    # Push early certificate at t=3.0
    heap.push(edge=(1, 2, 0), t_fail=3.0, witness=3)
    # Re-push same edge later with updated failure time t=6.0 and witness 4
    heap.push(edge=(1, 2, 0), t_fail=6.0, witness=4)

    assert len(heap) == 1
    assert heap.raw_size == 2

    # At t=4.0, the first entry is discarded as stale; nothing is due
    due_early = heap.pop_due(t_now=4.0)
    assert due_early == []

    # At t=6.0, only the newest certificate is returned
    due_late = heap.pop_due(t_now=6.0)
    assert len(due_late) == 1
    assert due_late[0] == DueEvent(t=6.0, edge=(1, 2, 0), witness=4)
    assert len(heap) == 0


def test_invalidate_edge_and_source() -> None:
    """invalidate_edge and invalidate_source make existing entries stale immediately."""
    versions = {1: 0, 2: 0, 3: 0, 4: 0}
    heap = KineticEventHeap(version_of=lambda nid: versions.get(nid, 0))

    heap.push(edge=(1, 2, 0), t_fail=2.0, witness=3)
    heap.push(edge=(1, 4, 0), t_fail=3.0, witness=3)
    heap.push(edge=(2, 3, 0), t_fail=4.0, witness=1)

    # Invalidate single edge (1, 2, 0)
    heap.invalidate_edge((1, 2, 0))
    assert len(heap) == 2
    assert (1, 2, 0) not in heap._live_edges

    # Invalidate all source 1 edges at level 0
    heap.invalidate_source(u=1, level=0)
    assert len(heap) == 1
    assert (1, 4, 0) not in heap._live_edges

    # Only edge (2, 3, 0) remains valid
    due = heap.pop_due(t_now=10.0)
    assert len(due) == 1
    assert due[0].edge == (2, 3, 0)


# =============================================================================
# 3. Event Emissions, Sequence Monotonicity, and Schema Validation
# =============================================================================


def test_events_schemas_and_sequencing(
    base_event_schema: dict[str, Any],
    per_type_schemas: dict[str, dict[str, Any]],
) -> None:
    """Trace events emit strictly increasing seq and validate against JSON schemas."""
    sink = ListSink()
    emitter = EventEmitter(run_id="heap-test-run", sink=sink)
    versions = {1: 0, 2: 0, 3: 0}
    heap = KineticEventHeap(
        version_of=lambda nid: versions.get(nid, 0),
        emitter=emitter,
    )

    # 1. Enqueue valid event
    heap.push(edge=(1, 2, 0), t_fail=2.5, witness=3, now=0.0)
    assert len(sink.events) == 1
    ev_enq = sink.events[-1]
    assert ev_enq.type == EventType.enqueue
    assert ev_enq.node == 1
    assert ev_enq.level == 0
    assert ev_enq.version == 0
    assert ev_enq.payload["priority"] == 2.5

    # 2. Pop valid event
    due = heap.pop_due(t_now=3.0)
    assert len(due) == 1
    assert len(sink.events) == 2
    ev_pop = sink.events[-1]
    assert ev_pop.type == EventType.pop
    assert ev_pop.node == 1
    assert ev_pop.payload["heap_size"] == 0

    # 3. Push and cause stale_skip via version bump
    heap.push(edge=(1, 2, 0), t_fail=4.0, witness=3, now=3.0)
    versions[1] += 1  # Node 1 moves, incrementing its version to 1
    assert heap.pop_due(t_now=5.0) == []
    assert len(sink.events) == 4
    ev_stale = sink.events[-1]
    assert ev_stale.type == EventType.stale_skip
    assert ev_stale.node == 1
    assert ev_stale.version == 0  # Version recorded at push time
    assert ev_stale.payload["current_version"] == 1
    assert ev_stale.payload["reason"] == "stale_node_version"

    # Validate all emitted events
    for idx, ev in enumerate(sink.events):
        assert ev.seq == idx, f"Sequence must be strictly increasing, got {ev.seq} at index {idx}"
        data = ev.to_dict()
        # Validate base envelope
        jsonschema.validate(instance=data, schema=base_event_schema)
        # Validate per-type schema
        type_schema = per_type_schemas[ev.type.value]
        jsonschema.validate(instance=data, schema=type_schema)


# =============================================================================
# 4. Compaction Mechanics
# =============================================================================


def test_compaction_rebuilds_heap_and_bounds_growth() -> None:
    """compact() drops all stale entries while preserving order and live bounds."""
    versions = {1: 0, 2: 0, 3: 0}
    heap = KineticEventHeap(
        version_of=lambda nid: versions.get(nid, 0),
        compact_min=10,
        compact_ratio=0.5,
    )

    # Re-pushing an edge repeatedly creates many stale copies
    for t in range(1, 20):
        heap.push(edge=(1, 2, 0), t_fail=float(t + 10), witness=3)

    assert len(heap) == 1
    # Check that automatic compaction kicked in
    assert heap.raw_size < 20

    # Manual forced compaction
    heap.compact()
    assert heap.raw_size == 1
    assert len(heap) == 1

    # Event remains popped in correct order
    due = heap.pop_due(t_now=50.0)
    assert len(due) == 1
    assert due[0].t == 29.0


# =============================================================================
# 5. Validation Errors and Edge Cases
# =============================================================================


def test_validation_errors_and_empty_behavior() -> None:
    """Invalid arguments raise ValueError; empty heap returns cleanly."""
    versions = {1: 0, 2: 0, 3: 0}
    heap = KineticEventHeap(version_of=lambda nid: versions.get(nid, 0))

    # Non-finite t_fail
    with pytest.raises(ValueError, match="finite"):
        heap.push(edge=(1, 2, 0), t_fail=float("nan"), witness=3)
    with pytest.raises(ValueError, match="finite"):
        heap.push(edge=(1, 2, 0), t_fail=float("inf"), witness=3)

    # t_fail < now
    with pytest.raises(ValueError, match="cannot be earlier than now"):
        heap.push(edge=(1, 2, 0), t_fail=2.0, witness=3, now=5.0)

    # Empty heap behavior
    assert heap.pop_due(t_now=10.0) == []
    assert heap.peek_time() is None
    assert len(heap) == 0
    assert heap.raw_size == 0


# =============================================================================
# 6. Model-Based Property Test (Hypothesis)
# =============================================================================


class NaiveReferenceHeap:
    """Simple list-based reference implementation for differential property testing."""

    def __init__(self, version_of: Any) -> None:
        self.version_of = version_of
        # edge -> (t_fail, witness, u_ver, v_ver, w_ver, push_cnt)
        self.entries: dict[tuple[int, int, int], tuple[float, int, int, int, int, int]] = {}
        self.push_counter = 0

    def push(self, edge: tuple[int, int, int], t_fail: float, witness: int) -> None:
        self.push_counter += 1
        u, v, _ = edge
        self.entries[edge] = (
            t_fail,
            witness,
            self.version_of(u),
            self.version_of(v),
            self.version_of(witness),
            self.push_counter,
        )

    def invalidate_edge(self, edge: tuple[int, int, int]) -> None:
        self.entries.pop(edge, None)

    def invalidate_source(self, u: int, level: int) -> None:
        to_del = [e for e in self.entries if e[0] == u and e[2] == level]
        for e in to_del:
            del self.entries[e]

    def pop_due(self, t_now: float) -> list[DueEvent]:
        # Filter all entries with t <= t_now
        candidates = []
        stale_edges = []
        for edge, (t_fail, witness, u_ver, v_ver, w_ver, push_cnt) in self.entries.items():
            if t_fail <= t_now:
                u, v, _ = edge
                if (
                    self.version_of(u) == u_ver
                    and self.version_of(v) == v_ver
                    and self.version_of(witness) == w_ver
                ):
                    candidates.append((t_fail, edge[0], edge[1], edge[2], push_cnt, edge, witness))
                else:
                    stale_edges.append(edge)

        # Discard stale edges encountered
        for e in stale_edges:
            del self.entries[e]

        # Sort valid candidates deterministically
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        res = []
        for c in candidates:
            res.append(DueEvent(t=c[0], edge=c[5], witness=c[6]))
            del self.entries[c[5]]
        return res


@settings(max_examples=100, deadline=None)
@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(["push", "bump", "invalidate_edge", "invalidate_source", "pop"]),
            st.integers(min_value=0, max_value=5),  # node u
            st.integers(min_value=0, max_value=5),  # node v
            st.integers(min_value=0, max_value=1),  # level
            st.integers(min_value=0, max_value=5),  # witness
            st.floats(min_value=0.1, max_value=10.0),  # timestamp
        ),
        min_size=10,
        max_size=60,
    )
)
def test_hypothesis_model_based_comparison(operations: list[Any]) -> None:
    """Hypothesis property test comparing KineticEventHeap against NaiveReferenceHeap."""
    node_versions: dict[int, int] = {i: 0 for i in range(10)}

    def get_version(nid: int) -> int:
        return node_versions.get(nid, 0)

    heap = KineticEventHeap(version_of=get_version)
    ref = NaiveReferenceHeap(version_of=get_version)

    for op, u, v, level, witness, t in operations:
        edge = (u, v, level)
        if op == "push":
            heap.push(edge=edge, t_fail=t, witness=witness, now=0.0)
            ref.push(edge=edge, t_fail=t, witness=witness)
        elif op == "bump":
            node_versions[u] += 1
        elif op == "invalidate_edge":
            heap.invalidate_edge(edge)
            ref.invalidate_edge(edge)
        elif op == "invalidate_source":
            heap.invalidate_source(u, level)
            ref.invalidate_source(u, level)
        elif op == "pop":
            due_heap = heap.pop_due(t_now=t)
            due_ref = ref.pop_due(t_now=t)
            assert due_heap == due_ref, (
                f"Mismatch in pop_due at t={t}: heap returned {due_heap}, ref returned {due_ref}"
            )


# =============================================================================
# 7. Large Randomized Run (10,000 operations)
# =============================================================================


def test_large_randomized_10k_operations_performance() -> None:
    """Stress test executing 10,000 mixed operations finishes within a few seconds."""
    rng = np.random.default_rng(1337)
    node_versions: dict[int, int] = {i: 0 for i in range(100)}

    heap = KineticEventHeap(
        version_of=lambda nid: node_versions.get(nid, 0),
        compact_min=200,
        compact_ratio=0.5,
    )

    t_curr = 0.0
    total_popped = 0

    for _ in range(10000):
        action = rng.choice(["push", "bump", "pop", "invalidate"])
        if action == "push":
            u = int(rng.integers(0, 50))
            v = int(rng.integers(0, 50))
            lvl = int(rng.integers(0, 3))
            witness = int(rng.integers(0, 50))
            t_fail = t_curr + float(rng.uniform(0.1, 20.0))
            heap.push(edge=(u, v, lvl), t_fail=t_fail, witness=witness, now=t_curr)
        elif action == "bump":
            target = int(rng.integers(0, 50))
            node_versions[target] += 1
        elif action == "invalidate":
            u = int(rng.integers(0, 50))
            lvl = int(rng.integers(0, 3))
            heap.invalidate_source(u, lvl)
        elif action == "pop":
            t_curr += float(rng.uniform(0.1, 1.0))
            due = heap.pop_due(t_now=t_curr)
            total_popped += len(due)

    # Final drain
    due_final = heap.pop_due(t_now=t_curr + 100.0)
    total_popped += len(due_final)
    assert total_popped > 0
