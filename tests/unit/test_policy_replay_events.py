"""Unit tests verifying replayable MaintenancePolicy event emission contracts.

Pure Python standard library and NumPy only in core; tests may use pytest and jsonschema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import jsonschema
import numpy as np
import pytest

from khnsw.core.detection.cost import CostMeter
from khnsw.core.events.schema import Event, EventEmitter, EventType, ListSink
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.scheduling.policy import MaintenancePolicy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_SCHEMA_PATH = REPO_ROOT / "docs" / "contracts" / "event.schema.json"
EDGE_ADD_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "contracts" / "schemas" / "events" / "edge_add.schema.json"
)
EDGE_REMOVE_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "contracts" / "schemas" / "events" / "edge_remove.schema.json"
)


def get_graph_adjacency(index: HNSWIndex) -> dict[tuple[int, int], set[int]]:
    """Return dictionary of (node_id, level) -> set of outgoing neighbor IDs."""
    adj: dict[tuple[int, int], set[int]] = {}
    for u in index.ids():
        for lvl in range(index.node_level(u) + 1):
            adj[(u, lvl)] = set(index.neighbors(u, lvl))
    return adj


def replay_events_on_adjacency(
    initial_adj: dict[tuple[int, int], set[int]],
    events: Sequence[Event],
    max_degrees: dict[int, int] | None = None,
) -> dict[tuple[int, int], set[int]]:
    """Replay sequence of edge_remove and edge_add events onto initial adjacency.

    Enforces degree limits if max_degrees is provided.
    """
    adj = {k: set(v) for k, v in initial_adj.items()}
    for ev in events:
        u = ev.node
        lvl = ev.level
        assert u is not None, "Event node must not be None"
        assert lvl is not None, "Event level must not be None"
        key = (u, lvl)
        target = ev.payload["target_id"]

        if ev.type == EventType.edge_remove:
            assert target in adj[key], f"Edge ({u}, {lvl}) -> {target} to remove was not in graph"
            adj[key].remove(target)
        elif ev.type == EventType.edge_add:
            assert target not in adj[key], f"Edge ({u}, {lvl}) -> {target} to add already exists"
            adj[key].add(target)
            if max_degrees is not None and lvl in max_degrees:
                assert len(adj[key]) <= max_degrees[lvl], (
                    f"Degree cap exceeded at level {lvl} for node {u}: "
                    f"{len(adj[key])} > {max_degrees[lvl]}"
                )
        else:
            raise ValueError(f"Unexpected event type {ev.type} in policy replay")

    return adj


def build_test_index(n: int = 50, d: int = 4, seed: int = 42) -> tuple[HNSWIndex, np.ndarray]:
    """Build deterministic test HNSW index with n nodes."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(n, d)).astype(np.float32)
    idx = HNSWIndex(dim=d, m=4, ef_construction=16, seed=seed)
    idx.add(np.arange(n, dtype=np.int64), X0)
    return idx, X0


@pytest.mark.parametrize(
    "policy_name,kwargs,drift_step",
    [
        ("round_robin", {"B": 5}, 0.05),
        ("periodic", {"K": 2}, 0.05),
        ("lazy", {"theta": 0.05}, 0.20),
        ("budgeted_priority", {"B": 5}, 0.05),
        ("round_robin", {"B": 5, "pre_repair": True}, 0.05),
    ],
)
def test_replay_equivalence(policy_name: str, kwargs: dict[str, Any], drift_step: float) -> None:
    """Test 1: Replay equivalence across all maintenance policies and pre-repair."""
    n, d, steps = 50, 4, 3
    idx, X0 = build_test_index(n=n, d=d, seed=42)
    init_adj = get_graph_adjacency(idx)

    sink = ListSink()
    emitter = EventEmitter(run_id=f"test_replay_{policy_name}", sink=sink)
    policy = MaintenancePolicy(policy=policy_name, emitter=emitter, seed=42, **kwargs)
    policy.register(idx, X0)

    for t in range(1, steps + 1):
        Xt = (X0 + drift_step * t).astype(np.float32)
        policy.step(idx, Xt, t=float(t))

    final_adj = get_graph_adjacency(idx)

    # Degree caps from store
    max_degrees = {0: idx._store.max_m_for_level(0), 1: idx._store.max_m_for_level(1)}
    replayed_adj = replay_events_on_adjacency(init_adj, sink.events, max_degrees=max_degrees)

    assert len(sink.events) > 0, f"Expected events emitted for {policy_name}"
    assert replayed_adj == final_adj, (
        f"Replayed adjacency did not match final index for {policy_name}"
    )


def test_event_count_matches_actual_neighbour_diffs() -> None:
    """Test 2: Event count equals actual sum of neighbour-list additions and removals."""
    n, d = 50, 4
    idx, X0 = build_test_index(n=n, d=d, seed=123)

    sink = ListSink()
    emitter = EventEmitter(run_id="test_counts", sink=sink)
    policy = MaintenancePolicy(policy="round_robin", B=1, emitter=emitter, seed=123)
    policy.register(idx, X0)

    # Step 1: Shift position of node 0 and record difference manually
    X1 = X0.copy()
    X1[0] += 0.5

    # Manual pre-snapshot
    before_adj = get_graph_adjacency(idx)
    reinserted = policy.step(idx, X1, t=1.0)
    assert reinserted == [0]
    after_adj = get_graph_adjacency(idx)

    # Manual net difference count for this reinsertion
    diff_count = 0
    all_keys = set(before_adj.keys()) | set(after_adj.keys())
    for k in all_keys:
        b_set = before_adj.get(k, set())
        a_set = after_adj.get(k, set())
        diff_count += len(b_set - a_set) + len(a_set - b_set)

    assert len(sink.events) == diff_count, (
        f"Event count {len(sink.events)} does not equal manual diff count {diff_count}"
    )
    removes = [e for e in sink.events if e.type == EventType.edge_remove]
    adds = [e for e in sink.events if e.type == EventType.edge_add]
    assert len(sink.events) == len(removes) + len(adds)


def test_seq_strictly_increasing_and_t_non_decreasing() -> None:
    """Test 3: seq is strictly increasing from 0 and timestamp t is non-decreasing."""
    n, d = 50, 4
    idx, X0 = build_test_index(n=n, d=d, seed=77)

    sink = ListSink()
    emitter = EventEmitter(run_id="test_seq_time", sink=sink)
    policy = MaintenancePolicy(policy="lazy", theta=0.05, emitter=emitter, seed=77)
    policy.register(idx, X0)

    for t in range(1, 4):
        Xt = (X0 + 0.2 * t).astype(np.float32)
        policy.step(idx, Xt, t=float(t))

    assert len(sink.events) > 0
    for i, ev in enumerate(sink.events):
        assert ev.seq == i, f"Expected seq {i}, got {ev.seq}"
        if i > 0:
            assert ev.t >= sink.events[i - 1].t, f"Timestamp decreased at seq {i}"


def test_no_emitter_costs_and_graph_identical() -> None:
    """Test 4: No emitter produces no events and exact same distance counts and topology."""
    n, d, steps = 50, 4, 3

    # Run without emitter
    idx_no_emit, X0_1 = build_test_index(n=n, d=d, seed=99)
    meter_no_emit = CostMeter()
    policy_no_emit = MaintenancePolicy(
        policy="round_robin", B=5, meter=meter_no_emit, emitter=None, seed=99
    )
    policy_no_emit.register(idx_no_emit, X0_1)

    # Run with emitter
    idx_with_emit, X0_2 = build_test_index(n=n, d=d, seed=99)
    meter_with_emit = CostMeter()
    sink = ListSink()
    emitter = EventEmitter(run_id="test_perf_parity", sink=sink)
    policy_with_emit = MaintenancePolicy(
        policy="round_robin", B=5, meter=meter_with_emit, emitter=emitter, seed=99
    )
    policy_with_emit.register(idx_with_emit, X0_2)

    for t in range(1, steps + 1):
        Xt = (X0_1 + 0.05 * t).astype(np.float32)
        policy_no_emit.step(idx_no_emit, Xt, t=float(t))
        policy_with_emit.step(idx_with_emit, Xt, t=float(t))

    assert len(sink.events) > 0
    # Exactly identical distance counts
    assert meter_no_emit.count == meter_with_emit.count
    assert idx_no_emit.distance_computations == idx_with_emit.distance_computations
    # Exactly identical graph topology
    assert get_graph_adjacency(idx_no_emit) == get_graph_adjacency(idx_with_emit)


def test_emitted_events_validate_against_schema() -> None:
    """Test 5: Emitted events conform strictly to JSON schema contracts."""
    assert BASE_SCHEMA_PATH.exists()
    assert EDGE_ADD_SCHEMA_PATH.exists()
    assert EDGE_REMOVE_SCHEMA_PATH.exists()

    with open(BASE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        base_schema = json.load(f)
    with open(EDGE_ADD_SCHEMA_PATH, "r", encoding="utf-8") as f:
        edge_add_schema = json.load(f)
    with open(EDGE_REMOVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        edge_remove_schema = json.load(f)

    n, d = 50, 4
    idx, X0 = build_test_index(n=n, d=d, seed=55)

    sink = ListSink()
    emitter = EventEmitter(run_id="test_schema_validation", sink=sink)
    policy = MaintenancePolicy(policy="round_robin", B=5, emitter=emitter, seed=55)
    policy.register(idx, X0)

    Xt = (X0 + 0.1).astype(np.float32)
    policy.step(idx, Xt, t=1.0)

    assert len(sink.events) > 0

    for ev in sink.events:
        ev_dict = ev.to_dict()
        # 1. Base envelope validation
        jsonschema.validate(instance=ev_dict, schema=base_schema)

        # 2. Per-type schema validation
        if ev.type == EventType.edge_add:
            jsonschema.validate(instance=ev_dict, schema=edge_add_schema)
            assert "target_id" in ev.payload
            assert "reason" not in ev.payload
        elif ev.type == EventType.edge_remove:
            jsonschema.validate(instance=ev_dict, schema=edge_remove_schema)
            assert "target_id" in ev.payload
            assert ev.payload["reason"] == "policy_reinsert"
