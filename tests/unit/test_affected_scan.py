"""Unit and property tests for affected edge scanning under kinetic motion and rewiring."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from khnsw.core.detection.affected_scan import (
    affected_after_move,
    affected_after_neighbor_change,
)
from khnsw.core.detection.certificates import is_violated_static
from tests.fixtures.reference_graph import ReferenceGraph


def _brute_force_all_statuses(graph: ReferenceGraph) -> dict[tuple[int, int, int], bool]:
    """Independent oracle computing static certificate validity for every edge in graph."""
    statuses: dict[tuple[int, int, int], bool] = {}
    for lvl in range(graph.max_level() + 1):
        for u in graph.ids():
            nbrs = graph.neighbors(u, lvl)
            if not nbrs:
                continue
            vec_u = graph.vector(u)
            for v in nbrs:
                vec_v = graph.vector(v)
                d_uv = float(np.sum((vec_u - vec_v) ** 2))
                is_violated = False
                for w in nbrs:
                    if w == v:
                        continue
                    vec_w = graph.vector(w)
                    d_uw = float(np.sum((vec_u - vec_w) ** 2))
                    d_vw = float(np.sum((vec_v - vec_w) ** 2))
                    if is_violated_static(d_uv, d_uw, d_vw):
                        is_violated = True
                        break
                statuses[(u, v, lvl)] = is_violated
    return statuses


# =============================================================================
# T1: Completeness After Move (Hypothesis Property Test)
# =============================================================================


@settings(max_examples=350, deadline=None)
@given(
    d=st.sampled_from([2, 4, 16]),
    M=st.integers(min_value=3, max_value=8),
    mimic=st.booleans(),
    seed=st.integers(min_value=0, max_value=10000),
    disp_mode=st.sampled_from(["tiny", "large", "duplicate"]),
)
def test_affected_after_move_completeness(
    d: int,
    M: int,
    mimic: bool,
    seed: int,
    disp_mode: str,
) -> None:
    """T1: The affected_after_move scan is a superset of all edges whose status changed."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(12, 22))
    X = rng.standard_normal((n, d)).astype(np.float32)

    g = ReferenceGraph.from_points(X, M=M, seed=seed, mimic_reverse_append=mimic)
    all_ids = g.ids()
    target_node = int(rng.choice(all_ids))

    # Formulate new vector based on displacement mode
    orig_vec = g.vector(target_node)
    if disp_mode == "tiny":
        new_vec = orig_vec + np.float32(1e-4) * rng.standard_normal(d).astype(np.float32)
    elif disp_mode == "large":
        new_vec = orig_vec + np.float32(50.0) * rng.standard_normal(d).astype(np.float32)
    else:
        other_node = int(rng.choice([i for i in all_ids if i != target_node]))
        new_vec = g.vector(other_node).copy()

    status_before = _brute_force_all_statuses(g)
    predicted_affected = affected_after_move(g, target_node)

    g.move(target_node, new_vec)
    status_after = _brute_force_all_statuses(g)

    # Any edge whose status toggled must be contained in the predicted affected scan
    changed_edges = {e for e in status_before if status_before[e] != status_after[e]}
    assert changed_edges.issubset(predicted_affected), (
        f"Completeness violated! Changed edges {changed_edges - predicted_affected} "
        f"were omitted by affected_after_move({target_node})"
    )


# =============================================================================
# T2: Completeness After Neighbor Change
# =============================================================================


@settings(max_examples=100, deadline=None)
@given(
    d=st.sampled_from([2, 4]),
    M=st.integers(min_value=3, max_value=6),
    seed=st.integers(min_value=0, max_value=10000),
    op=st.sampled_from(["set_neighbors", "add_edge", "remove_edge"]),
)
def test_affected_after_neighbor_change_completeness(
    d: int,
    M: int,
    seed: int,
    op: str,
) -> None:
    """T2: affected_after_neighbor_change covers all status changes after rewiring N_l(u)."""
    rng = np.random.default_rng(seed)
    n = 15
    X = rng.standard_normal((n, d)).astype(np.float32)
    g = ReferenceGraph.from_points(X, M=M, seed=seed, mimic_reverse_append=True)

    u = int(rng.choice(g.ids()))
    lvl = int(rng.integers(0, g.node_level(u) + 1))
    eligible = [v for v in g.ids() if v != u and g.node_level(v) >= lvl]
    degree_limit = g.max_degree(lvl)

    status_before = _brute_force_all_statuses(g)

    # Execute topological operation
    if op == "set_neighbors":
        k = int(rng.integers(0, min(len(eligible), degree_limit) + 1))
        chosen = list(rng.choice(eligible, size=k, replace=False)) if eligible else []
        g.set_neighbors(u, lvl, [int(x) for x in chosen])
    elif op == "add_edge":
        curr = g.neighbors(u, lvl)
        addable = [v for v in eligible if v not in curr]
        if addable and len(curr) < degree_limit:
            target_v = int(rng.choice(addable))
            g.add_edge(u, target_v, lvl)
        else:
            return
    elif op == "remove_edge":
        curr = g.neighbors(u, lvl)
        if curr:
            target_v = int(rng.choice(curr))
            g.remove_edge(u, target_v, lvl)
        else:
            return

    status_after = _brute_force_all_statuses(g)
    predicted_affected = affected_after_neighbor_change(g, u, lvl)

    # Persisting edges whose boolean validity toggled
    persisting_changed = {
        e for e in status_before if e in status_after and status_before[e] != status_after[e]
    }
    assert persisting_changed.issubset(predicted_affected), (
        f"Completeness violated on rewiring! Edges {persisting_changed - predicted_affected} "
        f"were omitted by affected_after_neighbor_change({u}, {lvl})"
    )


# =============================================================================
# T3: Non-Vacuity
# =============================================================================


def test_affected_scan_non_vacuity() -> None:
    """T3: On n >= 200, scan size is strictly smaller than total graph edge count."""
    rng = np.random.default_rng(42)
    n = 200
    dim = 4
    M = 8
    X = rng.standard_normal((n, dim)).astype(np.float32)
    g = ReferenceGraph.from_points(X, M=M, seed=42, mimic_reverse_append=True)

    total_edges = sum(len(g.neighbors(u, lvl)) for lvl in range(g.max_level() + 1) for u in g.ids())
    assert total_edges > 0

    fractions: list[float] = []
    # Test scan across a representative sample of nodes
    sample_nodes = g.ids()[:10]
    for node in sample_nodes:
        scanned = affected_after_move(g, node)
        scanned_count = len(scanned)
        frac = scanned_count / total_edges
        fractions.append(frac)
        assert scanned_count < total_edges, f"Node {node} scanned entire graph!"
        print(f"Node {node}: scanned={scanned_count}/{total_edges} ({frac:.2%})")

    mean_frac = float(np.mean(fractions))
    print(f"\n[Non-Vacuity] Mean scanned edge fraction across sample: {mean_frac:.2%}")
    assert mean_frac < 0.20, f"Expected localized scan under 20%, got {mean_frac:.2%}"
