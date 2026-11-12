"""Unit tests verifying HNSWIndex conformance to MutableGraphIndex protocol.

Validates that HNSWIndex satisfies runtime type checks, delegates mutation
operations (set_neighbors, add_edge, remove_edge) to GraphStore while maintaining
bidirectional adjacency symmetry, invariant validity, version semantics (monotonic
versions increment only on move, never on edge mutations), and distance counter
invariance (mutation operations perform zero distance computations).
"""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.core.graph.audit import validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.interfaces import ANNIndex, GraphIndex, MutableGraphIndex


def _build_test_index(n: int = 50, dim: int = 4, m: int = 8, seed: int = 42) -> HNSWIndex:
    rng = np.random.default_rng(seed)
    idx = HNSWIndex(dim=dim, m=m, seed=seed)
    ids = np.arange(n, dtype=np.int64)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    idx.add(ids, vectors)
    return idx


def test_hnsw_isinstance_mutable_graph_index() -> None:
    """HNSWIndex must satisfy ANNIndex, GraphIndex, and MutableGraphIndex protocols."""
    idx = HNSWIndex(dim=4, m=8)
    assert isinstance(idx, ANNIndex)
    assert isinstance(idx, GraphIndex)
    assert isinstance(idx, MutableGraphIndex)


def test_hnsw_set_neighbors_contract() -> None:
    """set_neighbors must update forward and reverse edges symmetrically at levels 0 and >0."""
    idx = _build_test_index(n=50, dim=4, m=8, seed=42)
    assert idx.max_level() >= 1, "Expected test index to have levels > 0"

    # --- Test at level 0 ---
    u0 = 0
    cur0 = list(idx.neighbors(u0, 0))
    all_lvl0 = [nid for nid in idx.ids() if nid != u0]

    # Pick a candidate not currently in cur0 to guarantee non-vacuity
    candidates0 = [nid for nid in all_lvl0 if nid not in cur0]
    assert len(candidates0) > 0, "Non-vacuity precondition: candidate outside cur0 must exist"

    # Build new neighbor list with at least one added and/or removed edge
    new_nbrs0 = sorted(cur0[: max(0, len(cur0) - 1)] + [candidates0[0]])
    assert set(new_nbrs0) != set(cur0), "Precondition non-vacuity: new_nbrs must differ from cur0"
    assert len(new_nbrs0) <= idx.max_degree(0)

    v_before = {nid: idx.version(nid) for nid in idx.ids()}
    dist_before = idx.distance_computations

    idx.set_neighbors(u0, 0, new_nbrs0)

    assert sorted(idx.neighbors(u0, 0)) == new_nbrs0
    assert validate_invariants(idx._store) == []

    # In-neighbors consistency
    for w in new_nbrs0:
        assert u0 in idx.in_neighbors(w, 0)
    for removed in set(cur0) - set(new_nbrs0):
        assert u0 not in idx.in_neighbors(removed, 0)

    # Invariance of version and distance
    assert idx.distance_computations == dist_before
    for nid in idx.ids():
        assert idx.version(nid) == v_before[nid]

    # --- Test at level > 0 ---
    # Find level > 0 with at least 2 nodes
    target_lvl = -1
    for lvl in range(idx.max_level(), 0, -1):
        nodes_at_lvl = [nid for nid in idx.ids() if idx.node_level(nid) >= lvl]
        if len(nodes_at_lvl) >= 2:
            target_lvl = lvl
            break
    assert target_lvl > 0, "Expected at least one upper level with >= 2 nodes"

    nodes_at_lvl = [nid for nid in idx.ids() if idx.node_level(nid) >= target_lvl]
    u_high = nodes_at_lvl[0]
    cur_high = list(idx.neighbors(u_high, target_lvl))
    candidates_high = [nid for nid in nodes_at_lvl if nid != u_high and nid not in cur_high]

    if candidates_high:
        new_nbrs_high = sorted(cur_high[: max(0, len(cur_high) - 1)] + [candidates_high[0]])
    else:
        # If all candidates were connected, drop one to ensure difference
        assert len(cur_high) > 0, "cur_high must have edges if all nodes connected"
        new_nbrs_high = sorted(cur_high[:-1])

    assert set(new_nbrs_high) != set(cur_high), "Precondition non-vacuity for level > 0"
    assert len(new_nbrs_high) <= idx.max_degree(target_lvl)

    dist_before_high = idx.distance_computations
    idx.set_neighbors(u_high, target_lvl, new_nbrs_high)

    assert sorted(idx.neighbors(u_high, target_lvl)) == new_nbrs_high
    assert validate_invariants(idx._store) == []
    for w in new_nbrs_high:
        assert u_high in idx.in_neighbors(w, target_lvl)
    for removed in set(cur_high) - set(new_nbrs_high):
        assert u_high not in idx.in_neighbors(removed, target_lvl)

    assert idx.distance_computations == dist_before_high
    for nid in idx.ids():
        assert idx.version(nid) == v_before[nid]


def test_hnsw_add_edge_remove_edge_contract() -> None:
    """add_edge and remove_edge must keep forward and reverse in sync and enforce errors."""
    idx = _build_test_index(n=50, dim=4, m=8, seed=42)

    # Find a node u at level 0 with degree < max_degree(0)
    u = -1
    v = -1
    for nid in idx.ids():
        nbrs = set(idx.neighbors(nid, 0))
        if len(nbrs) < idx.max_degree(0):
            candidates = [c for c in idx.ids() if c != nid and c not in nbrs]
            if candidates:
                u = nid
                v = candidates[0]
                break

    assert u != -1 and v != -1, "Expected to find an unconnected pair with degree headroom"

    dist_before = idx.distance_computations
    versions_before = {nid: idx.version(nid) for nid in idx.ids()}

    # --- add_edge ---
    idx.add_edge(u, v, 0)
    assert v in idx.neighbors(u, 0)
    assert u in idx.in_neighbors(v, 0)
    assert validate_invariants(idx._store) == []
    assert idx.distance_computations == dist_before
    assert {nid: idx.version(nid) for nid in idx.ids()} == versions_before

    # Idempotence: re-adding an existing edge does not error or duplicate
    idx.add_edge(u, v, 0)
    assert idx.neighbors(u, 0).count(v) == 1
    assert validate_invariants(idx._store) == []

    # --- remove_edge ---
    idx.remove_edge(u, v, 0)
    assert v not in idx.neighbors(u, 0)
    assert u not in idx.in_neighbors(v, 0)
    assert validate_invariants(idx._store) == []
    assert idx.distance_computations == dist_before
    assert {nid: idx.version(nid) for nid in idx.ids()} == versions_before

    # Removing non-existent edge raises KeyError
    with pytest.raises(KeyError, match="does not exist"):
        idx.remove_edge(u, v, 0)

    # Degree bound enforcement: fill node to max_degree(0) then attempt to add
    max_d = idx.max_degree(0)
    current_nbrs = idx.neighbors(u, 0)
    pool = [nid for nid in idx.ids() if nid != u and nid not in current_nbrs]
    needed = max_d - len(current_nbrs)
    for p in pool[:needed]:
        idx.add_edge(u, p, 0)
    assert len(idx.neighbors(u, 0)) == max_d

    overflow_cand = pool[needed]
    with pytest.raises(ValueError, match="reached max out-degree"):
        idx.add_edge(u, overflow_cand, 0)

    # Self-loops prohibited
    with pytest.raises(ValueError, match="Self-loops"):
        idx.add_edge(u, u, 0)

    # Missing nodes
    with pytest.raises(KeyError):
        idx.add_edge(999999, u, 0)
    with pytest.raises(KeyError):
        idx.remove_edge(999999, u, 0)


def test_hnsw_version_and_distance_invariance() -> None:
    """Verifies version semantics: move increments version, topology mutations do not."""
    idx = _build_test_index(n=20, dim=4, m=8, seed=42)

    # Initial versions are 0
    for nid in idx.ids():
        assert idx.version(nid) == 0

    target = 5
    old_dist = idx.distance_computations

    # Edge mutation: remove and add
    nbrs = idx.neighbors(target, 0)
    if nbrs:
        removed_nbr = nbrs[0]
        idx.remove_edge(target, removed_nbr, 0)
        assert idx.version(target) == 0
        assert idx.distance_computations == old_dist

        idx.add_edge(target, removed_nbr, 0)
        assert idx.version(target) == 0
        assert idx.distance_computations == old_dist

    # move() increments version(target) by 1 and leaves others unchanged
    new_vec = np.zeros(4, dtype=np.float32)
    idx.move(target, new_vec)
    assert idx.version(target) == 1
    for nid in idx.ids():
        if nid != target:
            assert idx.version(nid) == 0

    # set_neighbors after move does not bump version
    idx.set_neighbors(target, 0, nbrs)
    assert idx.version(target) == 1
    assert idx.distance_computations == old_dist
