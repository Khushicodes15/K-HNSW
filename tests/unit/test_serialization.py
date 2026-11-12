"""Unit tests for flat array graph serialization (src/khnsw/core/graph/serialization.py).

Covers:
  - Round trip on seeded HNSWIndex (exact parity for IDs, vectors, levels,
    entry point, forward/reverse neighbors, and search query results).
  - Empty index serialization round trip.
  - Single node index round trip.
  - keep_pruned_connections parameter preserved for both True and False settings.
  - Corrupted/truncated file detection.
  - Unsupported format_version rejection.
  - Hand-edited invalid edges (self-loop, over-degree, non-existent target ID) raising errors.
  - Loaded index accepting subsequent add() batches and maintaining invariants.
  - Hypothesis property testing on random index builds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from khnsw.core.graph.audit import validate_invariants
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.graph.serialization import load, save


def test_round_trip_seeded_hnsw_exact_parity(tmp_path: Path) -> None:
    """Seeded HNSWIndex round trip preserves exact topology, vectors, and query results."""
    seed = 42
    n = 60
    dim = 8
    m = 4
    ef_construction = 32

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    original = HNSWIndex(dim=dim, m=m, ef_construction=ef_construction, seed=seed)
    original.add(ids, data)

    file_path = tmp_path / "test_seeded_hnsw.npz"
    save(original, file_path)

    loaded = load(file_path)

    # 1. Structural invariants
    assert validate_invariants(loaded) == []

    # 2. Metadata parity
    assert loaded.dim == original.dim
    assert loaded.size() == original.size()
    assert loaded.max_level() == original.max_level()
    assert loaded.entry_point() == original.entry_point()
    assert loaded.ids() == original.ids()

    # 3. Node vectors, levels, and forward/reverse neighbors parity
    for nid in ids:
        nid_int = int(nid)
        np.testing.assert_array_equal(loaded.vector(nid_int), original.vector(nid_int))
        assert loaded.node_level(nid_int) == original.node_level(nid_int)
        assert loaded.version(nid_int) == original.version(nid_int)

        max_l = loaded.node_level(nid_int)
        for lvl in range(max_l + 1):
            assert loaded.neighbors(nid_int, lvl) == original.neighbors(nid_int, lvl)
            assert loaded.in_neighbors(nid_int, lvl) == original.in_neighbors(nid_int, lvl)

    # 4. Search query result parity
    queries = rng.standard_normal((10, dim)).astype(np.float32)
    for q in queries:
        orig_ids, orig_dists = original.search(q, k=5, ef=16)
        loaded_ids, loaded_dists = loaded.search(q, k=5, ef=16)
        np.testing.assert_array_equal(loaded_ids, orig_ids)
        np.testing.assert_allclose(loaded_dists, orig_dists, rtol=1e-5, atol=1e-5)


def test_round_trip_empty_index(tmp_path: Path) -> None:
    """Empty index can be serialized and deserialized cleanly."""
    orig = HNSWIndex(dim=6, m=4)
    file_path = tmp_path / "empty.npz"
    save(orig, file_path)

    loaded = load(file_path)
    assert validate_invariants(loaded) == []
    assert loaded.size() == 0
    assert loaded.max_level() == -1
    assert loaded.entry_point() is None
    assert loaded.ids() == []
    assert loaded.dim == 6

    # Queries on empty loaded index return empty arrays
    res_ids, res_dists = loaded.search(np.zeros(6, dtype=np.float32), k=3, ef=5)
    assert len(res_ids) == 0
    assert len(res_dists) == 0


def test_round_trip_single_node(tmp_path: Path) -> None:
    """Single node index round trip."""
    orig = HNSWIndex(dim=4, m=4)
    vec = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    orig.add(np.array([101], dtype=np.int64), vec)

    file_path = tmp_path / "single_node.npz"
    save(orig, file_path)

    loaded = load(file_path)
    assert validate_invariants(loaded) == []
    assert loaded.size() == 1
    assert loaded.ids() == [101]
    assert loaded.entry_point() == 101
    np.testing.assert_array_equal(loaded.vector(101), vec[0])


@pytest.mark.parametrize("setting", [False, True])
def test_keep_pruned_connections_setting_preserved(tmp_path: Path, setting: bool) -> None:
    """keep_pruned_connections setting is preserved across save and load."""
    orig = HNSWIndex(dim=4, m=4, keep_pruned_connections=setting)
    orig.add(np.array([1], dtype=np.int64), np.ones((1, 4), dtype=np.float32))

    file_path = tmp_path / f"keep_pruned_{setting}.npz"
    save(orig, file_path)

    loaded = load(file_path)
    assert loaded.keep_pruned_connections is setting


def test_corrupted_file_raises_clear_error(tmp_path: Path) -> None:
    """Corrupted or truncated file raises ValueError."""
    corrupt_path = tmp_path / "corrupt.npz"
    corrupt_path.write_bytes(b"PK\x03\x04garbage_truncated_zip_content")

    with pytest.raises(ValueError, match="Failed to load NPZ archive"):
        load(corrupt_path)


def test_wrong_format_version_raises_error(tmp_path: Path) -> None:
    """Unsupported format_version raises ValueError."""
    orig = HNSWIndex(dim=4, m=4)
    file_path = tmp_path / "version_test.npz"
    save(orig, file_path)

    # Load arrays, tamper with format_version, and re-save
    with np.load(file_path, allow_pickle=False) as data:
        tampered = {k: data[k] for k in data.files}
    tampered["params"][6] = 999  # Unsupported version
    np.savez_compressed(file_path, **tampered)

    with pytest.raises(ValueError, match="Unsupported format_version 999"):
        load(file_path)


def test_hand_edited_bad_edges_raise_errors(tmp_path: Path) -> None:
    """Tampered edge arrays (non-existent target, self-loop, over-degree) raise errors."""
    rng = np.random.default_rng(7)
    orig = HNSWIndex(dim=4, m=4, ef_construction=16, seed=7)
    orig.add(np.arange(10, dtype=np.int64), rng.standard_normal((10, 4)).astype(np.float32))

    file_path = tmp_path / "valid.npz"
    save(orig, file_path)

    with np.load(file_path, allow_pickle=False) as data:
        base_arrays = {k: data[k] for k in data.files}

    # Case 1: Non-existent neighbor target ID 9999
    bad_edge = dict(base_arrays)
    indices = bad_edge["edge_level_0_indices"].copy()
    indices[0] = 9999
    bad_edge["edge_level_0_indices"] = indices
    bad_path1 = tmp_path / "bad_target.npz"
    np.savez_compressed(bad_path1, **bad_edge)
    with pytest.raises(ValueError, match="references non-existent neighbor 9999"):
        load(bad_path1)

    # Case 2: Self-loop (node 0 has neighbor 0)
    bad_self = dict(base_arrays)
    indices_self = bad_self["edge_level_0_indices"].copy()
    indices_self[0] = 0  # node 0 connects to node 0
    bad_self["edge_level_0_indices"] = indices_self
    bad_path2 = tmp_path / "bad_self.npz"
    np.savez_compressed(bad_path2, **bad_self)
    with pytest.raises(ValueError, match="Self-loop detected on node 0"):
        load(bad_path2)

    # Case 3: Over-degree (node 0 out-degree exceeds m0)
    bad_deg = dict(base_arrays)
    # Give node 0 an out-degree exceeding m0 (m0 = 8) while preserving monotonicity
    extra_edges = 20
    indptr_over = bad_deg["edge_level_0_indptr"].copy()
    indptr_over[1:] += extra_edges
    new_indices = np.zeros(int(indptr_over[-1]), dtype=np.int64)
    bad_deg["edge_level_0_indptr"] = indptr_over
    bad_deg["edge_level_0_indices"] = new_indices
    bad_path3 = tmp_path / "bad_deg.npz"
    np.savez_compressed(bad_path3, **bad_deg)
    with pytest.raises(ValueError, match="out-degree.*> max"):
        load(bad_path3)


def test_loaded_index_accepts_further_add_and_maintains_invariants(tmp_path: Path) -> None:
    """Reconstituted index can accept subsequent add() batches and maintain invariants."""
    rng = np.random.default_rng(21)
    dim = 8
    orig = HNSWIndex(dim=dim, m=4, ef_construction=32, seed=21)
    orig.add(np.arange(20, dtype=np.int64), rng.standard_normal((20, dim)).astype(np.float32))

    file_path = tmp_path / "partial.npz"
    save(orig, file_path)

    loaded = load(file_path)
    assert loaded.size() == 20

    # Add 10 more nodes
    new_ids = np.arange(20, 30, dtype=np.int64)
    new_vecs = rng.standard_normal((10, dim)).astype(np.float32)
    loaded.add(new_ids, new_vecs)

    assert loaded.size() == 30
    assert validate_invariants(loaded) == []

    # Newly added nodes must be queryable
    q = new_vecs[0]
    ret_ids, _ = loaded.search(q, k=1, ef=10)
    assert ret_ids[0] == 20


# ---------------------------------------------------------------------------
# Hypothesis: random small builds round trip cleanly
# ---------------------------------------------------------------------------


class TestHypothesisSerialization:
    @given(
        n=st.integers(min_value=1, max_value=40),
        dim=st.integers(min_value=2, max_value=8),
        m=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_build_round_trip(
        self, tmp_path_factory: pytest.TempPathFactory, n: int, dim: int, m: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        idx = HNSWIndex(dim=dim, m=m, ef_construction=max(m, 16), seed=seed)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        idx.add(np.arange(n, dtype=np.int64), vecs)

        tmp_dir = tmp_path_factory.mktemp("hypo_ser")
        file_path = tmp_dir / f"idx_{seed}.npz"
        save(idx, file_path)

        loaded = load(file_path)
        assert validate_invariants(loaded) == []
        assert loaded.size() == idx.size()
        assert loaded.max_level() == idx.max_level()
        assert loaded.entry_point() == idx.entry_point()
        assert loaded.ids() == idx.ids()
