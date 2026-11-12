"""Parity and baseline integration tests against FAISS and hnswlib.

Tests compare K-HNSW (pure Python/NumPy) against external C++ reference implementations
on identical seeded datasets to ensure recall and structural behavior match.
"""

from __future__ import annotations

import numpy as np
import pytest

from khnsw.adapters.faiss_adapter import FaissAdapter
from khnsw.adapters.hnswlib_adapter import HnswlibAdapter
from khnsw.core.graph.hnsw import HNSWIndex
from khnsw.core.interfaces import ANNIndex

# ============================================================================
# hnswlib Parity Tests (skipped if hnswlib is not installed)
# ============================================================================


def test_hnswlib_adapter_crud() -> None:
    """Test basic lifecycle and CRUD methods of HnswlibAdapter."""
    pytest.importorskip("hnswlib")

    adapter = HnswlibAdapter(dim=4, m=16, ef_construction=100, seed=42)
    assert isinstance(adapter, ANNIndex)
    assert adapter.dim == 4
    assert adapter.size() == 0

    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = np.array([101, 102, 103], dtype=np.int64)

    adapter.add(ids, vecs)
    assert adapter.size() == 3
    assert adapter.contains(101)
    assert adapter.contains(102)
    assert not adapter.contains(999)

    # Search exact match
    labels, dists = adapter.search(vecs[0], k=1, ef=16)
    assert labels[0] == 101
    assert np.isclose(dists[0], 0.0, atol=1e-5)

    # Move node 101 to new position
    new_v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter.move(101, new_v)
    assert adapter.contains(101)
    labels, dists = adapter.search(new_v, k=1, ef=16)
    assert labels[0] == 101
    assert np.isclose(dists[0], 0.0, atol=1e-5)

    # Remove node 101
    adapter.remove(101)
    assert not adapter.contains(101)
    assert adapter.size() == 2


def test_hnswlib_vs_khnsw_parity() -> None:
    """Compare HNSWIndex (keep_pruned_connections=False) vs hnswlib on seeded data."""
    pytest.importorskip("hnswlib")

    rng = np.random.default_rng(42)
    n, dim, m, ef_c = 500, 16, 16, 100
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    khnsw_idx = HNSWIndex(
        dim=dim,
        m=m,
        ef_construction=ef_c,
        seed=42,
        keep_pruned_connections=False,
    )
    khnsw_idx.add(ids, vectors)

    hnsw_idx = HnswlibAdapter(
        space="l2",
        dim=dim,
        m=m,
        ef_construction=ef_c,
        seed=42,
        max_elements=n,
    )
    hnsw_idx.add(ids, vectors)

    n_queries = 50
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    # Brute force ground truth top 10
    bf_top10: list[set[int]] = []
    for q in queries:
        dists = np.sum((vectors - q) ** 2, axis=1)
        top10 = np.argsort(dists)[:10]
        bf_top10.append(set(ids[top10]))

    for ef in [10, 64]:
        khnsw_recalls = []
        hnsw_recalls = []
        for i, q in enumerate(queries):
            gt = bf_top10[i]
            k_labels, _ = khnsw_idx.search(q, k=10, ef=ef)
            h_labels, _ = hnsw_idx.search(q, k=10, ef=ef)
            khnsw_recalls.append(len(gt & set(k_labels)) / 10.0)
            hnsw_recalls.append(len(gt & set(h_labels)) / 10.0)

        mean_k = float(np.mean(khnsw_recalls))
        mean_h = float(np.mean(hnsw_recalls))
        print(
            f"[hnswlib parity] ef={ef}: K-HNSW recall@10={mean_k:.4f}, "
            f"hnswlib recall@10={mean_h:.4f}"
        )

        if ef == 64:
            assert mean_k >= 0.85, f"K-HNSW recall@10 ({mean_k}) below 0.85"
            assert mean_h >= 0.85, f"hnswlib recall@10 ({mean_h}) below 0.85"
            assert abs(mean_k - mean_h) <= 0.15, (
                f"Recall disparity {abs(mean_k - mean_h)} exceeds 0.15"
            )


# ============================================================================
# FAISS Parity Tests (skipped if faiss is not installed)
# ============================================================================


def test_faiss_adapter_crud() -> None:
    """Test lifecycle, error checks, and CRUD methods of FaissAdapter."""
    pytest.importorskip("faiss")

    adapter = FaissAdapter(dim=4, m=16, ef_construction=100)
    assert isinstance(adapter, ANNIndex)
    assert adapter.dim == 4
    assert adapter.size() == 0

    # Search empty index returns empty arrays
    res_i, res_d = adapter.search(np.zeros(4, dtype=np.float32), k=5)
    assert len(res_i) == 0 and len(res_d) == 0

    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = np.array([201, 202, 203], dtype=np.int64)

    adapter.add(ids, vecs)
    assert adapter.size() == 3
    assert adapter.contains(201)
    assert adapter.contains(202)
    assert not adapter.contains(999)

    # Validation errors on add
    with pytest.raises(ValueError, match="Length mismatch"):
        adapter.add(np.array([1], dtype=np.int64), np.zeros((2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="does not match dim"):
        adapter.add(np.array([301], dtype=np.int64), np.zeros((1, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="already exists"):
        adapter.add(np.array([201], dtype=np.int64), np.zeros((1, 4), dtype=np.float32))

    # Search exact vector
    labels, dists = adapter.search(vecs[0], k=1, ef=16)
    assert labels[0] == 201
    assert np.isclose(dists[0], 0.0, atol=1e-5)

    # Move node 201 to new position
    new_v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter.move(201, new_v)
    assert adapter.contains(201)
    labels, dists = adapter.search(new_v, k=1, ef=16)
    assert labels[0] == 201
    assert np.isclose(dists[0], 0.0, atol=1e-5)

    with pytest.raises(KeyError, match="not found"):
        adapter.move(999, new_v)

    with pytest.raises(ValueError, match="Vector shape"):
        adapter.move(201, np.zeros(5, dtype=np.float32))

    # Remove node 201
    adapter.remove(201)
    assert not adapter.contains(201)
    assert adapter.size() == 2

    with pytest.raises(KeyError, match="not found"):
        adapter.remove(201)

    # Node 201 should no longer be returned
    labels, _ = adapter.search(new_v, k=2, ef=16)
    assert 201 not in labels


def test_faiss_vs_khnsw_parity() -> None:
    """Compare HNSWIndex (keep_pruned_connections=False) vs FAISS on seeded data."""
    pytest.importorskip("faiss")

    rng = np.random.default_rng(42)
    n, dim, m, ef_c = 500, 16, 16, 100
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)

    khnsw_idx = HNSWIndex(
        dim=dim,
        m=m,
        ef_construction=ef_c,
        seed=42,
        keep_pruned_connections=False,
    )
    khnsw_idx.add(ids, vectors)

    faiss_idx = FaissAdapter(
        dim=dim,
        m=m,
        ef_construction=ef_c,
    )
    faiss_idx.add(ids, vectors)

    n_queries = 50
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    # Brute force ground truth top 10
    bf_top10: list[set[int]] = []
    for q in queries:
        dists = np.sum((vectors - q) ** 2, axis=1)
        top10 = np.argsort(dists)[:10]
        bf_top10.append(set(ids[top10]))

    for ef in [10, 64]:
        khnsw_recalls = []
        faiss_recalls = []
        for i, q in enumerate(queries):
            gt = bf_top10[i]
            k_labels, _ = khnsw_idx.search(q, k=10, ef=ef)
            f_labels, _ = faiss_idx.search(q, k=10, ef=ef)
            khnsw_recalls.append(len(gt & set(k_labels)) / 10.0)
            faiss_recalls.append(len(gt & set(f_labels)) / 10.0)

        mean_k = float(np.mean(khnsw_recalls))
        mean_f = float(np.mean(faiss_recalls))
        print(
            f"[FAISS parity] ef={ef}: K-HNSW recall@10={mean_k:.4f}, FAISS recall@10={mean_f:.4f}"
        )

        if ef == 64:
            assert mean_k >= 0.85, f"K-HNSW recall@10 ({mean_k}) below 0.85"
            assert mean_f >= 0.85, f"FAISS recall@10 ({mean_f}) below 0.85"
            assert abs(mean_k - mean_f) <= 0.10, (
                f"Recall disparity {abs(mean_k - mean_f)} exceeds 0.10"
            )
