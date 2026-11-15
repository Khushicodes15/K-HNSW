"""Unit tests for benchmarks/datasets.py (Member 4 dataset pipeline).

Tests:
1. synthetic_gaussian shape, dtype, and reproducibility.
2. load_sift round-trips a hand-crafted .fvecs binary blob.
3. load_glove parses a minimal GloVe text fixture.
4. generate_sift_like shape, non-negativity, and dtype.
5. Error paths: file not found, bad format.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from benchmarks.datasets import (
    generate_sift_like,
    load_glove,
    load_sift,
    synthetic_gaussian,
)


# ---------------------------------------------------------------------------
# 1. synthetic_gaussian
# ---------------------------------------------------------------------------


class TestSyntheticGaussian:
    def test_shape_and_dtype(self) -> None:
        X = synthetic_gaussian(n=100, d=16, seed=0)
        assert X.shape == (100, 16)
        assert X.dtype == np.float32

    def test_reproducible(self) -> None:
        X1 = synthetic_gaussian(n=50, d=8, seed=42)
        X2 = synthetic_gaussian(n=50, d=8, seed=42)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self) -> None:
        X1 = synthetic_gaussian(n=50, d=8, seed=1)
        X2 = synthetic_gaussian(n=50, d=8, seed=2)
        assert not np.array_equal(X1, X2)

    def test_approximately_zero_mean(self) -> None:
        X = synthetic_gaussian(n=5000, d=32, seed=99)
        assert abs(float(X.mean())) < 0.05


# ---------------------------------------------------------------------------
# 2. load_sift — round-trip test with crafted binary blob
# ---------------------------------------------------------------------------


def _make_fvecs(vectors: list[list[float]]) -> bytes:
    """Encode a list-of-lists as .fvecs binary data."""
    parts: list[bytes] = []
    for vec in vectors:
        d = len(vec)
        parts.append(struct.pack("<i", d))
        parts.append(struct.pack(f"<{d}f", *vec))
    return b"".join(parts)


class TestLoadSift:
    def test_round_trip(self, tmp_path: Path) -> None:
        vecs = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        fvecs_bytes = _make_fvecs(vecs)
        p = tmp_path / "test.fvecs"
        p.write_bytes(fvecs_bytes)

        X = load_sift(p)
        assert X.shape == (2, 4)
        assert X.dtype == np.float32
        np.testing.assert_allclose(X[0], [1.0, 2.0, 3.0, 4.0], atol=1e-5)
        np.testing.assert_allclose(X[1], [5.0, 6.0, 7.0, 8.0], atol=1e-5)

    def test_single_vector(self, tmp_path: Path) -> None:
        vecs = [[10.0, 20.0]]
        p = tmp_path / "single.fvecs"
        p.write_bytes(_make_fvecs(vecs))
        X = load_sift(p)
        assert X.shape == (1, 2)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_sift("/nonexistent/path/file.fvecs")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.fvecs"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="too small"):
            load_sift(p)

    def test_dtype_is_float32(self, tmp_path: Path) -> None:
        vecs = [[0.1, 0.2, 0.3]]
        p = tmp_path / "dtype.fvecs"
        p.write_bytes(_make_fvecs(vecs))
        X = load_sift(p)
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. load_glove — parse a minimal text fixture
# ---------------------------------------------------------------------------


GLOVE_FIXTURE = """\
the 0.1 0.2 0.3 0.4
of -0.1 -0.2 -0.3 -0.4
and 0.5 0.6 0.7 0.8
"""


class TestLoadGlove:
    def test_shape_and_values(self, tmp_path: Path) -> None:
        p = tmp_path / "glove.txt"
        p.write_text(GLOVE_FIXTURE, encoding="utf-8")
        X = load_glove(p, d=4)
        assert X.shape == (3, 4)
        assert X.dtype == np.float32
        np.testing.assert_allclose(X[0], [0.1, 0.2, 0.3, 0.4], atol=1e-5)

    def test_wrong_dim_skipped(self, tmp_path: Path) -> None:
        # Only lines with exactly d=3 floats should be kept
        content = "word1 1.0 2.0 3.0\nword2 1.0 2.0\nword3 4.0 5.0 6.0\n"
        p = tmp_path / "partial.txt"
        p.write_text(content, encoding="utf-8")
        X = load_glove(p, d=3)
        assert X.shape == (2, 3)  # word2 is skipped

    def test_max_vectors(self, tmp_path: Path) -> None:
        p = tmp_path / "glove.txt"
        p.write_text(GLOVE_FIXTURE, encoding="utf-8")
        X = load_glove(p, d=4, max_vectors=2)
        assert X.shape[0] == 2

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_glove("/no/such/file.txt", d=50)

    def test_no_valid_vectors_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.txt"
        p.write_text("word 1.0 2.0\n", encoding="utf-8")  # d=2, we ask for d=5
        with pytest.raises(ValueError, match="No valid"):
            load_glove(p, d=5)


# ---------------------------------------------------------------------------
# 4. generate_sift_like
# ---------------------------------------------------------------------------


class TestGenerateSiftLike:
    def test_shape_and_dtype(self) -> None:
        X = generate_sift_like(n=200, d=128, seed=0)
        assert X.shape == (200, 128)
        assert X.dtype == np.float32

    def test_non_negative(self) -> None:
        X = generate_sift_like(n=500, d=128, seed=1)
        assert float(X.min()) >= 0.0

    def test_reproducible(self) -> None:
        X1 = generate_sift_like(n=50, d=16, seed=5)
        X2 = generate_sift_like(n=50, d=16, seed=5)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self) -> None:
        X1 = generate_sift_like(n=50, d=16, seed=5)
        X2 = generate_sift_like(n=50, d=16, seed=6)
        assert not np.array_equal(X1, X2)

    def test_custom_dimension(self) -> None:
        X = generate_sift_like(n=100, d=64, seed=0)
        assert X.shape == (100, 64)
