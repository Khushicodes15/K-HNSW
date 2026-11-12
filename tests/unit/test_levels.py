"""Unit tests for hierarchical layer assignment via LevelGenerator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from khnsw.core.graph.levels import LevelGenerator, assign_level, compute_default_ml


def test_compute_default_ml() -> None:
    ml16 = compute_default_ml(16)
    assert math.isclose(ml16, 1.0 / math.log(16))

    with pytest.raises(ValueError, match="m must be > 1"):
        compute_default_ml(1)
    with pytest.raises(ValueError, match="m must be > 1"):
        compute_default_ml(0)


def test_assign_level_math_and_boundaries() -> None:
    ml = 1.0 / math.log(16)

    # U = 1.0 must give level 0 (floor(-ln(1.0) * mL) = floor(0) = 0)
    assert assign_level(1.0, ml) == 0

    # Extremely close to 1.0
    assert assign_level(0.999999, ml) == 0

    # Small U should give higher levels
    # For U = 1/16, -ln(1/16) * (1/ln(16)) = 1.0, floor(1.0) = 1
    assert assign_level(1.0 / 16.0, ml) == 1
    # For U = (1/16)^2, level = 2
    assert assign_level(1.0 / 256.0, ml) == 2


def test_assign_level_rejects_zero_and_invalid_u() -> None:
    ml = 1.0

    # U = 0 is explicitly forbidden to prevent -ln(0) = inf
    with pytest.raises(ValueError, match="U must be in"):
        assign_level(0.0, ml)

    # Negative U is forbidden
    with pytest.raises(ValueError, match="U must be in"):
        assign_level(-0.5, ml)

    # U > 1.0 is forbidden
    with pytest.raises(ValueError, match="U must be in"):
        assign_level(1.0001, ml)

    # m_l <= 0 is forbidden
    with pytest.raises(ValueError, match="m_l must be positive"):
        assign_level(0.5, 0.0)
    with pytest.raises(ValueError, match="m_l must be positive"):
        assign_level(0.5, -1.0)


def test_same_seed_gives_same_sequence() -> None:
    gen1 = LevelGenerator(m=16, seed=42)
    gen2 = LevelGenerator(m=16, seed=42)

    seq1 = [gen1.sample() for _ in range(100)]
    seq2 = [gen2.sample() for _ in range(100)]
    assert seq1 == seq2

    # Calling instance directly also samples
    gen3 = LevelGenerator(m=16, seed=42)
    seq3 = [gen3() for _ in range(100)]
    assert seq3 == seq1

    # Different seeds produce different sequences
    gen_other = LevelGenerator(m=16, seed=999)
    seq_other = [gen_other.sample() for _ in range(100)]
    assert seq_other != seq1


def test_generator_accepts_numpy_generator() -> None:
    rng = np.random.default_rng(123)
    gen = LevelGenerator(m=8, seed=rng)
    assert gen.m == 8
    lvl = gen.sample()
    assert isinstance(lvl, int) and lvl >= 0


def test_levels_are_non_negative_ints() -> None:
    gen = LevelGenerator(m=16, seed=12345)
    samples = gen.sample_many(1000)

    for lvl in samples:
        assert isinstance(lvl, int)
        assert not isinstance(lvl, bool)
        assert lvl >= 0


def test_geometric_decay_distribution() -> None:
    # Theoretical: P(level >= 1) = exp(-1 / m_L) = 1 / M
    m = 16
    gen = LevelGenerator(m=m, seed=2026)
    n_samples = 10000
    samples = gen.sample_many(n_samples)

    # Fraction at level >= 1
    count_ge_1 = sum(1 for lvl in samples if lvl >= 1)
    fraction_ge_1 = count_ge_1 / n_samples

    # Expected: 1/16 = 0.0625. Generous tolerance [0.045, 0.080]
    assert 0.045 <= fraction_ge_1 <= 0.080

    # Fraction at level >= 2 should be approximately 1 / (M^2) = 1/256 ~= 0.0039
    count_ge_2 = sum(1 for lvl in samples if lvl >= 2)
    fraction_ge_2 = count_ge_2 / n_samples
    assert 0.001 <= fraction_ge_2 <= 0.010

    # Test with M = 4: Expected fraction at level >= 1 is 1/4 = 0.25
    gen4 = LevelGenerator(m=4, seed=42)
    samples4 = gen4.sample_many(n_samples)
    fraction4 = sum(1 for lvl in samples4 if lvl >= 1) / n_samples
    assert 0.22 <= fraction4 <= 0.28


def test_generator_parameter_validation() -> None:
    with pytest.raises(ValueError, match="m must be > 1"):
        LevelGenerator(m=1)
    with pytest.raises(ValueError, match="m_l must be positive"):
        LevelGenerator(m=16, m_l=0.0)
    with pytest.raises(ValueError, match="size must be non-negative"):
        LevelGenerator(m=16).sample_many(-1)
