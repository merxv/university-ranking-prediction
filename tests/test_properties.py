"""Property-based tests: invariants that must hold for any input, not just examples."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from urps.data import parse_number, parse_rank
from urps.model import rank_band


@given(st.integers(min_value=1, max_value=5000), st.booleans())
def test_single_rank_roundtrip(rank, tied):
    text = f"={rank}" if tied else str(rank)
    assert parse_rank(text) == (rank, rank)


@given(st.integers(min_value=1, max_value=5000), st.integers(min_value=0, max_value=500))
def test_band_rank_roundtrip(lower, width):
    assert parse_rank(f"{lower}-{lower + width}") == (lower, lower + width)


@given(st.integers(min_value=0, max_value=10**9))
def test_thousands_separator_roundtrip(n):
    assert parse_number(f"{n:,}") == n


@given(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=1, max_size=400))
def test_rank_band_is_monotonic(scores):
    """A higher predicted score never receives a worse (larger) rank."""
    labels = rank_band(np.array(scores))
    lower = [int(label.split("-")[0]) for label in labels]
    order = np.argsort(-np.array(scores), kind="stable")
    ranked = [lower[i] for i in order]
    assert ranked == sorted(ranked)
    assert len(labels) == len(scores)
