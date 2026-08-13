"""Shared fixtures for the Role 2 filter tests.

Every unit test here builds its chunks by hand rather than calling retrieve(),
so the suite runs without a database, without network access and without
loading the embedding model. Only tests marked `integration` touch the stores.
"""

import numpy as np
import pytest


# Small enough to write vectors out by hand, large enough that basis vectors
# are genuinely orthogonal.
DIMENSIONS = 6


def basis(index: int, dimensions: int = DIMENSIONS) -> list[float]:
    """A unit vector pointing along one axis.

    Two different basis vectors are exactly orthogonal, so their cosine
    distance is exactly 1.0 — no floating-point slack for a test to trip on.
    """
    vector = np.zeros(dimensions)
    vector[index] = 1.0
    return vector.tolist()


def make_chunk(chunk_id: int, vector: list[float], poisoned: bool = False) -> dict:
    """A chunk in the shape retrieve() returns.

    Carries every field the filters read, so a test never passes because a key
    happened to be missing.
    """
    return {
        "chunk_id": chunk_id,
        "text": f"chunk {chunk_id}",
        "hash": f"hash-{chunk_id}",
        "created_at": "2026-01-01T00:00:00Z",
        "poisoned": poisoned,
        "score": 0.9,
        "vector": list(vector),
    }


def make_chunks(vectors: list[list[float]], poisoned_indices: set[int] = frozenset()) -> list[dict]:
    """Chunks numbered from 100, so ids are visibly distinct from list indices."""
    return [
        make_chunk(100 + i, vector, poisoned=i in poisoned_indices)
        for i, vector in enumerate(vectors)
    ]


@pytest.fixture
def cluster_and_outlier() -> list[dict]:
    """Four chunks on one axis plus one orthogonal to them.

    The four are identical, so they are at distance 0 from each other and form
    an unambiguous dense group; the fifth is at distance 1.0 from all of them.
    This is the clearest possible case of a single outlier.
    """
    return make_chunks([basis(0), basis(0), basis(0), basis(0), basis(1)])


@pytest.fixture
def majority_and_minority() -> list[dict]:
    """Three identical chunks on one axis, two identical on another.

    A majority group and a competing minority group, with nothing in between.
    """
    return make_chunks([basis(0), basis(0), basis(0), basis(1), basis(1)])


@pytest.fixture
def coherent_set() -> list[dict]:
    """Five identical chunks — one group, no outlier, nothing to drop."""
    return make_chunks([basis(0)] * 5)
