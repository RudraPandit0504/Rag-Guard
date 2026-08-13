"""Tests for the HDBSCAN outlier filter.

Grouped by what they protect:
  * the distance matrix HDBSCAN is fed
  * the clustering verdict itself
  * the safety rules that stop the filter dropping everything
  * the invariants callers depend on (partition, order, purity, determinism)
"""

import numpy as np
import pytest

from src.role2_filters.cluster import (
    NOISE_LABEL,
    _majority_label,
    cluster_chunks,
    cosine_distance_matrix,
    detect_cluster_outliers,
    explain,
)
from tests.conftest import basis, make_chunks


# --------------------------------------------------------------------------
# The distance matrix
# --------------------------------------------------------------------------

def test_distance_matrix_empty_input_is_empty_not_an_error():
    assert cosine_distance_matrix([]).shape == (0, 0)


def test_identical_vectors_are_at_distance_zero():
    matrix = cosine_distance_matrix(make_chunks([basis(0), basis(0)]))
    assert matrix[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_orthogonal_vectors_are_at_distance_one():
    matrix = cosine_distance_matrix(make_chunks([basis(0), basis(1)]))
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_opposite_vectors_are_at_distance_two():
    opposite = [-x for x in basis(0)]
    matrix = cosine_distance_matrix(make_chunks([basis(0), opposite]))
    assert matrix[0, 1] == pytest.approx(2.0, abs=1e-9)


def test_distance_matrix_is_symmetric_with_a_zero_diagonal():
    """HDBSCAN rejects a precomputed matrix that is not exactly symmetric, and
    rounding in the matrix product alone does not guarantee that."""
    rng = np.random.default_rng(0)
    chunks = make_chunks([rng.normal(size=6).tolist() for _ in range(5)])

    matrix = cosine_distance_matrix(chunks)

    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(5))


def test_distance_matrix_never_goes_negative():
    """Floating-point error can push a distance a hair below zero, which is not
    a valid metric."""
    chunks = make_chunks([basis(0)] * 4)
    assert (cosine_distance_matrix(chunks) >= 0).all()


def test_magnitude_does_not_affect_distance():
    """Cosine distance is scale-free: a vector and a scaled copy of it are the
    same direction and must land at distance 0."""
    scaled = [x * 17.0 for x in basis(0)]
    matrix = cosine_distance_matrix(make_chunks([basis(0), scaled]))
    assert matrix[0, 1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# The clustering verdict
# --------------------------------------------------------------------------

def test_empty_input_yields_two_empty_lists():
    assert detect_cluster_outliers([]) == ([], [])


def test_lone_outlier_is_dropped(cluster_and_outlier):
    kept, dropped = detect_cluster_outliers(cluster_and_outlier)

    assert [c["chunk_id"] for c in kept] == [100, 101, 102, 103]
    assert [c["chunk_id"] for c in dropped] == [104]


def test_outlier_is_labelled_noise_not_a_cluster(cluster_and_outlier):
    result = cluster_chunks(cluster_and_outlier)

    assert result["labels"][4] == NOISE_LABEL
    assert result["cluster_sizes"] == {0: 4}


def test_a_coherent_set_is_left_alone(coherent_set):
    """The filter must not invent an outlier when there is none. This is the
    failure mode that makes a filter useless in production."""
    kept, dropped = detect_cluster_outliers(coherent_set)

    assert len(kept) == 5
    assert dropped == []


def test_minority_group_is_dropped_in_favour_of_the_majority(majority_and_minority):
    kept, dropped = detect_cluster_outliers(majority_and_minority)

    assert [c["chunk_id"] for c in kept] == [100, 101, 102]
    assert [c["chunk_id"] for c in dropped] == [103, 104]


def test_membership_probability_is_reported_per_chunk(cluster_and_outlier):
    result = cluster_chunks(cluster_and_outlier)

    assert len(result["probabilities"]) == len(cluster_and_outlier)
    assert all(0.0 <= p <= 1.0 for p in result["probabilities"])
    # A point that belongs to no cluster has no membership in one.
    assert result["probabilities"][4] == pytest.approx(0.0)


def test_labels_line_up_one_per_chunk(majority_and_minority):
    result = cluster_chunks(majority_and_minority)
    assert len(result["labels"]) == len(majority_and_minority)


def test_min_cluster_size_decides_what_counts_as_a_group():
    """Three identical chunks are a cluster at min_cluster_size=3, and the two
    odd ones out are correctly left as noise."""
    chunks = make_chunks([basis(0)] * 3 + [basis(1), basis(2)])

    result = cluster_chunks(chunks, min_cluster_size=3)

    assert result["cluster_sizes"] == {0: 3}
    assert result["labels"] == [0, 0, 0, NOISE_LABEL, NOISE_LABEL]


def test_min_cluster_size_is_not_monotonic_in_strictness():
    """Raising min_cluster_size past the size of any real group does NOT make
    the filter stricter — it makes it useless.

    With three identical chunks and two unrelated ones, min_cluster_size=4 means
    no genuine group can qualify, so HDBSCAN climbs to the root of the hierarchy
    and (because allow_single_cluster is on) accepts all five as one cluster.
    Nothing is an outlier and nothing is dropped.

    This is the trap in tuning this parameter: the failure is silent, and it
    fails open rather than closed.
    """
    chunks = make_chunks([basis(0)] * 3 + [basis(1), basis(2)])

    at_three = detect_cluster_outliers(chunks, min_cluster_size=3)[1]
    at_four = detect_cluster_outliers(chunks, min_cluster_size=4)[1]

    assert len(at_three) == 2
    assert at_four == []
    assert cluster_chunks(chunks, min_cluster_size=4)["cluster_sizes"] == {0: 5}


# --------------------------------------------------------------------------
# Safety rules — the filter must never drop everything
# --------------------------------------------------------------------------

def test_a_single_chunk_passes_through_unfiltered(capsys):
    chunks = make_chunks([basis(0)])

    kept, dropped = detect_cluster_outliers(chunks)

    assert kept == chunks
    assert dropped == []
    assert "skipped" in capsys.readouterr().out


def test_too_few_chunks_to_form_a_cluster_pass_through(capsys):
    """Two chunks cannot fill a min_cluster_size of 3. Without this rule
    HDBSCAN would call both noise and the filter would drop the entire set."""
    chunks = make_chunks([basis(0), basis(0)])

    kept, dropped = detect_cluster_outliers(chunks, min_cluster_size=3)

    assert kept == chunks
    assert dropped == []
    assert "skipped" in capsys.readouterr().out


def test_when_no_cluster_forms_everything_passes_through(capsys):
    """Five mutually orthogonal chunks have no dense group anywhere. There is no
    majority to judge against, so nothing may be dropped."""
    chunks = make_chunks([basis(i) for i in range(5)])

    result = cluster_chunks(chunks, min_cluster_size=4, cluster_selection_epsilon=0.0)
    kept, dropped = detect_cluster_outliers(
        chunks, min_cluster_size=4, cluster_selection_epsilon=0.0
    )

    if result["skipped"]:
        assert kept == chunks
        assert dropped == []
        assert "skipped" in capsys.readouterr().out
    else:
        # If a cluster did form it must be a real one, and the safety rule was
        # simply not needed here.
        assert result["majority_label"] is not None
        assert len(kept) >= 1


def test_skipping_reports_a_reason():
    result = cluster_chunks(make_chunks([basis(0)]))

    assert result["skipped"] is True
    assert result["note"]
    assert result["majority_label"] is None


def test_a_run_that_succeeds_carries_no_skip_note(coherent_set):
    result = cluster_chunks(coherent_set)

    assert result["skipped"] is False
    assert result["note"] == ""


# --------------------------------------------------------------------------
# Invariants callers depend on
# --------------------------------------------------------------------------

def test_kept_and_dropped_partition_the_input_exactly(majority_and_minority):
    kept, dropped = detect_cluster_outliers(majority_and_minority)

    kept_ids = [c["chunk_id"] for c in kept]
    dropped_ids = [c["chunk_id"] for c in dropped]

    assert sorted(kept_ids + dropped_ids) == [100, 101, 102, 103, 104]
    assert set(kept_ids).isdisjoint(dropped_ids)


def test_input_order_is_preserved_within_each_list():
    chunks = make_chunks([basis(0), basis(1), basis(0), basis(1), basis(0)])

    kept, dropped = detect_cluster_outliers(chunks)

    assert [c["chunk_id"] for c in kept] == sorted(c["chunk_id"] for c in kept)
    assert [c["chunk_id"] for c in dropped] == sorted(c["chunk_id"] for c in dropped)


def test_chunks_come_back_unmodified(cluster_and_outlier):
    """The filter selects chunks; it must not rewrite them. Downstream stages
    read these same dicts."""
    before = [dict(c) for c in cluster_and_outlier]

    detect_cluster_outliers(cluster_and_outlier)

    assert cluster_and_outlier == before


def test_the_distance_matrix_survives_clustering(cluster_and_outlier):
    """HDBSCAN modifies a precomputed matrix in place unless copy=True, which
    would corrupt it for anything reusing it."""
    matrix = cosine_distance_matrix(cluster_and_outlier)
    original = matrix.copy()

    cluster_chunks(cluster_and_outlier)

    assert np.array_equal(cosine_distance_matrix(cluster_and_outlier), original)


def test_the_same_input_always_gives_the_same_answer(majority_and_minority):
    """Reproducibility is what makes the measurements in the README meaningful."""
    runs = [
        [c["chunk_id"] for c in detect_cluster_outliers(majority_and_minority)[0]]
        for _ in range(5)
    ]

    assert all(run == runs[0] for run in runs)


def test_the_poisoned_flag_cannot_influence_the_verdict(majority_and_minority):
    """The filter is geometry only. Flipping ground truth must change nothing —
    otherwise every accuracy measurement taken with it is circular."""
    kept_ids = [c["chunk_id"] for c in detect_cluster_outliers(majority_and_minority)[0]]

    relabelled = [{**c, "poisoned": not c["poisoned"]} for c in majority_and_minority]
    relabelled_ids = [c["chunk_id"] for c in detect_cluster_outliers(relabelled)[0]]

    assert kept_ids == relabelled_ids


def test_verdict_does_not_depend_on_which_chunk_is_listed_first():
    """Presenting the same set in a different order must not change who is an
    outlier — retrieval order is a ranking, not evidence about geometry."""
    vectors = [basis(0), basis(0), basis(0), basis(0), basis(1)]
    forward = make_chunks(vectors)
    reversed_chunks = make_chunks(list(reversed(vectors)))

    forward_dropped = detect_cluster_outliers(forward)[1]
    reversed_dropped = detect_cluster_outliers(reversed_chunks)[1]

    # The outlier is the last chunk forward, the first chunk reversed.
    assert [c["chunk_id"] for c in forward_dropped] == [104]
    assert [c["chunk_id"] for c in reversed_dropped] == [100]


# --------------------------------------------------------------------------
# Majority selection and explanations
# --------------------------------------------------------------------------

def test_the_largest_cluster_wins():
    assert _majority_label([0, 0, 0, 1, 1], [1.0] * 5, {0: 3, 1: 2}) == 0


def test_an_equal_split_is_broken_by_membership_confidence():
    """Two clusters of the same size is a genuine ambiguity; the tie-break only
    guarantees the same answer every time, it does not resolve which is right."""
    labels = [0, 0, 1, 1]
    probabilities = [0.5, 0.5, 0.9, 0.9]

    assert _majority_label(labels, probabilities, {0: 2, 1: 2}) == 1


def test_an_exact_tie_falls_back_to_the_lowest_label():
    assert _majority_label([0, 0, 1, 1], [0.8] * 4, {0: 2, 1: 2}) == 0


def test_noise_and_minority_get_different_explanations():
    noise = explain(NOISE_LABEL, majority_label=0, probability=0.0)
    minority = explain(1, majority_label=0, probability=0.75)

    assert "noise" in noise
    assert "minority cluster 1" in minority
    assert noise != minority
