"""Tests for apply_math_filters — the two-stage chain Role 2 exposes.

These cover the contract the rest of the pipeline relies on: the shape that
comes back, the fields on a dropped chunk, and the fact that swapping the
outlier algorithm does not change any of it.
"""

import pytest

from src.config import MIN_CLUSTER_SIZE
from src.role2_filters.filters import apply_math_filters, apply_math_filters_verbose
from tests.conftest import basis, make_chunks


RETRIEVE_FIELDS = {"chunk_id", "text", "hash", "created_at", "poisoned", "score", "vector"}


def test_empty_input_survives_both_stages():
    assert apply_math_filters([]) == []


def test_output_is_the_same_shape_retrieve_produces(coherent_set):
    """The stages chain only because every one of them takes and returns this
    exact shape."""
    survivors = apply_math_filters(coherent_set)

    assert isinstance(survivors, list)
    for chunk in survivors:
        assert RETRIEVE_FIELDS <= set(chunk)


def test_surviving_chunks_keep_their_original_fields_untouched(coherent_set):
    survivors = apply_math_filters(coherent_set)

    by_id = {c["chunk_id"]: c for c in coherent_set}
    for chunk in survivors:
        assert chunk == by_id[chunk["chunk_id"]]


def test_the_plain_function_returns_exactly_the_verbose_survivors(majority_and_minority):
    """The plain function wraps the verbose one, so the two can never disagree.
    A test pins that, because a future edit could easily split them."""
    plain = apply_math_filters(majority_and_minority)
    verbose = apply_math_filters_verbose(majority_and_minority)["kept"]

    assert plain == verbose


def test_every_chunk_is_either_kept_or_dropped(majority_and_minority):
    result = apply_math_filters_verbose(majority_and_minority)

    kept_ids = [c["chunk_id"] for c in result["kept"]]
    dropped_ids = [c["chunk_id"] for c in result["dropped"]]

    assert sorted(kept_ids + dropped_ids) == [100, 101, 102, 103, 104]
    assert set(kept_ids).isdisjoint(dropped_ids)


def test_a_dropped_chunk_says_which_filter_removed_it_and_why(majority_and_minority):
    result = apply_math_filters_verbose(majority_and_minority)

    assert result["dropped"], "expected the minority group to be dropped"
    for chunk in result["dropped"]:
        assert chunk["filter"] in {"cluster", "consistency"}
        assert chunk["reason"]
        # The diagnostic fields are added to the chunk, not substituted for it.
        assert RETRIEVE_FIELDS <= set(chunk)


def test_the_cluster_stage_names_itself_in_its_reason(cluster_and_outlier):
    result = apply_math_filters_verbose(cluster_and_outlier)

    cluster_drops = [c for c in result["dropped"] if c["filter"] == "cluster"]
    assert cluster_drops
    assert all("noise" in c["reason"] or "cluster" in c["reason"] for c in cluster_drops)


# --------------------------------------------------------------------------
# Choosing the outlier algorithm
# --------------------------------------------------------------------------

def test_hdbscan_is_the_default(cluster_and_outlier):
    """The default path must be the clustering one, not the old centroid rule."""
    explicit = apply_math_filters_verbose(cluster_and_outlier, outlier_method="hdbscan")
    default = apply_math_filters_verbose(cluster_and_outlier)

    assert default == explicit
    assert any(c["filter"] == "cluster" for c in default["dropped"])


def test_the_centroid_method_is_still_available(cluster_and_outlier):
    """Kept so the two algorithms can be measured against each other."""
    result = apply_math_filters_verbose(cluster_and_outlier, outlier_method="centroid")

    assert all(c["filter"] != "cluster" for c in result["dropped"])
    assert any(c["filter"] == "outlier" for c in result["dropped"])


def test_the_two_methods_disagree_on_this_input(cluster_and_outlier):
    """If they agreed everywhere, swapping them would be pointless. The
    orthogonal chunk sits at distance 0.8 from the centroid, past the 0.30
    cutoff, so both drop it — but they record different reasons."""
    hdbscan = apply_math_filters_verbose(cluster_and_outlier, outlier_method="hdbscan")
    centroid = apply_math_filters_verbose(cluster_and_outlier, outlier_method="centroid")

    assert {c["filter"] for c in hdbscan["dropped"]} != {c["filter"] for c in centroid["dropped"]}


def test_an_unknown_method_fails_loudly(coherent_set):
    """A typo must not silently fall through to one algorithm or the other."""
    with pytest.raises(ValueError, match="unknown outlier_method"):
        apply_math_filters(coherent_set, outlier_method="kmeans")


def test_min_cluster_size_reaches_the_cluster_stage():
    """The parameter has to travel from the call site through to HDBSCAN, which
    is easy to break silently."""
    chunks = make_chunks([basis(0)] * 3 + [basis(1), basis(2)])

    strict = apply_math_filters_verbose(chunks, min_cluster_size=4)
    normal = apply_math_filters_verbose(chunks, min_cluster_size=3)

    # At 4 no cluster forms, the safety rule fires and nothing is dropped by
    # the cluster stage; at 3 the group of 3 is the majority.
    assert not [c for c in strict["dropped"] if c["filter"] == "cluster"]
    assert [c for c in normal["dropped"] if c["filter"] == "cluster"]


def test_the_centroid_threshold_still_bites_when_that_method_is_chosen(cluster_and_outlier):
    """Counts only the outlier stage's own drops. The totals are equal either
    way, because a chunk the loose threshold spares is then removed by the
    consistency stage instead — which is exactly why the two must be counted
    apart."""
    def outlier_drops(threshold):
        result = apply_math_filters_verbose(
            cluster_and_outlier, outlier_method="centroid", outlier_threshold=threshold
        )
        return [c for c in result["dropped"] if c["filter"] == "outlier"]

    assert outlier_drops(1.5) == []
    assert len(outlier_drops(0.1)) == 1


def test_the_configured_default_matches_the_signature_default(coherent_set):
    """apply_math_filters defaults to config.MIN_CLUSTER_SIZE; if the two ever
    drift apart the config file stops describing what actually runs."""
    from_signature = apply_math_filters_verbose(coherent_set)
    from_config = apply_math_filters_verbose(coherent_set, min_cluster_size=MIN_CLUSTER_SIZE)

    assert from_signature == from_config


# --------------------------------------------------------------------------
# Stage ordering
# --------------------------------------------------------------------------

def test_the_consistency_stage_only_sees_cluster_survivors(cluster_and_outlier):
    """Order matters: the cheap geometric check runs first and the second stage
    judges what is left, not the original set."""
    result = apply_math_filters_verbose(cluster_and_outlier)

    cluster_dropped_ids = {c["chunk_id"] for c in result["dropped"] if c["filter"] == "cluster"}
    consistency_dropped_ids = {
        c["chunk_id"] for c in result["dropped"] if c["filter"] == "consistency"
    }

    assert cluster_dropped_ids.isdisjoint(consistency_dropped_ids)


def test_a_chunk_is_never_reported_as_dropped_twice(majority_and_minority):
    result = apply_math_filters_verbose(majority_and_minority)

    dropped_ids = [c["chunk_id"] for c in result["dropped"]]
    assert len(dropped_ids) == len(set(dropped_ids))
