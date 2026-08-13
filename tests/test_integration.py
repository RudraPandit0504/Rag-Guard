"""End-to-end checks against the real corpus.

These are the only tests that touch Qdrant, MongoDB and the embedding model.
They skip rather than fail when the stores are unreachable or empty, so the
unit suite stays runnable on a laptop with no credentials:

    pytest -m "not integration"     # unit tests only
    pytest -m integration           # these

They assert properties that must hold on real data, not specific counts —
poison counts change whenever the attacker or the corpus changes, and a test
that pins them would fail for the wrong reason.
"""

import pytest

from src.config import DEFAULT_TOP_K
from src.role2_filters.cluster import cluster_chunks, detect_cluster_outliers
from src.role2_filters.filters import apply_math_filters, apply_math_filters_verbose

pytestmark = pytest.mark.integration

QUERY = "How long do I have to return a product?"


@pytest.fixture(scope="module")
def retrieved() -> list[dict]:
    """Top-K chunks for the target query, or a skip if the stores aren't there."""
    try:
        from src.role1_ingestion.retriever import retrieve
        chunks = retrieve(QUERY, top_k=DEFAULT_TOP_K)
    except Exception as error:
        pytest.skip(f"retrieval unavailable: {type(error).__name__}: {error}")

    if not chunks:
        pytest.skip("corpus is empty — run src.role1_ingestion.ingest first")

    return chunks


def test_retrieval_returns_usable_vectors(retrieved):
    """Everything downstream reads chunk["vector"]; if retrieval stops
    returning it the filters break in a confusing way rather than an obvious
    one."""
    for chunk in retrieved:
        assert chunk["vector"]
        assert len(chunk["vector"]) == len(retrieved[0]["vector"])


def test_clustering_runs_on_real_embeddings(retrieved):
    result = cluster_chunks(retrieved)

    assert len(result["labels"]) == len(retrieved)
    assert len(result["probabilities"]) == len(retrieved)


def test_the_filter_never_empties_a_real_retrieval(retrieved):
    """An answer needs context. A filter that removes every chunk has not made
    the system safe, it has broken it."""
    survivors = apply_math_filters(retrieved)

    assert len(survivors) >= 1


def test_the_full_chain_partitions_a_real_retrieval(retrieved):
    result = apply_math_filters_verbose(retrieved)

    kept_ids = [c["chunk_id"] for c in result["kept"]]
    dropped_ids = [c["chunk_id"] for c in result["dropped"]]
    original_ids = [c["chunk_id"] for c in retrieved]

    assert sorted(kept_ids + dropped_ids) == sorted(original_ids)


def test_both_methods_run_on_real_data_and_agree_on_shape(retrieved):
    """Swapping the algorithm must not change the contract, only the verdict."""
    hdbscan = apply_math_filters_verbose(retrieved, outlier_method="hdbscan")
    centroid = apply_math_filters_verbose(retrieved, outlier_method="centroid")

    for result in (hdbscan, centroid):
        assert set(result) == {"kept", "dropped"}
        assert len(result["kept"]) + len(result["dropped"]) == len(retrieved)


def test_clustering_is_stable_across_repeated_real_retrievals(retrieved):
    first = [c["chunk_id"] for c in detect_cluster_outliers(retrieved)[0]]
    second = [c["chunk_id"] for c in detect_cluster_outliers(retrieved)[0]]

    assert first == second


def test_coherence_scores_real_chunks(retrieved):
    """Runs the real embedding model over real chunk text — the fake encoder in
    the unit tests cannot catch a change in how sentences are split or embedded."""
    from src.role2_filters.coherence import coherence_profile

    scored = [coherence_profile(c) for c in retrieved]

    assert any(not p["skipped"] for p in scored), "no chunk had enough sentences to judge"
    for profile in scored:
        if not profile["skipped"]:
            assert -1.0 <= profile["score"] <= 1.0
            assert profile["weakest"] in profile["sentences"]


def test_coherence_separates_poison_from_legitimate_chunks(retrieved):
    """The measured claim, checked against live data rather than restated.

    Skips instead of failing when the retrieval has no poison, so the suite is
    still meaningful on a clean corpus.
    """
    from src.role2_filters.coherence import coherence_profile

    scores = {}
    for chunk in retrieved:
        profile = coherence_profile(chunk)
        if not profile["skipped"]:
            scores.setdefault(chunk["poisoned"], []).append(profile["score"])

    if True not in scores or False not in scores:
        pytest.skip("need both poisoned and legitimate chunks to compare")

    assert max(scores[True]) < min(scores[False]), (
        "poisoned chunks no longer score below every legitimate one; "
        "COHERENCE_THRESHOLD needs re-measuring"
    )


def test_the_coherence_stage_can_be_switched_off(retrieved):
    """The evaluation isolates each stage this way, so the toggle has to work."""
    with_stage = apply_math_filters_verbose(retrieved, use_coherence=True)
    without_stage = apply_math_filters_verbose(retrieved, use_coherence=False)

    assert not [c for c in without_stage["dropped"] if c["filter"] == "coherence"]
    assert len(with_stage["kept"]) + len(with_stage["dropped"]) == len(retrieved)
    assert len(without_stage["kept"]) + len(without_stage["dropped"]) == len(retrieved)


def test_ground_truth_is_present_for_measurement(retrieved):
    """Every chunk must carry a poisoned flag or the evaluation cannot score
    anything. It is read here, in a test about measurement — never in a filter."""
    for chunk in retrieved:
        assert isinstance(chunk["poisoned"], bool)
