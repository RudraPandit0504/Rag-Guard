"""Module 3 — outlier detection by density clustering (HDBSCAN).

This replaces the centroid-distance rule in outlier.py. That rule asks a single
question of every chunk: how far is it from the mean of the retrieved set? A
mean is not robust — the poisoned chunks help compute the very centre they are
measured against — and it forces one global threshold onto every query,
regardless of how tight or loose that query's neighbourhood happens to be.

HDBSCAN asks a different question: is this chunk part of a dense group at all?
It builds a hierarchy over the mutual-reachability distances between chunks and
keeps the clusters that persist across it. Points that never join a dense group
come out labelled -1 (noise). There is no distance threshold to tune, and the
notion of "dense enough" is derived per retrieval rather than fixed in advance.

The keep rule is majority consensus: the largest cluster survives, noise is
dropped, and smaller competing clusters are dropped as well. Two safety rules
override it, because dropping everything is never a useful answer:

  * fewer chunks than a cluster could contain -> pass through unfiltered
  * no cluster found at all (every point noise) -> pass through unfiltered

In both cases the chunks are unfiltered, not approved, and a warning says so.

Geometry only. The poisoned flag on each chunk is never read here — it exists
purely so callers can measure this filter's accuracy afterward.
"""

import numpy as np
from sklearn.cluster import HDBSCAN

from ..config import (
    MIN_CLUSTER_SIZE, MIN_SAMPLES, CLUSTER_SELECTION_EPSILON, DEFAULT_TOP_K,
)

# HDBSCAN's own label for a point that belongs to no cluster.
NOISE_LABEL = -1


def cosine_distance_matrix(chunks: list[dict]) -> np.ndarray:
    """N x N matrix of cosine distance (1 - cosine similarity) between chunks.

    An empty input yields an empty 0x0 matrix rather than raising.

    HDBSCAN is fed this matrix directly via metric="precomputed" rather than
    raw vectors, so the clustering measures the same cosine geometry as the
    rest of Role 2. Floating-point error can push a self-distance a hair below
    zero or a similarity a hair above one, which HDBSCAN rejects as an invalid
    metric, so the result is clipped and its diagonal zeroed.
    """
    if not chunks:
        return np.zeros((0, 0))

    vectors = np.array([c["vector"] for c in chunks], dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms

    distances = 1.0 - (normalized @ normalized.T)
    distances = np.clip(distances, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)

    # Cosine distance is symmetric by definition; the matrix product is only
    # symmetric up to rounding, and HDBSCAN validates that it is exactly so.
    return (distances + distances.T) / 2.0


def cluster_chunks(
    chunks: list[dict],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int | None = MIN_SAMPLES,
    cluster_selection_epsilon: float = CLUSTER_SELECTION_EPSILON,
) -> dict:
    """Run HDBSCAN over the chunk vectors and describe the outcome.

    Returns a dict with:
      labels          cluster id per chunk, in input order; -1 means noise
      probabilities   HDBSCAN's confidence that a chunk belongs to its cluster
      majority_label  the id of the largest cluster, or None if none formed
      cluster_sizes   {label: count} for real clusters, noise excluded
      skipped         True when the filter could not run and chunks pass through
      note            why it was skipped, or "" when it ran

    Separated from detect_cluster_outliers() so a caller that needs both the
    verdict and the numbers behind it does not cluster the same set twice.
    """
    n = len(chunks)
    minimum_needed = max(2, min_cluster_size)

    if n < minimum_needed:
        return {
            "labels": [NOISE_LABEL] * n,
            "probabilities": [0.0] * n,
            "majority_label": None,
            "cluster_sizes": {},
            "skipped": True,
            "note": (
                f"{n} chunk(s) is fewer than the {minimum_needed} a cluster "
                "would need; nothing can be judged dense or sparse"
            ),
        }

    distances = cosine_distance_matrix(chunks)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="precomputed",
        # A Top-K retrieval for one query is usually a single topic. Without
        # this, HDBSCAN refuses to return one cluster and splits a coherent
        # set or calls all of it noise.
        #
        # The cost is that min_cluster_size is not monotonic in strictness. Set
        # it larger than any genuine group in the data and no group qualifies,
        # so selection climbs to the root of the hierarchy and accepts the whole
        # set as one cluster — the filter then drops nothing and does so
        # silently. It fails open, not closed. Keep min_cluster_size at or below
        # the size of the smallest group worth trusting.
        allow_single_cluster=True,
        # The precomputed matrix is modified in place otherwise, which would
        # corrupt it for any caller reusing it.
        copy=True,
    ).fit(distances)

    labels = [int(label) for label in clusterer.labels_]
    probabilities = [float(p) for p in clusterer.probabilities_]

    cluster_sizes: dict[int, int] = {}
    for label in labels:
        if label != NOISE_LABEL:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    if not cluster_sizes:
        return {
            "labels": labels,
            "probabilities": probabilities,
            "majority_label": None,
            "cluster_sizes": {},
            "skipped": True,
            "note": (
                "no dense cluster formed — every chunk is noise, so there is no "
                "majority to judge the others against"
            ),
        }

    return {
        "labels": labels,
        "probabilities": probabilities,
        "majority_label": _majority_label(labels, probabilities, cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "skipped": False,
        "note": "",
    }


def _majority_label(labels: list[int], probabilities: list[float], cluster_sizes: dict[int, int]) -> int:
    """The largest cluster; ties broken by mean membership confidence, then by
    lowest label id.

    Two clusters of equal size is a genuine ambiguity — the tie-break does not
    resolve which one is legitimate, it only guarantees the same answer for the
    same input. Callers that care should look at cluster_sizes and see the tie.
    """
    mean_probability = {}
    for label in cluster_sizes:
        members = [p for p, l in zip(probabilities, labels) if l == label]
        mean_probability[label] = float(np.mean(members))

    return max(
        cluster_sizes,
        key=lambda label: (cluster_sizes[label], mean_probability[label], -label),
    )


def detect_cluster_outliers(
    chunks: list[dict],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int | None = MIN_SAMPLES,
    cluster_selection_epsilon: float = CLUSTER_SELECTION_EPSILON,
) -> tuple[list[dict], list[dict]]:
    """Split chunks into (kept, dropped) by HDBSCAN cluster membership.

    Same signature shape and return contract as detect_outliers() in outlier.py,
    so the two are interchangeable inside apply_math_filters(). Input order is
    preserved within each list, and every input chunk appears in exactly one.
    """
    if not chunks:
        return [], []

    result = cluster_chunks(
        chunks,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    if result["skipped"]:
        print(
            f"WARNING: cluster filter skipped — {result['note']}; chunks pass "
            "through unfiltered"
        )
        return list(chunks), []

    majority = result["majority_label"]

    kept, dropped = [], []
    for chunk, label in zip(chunks, result["labels"]):
        if label == majority:
            kept.append(chunk)
        else:
            dropped.append(chunk)

    return kept, dropped


def explain(chunk_label: int, majority_label: int, probability: float) -> str:
    """One line saying why a chunk was dropped, for the verbose filter output."""
    if chunk_label == NOISE_LABEL:
        return "clustered as noise — not part of any dense group"
    return (
        f"in minority cluster {chunk_label} (membership {probability:.3f}), "
        f"not the majority cluster {majority_label}"
    )


def analyze(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    """Diagnostic table: cluster label, membership strength and verdict per chunk.

    This is a report, not a filter — nothing here changes retrieval results.
    """
    from ..role1_ingestion.retriever import retrieve

    chunks = retrieve(query, top_k=top_k)
    result = cluster_chunks(chunks)
    kept, _ = detect_cluster_outliers(chunks)
    kept_ids = {c["chunk_id"] for c in kept}

    print(f"Q: {query}\n")
    if result["skipped"]:
        print(f"filter skipped: {result['note']}\n")
    else:
        print(f"clusters: {result['cluster_sizes']}  majority: {result['majority_label']}\n")

    print(f"{'chunk_id':>10}  {'score':>7}  {'cluster':>7}  {'member':>7}  {'poisoned':>9}  verdict")

    for chunk, label, probability in zip(chunks, result["labels"], result["probabilities"]):
        verdict = "KEEP" if chunk["chunk_id"] in kept_ids else "DROP"
        label_text = "noise" if label == NOISE_LABEL else str(label)
        print(
            f"{chunk['chunk_id']:>10}  {chunk['score']:>7.3f}  {label_text:>7}  "
            f"{probability:>7.3f}  {str(chunk['poisoned']):>9}  {verdict}"
        )


def sweep(
    query: str,
    min_cluster_sizes: list[int] = [2, 3, 4],
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Compare keep/drop outcomes across min_cluster_size for one retrieval.

    HDBSCAN has no distance threshold to sweep, so this varies the one
    parameter that does change what counts as a cluster. Read-only and
    geometry-only: retrieve() and detect_cluster_outliers() aren't
    reimplemented here. The poisoned flag is only read after each decision is
    already made, to describe what that decision did.
    """
    from ..role1_ingestion.retriever import retrieve

    chunks = retrieve(query, top_k=top_k)

    print(f"Q: {query}\n")
    print(f"{'min_size':>8}  {'kept':>4}  {'dropped':>7}  {'poison dropped':>14}  {'legit dropped':>13}")

    for min_cluster_size in min_cluster_sizes:
        kept, dropped = detect_cluster_outliers(chunks, min_cluster_size=min_cluster_size)

        poison_dropped = any(c["poisoned"] for c in dropped)
        legit_dropped = any(not c["poisoned"] for c in dropped)

        print(
            f"{min_cluster_size:>8}  {len(kept):>4}  {len(dropped):>7}  "
            f"{str(poison_dropped):>14}  {str(legit_dropped):>13}"
        )


if __name__ == "__main__":
    analyze("How long do I have to return a product?")
    print()
    sweep("How long do I have to return a product?")
