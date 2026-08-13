from .cluster import cluster_chunks, detect_cluster_outliers, explain
from .coherence import coherence_profile, detect_incoherent
from .coherence import explain as explain_incoherence
from .outlier import detect_outliers, compute_centroid, distance_from_centroid
from .consistency import detect_inconsistent, mean_similarity_to_others
from ..config import (
    OUTLIER_THRESHOLD, CONSISTENCY_THRESHOLD, OUTLIER_METHOD, DEFAULT_TOP_K,
    MIN_CLUSTER_SIZE, COHERENCE_THRESHOLD,
)


def apply_math_filters(
    retrieved_chunks: list[dict],
    outlier_threshold: float = OUTLIER_THRESHOLD,
    consistency_threshold: float = CONSISTENCY_THRESHOLD,
    outlier_method: str = OUTLIER_METHOD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    coherence_threshold: float = COHERENCE_THRESHOLD,
    use_coherence: bool = True,
    encode=None,
) -> list[dict]:
    """Role 2's deliverable: the chunks that survived both math filters.

    Returns the same list-of-dicts shape retrieve() produces, so the pipeline
    chains directly:

        retrieve(query) -> apply_math_filters(...) -> apply_sandbox_filters(...)

    Use apply_math_filters_verbose() instead if you need to know what was
    dropped and why.
    """
    return apply_math_filters_verbose(
        retrieved_chunks,
        outlier_threshold=outlier_threshold,
        consistency_threshold=consistency_threshold,
        outlier_method=outlier_method,
        min_cluster_size=min_cluster_size,
        coherence_threshold=coherence_threshold,
        use_coherence=use_coherence,
        encode=encode,
    )["kept"]


def apply_math_filters_verbose(
    retrieved_chunks: list[dict],
    outlier_threshold: float = OUTLIER_THRESHOLD,
    consistency_threshold: float = CONSISTENCY_THRESHOLD,
    outlier_method: str = OUTLIER_METHOD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    coherence_threshold: float = COHERENCE_THRESHOLD,
    use_coherence: bool = True,
    encode=None,
) -> dict:
    """Run coherence, then the outlier filter, then consistency on the survivors.

    A chunk must pass all three to survive. Returns {"kept": [...],
    "dropped": [...]} where each dropped chunk carries its original fields plus
    "filter" (which check removed it) and "reason" (the number that decided it).

    Coherence runs first on purpose. It is the only stage that judges a chunk on
    its own text rather than against its neighbours, so it cannot be swayed by
    what else was retrieved — and removing obvious poison before the two
    consensus-based stages vote means the vote is taken on a cleaner set.

    outlier_method picks how the second stage decides:
      "hdbscan"   density clustering; keeps the majority cluster, drops noise
                  and minority clusters. outlier_threshold is unused.
      "centroid"  the original rule; drops chunks further than
                  outlier_threshold from the mean vector. min_cluster_size is
                  unused.

    use_coherence=False skips the first stage entirely, which is how the
    evaluation measures what each stage contributes on its own.
    """
    if outlier_method not in ("hdbscan", "centroid"):
        raise ValueError(
            f"unknown outlier_method {outlier_method!r}; expected 'hdbscan' or 'centroid'"
        )

    dropped = []
    surviving = retrieved_chunks

    if use_coherence:
        surviving, dropped_by_coherence = detect_incoherent(
            surviving, threshold=coherence_threshold, encode=encode
        )

        for chunk in dropped_by_coherence:
            profile = coherence_profile(chunk, encode=encode)
            dropped.append({
                **chunk,
                "filter": "coherence",
                "reason": explain_incoherence(profile, coherence_threshold),
            })

    if outlier_method == "hdbscan":
        kept_after_outliers, dropped_by_outlier = detect_cluster_outliers(
            surviving, min_cluster_size=min_cluster_size
        )

        if dropped_by_outlier:
            clustering = cluster_chunks(surviving, min_cluster_size=min_cluster_size)
            by_id = {
                chunk["chunk_id"]: (label, probability)
                for chunk, label, probability in zip(
                    surviving, clustering["labels"], clustering["probabilities"]
                )
            }
            for chunk in dropped_by_outlier:
                label, probability = by_id[chunk["chunk_id"]]
                dropped.append({
                    **chunk,
                    "filter": "cluster",
                    "reason": explain(label, clustering["majority_label"], probability),
                })
    else:
        kept_after_outliers, dropped_by_outlier = detect_outliers(
            surviving, threshold=outlier_threshold
        )

        if dropped_by_outlier:
            centroid = compute_centroid(surviving)
            for chunk in dropped_by_outlier:
                distance = distance_from_centroid(chunk["vector"], centroid)
                dropped.append({
                    **chunk,
                    "filter": "outlier",
                    "reason": f"distance {distance:.3f} exceeds threshold {outlier_threshold:.2f}",
                })

    kept_final, dropped_by_consistency = detect_inconsistent(
        kept_after_outliers, threshold=consistency_threshold
    )

    if dropped_by_consistency:
        means = mean_similarity_to_others(kept_after_outliers)
        mean_by_id = {
            chunk["chunk_id"]: mean_sim
            for chunk, mean_sim in zip(kept_after_outliers, means)
        }
        for chunk in dropped_by_consistency:
            mean_sim = mean_by_id[chunk["chunk_id"]]
            dropped.append({
                **chunk,
                "filter": "consistency",
                "reason": f"mean similarity {mean_sim:.3f} below threshold {consistency_threshold:.2f}",
            })

    return {"kept": kept_final, "dropped": dropped}


if __name__ == "__main__":
    from ..role1_ingestion.retriever import retrieve

    query = "How long do I have to return a product?"
    chunks = retrieve(query, top_k=DEFAULT_TOP_K)

    result = apply_math_filters_verbose(chunks)

    print(f"Q: {query}\n")
    print(f"{len(result['kept'])} kept, {len(result['dropped'])} dropped\n")

    print("Kept:")
    for c in result["kept"]:
        print(f"  chunk_id={c['chunk_id']} score={c['score']:.3f} poisoned={c['poisoned']}")

    print("\nDropped:")
    for c in result["dropped"]:
        print(f"  chunk_id={c['chunk_id']} poisoned={c['poisoned']} filter={c['filter']} reason={c['reason']}")
