from .outlier import detect_outliers, compute_centroid, distance_from_centroid
from .consistency import detect_inconsistent, mean_similarity_to_others
from ..config import OUTLIER_THRESHOLD, CONSISTENCY_THRESHOLD, DEFAULT_TOP_K
from ..role1_ingestion.retriever import retrieve


def apply_math_filters(
    retrieved_chunks: list[dict],
    outlier_threshold: float = OUTLIER_THRESHOLD,
    consistency_threshold: float = CONSISTENCY_THRESHOLD,
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
    )["kept"]


def apply_math_filters_verbose(
    retrieved_chunks: list[dict],
    outlier_threshold: float = OUTLIER_THRESHOLD,
    consistency_threshold: float = CONSISTENCY_THRESHOLD,
) -> dict:
    """Run the outlier filter, then the consistency filter on its survivors.

    A chunk must pass both to survive. Returns {"kept": [...], "dropped": [...]}
    where each dropped chunk carries its original fields plus "filter" (which
    check removed it) and "reason" (the number that decided it).
    """
    dropped = []

    kept_after_outliers, dropped_by_outlier = detect_outliers(
        retrieved_chunks, threshold=outlier_threshold
    )

    if dropped_by_outlier:
        centroid = compute_centroid(retrieved_chunks)
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
