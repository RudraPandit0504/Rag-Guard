import numpy as np

from ..config import OUTLIER_THRESHOLD, DEFAULT_TOP_K


def compute_centroid(chunks: list[dict]) -> np.ndarray:
    """Element-wise mean of all chunk vectors.

    An empty input yields an empty vector rather than a nan-filled one — there
    is no centre of mass without points to average.
    """
    if not chunks:
        return np.zeros(0)

    vectors = np.array([c["vector"] for c in chunks])
    return vectors.mean(axis=0)


def distance_from_centroid(vector: list[float], centroid: np.ndarray) -> float:
    """Cosine distance: 1 - cosine similarity."""
    vector = np.asarray(vector)
    similarity = np.dot(vector, centroid) / (np.linalg.norm(vector) * np.linalg.norm(centroid))
    return float(1 - similarity)


def detect_outliers(chunks: list[dict], threshold: float = OUTLIER_THRESHOLD) -> tuple[list[dict], list[dict]]:
    """Split chunks into (kept, dropped) by distance from the centroid.

    An empty input yields two empty lists — there is nothing to judge.

    Geometry only. The poisoned flag on each chunk is never read here — it
    exists purely so callers can measure this filter's accuracy afterward.
    """
    if not chunks:
        return [], []

    centroid = compute_centroid(chunks)

    kept, dropped = [], []
    for chunk in chunks:
        distance = distance_from_centroid(chunk["vector"], centroid)
        if distance > threshold:
            dropped.append(chunk)
        else:
            kept.append(chunk)

    return kept, dropped


def analyze(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    """Diagnostic table: distance from centroid and keep/drop verdict per chunk.

    This is a report, not a filter — nothing here changes retrieval results.
    """
    from ..role1_ingestion.retriever import retrieve

    chunks = retrieve(query, top_k=top_k)
    centroid = compute_centroid(chunks)
    kept, _ = detect_outliers(chunks)
    kept_ids = {c["chunk_id"] for c in kept}

    print(f"Q: {query}\n")
    print(f"{'chunk_id':>10}  {'score':>7}  {'distance':>9}  {'poisoned':>9}  verdict")

    for chunk in chunks:
        distance = distance_from_centroid(chunk["vector"], centroid)
        verdict = "KEEP" if chunk["chunk_id"] in kept_ids else "DROP"
        print(
            f"{chunk['chunk_id']:>10}  {chunk['score']:>7.3f}  {distance:>9.3f}  "
            f"{str(chunk['poisoned']):>9}  {verdict}"
        )


def sweep(
    query: str,
    thresholds: list[float] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Compare keep/drop outcomes across several thresholds for one retrieval.

    Read-only and geometry-only: retrieve() and detect_outliers() aren't
    reimplemented here. The poisoned flag is only read after each threshold's
    decision is already made, to describe what that decision did.
    """
    from ..role1_ingestion.retriever import retrieve

    chunks = retrieve(query, top_k=top_k)

    print(f"Q: {query}\n")
    print(f"{'threshold':>9}  {'kept':>4}  {'dropped':>7}  {'poison dropped':>14}  {'legit dropped':>13}")

    for threshold in thresholds:
        kept, dropped = detect_outliers(chunks, threshold=threshold)

        poison_dropped = any(c["poisoned"] for c in dropped)
        legit_dropped = any(not c["poisoned"] for c in dropped)

        print(
            f"{threshold:>9.2f}  {len(kept):>4}  {len(dropped):>7}  "
            f"{str(poison_dropped):>14}  {str(legit_dropped):>13}"
        )


if __name__ == "__main__":
    analyze("How long do I have to return a product?")
    print()
    sweep("How long do I have to return a product?")
