import numpy as np

from ..config import CONSISTENCY_THRESHOLD, DEFAULT_TOP_K
from ..role1_ingestion.retriever import retrieve


def pairwise_similarities(chunks: list[dict]) -> np.ndarray:
    """N x N matrix of cosine similarity between every pair of chunk vectors.

    An empty input yields an empty 0x0 matrix rather than raising.
    """
    if not chunks:
        return np.zeros((0, 0))

    vectors = np.array([c["vector"] for c in chunks])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    return normalized @ normalized.T


def mean_similarity_to_others(chunks: list[dict]) -> list[float]:
    """For each chunk, its average cosine similarity to every other chunk (self excluded).

    With fewer than 2 chunks a chunk has no peers, so its mean similarity is
    undefined and returned as nan. Callers must handle that explicitly —
    comparing nan against a threshold is always False, which would silently
    keep the chunk. detect_inconsistent() guards for it.
    """
    if len(chunks) < 2:
        return [float("nan")] * len(chunks)

    similarities = pairwise_similarities(chunks)
    n = len(chunks)

    means = []
    for i in range(n):
        others = [similarities[i, j] for j in range(n) if j != i]
        means.append(float(np.mean(others)))

    return means


def detect_inconsistent(chunks: list[dict], threshold: float = CONSISTENCY_THRESHOLD) -> tuple[list[dict], list[dict]]:
    """Split chunks into (kept, dropped) by mean similarity to the rest of the group.

    Consistency is a comparison between peers, so it cannot be assessed with
    fewer than 2 chunks. In that case every chunk is kept and a warning says
    the filter was skipped — the chunks are unfiltered, not approved.

    Geometry only. The poisoned flag on each chunk is never read here — it
    exists purely so callers can measure this filter's accuracy afterward.
    """
    if len(chunks) < 2:
        print(
            f"WARNING: consistency filter skipped — {len(chunks)} chunk(s) is too "
            "few to compare against peers; chunks pass through unfiltered"
        )
        return list(chunks), []

    means = mean_similarity_to_others(chunks)

    kept, dropped = [], []
    for chunk, mean_sim in zip(chunks, means):
        if mean_sim < threshold:
            dropped.append(chunk)
        else:
            kept.append(chunk)

    return kept, dropped


def analyze(query: str, top_k: int = DEFAULT_TOP_K) -> None:
    """Diagnostic table: mean similarity to others and keep/drop verdict per chunk.

    This is a report, not a filter — nothing here changes retrieval results.
    """
    chunks = retrieve(query, top_k=top_k)
    means = mean_similarity_to_others(chunks)
    kept, _ = detect_inconsistent(chunks)
    kept_ids = {c["chunk_id"] for c in kept}

    print(f"Q: {query}\n")
    print(f"{'chunk_id':>10}  {'score':>7}  {'mean_sim':>9}  {'poisoned':>9}  verdict")

    for chunk, mean_sim in zip(chunks, means):
        verdict = "KEEP" if chunk["chunk_id"] in kept_ids else "DROP"
        print(
            f"{chunk['chunk_id']:>10}  {chunk['score']:>7.3f}  {mean_sim:>9.3f}  "
            f"{str(chunk['poisoned']):>9}  {verdict}"
        )


def sweep(
    query: str,
    thresholds: list[float] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Compare keep/drop outcomes across several thresholds for one retrieval.

    Read-only and geometry-only: retrieve() and detect_inconsistent() aren't
    reimplemented here. The poisoned flag is only read after each threshold's
    decision is already made, to describe what that decision did.
    """
    chunks = retrieve(query, top_k=top_k)

    print(f"Q: {query}\n")
    print(f"{'threshold':>9}  {'kept':>4}  {'dropped':>7}  {'poison dropped':>14}  {'legit dropped':>13}")

    for threshold in thresholds:
        kept, dropped = detect_inconsistent(chunks, threshold=threshold)

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
