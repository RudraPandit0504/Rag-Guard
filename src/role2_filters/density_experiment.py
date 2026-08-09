"""Poison-density experiment for the Role 2 math filters.

All earlier measurements used 3 poisoned chunks against a 35-chunk corpus —
roughly 8% density, which is unrealistically high for a real knowledge base.
This module varies the poison count and measures whether the filters behave
differently when poison is rare, to find where their behaviour changes.

Unlike evaluate.py, this module WRITES to both stores: it clears and re-injects
poison between runs. It restores the 3-poison state when it finishes, including
on error. It lives apart from evaluate.py so that measurement stays read-only.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from pymongo import MongoClient

from .attacker import inject, mutate, POISON_TEMPLATES
from .evaluate import (
    TEST_QUERIES,
    evaluate_baseline,
    evaluate_defended,
    _vulnerability_score,  # single source of the score formula; do not restate it
    _format_score,
)
from .outlier import compute_centroid, distance_from_centroid
from ..config import (
    MONGO_URI, QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_NAME, DB_NAME, POISON_START_ID, DEFAULT_TOP_K,
)
from ..role1_ingestion.retriever import retrieve


# The query the poison was written and mutated against.
TARGET_QUERY = "How long do I have to return a product?"

POISON_COUNTS = [1, 2, 3]

# Qdrant has no payload index on chunk_id, so poison cannot be deleted by
# filter. Delete an explicit id span above POISON_START_ID instead — real
# chunk ids live far below it, and deleting absent ids is a no-op.
CLEAR_SPAN = 100


def _poison_set() -> list[str]:
    """The three poison chunks, strongest first, added cumulatively.

    N=1 uses the mutated subtle chunk (highest similarity to the target query),
    N=2 adds "SYSTEM OVERRIDE", N=3 adds "Forget your previous instructions".
    mutate() is deterministic, so this list is identical on every run.
    """
    mutated, _ = mutate(POISON_TEMPLATES[2], TARGET_QUERY, iterations=10)
    return [mutated, POISON_TEMPLATES[1], POISON_TEMPLATES[5]]


def clear_poison() -> int:
    """Remove every injected chunk from Qdrant and MongoDB. Returns docs removed."""
    collection = MongoClient(MONGO_URI)[DB_NAME]["chunks"]

    known_ids = {doc["chunk_id"] for doc in collection.find({"source": "injected"}, {"chunk_id": 1})}
    span_ids = set(range(POISON_START_ID, POISON_START_ID + CLEAR_SPAN))

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=sorted(known_ids | span_ids)),
    )

    return collection.delete_many({"source": "injected"}).deleted_count


def _distance_profile(queries: list[str], top_k: int) -> dict:
    """Mean distance-from-centroid for poisoned vs legitimate retrieved chunks.

    The geometry is computed for every chunk first; the poisoned flag is only
    used afterward to bucket the results, never to influence them.
    """
    poison_distances, legit_distances = [], []

    for query in queries:
        chunks = retrieve(query, top_k=top_k)
        if not chunks:
            continue

        centroid = compute_centroid(chunks)
        for chunk in chunks:
            distance = distance_from_centroid(chunk["vector"], centroid)
            if chunk["poisoned"]:
                poison_distances.append(distance)
            else:
                legit_distances.append(distance)

    return {
        "poison_mean": sum(poison_distances) / len(poison_distances) if poison_distances else None,
        "legit_mean": sum(legit_distances) / len(legit_distances) if legit_distances else None,
        "poison_observations": len(poison_distances),
    }


def _format_distance(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run_density_experiment(
    poison_counts: list[int] = POISON_COUNTS,
    queries: list[str] = TEST_QUERIES,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Measure filter behaviour at each poison density, then restore 3 poison chunks."""
    poison_set = _poison_set()
    rows = []

    try:
        for count in poison_counts:
            print(f"\n=== density: {count} poison chunk(s) ===")
            clear_poison()
            inject(poison_set[:count], start_id=POISON_START_ID)

            baseline = evaluate_baseline(queries, top_k=top_k)["total"]
            defended = evaluate_defended(queries, top_k=top_k)["total"]
            distances = _distance_profile(queries, top_k=top_k)

            rows.append({
                "poison_count": count,
                "poison_retrieved": baseline["poisoned_retrieved"],
                "poison_surviving": defended["poisoned_surviving"],
                "baseline_score": _vulnerability_score(baseline),
                "defended_score": _vulnerability_score(defended),
                "legit_removed": defended["legit_removed"],
                **distances,
            })
    finally:
        print("\nRestoring the 3-poison state...")
        clear_poison()
        inject(poison_set, start_id=POISON_START_ID)

    _print_summary(rows, queries, top_k)
    return rows


def _print_summary(rows: list[dict], queries: list[str], top_k: int) -> None:
    print(f"\n\nPoison-density summary ({len(queries)} queries, top_k={top_k})\n")

    header = (
        f"{'poison':>6}  {'retr':>5}  {'surv':>5}  {'baseline':>9}  {'defended':>9}  "
        f"{'legit_rm':>8}  {'poison_d':>8}  {'legit_d':>7}  {'gap':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        poison_d, legit_d = row["poison_mean"], row["legit_mean"]
        gap = "n/a" if poison_d is None or legit_d is None else f"{legit_d - poison_d:+.3f}"

        print(
            f"{row['poison_count']:>6}  {row['poison_retrieved']:>5}  {row['poison_surviving']:>5}  "
            f"{_format_score(row['baseline_score']):>9}  {_format_score(row['defended_score']):>9}  "
            f"{row['legit_removed']:>8}  {_format_distance(poison_d):>8}  "
            f"{_format_distance(legit_d):>7}  {gap:>7}"
        )

    print(
        "\nretr/surv = poisoned chunks retrieved / surviving the filters."
        "\nbaseline/defended = Vulnerability Score, lower is safer."
        "\npoison_d/legit_d = mean cosine distance from the retrieved centroid."
        "\ngap = legit_d - poison_d; positive means poison sits CLOSER to the"
        "\n      centroid than legitimate chunks, which is what defeats the filter."
    )


if __name__ == "__main__":
    run_density_experiment()
