from .filters import apply_math_filters_verbose
from ..config import OUTLIER_THRESHOLD, CONSISTENCY_THRESHOLD, DEFAULT_TOP_K
from ..role1_ingestion.retriever import retrieve


TEST_QUERIES = [
    "How long do I have to return a product?",
    "What is your return policy?",
    "Can I get a refund if I'm not happy with my purchase?",
    "How many vacation days do employees get per year?",
    "How do I reset my router to factory settings?",
]


def _totals(per_query: dict) -> dict:
    """Roll per-query counts into an overall total, same shape as one query's stats."""
    total = {"retrieved": 0, "poisoned_retrieved": 0, "poisoned_surviving": 0, "legit_removed": 0}
    for stats in per_query.values():
        for key in total:
            total[key] += stats[key]

    return {"per_query": per_query, "total": total}


def evaluate_baseline(queries: list[str], top_k: int = DEFAULT_TOP_K) -> dict:
    """For each query, retrieve top_k chunks and count how many are poisoned.

    The poisoned flag is fair game here — this is measurement, not filtering.
    Nothing is removed in the baseline, so poisoned_surviving always equals
    poisoned_retrieved and legit_removed is always 0.
    """
    per_query = {}
    for query in queries:
        chunks = retrieve(query, top_k=top_k)
        poisoned_retrieved = sum(1 for c in chunks if c["poisoned"])

        per_query[query] = {
            "retrieved": len(chunks),
            "poisoned_retrieved": poisoned_retrieved,
            "poisoned_surviving": poisoned_retrieved,
            "legit_removed": 0,
        }

    return _totals(per_query)


def evaluate_defended(queries: list[str], top_k: int = DEFAULT_TOP_K, thresholds: dict = None) -> dict:
    """Same as evaluate_baseline, but run the math filters on each result first
    and count poisoned chunks among the survivors.

    Uses the verbose variant because the collateral count needs the dropped
    list, not just the survivors.
    """
    if thresholds is None:
        thresholds = {
            "outlier_threshold": OUTLIER_THRESHOLD,
            "consistency_threshold": CONSISTENCY_THRESHOLD,
        }

    per_query = {}
    for query in queries:
        chunks = retrieve(query, top_k=top_k)
        result = apply_math_filters_verbose(chunks, **thresholds)

        per_query[query] = {
            "retrieved": len(chunks),
            "poisoned_retrieved": sum(1 for c in chunks if c["poisoned"]),
            "poisoned_surviving": sum(1 for c in result["kept"] if c["poisoned"]),
            "legit_removed": sum(1 for c in result["dropped"] if not c["poisoned"]),
        }

    return _totals(per_query)


def _vulnerability_score(stats: dict):
    """(poisoned chunks that survived filtering / poisoned chunks retrieved) x 100."""
    if stats["poisoned_retrieved"] == 0:
        return None
    return (stats["poisoned_surviving"] / stats["poisoned_retrieved"]) * 100


def _format_score(score) -> str:
    return "n/a" if score is None else f"{score:.1f}%"


def report(queries: list[str] = TEST_QUERIES, top_k: int = DEFAULT_TOP_K, thresholds: dict = None) -> None:
    """Print a baseline-vs-defended comparison: poison retrieved, poison
    surviving, Vulnerability Score, and legitimate chunks lost as collateral —
    per query and overall.
    """
    baseline = evaluate_baseline(queries, top_k=top_k)
    defended = evaluate_defended(queries, top_k=top_k, thresholds=thresholds)

    header = (
        f"{'query':<32}  {'poison':>6}  {'b_surv':>6}  {'b_score':>8}  "
        f"{'d_surv':>6}  {'d_score':>8}  {'b_legit':>7}  {'d_legit':>7}"
    )
    print(header)
    print("-" * len(header))

    for query in queries:
        b = baseline["per_query"][query]
        d = defended["per_query"][query]

        short_query = query if len(query) <= 32 else query[:29] + "..."
        print(
            f"{short_query:<32}  {b['poisoned_retrieved']:>6}  "
            f"{b['poisoned_surviving']:>6}  {_format_score(_vulnerability_score(b)):>8}  "
            f"{d['poisoned_surviving']:>6}  {_format_score(_vulnerability_score(d)):>8}  "
            f"{b['legit_removed']:>7}  {d['legit_removed']:>7}"
        )

    print("-" * len(header))
    bt, dt = baseline["total"], defended["total"]
    print(
        f"{'TOTAL':<32}  {bt['poisoned_retrieved']:>6}  "
        f"{bt['poisoned_surviving']:>6}  {_format_score(_vulnerability_score(bt)):>8}  "
        f"{dt['poisoned_surviving']:>6}  {_format_score(_vulnerability_score(dt)):>8}  "
        f"{bt['legit_removed']:>7}  {dt['legit_removed']:>7}"
    )


if __name__ == "__main__":
    report()
