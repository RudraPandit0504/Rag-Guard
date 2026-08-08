from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pymongo import MongoClient

from ..config import (
    MONGO_URI, QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_NAME, DB_NAME, MODEL_NAME,
)

model = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
mongo = MongoClient(MONGO_URI)
chunks_collection = mongo[DB_NAME]["chunks"]


def retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    """Find the chunks most similar in meaning to the query."""

    query_vector = model.encode(query_text).tolist()

    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_vectors=True,
    ).points

    results = []
    for hit in hits:
        chunk_id = hit.payload["chunk_id"]
        doc = chunks_collection.find_one({"chunk_id": chunk_id})

        if doc is None:
            continue

        results.append({
            "chunk_id": chunk_id,
            "text": doc["text"],
            "hash": doc["hash"],
            "created_at": doc["created_at"],
            "poisoned": doc.get("poisoned", False),
            "score": hit.score,
            "vector": hit.vector,
        })

    return results


if __name__ == "__main__":
    question = "How long do I have to return a product?"

    print(f"Q: {question}\n")

    for r in retrieve(question):
        print(f"[{r['chunk_id']}] score={r['score']:.3f}")
        print(f"    {r['text'][:110]}...\n")