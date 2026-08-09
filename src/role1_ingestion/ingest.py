from pathlib import Path
from datetime import datetime, timezone

from .loaders import load_document, LOADERS
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient

from ..config import (
    MONGO_URI, QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_NAME, DB_NAME, VECTOR_SIZE, MODEL_NAME,
)
from .chunker import chunk_text, compute_hash


def main():
    # 0. Injected poison lives only in the databases. The poisoned-hash restore
    # below can only re-flag chunks rebuilt from data/, and injected poison is
    # not in data/ — so a re-ingest destroys it for good. Ask first.
    injected = MongoClient(MONGO_URI)[DB_NAME]["chunks"].count_documents(
        {"source": "injected"}
    )
    if injected:
        print(
            f"WARNING: {injected} injected poison chunk(s) are currently stored.\n"
            "Re-ingesting wipes both Qdrant and MongoDB. Injected poison is NOT\n"
            "rebuilt from data/, so it will be lost permanently and would have to\n"
            "be recreated by re-running the attacker module."
        )
        try:
            answer = input("Type 'yes' to destroy it and continue: ").strip().lower()
        except EOFError:
            answer = ""

        if answer != "yes":
            print("Aborted — nothing was changed.")
            return

    # 1. Read the document
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    all_chunks = []
    all_sources = []

    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() not in LOADERS:
            print(f"Skipping {f.name}: unsupported extension '{f.suffix}'")
            continue

        raw = load_document(f)

        if len(raw.strip()) < 100:
            print(f"WARNING: {f.name} gave almost no text — possibly scanned")
            continue

        pieces = chunk_text(raw)
        for piece in pieces:
            all_chunks.append(piece)
            all_sources.append(f.name)

        print(f"{f.name}: {len(pieces)} chunks")

    print(f"Total: {len(all_chunks)} chunks")

    # 2. Turn each chunk into 384 numbers
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(all_chunks)
    print(f"Vectors: {vectors.shape}")

    # 3. Make a fresh collection in Qdrant
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created")

    # 4. Send the numbers to Qdrant
    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload={"chunk_id": i})
        for i in range(len(all_chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} vectors in Qdrant")

    # 5. Send the text, hash and date to MongoDB
    mongo = MongoClient(MONGO_URI)
    collection = mongo[DB_NAME]["chunks"]

    poisoned_hashes = {
        doc["hash"] for doc in collection.find({"poisoned": True}, {"hash": 1})
    }

    collection.delete_many({})
    collection.create_index("chunk_id")

    now = datetime.now(timezone.utc)
    docs = []
    for i in range(len(all_chunks)):
        chunk_hash = compute_hash(all_chunks[i])
        docs.append({
            "chunk_id": i,
            "text": all_chunks[i],
            "hash": chunk_hash,
            "created_at": now,
            "source": all_sources[i],
            "poisoned": chunk_hash in poisoned_hashes,
        })
    collection.insert_many(docs)
    print(f"Stored {len(docs)} documents in MongoDB")


if __name__ == "__main__":
    main()