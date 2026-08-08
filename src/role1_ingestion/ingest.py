import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pymongo import MongoClient

from config import (
    MONGO_URI, QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_NAME, DB_NAME, VECTOR_SIZE,
)
from chunker import chunk_text, compute_hash


# 1. Read the document
data_dir = Path(__file__).resolve().parent.parent.parent / "data"
raw = (data_dir / "sample.txt").read_text(encoding="utf-8")

chunks = chunk_text(raw)
print(f"Chunks: {len(chunks)}")

# 2. Turn each chunk into 384 numbers
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(chunks)
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
    for i in range(len(chunks))
]
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Stored {len(points)} vectors in Qdrant")

# 5. Send the text, hash and date to MongoDB
mongo = MongoClient(MONGO_URI)
collection = mongo[DB_NAME]["chunks"]
collection.delete_many({})

now = datetime.now(timezone.utc)
docs = [
    {
        "chunk_id": i,
        "text": chunks[i],
        "hash": compute_hash(chunks[i]),
        "created_at": now,
        "source": "sample.txt",
        "poisoned": False,
    }
    for i in range(len(chunks))
]
collection.insert_many(docs)
print(f"Stored {len(docs)} documents in MongoDB")