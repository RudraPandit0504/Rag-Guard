from pymongo import MongoClient
from qdrant_client import QdrantClient
from config import MONGO_URI, QDRANT_URL, QDRANT_API_KEY

print("Testing MongoDB...")
try:
    mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    mongo.admin.command("ping")
    print("  MongoDB OK")
except Exception as e:
    print(f"  MongoDB FAILED: {e}")

print("Testing Qdrant...")
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = qdrant.get_collections()
    print(f"  Qdrant OK — existing collections: {collections.collections}")
except Exception as e:
    print(f"  Qdrant FAILED: {e}")