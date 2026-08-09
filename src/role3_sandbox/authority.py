"""
Module 4 -- Authority scoring and tamper detection.
"""

import hashlib
from datetime import datetime, timezone
from pymongo import MongoClient
from ..config import MONGO_URI, DB_NAME

T_BASE = 1.0
LAMBDA = 0.005
MIN_AUTHORITY = 0.1

def _get_baseline_epoch() -> datetime:
    """Dynamically fetch the oldest chunk's creation time from MongoDB."""
    try:
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        collection = mongo[DB_NAME]["chunks"]
        oldest_chunk = collection.find_one(sort=[("created_at", 1)])
        if oldest_chunk and "created_at" in oldest_chunk:
            return oldest_chunk["created_at"].replace(tzinfo=timezone.utc)
    except Exception as e:
        print(f"[authority] Failed to fetch baseline epoch from DB: {e}")
    
    # Fallback to current time if DB is empty or unreachable
    return datetime.now(timezone.utc)

BASELINE_EPOCH = _get_baseline_epoch()

def compute_hash(text: str) -> str:
    """Same SHA-256 fingerprint Role 1 uses at ingestion."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def verify_hash(chunk: dict) -> bool:
    """True if the chunk's current text still matches its stored hash."""
    return compute_hash(chunk["text"]) == chunk["hash"]

def age_days(created_at: datetime) -> float:
    """Days *forward* from BASELINE_EPOCH."""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = (created_at - BASELINE_EPOCH).total_seconds() / 86400
    return max(0.0, delta)

def authority_score(chunk: dict) -> float:
    """Authority_Score = max(0.1, T_base - (lambda * Age_days))"""
    return max(MIN_AUTHORITY, T_BASE - LAMBDA * age_days(chunk["created_at"]))

def apply_authority_checks(chunks: list[dict]) -> list[dict]:
    """Attach `hash_valid` and `authority_score` to every chunk."""
    enriched = []
    for chunk in chunks:
        c = dict(chunk)
        c["hash_valid"] = verify_hash(chunk)
        c["authority_score"] = authority_score(chunk)
        enriched.append(c)
    return enriched