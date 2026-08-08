import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

_required = [
    ("MONGO_URI", MONGO_URI),
    ("QDRANT_URL", QDRANT_URL),
    ("QDRANT_API_KEY", QDRANT_API_KEY),
]
_missing = [name for name, value in _required if not value]
if _missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing)}.\n"
        "Copy .env.example to .env and fill in the values."
    )

COLLECTION_NAME = "ragguard_chunks"
DB_NAME = "ragguard"
VECTOR_SIZE = 384
MODEL_NAME = "all-MiniLM-L6-v2"