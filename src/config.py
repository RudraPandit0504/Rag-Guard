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

# Role 2 math filters. Both stages share the same default so that a chunk has
# to clear an equally strict bar on distance and on peer agreement.
OUTLIER_THRESHOLD = 0.30
CONSISTENCY_THRESHOLD = 0.30

# Injected poison starts far above real chunk ids so the two never collide.
POISON_START_ID = 10000

# How many chunks retrieval returns, and therefore how many the filters judge.
DEFAULT_TOP_K = 5