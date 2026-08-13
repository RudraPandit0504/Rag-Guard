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

# Which algorithm decides the outlier stage: "hdbscan" (density clustering) or
# "centroid" (the original mean-and-distance rule, kept for comparison).
OUTLIER_METHOD = "hdbscan"

# HDBSCAN parameters, sized for a Top-K retrieval rather than a whole corpus.
# With DEFAULT_TOP_K=5, min_cluster_size=3 means a cluster must be a majority of
# the retrieved set before it can exist at all. min_cluster_size=2 was measured
# and rejected: it treats any two chunks as agreement and shatters a coherent
# set into pairs.
MIN_CLUSTER_SIZE = 3

# None makes HDBSCAN reuse min_cluster_size. Raising it independently makes the
# algorithm more conservative about what counts as dense.
MIN_SAMPLES = None

# Cosine distance below which points merge regardless of density. Left at 0.0
# so cluster selection is decided entirely by the density hierarchy.
#
# This is not a neutral default; it was measured. Real retrieved chunks sit
# 0.31-0.78 cosine distance apart, so any epsilon large enough to merge them
# (>= 0.15) merges the poison too and the filter stops firing at all. Every
# positive value tested dropped zero poisoned chunks. See the README.
CLUSTER_SELECTION_EPSILON = 0.0

# Module 3b, intra-chunk coherence. A chunk scores the mean cosine similarity of
# its weakest sentence to the other sentences in the same chunk; below this it is
# treated as two topics welded together.
#
# Measured over the whole 38-chunk corpus: all 3 poisoned chunks scored <= 0.008,
# and 33 of 34 legitimate chunks scored >= 0.114. 0.06 sits in the middle of that
# gap rather than against either edge, so a slightly better-camouflaged payload
# still has 0.05 of margin to cross before it passes.
#
# The cost is one false positive, chunk 24, at 0.031 — a chunk the character-based
# chunker cut mid-word across two topics. Hugging the floor at 0.02 would spare it
# and catch the same 3 poisoned chunks, but leaves almost no margin. See the README.
COHERENCE_THRESHOLD = 0.06

# A chunk needs at least this many usable sentences before it can disagree with
# itself. Two is the minimum that defines a comparison at all.
MIN_SENTENCES = 2

# Shorter fragments than this are dropped before scoring. A stray "Yes." has
# almost no semantic content and its embedding would read as disagreement.
MIN_SENTENCE_LENGTH = 20

# Injected poison starts far above real chunk ids so the two never collide.
POISON_START_ID = 10000

# How many chunks retrieval returns, and therefore how many the filters judge.
DEFAULT_TOP_K = 5