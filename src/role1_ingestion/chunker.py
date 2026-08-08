import hashlib
from pathlib import Path

CHUNK_SIZE = 400
OVERLAP = 80


def compute_hash(text: str) -> str:
    """SHA-256 fingerprint of a chunk, used later to detect tampering."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking at word boundaries."""
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        piece = text[start:end]

        if end < len(text):
            last_space = piece.rfind(" ")
            if last_space > size * 0.5:
                piece = piece[:last_space]
                end = start + last_space

        piece = piece.strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break
        start = end - overlap

    return chunks


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    sample = data_dir / "sample.txt"

    raw = sample.read_text(encoding="utf-8")
    pieces = chunk_text(raw)

    print(f"Original: {len(raw)} chars")
    print(f"Chunks:   {len(pieces)}\n")

    for i, p in enumerate(pieces):
        print(f"[{i}] len={len(p)} hash={compute_hash(p)[:12]}...")
        print(f"    {p[:80]}...\n")