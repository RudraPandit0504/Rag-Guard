from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Quarterly revenue grew by twelve percent.",
]

vectors = model.encode(sentences)

print(f"Shape: {vectors.shape}")
print(f"First 5 numbers of sentence 1: {vectors[0][:5]}")

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print()
print(f"cat/feline    : {cosine(vectors[0], vectors[1]):.3f}")
print(f"cat/revenue   : {cosine(vectors[0], vectors[2]):.3f}")
print(f"cat/itself    : {cosine(vectors[0], vectors[0]):.3f}")