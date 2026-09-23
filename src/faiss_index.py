import json
import numpy as np
import faiss
from pathlib import Path


EMBEDDINGS_FILE = "data/processed/faq_embeddings.npy"
FAQ_FILE = "data/processed/customer_support_faq_processed.jsonl"

INDEX_FILE = "data/processed/faiss_index.bin"
METADATA_FILE = "data/processed/faq_metadata.json"


# Load embeddings
print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print("Embeddings shape:", embeddings.shape)


# FAISS expects float32
embeddings = embeddings.astype("float32")


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# Add embeddings to FAISS
index.add(embeddings)


print("FAISS index created!")
print("Number of vectors:", index.ntotal)


# Load FAQ metadata
records = []

with open(FAQ_FILE, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))


# Save FAISS index
Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

faiss.write_index(index, INDEX_FILE)


# Save metadata
with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(
        records,
        file,
        indent=2,
        ensure_ascii=False
    )


print("FAISS index saved to:", INDEX_FILE)
print("Metadata saved to:", METADATA_FILE)