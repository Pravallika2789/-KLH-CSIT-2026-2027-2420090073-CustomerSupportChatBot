import json
import faiss
import numpy as np
from backend.embeddings import generate_embedding


INDEX_FILE = "data/processed/faiss_index.bin"
METADATA_FILE = "data/processed/faq_metadata.json"


# Load FAISS index
index = faiss.read_index(INDEX_FILE)

# Load FAQ information
with open(METADATA_FILE, "r", encoding="utf-8") as file:
    records = json.load(file)


def retrieve_documents(query, top_k=3):

    # Convert customer question into embedding
    query_embedding = generate_embedding(query)

    # Convert to required format
    query_embedding = query_embedding.reshape(1, -1).astype("float32")

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        record = records[idx]

        results.append({
            "question": record["question"],
            "answer": record["answer"],
            "source": record["source"],
            "distance": float(distance)
        })

    return results