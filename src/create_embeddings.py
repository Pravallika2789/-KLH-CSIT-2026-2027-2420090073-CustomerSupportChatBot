import json
import numpy as np
from embeddings import generate_embeddings


INPUT_FILE = "data/processed/customer_support_faq_processed.jsonl"
OUTPUT_FILE = "data/processed/faq_embeddings.npy"


# Read processed FAQ data
records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))

print("Loaded records:", len(records))


# Get the text from every FAQ
texts = [record["text"] for record in records]

print("Generating embeddings for 200 FAQs...")

# Generate embeddings
embeddings = generate_embeddings(texts)

# Save embeddings
np.save(OUTPUT_FILE, embeddings)

print("Embeddings generated successfully!")
print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", embeddings.shape[1])
print("Saved to:", OUTPUT_FILE)