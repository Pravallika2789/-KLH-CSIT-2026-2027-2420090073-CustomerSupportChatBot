from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):
    """Generate embedding for one text."""
    return model.encode(text)


def generate_embeddings(texts):
    """Generate embeddings for multiple texts."""
    return model.encode(
        texts,
        show_progress_bar=True
    )