import numpy as np
from app.utils.logger import setup_logger
from app.config import get_settings

logger = setup_logger(__name__)

# Module-level model cache — loaded once, reused on every call
_model = None


def get_embedding_model():
    """Lazy load the embedding model once and cache it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded.")
    return _model


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Generates embeddings for a list of chunk dicts.
    Returns a float32 numpy array of shape (n_chunks, embedding_dim).
    """
    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity = dot product
        convert_to_numpy=True,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string.
    Returns a float32 numpy array of shape (1, embedding_dim).
    """
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.astype("float32")
