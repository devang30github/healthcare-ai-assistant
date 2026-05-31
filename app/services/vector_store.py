import json
import pickle
import numpy as np
from pathlib import Path
from app.utils.logger import setup_logger
from app.config import get_settings

logger = setup_logger(__name__)

# ── File paths inside vector_store/ ─────────────────────────
def _paths(vector_store_dir: str):
    base = Path(vector_store_dir)
    return {
        "faiss":    base / "index.faiss",
        "metadata": base / "metadata.json",
        "bm25":     base / "bm25.pkl",
    }


# ── Build & Save ─────────────────────────────────────────────

def build_and_save(
    chunks: list[dict],
    embeddings: np.ndarray,
    vector_store_dir: str,
) -> None:
    """
    Builds FAISS index and BM25 corpus from chunks + embeddings.
    Persists everything to disk.
    """
    import faiss
    from rank_bm25 import BM25Okapi

    paths = _paths(vector_store_dir)
    paths["faiss"].parent.mkdir(parents=True, exist_ok=True)

    # ── FAISS index ──────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)      # Inner product (cosine on normalized vectors)
    index.add(embeddings)
    faiss.write_index(index, str(paths["faiss"]))
    logger.info(f"FAISS index saved: {index.ntotal} vectors, dim={dim}")

    # ── Metadata ─────────────────────────────────────────────
    # Each entry maps 1:1 to a FAISS index position
    metadata = [
        {
            "faiss_id":    i,
            "filename":    chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "text":        chunk["text"],
        }
        for i, chunk in enumerate(chunks)
    ]
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Metadata saved: {len(metadata)} entries")

    # ── BM25 corpus ──────────────────────────────────────────
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(paths["bm25"], "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    logger.info("BM25 corpus saved.")


# ── Load ─────────────────────────────────────────────────────

def load_vector_store(vector_store_dir: str) -> dict:
    """
    Loads FAISS index, metadata, and BM25 from disk.
    Returns a dict with all three for use in retrieval.
    """
    import faiss
    import pickle

    paths = _paths(vector_store_dir)

    if not paths["faiss"].exists():
        raise FileNotFoundError("Vector store not found. Run /ingest first.")

    index    = faiss.read_index(str(paths["faiss"]))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    with open(paths["bm25"], "rb") as f:
        bm25_data = pickle.load(f)

    logger.info(f"Vector store loaded: {index.ntotal} vectors")

    return {
        "faiss_index": index,
        "metadata":    metadata,
        "bm25":        bm25_data["bm25"],
        "chunks":      bm25_data["chunks"],
    }


def is_vector_store_ready(vector_store_dir: str) -> bool:
    paths = _paths(vector_store_dir)
    return all(p.exists() for p in paths.values())
