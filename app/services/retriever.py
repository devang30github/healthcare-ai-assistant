import numpy as np
from app.utils.logger import setup_logger
from app.config import get_settings

logger = setup_logger(__name__)


def hybrid_retrieve(query: str, store: dict) -> list[dict]:
    """
    Runs BM25 + FAISS retrieval independently, merges with RRF.

    Args:
        query: raw user question string
        store: loaded vector store dict from vector_store.load_vector_store()

    Returns:
        Top-K chunks as list of dicts with keys:
        { filename, chunk_index, text, score, bm25_rank, faiss_rank }
    """
    settings = get_settings()
    bm25_top_k  = settings.bm25_top_k
    faiss_top_k = settings.faiss_top_k
    final_top_k = settings.top_k

    # ── Step 1: BM25 lexical retrieval ──────────────────────
    bm25_results = _bm25_search(query, store, top_k=bm25_top_k)
    logger.info(f"BM25 returned {len(bm25_results)} results")

    # ── Step 2: FAISS semantic retrieval ────────────────────
    faiss_results = _faiss_search(query, store, top_k=faiss_top_k)
    logger.info(f"FAISS returned {len(faiss_results)} results")

    # ── Step 3: RRF merge ────────────────────────────────────
    merged = _reciprocal_rank_fusion(bm25_results, faiss_results)
    logger.info(f"RRF merged to {len(merged)} unique chunks")

    # ── Step 4: Return top K ─────────────────────────────────
    top_chunks = merged[:final_top_k]
    for i, c in enumerate(top_chunks):
        logger.info(f"  Chunk {i+1}: [{c['filename']}] score={c['rrf_score']:.4f} cosine={c.get('cosine_score', 0):.4f}")

    return top_chunks


# ── BM25 ─────────────────────────────────────────────────────

def _bm25_search(query: str, store: dict, top_k: int) -> list[dict]:
    bm25   = store["bm25"]
    chunks = store["chunks"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get indices sorted by score descending
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        if scores[idx] > 0:                     # skip zero-score chunks
            results.append({
                **chunks[idx],
                "bm25_score": float(scores[idx]),
                "bm25_rank":  rank,
                "cosine_score": 0.0,            # filled by FAISS if overlap
            })
    return results


# ── FAISS ─────────────────────────────────────────────────────

def _faiss_search(query: str, store: dict, top_k: int) -> list[dict]:
    from app.services.embeddings import embed_query

    index    = store["faiss_index"]
    metadata = store["metadata"]

    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({
            "filename":    meta["filename"],
            "chunk_index": meta["chunk_index"],
            "text":        meta["text"],
            "cosine_score": float(dist),        # normalized → cosine similarity
            "faiss_rank":  rank,
            "bm25_score":  0.0,
        })
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────

def _reciprocal_rank_fusion(
    bm25_results: list[dict],
    faiss_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    RRF formula: score(d) = sum(1 / (k + rank(d)))
    Merges two ranked lists, deduplicates by (filename, chunk_index).
    """
    rrf_scores: dict[tuple, float] = {}
    chunk_map:  dict[tuple, dict]  = {}

    # Score BM25 list
    for rank, chunk in enumerate(bm25_results):
        key = (chunk["filename"], chunk["chunk_index"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[key]  = chunk

    # Score FAISS list (add to existing RRF score if already seen)
    for rank, chunk in enumerate(faiss_results):
        key = (chunk["filename"], chunk["chunk_index"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key in chunk_map:
            # Preserve cosine score from FAISS result
            chunk_map[key]["cosine_score"] = chunk["cosine_score"]
        else:
            chunk_map[key] = chunk

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    merged = []
    for key in sorted_keys:
        chunk = chunk_map[key]
        chunk["rrf_score"] = rrf_scores[key]
        merged.append(chunk)

    return merged
