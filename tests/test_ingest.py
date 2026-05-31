"""
Tests for POST /ingest endpoint and core ingestion services.
Requires /data folder to have at least one document.
"""
import pytest
import numpy as np
from pathlib import Path


def test_ingest_returns_200(client):
    r = client.post("/ingest")
    assert r.status_code == 200


def test_ingest_response_shape(client):
    r    = client.post("/ingest")
    data = r.json()
    assert "message"              in data
    assert "documents_processed"  in data
    assert "chunks_created"       in data


def test_ingest_processes_documents(client):
    r    = client.post("/ingest")
    data = r.json()
    assert data["documents_processed"] > 0


def test_ingest_creates_chunks(client):
    r    = client.post("/ingest")
    data = r.json()
    assert data["chunks_created"] > 0


def test_ingest_chunks_more_than_docs(client):
    """Each document should produce multiple chunks."""
    r    = client.post("/ingest")
    data = r.json()
    assert data["chunks_created"] >= data["documents_processed"]


def test_vector_store_created_after_ingest(client):
    client.post("/ingest")
    assert Path("vector_store/index.faiss").exists()
    assert Path("vector_store/metadata.json").exists()
    assert Path("vector_store/bm25.pkl").exists()


def test_health_shows_vector_store_loaded_after_ingest(client):
    client.post("/ingest")
    r    = client.get("/health")
    data = r.json()
    assert data["vector_store_loaded"] is True


# ── Unit tests for core services ──────────────────────────────

def test_chunker_produces_chunks():
    from app.services.chunker import chunk_documents
    docs   = [{"filename": "test.txt", "content": "A " * 500}]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


def test_chunker_preserves_filename():
    from app.services.chunker import chunk_documents
    docs   = [{"filename": "hipaa.txt", "content": "X " * 500}]
    chunks = chunk_documents(docs)
    assert all(c["filename"] == "hipaa.txt" for c in chunks)


def test_chunker_chunk_index_sequential():
    from app.services.chunker import chunk_documents
    docs   = [{"filename": "test.txt", "content": "A " * 1000}]
    chunks = chunk_documents(docs)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_rrf_merge_deduplicates():
    from app.services.retriever import _reciprocal_rank_fusion
    chunk = {"filename": "a.txt", "chunk_index": 0, "text": "hello", "bm25_score": 1.0, "cosine_score": 0.8}
    merged = _reciprocal_rank_fusion([chunk], [chunk])
    assert len(merged) == 1


def test_confidence_high():
    from app.services.confidence import compute_confidence
    chunks = [{"cosine_score": 0.9}, {"cosine_score": 0.85}]
    assert compute_confidence(chunks) == "high"


def test_confidence_medium():
    from app.services.confidence import compute_confidence
    chunks = [{"cosine_score": 0.6}, {"cosine_score": 0.65}]
    assert compute_confidence(chunks) == "medium"


def test_confidence_low():
    from app.services.confidence import compute_confidence
    chunks = [{"cosine_score": 0.3}, {"cosine_score": 0.2}]
    assert compute_confidence(chunks) == "low"


def test_confidence_none_when_no_scores():
    from app.services.confidence import compute_confidence
    chunks = [{"cosine_score": 0.0}]
    assert compute_confidence(chunks) == "none"
