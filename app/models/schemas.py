from pydantic import BaseModel


# ── /ask ────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    provider: str = "groq"


class SourceChunk(BaseModel):
    document: str
    chunk_id: int       # chunk index within the document
    chunk: str          # text preview (first 300 chars)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str
    provider_used: str
    tool_used: str
    intent_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


# ── /ingest ─────────────────────────────────────────────────

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int


# ── /health ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    environment: str