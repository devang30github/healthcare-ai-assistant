import time
from fastapi import APIRouter, HTTPException

from app.models.schemas import AskRequest, AskResponse, SourceChunk
from app.services.vector_store import load_vector_store, is_vector_store_ready
from app.utils.logger import setup_logger
from app.config import get_settings

router = APIRouter()
logger = setup_logger(__name__)

_store: dict | None = None


def load_store_at_startup() -> None:
    global _store
    settings = get_settings()
    if is_vector_store_ready(settings.vector_store_dir):
        logger.info("Loading vector store into memory at startup...")
        _store = load_vector_store(settings.vector_store_dir)
        logger.info("Vector store cached in memory.")
    else:
        logger.info("No vector store found at startup — run POST /ingest first.")


def get_cached_store() -> dict:
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not ready. Please run POST /ingest first."
        )
    return _store


def invalidate_store_cache() -> None:
    global _store
    _store = None
    logger.info("Vector store cache invalidated.")


def reload_store_cache() -> None:
    global _store
    settings = get_settings()
    logger.info("Reloading vector store into memory...")
    _store = load_vector_store(settings.vector_store_dir)
    logger.info("Vector store reloaded.")


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Agentic /ask endpoint — async so FastAPI doesn't block the event loop.
    All business logic lives in agent.py.
    """
    import asyncio
    from app.services.agent import run_agent

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Question: '{question}' | Provider: {request.provider}")
    total_start = time.time()

    # Run sync agent in thread pool — keeps event loop free
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_agent(question, provider=request.provider)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    total_ms = round((time.time() - total_start) * 1000, 2)
    logger.info(
        f"tool={result['tool_used']} | intent={result['intent_ms']}ms | "
        f"retrieval={result['retrieval_ms']}ms | generation={result['generation_ms']}ms | "
        f"total={total_ms}ms | confidence={result['confidence']}"
    )

    sources = [
        SourceChunk(
            document=chunk["filename"],
            chunk_id=chunk["chunk_index"],
            chunk=chunk["text"][:300] + "..."
            if len(chunk["text"]) > 300 else chunk["text"],
        )
        for chunk in result["chunks"]
    ]

    return AskResponse(
        answer=result["answer"],
        sources=sources,
        confidence=result["confidence"],
        provider_used=request.provider,
        tool_used=result["tool_used"],
        intent_ms=result["intent_ms"],
        retrieval_time_ms=result["retrieval_ms"],
        generation_time_ms=result["generation_ms"],
        total_time_ms=total_ms,
    )