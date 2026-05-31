import time
import json
import concurrent.futures
from openai import OpenAI
from app.utils.logger import setup_logger
from app.config import get_settings

logger = setup_logger(__name__)


def stream_agent(question: str, provider: str = "groq"):
    """
    Generator that yields Server-Sent Events (SSE).

    Flow:
    1. Intent + retrieval run concurrently (same as non-streaming)
    2. If appointment → yield full answer as single event
    3. If knowledge   → stream LLM tokens as they arrive

    SSE event types:
      event: metadata  — intent, retrieval_ms, confidence, sources, tool_used
      event: token     — one LLM token chunk
      event: done      — signals stream end with final timing
      event: error     — something went wrong
    """
    from app.services.agent import _classify_intent, _run_retrieval, _handle_appointment
    from app.prompts.rag_prompt import build_prompt, FALLBACK_RESPONSE
    from app.services.confidence import compute_confidence

    total_start = time.time()

    # ── Step 1: Concurrent intent + retrieval ────────────────
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            intent_future    = executor.submit(_classify_intent, question)
            retrieval_future = executor.submit(_run_retrieval, question)
            intent, intent_ms = intent_future.result()
            retrieval_result  = retrieval_future.result()
    except Exception as e:
        yield _sse("error", {"message": str(e)})
        return

    logger.info(f"Stream | intent={intent} ({intent_ms}ms) | retrieval={retrieval_result['retrieval_ms']}ms")

    # ── Step 2: Appointment path (no streaming needed) ───────
    if intent == "appointment":
        t0     = time.time()
        answer = _handle_appointment(question, provider)
        gen_ms = round((time.time() - t0) * 1000, 2)

        yield _sse("metadata", {
            "tool_used":    "appointment_tool",
            "intent_ms":    intent_ms,
            "retrieval_ms": 0.0,
            "confidence":   "none",
            "sources":      [],
        })
        yield _sse("token", {"text": answer})
        yield _sse("done",  {
            "generation_ms": gen_ms,
            "total_ms":      round((time.time() - total_start) * 1000, 2),
        })
        return

    # ── Step 3: RAG path — stream tokens ────────────────────
    chunks     = retrieval_result["chunks"]
    confidence = retrieval_result["confidence"]

    if not chunks or confidence == "none":
        yield _sse("metadata", {
            "tool_used":    "rag_search",
            "intent_ms":    intent_ms,
            "retrieval_ms": retrieval_result["retrieval_ms"],
            "confidence":   "none",
            "sources":      [],
        })
        yield _sse("token", {"text": FALLBACK_RESPONSE})
        yield _sse("done",  {"generation_ms": 0.0, "total_ms": round((time.time() - total_start) * 1000, 2)})
        return

    # Build sources metadata
    sources = [
        {
            "document": chunk["filename"],
            "chunk_id": chunk["chunk_index"],
            "chunk":    chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
        }
        for chunk in chunks
    ]

    # Send metadata first so frontend can render confidence + sources immediately
    yield _sse("metadata", {
        "tool_used":    "rag_search",
        "intent_ms":    intent_ms,
        "retrieval_ms": retrieval_result["retrieval_ms"],
        "confidence":   confidence,
        "sources":      sources,
    })

    # ── Stream LLM tokens ────────────────────────────────────
    system_prompt, user_message = build_prompt(question, chunks)
    client, model               = _get_streaming_client(provider)

    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.1,
            max_tokens=512,
            stream=True,           # key difference from non-streaming
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield _sse("token", {"text": delta.content})

    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        yield _sse("error", {"message": f"Generation failed: {str(e)}"})
        return

    gen_ms = round((time.time() - t0) * 1000, 2)
    logger.info(f"Stream complete | generation={gen_ms}ms")

    yield _sse("done", {
        "generation_ms": gen_ms,
        "total_ms":      round((time.time() - total_start) * 1000, 2),
    })


# ── Helpers ───────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _get_streaming_client(provider: str) -> tuple:
    """Returns (OpenAI client, model string) for the given provider."""
    settings = get_settings()
    p        = provider.lower().strip()

    if p == "groq":
        return OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        ), "llama-3.3-70b-versatile"

    elif p == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
        ), settings.ollama_model


    else:
        raise ValueError(f"Unsupported provider: {provider}")