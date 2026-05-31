import time
import concurrent.futures
from openai import OpenAI
from app.tools.appointment import check_available_slots as _check_slots
from app.utils.logger import setup_logger
from app.config import get_settings

logger = setup_logger(__name__)

# ── LLM Constants ─────────────────────────────────────────────
GROQ_BASE_URL    = "https://api.groq.com/openai/v1"
GROQ_GEN_MODEL   = "llama-3.3-70b-versatile"   # generation
GROQ_INTENT_MODEL = "llama-3.1-8b-instant"     # LLM intent fallback only

# ── Module-level cached clients ───────────────────────────────
# Created once on first use — reuses TCP connection pool across all requests.
# Creating a new OpenAI() per request adds 200-400ms (TLS handshake).
_groq_client:   OpenAI | None = None
_ollama_client: OpenAI | None = None


def _get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is None:
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        _groq_client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=GROQ_BASE_URL,
        )
        logger.info("Groq OpenAI client initialized (cached)")
    return _groq_client


def _get_ollama_client() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        settings = get_settings()
        _ollama_client = OpenAI(
            api_key="ollama",                          # required but unused by Ollama
            base_url=f"{settings.ollama_base_url}/v1",
        )
        logger.info(f"Ollama OpenAI client initialized (cached) | url={settings.ollama_base_url}")
    return _ollama_client


# ── Intent Keywords ───────────────────────────────────────────
# Strong signals score ±2, weak signals score ±1.
# Score >= 2  → appointment  (high confidence)
# Score == 1  → ambiguous    → LLM fallback
# Score <= 0  → knowledge
#
# Negative scores come from KNOWLEDGE_STRONG pulling away from appointment,
# which handles cases like: "What is the HIPAA policy for appointment data?"
# (appointment=+1, policy=-2 → score=-1 → knowledge ✓)

APPOINTMENT_STRONG = {
    "book", "schedule", "slot", "slots", "reserve",
    "reschedule", "cancel appointment", "book appointment",
    "make appointment", "set appointment",
}
APPOINTMENT_WEAK = {
    "appointment", "visit", "available", "availability",
    "when can i", "open slot", "free slot",
}
KNOWLEDGE_STRONG = {
    "policy", "hipaa", "what is", "how does", "guidelines",
    "refill", "medication", "insurance", "discharge", "privacy",
    "coverage", "compliance", "procedure", "explain", "definition",
    "telehealth policy", "can i", "do i", "is it", "what are",
}

INTENT_PROMPT = """You are an intent classifier for a healthcare assistant.

Classify the user question into exactly one of these intents:
- "appointment" — user wants to book, schedule, check, or inquire about appointment slots
- "knowledge"   — anything else: policies, guidelines, medical info, HIPAA, medication, etc.

Respond with ONLY one word: appointment or knowledge. Nothing else.

User question: {question}"""


# ── Public Entry Point ────────────────────────────────────────

def run_agent(question: str, provider: str = "groq") -> dict:
    """
    Optimized two-step agentic workflow.
    Intent classification and retrieval run CONCURRENTLY.

    Returns:
    {
        answer, tool_used, chunks, confidence,
        intent_ms, retrieval_ms, generation_ms
    }
    """
    logger.info(f"Agent received question: '{question}' | provider: {provider}")

    # ── Concurrent: intent + retrieval ──────────────────────
    # Retrieval always runs — even if intent turns out to be appointment,
    # the cost is only ~80ms wasted, but saves ~250ms on knowledge queries.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        intent_future    = executor.submit(_classify_intent, question)
        retrieval_future = executor.submit(_run_retrieval, question)

        intent, intent_ms = intent_future.result()
        retrieval_result  = retrieval_future.result()

    logger.info(
        f"Intent: '{intent}' in {intent_ms}ms | "
        f"Retrieval: {retrieval_result['retrieval_ms']}ms (concurrent)"
    )

    # ── Route ────────────────────────────────────────────────
    if intent == "appointment":
        t0     = time.time()
        answer = _handle_appointment(question, provider)
        gen_ms = round((time.time() - t0) * 1000, 2)
        return {
            "answer":        answer,
            "tool_used":     "appointment_tool",
            "chunks":        [],
            "confidence":    "none",
            "intent_ms":     intent_ms,
            "retrieval_ms":  0.0,
            "generation_ms": gen_ms,
        }
    else:
        result = _handle_generation(question, provider, retrieval_result)
        return {
            "answer":        result["answer"],
            "tool_used":     "rag_search",
            "chunks":        result["chunks"],
            "confidence":    result["confidence"],
            "intent_ms":     intent_ms,
            "retrieval_ms":  retrieval_result["retrieval_ms"],
            "generation_ms": result["generation_ms"],
        }


# ── Intent Classifier ─────────────────────────────────────────

def _classify_intent(question: str) -> tuple[str, float]:
    """
    Hybrid intent classification:

    Step 1 — Weighted keyword scoring (<1ms, no network):
        score >= 2  → appointment  (high confidence, return immediately)
        score <= 0  → knowledge    (high confidence, return immediately)
        score == 1  → ambiguous    → fall through to Step 2

    Step 2 — LLM fallback (only for ambiguous cases ~5-10% of queries):
        Uses llama-3.1-8b-instant via cached Groq client.
        If LLM also fails → default to "knowledge" (safer fallback).
    """
    t0    = time.time()
    q     = question.lower()
    words = set(q.split())

    # ── Step 1: Keyword scoring ──────────────────────────────
    score = 0
    score += sum(2 for kw in APPOINTMENT_STRONG if kw in q)
    score += sum(1 for kw in APPOINTMENT_WEAK   if kw in words)
    score -= sum(2 for kw in KNOWLEDGE_STRONG   if kw in q)

    if score >= 2:
        intent = "appointment"
        logger.info(f"Intent (keyword/strong): '{intent}' | score={score}")
        return intent, round((time.time() - t0) * 1000, 2)

    if score <= 0:
        intent = "knowledge"
        logger.info(f"Intent (keyword): '{intent}' | score={score}")
        return intent, round((time.time() - t0) * 1000, 2)

    # ── Step 2: Ambiguous (score == 1) → LLM fallback ───────
    logger.info(f"Intent ambiguous (score={score}), calling LLM fallback...")
    intent = _llm_intent_fallback(question)
    logger.info(f"Intent (LLM fallback): '{intent}'")
    return intent, round((time.time() - t0) * 1000, 2)


def _llm_intent_fallback(question: str) -> str:
    """
    Called only when keyword scoring is ambiguous (score == 1).
    Uses cached Groq client + small fast model.
    Defaults to 'knowledge' on any failure.
    """
    try:
        client = _get_groq_client()
        resp   = client.chat.completions.create(
            model=GROQ_INTENT_MODEL,
            messages=[{"role": "user", "content": INTENT_PROMPT.format(question=question)}],
            temperature=0.0,
            max_tokens=5,       # "appointment" and "knowledge" are single tokens
            timeout=5,
        )
        raw = resp.choices[0].message.content.strip().lower()
        return "appointment" if "appointment" in raw else "knowledge"
    except Exception as e:
        logger.warning(f"LLM intent fallback failed, defaulting to knowledge: {e}")
        return "knowledge"


# ── Retrieval ─────────────────────────────────────────────────

def _run_retrieval(question: str) -> dict:
    """
    Runs hybrid BM25 + FAISS retrieval.
    Uses cached vector store — no disk read per request.
    Runs concurrently with intent classification.
    """
    from app.services.retriever import hybrid_retrieve
    from app.services.confidence import compute_confidence
    from app.routes.ask import get_cached_store

    t0 = time.time()
    try:
        store      = get_cached_store()
        chunks     = hybrid_retrieve(question, store)
        confidence = compute_confidence(chunks)
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        return {"chunks": [], "confidence": "none", "retrieval_ms": 0.0}

    retrieval_ms = round((time.time() - t0) * 1000, 2)
    logger.info(f"Retrieval: {retrieval_ms}ms | confidence={confidence} | chunks={len(chunks)}")
    return {"chunks": chunks, "confidence": confidence, "retrieval_ms": retrieval_ms}


# ── Generation ────────────────────────────────────────────────

def _handle_generation(question: str, provider: str, retrieval_result: dict) -> dict:
    """
    Runs LLM generation using the pre-fetched retrieval result.
    No retrieval happens here — it already ran concurrently.
    """
    from app.prompts.rag_prompt import build_prompt, FALLBACK_RESPONSE

    chunks     = retrieval_result["chunks"]
    confidence = retrieval_result["confidence"]

    if not chunks or confidence == "none":
        return {
            "answer":        FALLBACK_RESPONSE,
            "chunks":        [],
            "confidence":    "none",
            "generation_ms": 0.0,
        }

    system_prompt, user_message = build_prompt(question, chunks)

    t0     = time.time()
    answer = _llm_generate(provider, system_prompt, user_message)
    gen_ms = round((time.time() - t0) * 1000, 2)
    logger.info(f"Generation: {gen_ms}ms")

    return {
        "answer":        answer,
        "chunks":        chunks,
        "confidence":    confidence,
        "generation_ms": gen_ms,
    }


# ── Appointment Handler ───────────────────────────────────────

def _handle_appointment(question: str, provider: str) -> str:
    """
    Extracts department + day from the question via LLM,
    then calls the mock scheduling tool.
    """
    extract_prompt = f"""Extract the department and day from this appointment question.
Respond in exactly this format: department=<name>,day=<day>
Use only these departments: cardiology, general, dermatology, telehealth
Use lowercase day names: monday, tuesday, wednesday, thursday, friday, saturday
If department is unclear use: general. If day is unclear use: monday.
Question: {question}"""

    try:
        raw = _llm_generate(provider, "", extract_prompt, max_tokens=20)
        raw = raw.strip().lower()
        logger.info(f"Appointment extraction raw: '{raw}'")

        department = "general"
        day        = "monday"
        parsed     = {"department": False, "day": False}

        for part in raw.replace(" ", "").split(","):
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key == "department" and val:
                    department = val
                    parsed["department"] = True
                elif key == "day" and val:
                    day = val
                    parsed["day"] = True

        if not parsed["department"]:
            logger.warning("Department not extracted from LLM response, using default: general")
        if not parsed["day"]:
            logger.warning("Day not extracted from LLM response, using default: monday")

        return _check_slots(department, day)

    except Exception as e:
        logger.error(f"Appointment handler failed: {e}")
        return "I was unable to check appointment availability. Please contact our scheduling desk directly."


# ── Unified LLM Call ──────────────────────────────────────────

def _llm_generate(
    provider: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 300,      
) -> str:
    """
    Single function for all LLM generation calls.
    Uses module-level cached clients — no new TCP/TLS handshake per request.
    """
    p = provider.lower().strip()

    if p == "groq":
        client = _get_groq_client()
        model  = GROQ_GEN_MODEL

    elif p == "ollama":
        client = _get_ollama_client()
        model  = get_settings().ollama_model

    else:
        raise ValueError(f"Unsupported provider: '{p}'. Choose from: groq, ollama")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()