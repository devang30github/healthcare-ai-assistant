"""
Tests for POST /ask endpoint.
LLM calls are mocked — no API keys or network needed.
Only Groq and Ollama providers supported.
"""
import pytest
from unittest.mock import patch


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="module")
def ensure_ingested(client):
    """Make sure vector store is ready before ask tests run."""
    client.post("/ingest")


# ── Request validation ─────────────────────────────────────────

def test_ask_empty_question_returns_400(client):
    r = client.post("/ask", json={"question": "", "provider": "groq"})
    assert r.status_code == 400


def test_ask_missing_question_returns_422(client):
    r = client.post("/ask", json={"provider": "groq"})
    assert r.status_code == 422


def test_ask_invalid_provider_returns_error(client):
    """Invalid provider should return 400 or 500 — not crash the server."""
    with patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r = client.post("/ask", json={"question": "What is HIPAA?", "provider": "invalid"})
    assert r.status_code in (400, 500)
    assert "detail" in r.json()


# ── Response shape ─────────────────────────────────────────────

def test_ask_response_has_required_fields(client):
    with patch("app.services.agent._llm_generate", return_value="HIPAA protects patient data."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
        data = r.json()
    assert "answer"             in data
    assert "sources"            in data
    assert "confidence"         in data
    assert "provider_used"      in data
    assert "tool_used"          in data
    assert "intent_ms"          in data
    assert "retrieval_time_ms"  in data
    assert "generation_time_ms" in data
    assert "total_time_ms"      in data


def test_ask_answer_is_string(client):
    with patch("app.services.agent._llm_generate", return_value="Test answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
        data = r.json()
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_ask_sources_is_list(client):
    with patch("app.services.agent._llm_generate", return_value="Answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
        data = r.json()
    assert isinstance(data["sources"], list)


def test_ask_source_has_chunk_id(client):
    with patch("app.services.agent._llm_generate", return_value="Answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
        data = r.json()
    for src in data["sources"]:
        assert "document" in src
        assert "chunk_id" in src
        assert "chunk"    in src


def test_ask_confidence_is_valid(client):
    with patch("app.services.agent._llm_generate", return_value="Answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
        data = r.json()
    assert data["confidence"] in ("high", "medium", "low", "none")


def test_ask_tool_used_is_valid(client):
    with patch("app.services.agent._llm_generate", return_value="Answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
        data = r.json()
    assert data["tool_used"] in ("rag_search", "appointment_tool", "unknown")


def test_ask_timing_fields_are_positive(client):
    with patch("app.services.agent._llm_generate", return_value="Answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
        data = r.json()
    assert data["total_time_ms"]      >= 0
    assert data["retrieval_time_ms"]  >= 0
    assert data["generation_time_ms"] >= 0
    assert data["intent_ms"]          >= 0


# ── Fallback behaviour ─────────────────────────────────────────

def test_ask_returns_fallback_for_unknown_question(client):
    with patch("app.services.agent._llm_generate", return_value="I could not find this information in the provided documents."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r    = client.post("/ask", json={
            "question": "What is the airspeed velocity of an unladen swallow?",
            "provider": "groq"
        })
        data = r.json()
    assert r.status_code == 200
    assert isinstance(data["answer"], str)


# ── Appointment routing ────────────────────────────────────────

def test_ask_routes_appointment_correctly(client):
    with patch("app.services.agent._classify_intent", return_value=("appointment", 50.0)), \
         patch("app.services.agent._handle_appointment", return_value="Available slots: 9AM, 11AM"):
        r    = client.post("/ask", json={
            "question": "Book a cardiology appointment for Monday",
            "provider": "groq"
        })
        data = r.json()
    assert r.status_code == 200
    assert data["tool_used"]  == "appointment_tool"
    assert data["sources"]    == []
    assert data["confidence"] == "none"


def test_appointment_answer_contains_slots(client):
    with patch("app.services.agent._classify_intent", return_value=("appointment", 50.0)), \
         patch("app.services.agent._handle_appointment", return_value="Available: 9:00 AM, 11:00 AM"):
        r    = client.post("/ask", json={
            "question": "Book cardiology Monday",
            "provider": "groq"
        })
        data = r.json()
    assert "Available" in data["answer"]


# ── Provider validation ────────────────────────────────────────

def test_groq_is_valid_provider(client):
    with patch("app.services.agent._llm_generate", return_value="Answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r = client.post("/ask", json={"question": "What is HIPAA?", "provider": "groq"})
    assert r.status_code == 200
    assert r.json()["provider_used"] == "groq"


def test_ollama_is_valid_provider(client):
    with patch("app.services.agent._llm_generate", return_value="Answer."), \
         patch("app.services.agent._classify_intent", return_value=("knowledge", 50.0)):
        r = client.post("/ask", json={"question": "What is HIPAA?", "provider": "ollama"})
    assert r.status_code == 200
    assert r.json()["provider_used"] == "ollama"


# ── Prompt builder ─────────────────────────────────────────────

def test_prompt_builder_no_source_labels():
    """Prompt context must not contain [Source N:] labels."""
    from app.prompts.rag_prompt import build_prompt
    chunks = [
        {"filename": "hipaa.txt",      "chunk_index": 0, "text": "HIPAA protects PHI."},
        {"filename": "telehealth.txt",  "chunk_index": 1, "text": "Telehealth is remote care."},
    ]
    system, _ = build_prompt("What is HIPAA?", chunks)
    assert "[Source" not in system


def test_prompt_includes_chunk_text():
    from app.prompts.rag_prompt import build_prompt
    chunks = [{"filename": "a.txt", "chunk_index": 0, "text": "Unique content XYZ123"}]
    system, _ = build_prompt("question", chunks)
    assert "Unique content XYZ123" in system