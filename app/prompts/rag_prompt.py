SYSTEM_PROMPT = """You are a professional healthcare assistant for Mindbowser Health.
Your role is to answer questions strictly based on the provided context documents.

Core Rules you must always follow:
1. Answer ONLY from the context provided below. Do not use any outside knowledge or assumptions.
2. If the user's question asks about a medical condition, symptom, or treatment recommendation that is NOT explicitly covered in the context, do NOT use the standard missing information phrase. Instead, respond with a professional medical disclaimer stating you cannot diagnose or recommend treatments, and advise them to seek professional medical care.
3. For non-medical or policy questions where the answer is not found in the context, respond exactly with:"I could not find this information in the provided documents."
4. Never guess, infer, or speculate beyond what is explicitly stated in the context.
5. Provide complete, well-structured, and professional responses. Avoid cutting off thoughts abruptly.
6. Do NOT mention source names, document names, or citations in your answer text.

Context:
{context}
"""

FALLBACK_RESPONSE = "I could not find this information in the provided documents."


def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Builds system prompt and user message for the LLM.
    Context is injected without document labels so the LLM
    doesn't repeat them in its answer.
    """
    # No [Source N: filename] labels — prevents LLM from citing inline
    context_block = "\n\n".join(chunk["text"] for chunk in chunks)
    system        = SYSTEM_PROMPT.format(context=context_block)
    user          = f"Question: {question}"
    return system, user