from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_confidence(chunks: list[dict]) -> str:
    """
    Derives a confidence level from the average cosine similarity
    of the retrieved top-K chunks.

    Thresholds (configurable in .env):
        High   >= confidence_high  (default 0.75)
        Medium >= confidence_low   (default 0.50)
        Low    <  confidence_low

    If no chunks have a cosine score, returns 'none' — triggers fallback.
    """
    settings = get_settings()

    cosine_scores = [
        c["cosine_score"]
        for c in chunks
        if c.get("cosine_score", 0.0) > 0.0
    ]

    if not cosine_scores:
        logger.info("Confidence: none (no cosine scores available)")
        return "none"

    avg_score = sum(cosine_scores) / len(cosine_scores)
    logger.info(f"Avg cosine score: {avg_score:.4f}")

    if avg_score >= settings.confidence_high:
        level = "high"
    elif avg_score >= settings.confidence_low:
        level = "medium"
    else:
        level = "low"

    logger.info(f"Confidence level: {level}")
    return level
