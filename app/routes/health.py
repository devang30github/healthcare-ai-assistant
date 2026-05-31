from fastapi import APIRouter
from pathlib import Path
from app.models.schemas import HealthResponse
from app.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    vector_store_path = Path(settings.vector_store_dir) / "index.faiss"
    return HealthResponse(
        status="ok",
        vector_store_loaded=vector_store_path.exists(),
        environment=settings.app_env,
    )