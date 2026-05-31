from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM Providers
    groq_api_key: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str    = "phi3:mini"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval
    chunk_size: int    = 400
    chunk_overlap: int = 80
    top_k: int         = 3
    bm25_top_k: int    = 5
    faiss_top_k: int   = 5

    # Confidence thresholds
    confidence_high: float = 0.75
    confidence_low: float  = 0.50

    # App
    app_env: str   = "development"
    log_level: str = "INFO"

    # Paths
    data_dir:        str = "data"
    vector_store_dir: str = "vector_store"
    logs_dir:        str = "logs"


@lru_cache()
def get_settings() -> Settings:
    return Settings()