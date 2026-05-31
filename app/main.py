from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from contextlib import asynccontextmanager

from app.routes.health import router as health_router
from app.routes.ingest import router as ingest_router
from app.routes.ask    import router as ask_router
from app.routes.stream import router as stream_router
from app.utils.logger  import setup_logger
from app.config        import get_settings

logger   = setup_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Replaces deprecated @app.on_event("startup").
    Runs once at startup before accepting requests.
    """
    logger.info("Healthcare AI Assistant starting up...")
    logger.info(f"Environment: {settings.app_env}")
 
    # Pre-load embedding model — avoids cold start on first /ask
    logger.info("Pre-loading embedding model...")
    from app.services.embeddings import get_embedding_model
    get_embedding_model()
    logger.info("Embedding model ready.")
 
    # Pre-load vector store into memory at startup
    from app.routes.ask import load_store_at_startup
    load_store_at_startup()
 
    yield   # app runs here
 
    # Shutdown logic goes here if needed
    logger.info("Healthcare AI Assistant shutting down.")
    
app = FastAPI(
    title="Healthcare AI Assistant",
    description="RAG-based healthcare assistant — hybrid search, multi-provider LLM, streaming.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(ask_router)
app.include_router(stream_router)

# ── Static frontend ───────────────────────────────────────────
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


