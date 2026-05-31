import time
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List

from app.models.schemas import IngestResponse
from app.services.loader import load_documents
from app.services.chunker import chunk_documents
from app.services.embeddings import embed_chunks
from app.services.vector_store import build_and_save
from app.utils.logger import setup_logger
from app.config import get_settings

router = APIRouter()
logger = setup_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Uploads one or more documents to the /data folder.
    Accepts: .pdf, .txt, .docx
    Does NOT ingest — call POST /ingest after uploading.
    """
    settings   = get_settings()
    data_path  = Path(settings.data_dir)
    data_path.mkdir(exist_ok=True)

    saved    = []
    rejected = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            rejected.append(file.filename)
            continue
        dest = data_path / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
        logger.info(f"Uploaded: {file.filename}")

    return {
        "saved":    saved,
        "rejected": rejected,
        "message":  f"{len(saved)} file(s) uploaded. Run POST /ingest to process."
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Reads all documents from /data, chunks, embeds,
    saves FAISS + BM25 + metadata, reloads store into memory.
    """
    import asyncio
    settings = get_settings()
    start    = time.time()

    try:
        logger.info("Starting ingestion pipeline...")

        # Run blocking ingestion in thread pool
        def _ingest():
            documents  = load_documents(settings.data_dir)
            if not documents:
                raise ValueError("No documents found in /data.")
            chunks     = chunk_documents(documents)
            embeddings = embed_chunks(chunks)
            build_and_save(chunks, embeddings, settings.vector_store_dir)
            return documents, chunks

        documents, chunks = await asyncio.get_event_loop().run_in_executor(None, _ingest)

        from app.routes.ask import reload_store_cache
        reload_store_cache()

        elapsed = round((time.time() - start) * 1000, 2)
        logger.info(f"Ingestion complete in {elapsed}ms")

        return IngestResponse(
            message="Ingestion complete.",
            documents_processed=len(documents),
            chunks_created=len(chunks),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/clear-db")
async def clear_vector_store():
    """Deletes FAISS index, BM25 corpus, and metadata from disk."""
    from app.routes.ask import invalidate_store_cache

    settings = get_settings()
    vs_path  = Path(settings.vector_store_dir)

    try:
        deleted = []
        for f in ["index.faiss", "metadata.json", "bm25.pkl"]:
            fp = vs_path / f
            if fp.exists():
                fp.unlink()
                deleted.append(f)

        invalidate_store_cache()
        logger.info(f"Vector store cleared: {deleted}")
        return {"message": f"Cleared: {', '.join(deleted) if deleted else 'nothing to clear'}"}

    except Exception as e:
        logger.error(f"Clear DB failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))