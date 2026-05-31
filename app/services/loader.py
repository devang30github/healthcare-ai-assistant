from pathlib import Path
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_documents(data_dir: str) -> list[dict]:
    """
    Reads all PDF, DOCX, and TXT files from data_dir.
    Returns a list of dicts: { "filename": str, "content": str }
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = []
    supported = {".pdf", ".docx", ".txt"}

    for file_path in sorted(data_path.iterdir()):
        if file_path.suffix.lower() not in supported:
            logger.warning(f"Skipping unsupported file: {file_path.name}")
            continue

        try:
            content = _read_file(file_path)
            if content.strip():
                documents.append({
                    "filename": file_path.name,
                    "content": content.strip()
                })
                logger.info(f"Loaded: {file_path.name} ({len(content)} chars)")
            else:
                logger.warning(f"Empty content, skipping: {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def _read_file(file_path: Path) -> str:
    """Dispatch to the correct reader based on file extension."""
    ext = file_path.suffix.lower()

    if ext == ".txt":
        return _read_txt(file_path)
    elif ext == ".pdf":
        return _read_pdf(file_path)
    elif ext == ".docx":
        return _read_docx(file_path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
