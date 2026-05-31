from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.logger import setup_logger
from app.config import get_settings

logger = setup_logger(__name__)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Splits each document into overlapping text chunks using LangChain's 
    RecursiveCharacterTextSplitter to preserve layout structures (lists, tables).

    Input:  [{ "filename": str, "content": str }]
    Output: [{ "filename": str, "chunk_index": int, "text": str }]
    """
    settings = get_settings()
    
    # Initialize the recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Smart hierarchy
    )

    all_chunks = []

    for doc in documents:
        filename = doc["filename"]
        content = doc["content"]

        # LangChain outputs a list of strings
        chunks = splitter.split_text(content)

        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "filename": filename,
                "chunk_index": idx,
                "text": chunk_text.strip()
            })

        logger.info(f"{filename} → {len(chunks)} chunks")

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks