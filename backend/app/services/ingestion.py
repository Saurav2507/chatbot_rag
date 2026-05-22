import logging
from pathlib import Path

from app.services.chunking import create_chunks_with_metadata
from app.services.indexing import index_chunks
from app.services.loaders import iter_supported_documents, load_document

logger = logging.getLogger(__name__)


def process_document(file_path: str | Path) -> list[dict]:
    """Compatibility wrapper: load one supported document into page records."""
    return load_document(file_path)


def ingest_document(file_path: str | Path, replace_existing: bool = True) -> dict:
    """Load, preprocess, chunk, embed, and index a single document."""
    path = Path(file_path)
    logger.info("Starting ingestion for %s", path.name)

    pages = load_document(path)
    if not pages:
        logger.warning("No text extracted from %s", path.name)
        return {"filename": path.name, "total_pages": 0, "total_chunks": 0, "status": "empty"}

    chunks = create_chunks_with_metadata(pages, path.name)
    indexed = index_chunks(chunks, path.name, replace_existing=replace_existing)
    logger.info("Finished ingestion for %s: %s pages, %s chunks", path.name, len(pages), indexed)

    return {
        "filename": path.name,
        "total_pages": len(pages),
        "total_chunks": indexed,
        "status": "indexed",
    }


def ingest_folder(folder: str | Path, replace_existing: bool = True) -> list[dict]:
    results = []
    for file_path in iter_supported_documents(folder):
        results.append(ingest_document(file_path, replace_existing=replace_existing))
    return results
