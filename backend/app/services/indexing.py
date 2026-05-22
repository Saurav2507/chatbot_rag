import logging

from app.db.qdrant_client import delete_source, insert_chunks
from app.services.embeddings import embedding_models

logger = logging.getLogger(__name__)


def index_chunks(chunks: list[dict], filename: str, replace_existing: bool = True) -> int:
    """Embed and write chunks to Qdrant in deterministic batches."""
    if not chunks:
        return 0

    if replace_existing:
        delete_source(filename)

    batch_size = 32
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.get("embedding_text") or chunk["text"] for chunk in batch]
        embeddings = embedding_models.embed_corpus(texts)
        insert_chunks(batch, embeddings)
        total += len(batch)
        logger.info("Indexed batch %s (%s chunks)", i // batch_size + 1, len(batch))
    return total
