import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings

logger = logging.getLogger(__name__)

QDRANT_HOST = settings.qdrant_host
QDRANT_PORT = settings.qdrant_port
COLLECTION_NAME = settings.collection_name

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant():
    """Initializes the Qdrant collection if it doesn't exist.
    Uses optimized HNSW parameters for faster search."""
    try:
        collection = qdrant.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}'...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
            # Optimized HNSW index for faster search with good recall
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            # Optimize for faster search at the cost of slightly more memory
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10000,
            ),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created with optimized HNSW.")
        return

    logger.info(f"Collection '{COLLECTION_NAME}' exists.")
    vector_config = collection.config.params.vectors
    if hasattr(vector_config, "size") and vector_config.size != settings.embedding_dimension:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' has vector size {vector_config.size}, "
            f"but EMBEDDING_DIMENSION is {settings.embedding_dimension}. "
            "Use a new COLLECTION_NAME or recreate the collection."
        )

def insert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """Inserts chunks with their embeddings into Qdrant."""
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk.get("chunk_index"),
                    "source_type": chunk.get("source_type", "pdf"),
                    "extraction_mode": chunk.get("extraction_mode", "native"),
                    "start_char": chunk.get("start_char"),
                    "end_char": chunk.get("end_char"),
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


def delete_source(filename: str) -> None:
    """Remove existing chunks for a document before deterministic re-indexing."""
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    )
                ]
            )
        ),
        wait=True,
    )


def search_dense(query_embedding: list[float], top_k: int = 10, score_threshold: float | None = None) -> list[dict]:
    """Searches Qdrant using dense vector search.
    Falls back to no score threshold if the filtered search returns zero results.
    """
    score_threshold = settings.retrieval_score_threshold if score_threshold is None else score_threshold
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    # Fallback: if score threshold filtered everything out, retry without threshold
    if not search_result:
        logger.warning(
            "No results above score_threshold=%.2f — retrying without threshold", score_threshold
        )
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=None,
            with_payload=True,
        )

    results = []
    for hit in search_result:
        doc = dict(hit.payload)
        doc["score"] = hit.score
        results.append(doc)
    return results
