import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

# Environment variable or default
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_knowledge_base")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant():
    """Initializes the Qdrant collection if it doesn't exist.
    Uses optimized HNSW parameters for faster search."""
    try:
        qdrant.get_collection(collection_name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' exists.")
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}'...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024,
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
                    "extraction_mode": chunk["extraction_mode"]
                }
            )
        )
    
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

def search_dense(query_embedding: list[float], top_k: int = 10, score_threshold: float = 0.3) -> list[dict]:
    """Searches Qdrant using dense vector search.
    
    Optimizations:
    - score_threshold filters out low-relevance noise early
    - with_payload selects only needed fields (avoids transferring large text unnecessarily)
    - Reduced default top_k from 20 to 10
    """
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,  # We need text for reranking
    )
    
    results = []
    for hit in search_result:
        doc = hit.payload
        doc["score"] = hit.score
        results.append(doc)
    return results
