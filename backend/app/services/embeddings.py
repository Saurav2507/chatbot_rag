import logging
import time
import torch
from functools import lru_cache
from FlagEmbedding import FlagModel, FlagReranker

logger = logging.getLogger(__name__)

class BGEModel:
    def __init__(self):
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"BGE models will use device: {self.device}")
        
        # Load embedding model (always needed)
        logger.info("Loading BGE-M3 embedding model...")
        t0 = time.time()
        self.embed_model = FlagModel(
            'BAAI/bge-m3', 
            use_fp16=True if self.device == "cuda" else False,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: "
        )
        logger.info(f"BGE-M3 loaded in {time.time() - t0:.1f}s")
        
        # Lazy-load reranker (only loaded on first rerank call)
        self._reranker_model = None

    @property
    def reranker_model(self):
        """Lazy-load reranker to speed up server startup."""
        if self._reranker_model is None:
            logger.info("Loading BGE reranker model (first use)...")
            t0 = time.time()
            self._reranker_model = FlagReranker(
                'BAAI/bge-reranker-base',
                use_fp16=True if self.device == "cuda" else False
            )
            logger.info(f"BGE reranker loaded in {time.time() - t0:.1f}s")
        return self._reranker_model

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query. Uses cache for repeated queries."""
        return self._cached_embed_query(query)

    @lru_cache(maxsize=128)
    def _cached_embed_query(self, query: str) -> tuple:
        """Cache query embeddings — same question = instant lookup.
        Returns tuple for hashability, converted back to list by caller."""
        t0 = time.time()
        embedding = self.embed_model.encode_queries([query])[0]
        elapsed = (time.time() - t0) * 1000
        logger.info(f"Query embedding: {elapsed:.0f}ms")
        return tuple(embedding.tolist())

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple queries (for batch use)."""
        embeddings = self.embed_model.encode_queries(queries)
        return embeddings.tolist()
        
    def embed_corpus(self, texts: list[str]) -> list[list[float]]:
        """For corpus/document embedding during ingestion."""
        embeddings = self.embed_model.encode_corpus(texts)
        return embeddings.tolist()

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Rerank candidates based on cross-encoder.
        Optimized: uses lighter bge-reranker-base model.
        """
        if not candidates:
            return []
        
        t0 = time.time()
        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.reranker_model.compute_score(pairs, normalize=True)
        
        # Handle single vs multiple candidate outputs
        if isinstance(scores, float):
            scores = [scores]
            
        # Attach scores to candidates
        for idx, score in enumerate(scores):
            candidates[idx]["relevance_score"] = score
            
        # Sort descending
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        elapsed = (time.time() - t0) * 1000
        logger.info(f"Reranking {len(candidates)} candidates: {elapsed:.0f}ms")
        return candidates

# Singleton pattern for the model loader
bge_models = BGEModel()
