import os
import logging
import time
from functools import lru_cache

from app.core.config import settings
from app.services.preprocessing import transliteration_variants, normalize_sanskrit_text

logger = logging.getLogger(__name__)


class SentenceTransformerModel:
    def __init__(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        self.device = "cpu"
        logger.info(f"Embedding models will use device: {self.device}")

        model_id = settings.embedding_model_id
        logger.info(f"Loading embedding model: {model_id}...")
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. "
                "Install backend dependencies with: pip install -r backend/requirements.txt"
            ) from exc
        self.embed_model = SentenceTransformer(model_id, device=self.device)
        logger.info(f"Embedding model loaded in {time.time() - t0:.1f}s")

    def _query_texts(self, query: str) -> list[str]:
        """Return e5-formatted query strings for all transliteration variants.
        Each variant gets its own 'query: ' prefix so the model sees valid inputs."""
        variants = transliteration_variants(query)
        if not variants:
            variants = [normalize_sanskrit_text(query) or query]
        return [f"query: {v}" for v in variants]

    def _passage_texts(self, text: str) -> list[str]:
        """Return e5-formatted passage strings for all transliteration variants."""
        variants = transliteration_variants(text)
        if not variants:
            variants = [normalize_sanskrit_text(text) or text]
        return [f"passage: {v}" for v in variants]

    @lru_cache(maxsize=256)
    def _cached_query_embedding(self, query: str) -> tuple[float, ...]:
        """Encode all query variants and average the embeddings."""
        texts = self._query_texts(query)
        embeddings = self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            device=self.device,
        )
        # Average across variants then re-normalize for cosine similarity
        import numpy as np
        avg = embeddings.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return tuple(float(x) for x in avg)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        t0 = time.time()
        normalized = normalize_sanskrit_text(query)
        embedding = self._cached_query_embedding(normalized)
        elapsed = (time.time() - t0) * 1000
        logger.info(f"Query embedding: {elapsed:.0f}ms")
        return list(embedding)

    def embed_corpus(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus texts. Each text may already be a pre-built retrieval_text string
        (multi-line variants) from ingestion. We split on newlines and encode each variant
        separately, then average — matching how queries are embedded."""
        import numpy as np
        all_embeddings = []
        for text in texts:
            # Support pre-built multi-variant strings (lines separated by newlines)
            raw_variants = [v.strip() for v in text.split("\n") if v.strip()]
            if not raw_variants:
                raw_variants = [text]
            formatted = [f"passage: {v}" for v in raw_variants]
            embs = self.embed_model.encode(
                formatted,
                normalize_embeddings=True,
                device=self.device,
                show_progress_bar=False,
            )
            avg = embs.mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            all_embeddings.append(avg.tolist())
        return all_embeddings


class LazySentenceTransformerModel:
    def __init__(self):
        self._model: SentenceTransformerModel | None = None

    def _get_model(self) -> SentenceTransformerModel:
        if self._model is None:
            self._model = SentenceTransformerModel()
        return self._model

    def embed_query(self, query: str) -> list[float]:
        return self._get_model().embed_query(query)

    def embed_corpus(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().embed_corpus(texts)


embedding_models = LazySentenceTransformerModel()
