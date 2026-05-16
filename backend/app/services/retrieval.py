import os
import time
import json
import logging
from functools import lru_cache
from app.services.embeddings import bge_models
from app.db.qdrant_client import search_dense
from app.services.generation import llm_generator

logger = logging.getLogger(__name__)

# Configurable defaults
DEFAULT_RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
MAX_CHUNK_CHARS = 500  # Truncate chunk text sent to LLM to cap context size

SYSTEM_MSG = (
    "You are a reliable document analyst. "
    "Answer the user's question using ONLY the provided context chunks. "
    "If the answer is not contained in the context, say so clearly instead of guessing. "
    "Every factual claim must include a citation in the exact format: [filename, page X]."
)

def _build_prompt(query: str, top_chunks: list[dict]) -> str:
    """Build a concise prompt with truncated chunks."""
    context_parts = []
    for i, c in enumerate(top_chunks):
        text = c["text"][:MAX_CHUNK_CHARS]
        context_parts.append(
            f"[{i+1}] {c['filename']} p.{c['page_number']}: {text}"
        )
    context_str = "\n\n".join(context_parts)
    return f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer concisely with citations:"


def _retrieve_chunks(query: str, retrieval_top_k: int, rerank_top_k: int) -> tuple:
    """Core retrieval logic: embed → search → rerank. Returns (top_chunks, timing_dict)."""
    timings = {}

    # 1. Embed Query (cached for repeated queries)
    t0 = time.time()
    query_embedding = list(bge_models.embed_query(query))
    timings["embedding"] = (time.time() - t0) * 1000

    # 2. Retrieve candidates from Qdrant
    t0 = time.time()
    candidates = search_dense(query_embedding, top_k=retrieval_top_k)
    timings["retrieval"] = (time.time() - t0) * 1000

    if not candidates:
        logger.warning("No candidates returned from Qdrant.")
        return [], timings

    # 3. Rerank candidates
    t0 = time.time()
    reranked = bge_models.rerank(query, candidates)
    top_chunks = reranked[:rerank_top_k]
    timings["reranking"] = (time.time() - t0) * 1000

    return top_chunks, timings


def retrieve_and_generate(query: str, top_k: int = None) -> dict:
    """Full pipeline: retrieve + generate (synchronous)."""
    rerank_top_k = top_k or DEFAULT_RERANK_TOP_K
    start_time = time.time()
    
    top_chunks, timings = _retrieve_chunks(query, DEFAULT_RETRIEVAL_TOP_K, rerank_top_k)

    if not top_chunks:
        return {
            "answer": "I couldn't find relevant information in the documents to answer your question.",
            "citations": [],
            "latency_ms": {**timings, "generation": 0, "total": (time.time() - start_time) * 1000}
        }

    # 4. Generate Answer
    prompt = _build_prompt(query, top_chunks)
    t0 = time.time()
    answer = llm_generator.generate(prompt, SYSTEM_MSG)
    timings["generation"] = (time.time() - t0) * 1000
    timings["total"] = (time.time() - start_time) * 1000
    
    # Log all timings
    logger.info(
        f"Query latency breakdown: "
        f"embed={timings['embedding']:.0f}ms, "
        f"search={timings['retrieval']:.0f}ms, "
        f"rerank={timings['reranking']:.0f}ms, "
        f"gen={timings['generation']:.0f}ms, "
        f"total={timings['total']:.0f}ms"
    )
    
    citations = [
        {
            "filename": c["filename"],
            "page_number": c["page_number"],
            "text_snippet": c["text"][:200] + "...",
            "relevance_score": c.get("relevance_score", c.get("score", 0.0))
        } for c in top_chunks
    ]
    
    return {
        "answer": answer,
        "citations": citations,
        "latency_ms": timings
    }


def retrieve_and_generate_stream(query: str, top_k: int = None):
    """Streaming variant: yields SSE-formatted events.
    
    Events:
      - {"type": "citations", "data": [...]}   — sent first with source info
      - {"type": "token", "data": "..."}       — each generated token
      - {"type": "done", "data": {...}}        — final latency info
    """
    rerank_top_k = top_k or DEFAULT_RERANK_TOP_K
    start_time = time.time()
    
    # Retrieve (non-streaming part)
    top_chunks, timings = _retrieve_chunks(query, DEFAULT_RETRIEVAL_TOP_K, rerank_top_k)

    if not top_chunks:
        yield json.dumps({
            "type": "token",
            "data": "I couldn't find relevant information in the documents to answer your question."
        })
        yield json.dumps({
            "type": "done",
            "data": {"latency_ms": {**timings, "generation": 0, "total": (time.time() - start_time) * 1000}}
        })
        return

    # Send citations first so frontend can display sources immediately
    citations = [
        {
            "filename": c["filename"],
            "page_number": c["page_number"],
            "text_snippet": c["text"][:200] + "...",
            "relevance_score": c.get("relevance_score", c.get("score", 0.0))
        } for c in top_chunks
    ]
    yield json.dumps({"type": "citations", "data": citations})

    # Stream generation tokens
    prompt = _build_prompt(query, top_chunks)
    gen_start = time.time()
    
    for token in llm_generator.generate_stream(prompt, SYSTEM_MSG):
        yield json.dumps({"type": "token", "data": token})
    
    timings["generation"] = (time.time() - gen_start) * 1000
    timings["total"] = (time.time() - start_time) * 1000
    
    logger.info(
        f"[Stream] Query latency: "
        f"embed={timings['embedding']:.0f}ms, "
        f"search={timings['retrieval']:.0f}ms, "
        f"rerank={timings['reranking']:.0f}ms, "
        f"gen={timings['generation']:.0f}ms, "
        f"total={timings['total']:.0f}ms"
    )
    
    yield json.dumps({"type": "done", "data": {"latency_ms": timings}})
