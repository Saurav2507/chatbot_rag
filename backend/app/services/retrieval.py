import json
import logging
import time

from app.core.config import settings
from app.db.qdrant_client import search_dense
from app.services.embeddings import embedding_models
from app.services.generation import get_llm_generator
from app.services.preprocessing import normalize_sanskrit_text
from app.services.reranking import lightweight_rerank

logger = logging.getLogger(__name__)

SYSTEM_MSG = (
    "You are a Sanskrit document question-answering assistant. "
    "Use only the provided context chunks. If the answer is not supported by the context, "
    "say that the available documents do not contain enough information. "
    "Do not invent facts. Cite evidence as [filename, page X]."
)


def _chunk_label(chunk: dict) -> str:
    return f"{chunk.get('filename', 'unknown')}, page {chunk.get('page_number', '?')}"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token on average for mixed Sanskrit/English."""
    return max(1, len(text) // 4)


def _build_prompt(query: str, top_chunks: list[dict]) -> str:
    """Build a context prompt that fits within the LLM context window.

    Reserves space for:
      - System message overhead
      - The question + fixed prompt text
      - Output tokens (llm_max_tokens)
    Progressively trims or drops chunks to ensure the total stays safe.
    """
    # Budget: context_window - output_tokens - system/question overhead
    SYSTEM_OVERHEAD_TOKENS = _estimate_tokens(SYSTEM_MSG) + 64  # role tokens + safety margin
    QUESTION_OVERHEAD = _estimate_tokens(query) + 32
    max_prompt_tokens = (
        settings.llm_n_ctx
        - settings.llm_max_tokens
        - SYSTEM_OVERHEAD_TOKENS
        - QUESTION_OVERHEAD
        - 64  # extra buffer
    )
    max_prompt_tokens = max(200, max_prompt_tokens)  # floor so we always try something

    # Per-chunk cap (from settings, in chars)
    char_cap = settings.max_context_chars_per_chunk

    # Build parts greedily, dropping chunks that don't fit
    used_tokens = 0
    context_parts = []
    for index, chunk in enumerate(top_chunks, start=1):
        text = chunk["text"][:char_cap]
        part = f"[{index}] Source: {_chunk_label(chunk)}\n{text}"
        part_tokens = _estimate_tokens(part)
        if used_tokens + part_tokens > max_prompt_tokens:
            # Try trimming this chunk to whatever space is left
            remaining_chars = max(0, (max_prompt_tokens - used_tokens) * 4 - 60)
            if remaining_chars > 100:
                text = chunk["text"][:remaining_chars]
                part = f"[{index}] Source: {_chunk_label(chunk)}\n{text}"
                context_parts.append(part)
            break
        context_parts.append(part)
        used_tokens += part_tokens

    if not context_parts:
        # Last resort: just use first 200 chars of top chunk
        text = top_chunks[0]["text"][:800]
        context_parts.append(f"[1] Source: {_chunk_label(top_chunks[0])}\n{text}")

    context_str = "\n\n".join(context_parts)
    prompt = (
        "Context chunks:\n"
        f"{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely. Include citations in the form [filename, page X]."
    )
    logger.debug("Prompt token estimate: ~%d / %d budget", _estimate_tokens(prompt), max_prompt_tokens)
    return prompt


def _citation(chunk: dict) -> dict:
    snippet = chunk.get("text", "")[:240]
    if len(chunk.get("text", "")) > 240:
        snippet += "..."
    return {
        "filename": chunk.get("filename", ""),
        "page_number": chunk.get("page_number", 1),
        "chunk_id": chunk.get("chunk_id"),
        "source_type": chunk.get("source_type"),
        "text_snippet": snippet,
        "relevance_score": float(chunk.get("score", 0.0)),
    }


def retrieve_chunks(query: str, top_k: int | None = None) -> dict:
    retrieval_top_k = top_k or settings.retrieval_top_k
    candidate_k = max(retrieval_top_k, settings.retrieval_candidate_k)
    timings = {}

    # embed_query handles normalization and transliteration variants internally
    t0 = time.time()
    query_embedding = embedding_models.embed_query(query)
    timings["embedding"] = (time.time() - t0) * 1000

    t0 = time.time()
    candidates = search_dense(query_embedding, top_k=candidate_k)
    timings["retrieval"] = (time.time() - t0) * 1000

    t0 = time.time()
    # Use the original query for lexical reranking so terms are preserved
    chunks = lightweight_rerank(normalize_sanskrit_text(query), candidates, retrieval_top_k)
    timings["reranking"] = (time.time() - t0) * 1000

    return {"chunks": chunks, "latency_ms": timings}


def retrieve_and_generate(query: str, top_k: int | None = None) -> dict:
    start_time = time.time()
    retrieval = retrieve_chunks(query, top_k)
    top_chunks = retrieval["chunks"]
    timings = retrieval["latency_ms"]

    if not top_chunks:
        timings["generation"] = 0
        timings["total"] = (time.time() - start_time) * 1000
        return {
            "answer": "The available documents do not contain enough information to answer this question.",
            "citations": [],
            "retrieved_chunks": [],
            "latency_ms": timings,
        }

    prompt = _build_prompt(query, top_chunks)
    t0 = time.time()
    answer = get_llm_generator().generate(prompt, SYSTEM_MSG)
    timings["generation"] = (time.time() - t0) * 1000
    timings["total"] = (time.time() - start_time) * 1000

    logger.info(
        "Query latency: embed=%.0fms, search=%.0fms, rerank=%.0fms, gen=%.0fms, total=%.0fms",
        timings.get("embedding", 0),
        timings.get("retrieval", 0),
        timings.get("reranking", 0),
        timings.get("generation", 0),
        timings.get("total", 0),
    )

    citations = [_citation(chunk) for chunk in top_chunks]
    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": top_chunks,
        "latency_ms": timings,
    }


def retrieve_and_generate_stream(query: str, top_k: int | None = None):
    start_time = time.time()
    retrieval = retrieve_chunks(query, top_k)
    top_chunks = retrieval["chunks"]
    timings = retrieval["latency_ms"]

    if not top_chunks:
        yield json.dumps(
            {
                "type": "token",
                "data": "The available documents do not contain enough information to answer this question.",
            }
        )
        timings["generation"] = 0
        timings["total"] = (time.time() - start_time) * 1000
        yield json.dumps({"type": "done", "data": {"latency_ms": timings}})
        return

    citations = [_citation(chunk) for chunk in top_chunks]
    yield json.dumps({"type": "citations", "data": citations})
    yield json.dumps({"type": "chunks", "data": top_chunks})

    prompt = _build_prompt(query, top_chunks)
    gen_start = time.time()
    for token in get_llm_generator().generate_stream(prompt, SYSTEM_MSG):
        yield json.dumps({"type": "token", "data": token})

    timings["generation"] = (time.time() - gen_start) * 1000
    timings["total"] = (time.time() - start_time) * 1000
    yield json.dumps({"type": "done", "data": {"latency_ms": timings}})
