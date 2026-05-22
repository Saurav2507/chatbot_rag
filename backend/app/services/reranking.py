from app.core.config import settings
from app.services.preprocessing import lexical_terms


def lexical_score(query: str, text: str) -> float:
    query_terms = lexical_terms(query)
    if not query_terms:
        return 0.0
    text_terms = lexical_terms(text)
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


def lightweight_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """CPU-cheap hybrid rerank: dense score plus small lexical overlap boost."""
    reranked = []
    for chunk in chunks:
        lexical = lexical_score(query, chunk.get("text", ""))
        dense = float(chunk.get("score", 0.0))
        combined = dense + (settings.lexical_boost_weight * lexical)
        reranked.append({**chunk, "dense_score": dense, "lexical_score": lexical, "score": combined})
    return sorted(reranked, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
