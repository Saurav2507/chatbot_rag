import hashlib
import re

from app.core.config import settings
from app.services.preprocessing import normalize_sanskrit_text, retrieval_text


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\u0964\u0965.!?])\s+")


def _token_count(text: str) -> int:
    pieces = text.split()
    if pieces:
        return len(pieces)
    return max(1, len(text) // 4)


def _tail_by_tokens(text: str, overlap: int) -> str:
    pieces = text.split()
    if pieces:
        return " ".join(pieces[-overlap:])
    return text[-overlap * 4 :]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """
    Chunk text into roughly 300-800 token windows with overlap.
    The returned offsets are character offsets inside the page text.
    """
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = overlap if overlap is not None else settings.chunk_overlap_tokens
    text = normalize_sanskrit_text(text)
    if not text or not text.strip():
        return []

    sentences = SENTENCE_BOUNDARY_RE.split(text)
    chunks = []
    current_chunk = ""
    current_start = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_start = text.find(sentence, current_start)
        if sentence_start == -1:
            sentence_start = current_start

        candidate = f"{current_chunk} {sentence}".strip()
        if _token_count(candidate) <= chunk_size:
            if not current_chunk:
                current_start = sentence_start
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "start_char": current_start,
                        "end_char": current_start + len(current_chunk),
                    }
                )

            if _token_count(sentence) > chunk_size:
                words = sentence.split()
                if words:
                    step = max(1, chunk_size - overlap)
                    for i in range(0, len(words), step):
                        part = " ".join(words[i : i + chunk_size]).strip()
                        if part:
                            part_start = text.find(part, sentence_start)
                            chunks.append(
                                {
                                    "text": part,
                                    "start_char": part_start if part_start >= 0 else sentence_start,
                                    "end_char": (part_start if part_start >= 0 else sentence_start) + len(part),
                                }
                            )
                    current_chunk = _tail_by_tokens(sentence, overlap)
                    current_start = max(sentence_start, text.find(current_chunk, sentence_start))
                else:
                    step = max(1, (chunk_size - overlap) * 4)
                    size = chunk_size * 4
                    for i in range(0, len(sentence), step):
                        part = sentence[i : i + size].strip()
                        part_start = sentence_start + i
                        if part:
                            chunks.append({"text": part, "start_char": part_start, "end_char": part_start + len(part)})
                    current_chunk = sentence[-overlap * 4 :]
                    current_start = sentence_start + max(0, len(sentence) - len(current_chunk))
            else:
                tail = _tail_by_tokens(current_chunk, overlap)
                current_chunk = f"{tail} {sentence}".strip() if tail else sentence
                current_start = text.find(current_chunk, max(0, sentence_start - len(tail) - 2))
                if current_start == -1:
                    current_start = sentence_start

    if current_chunk.strip():
        chunks.append(
            {
                "text": current_chunk.strip(),
                "start_char": current_start,
                "end_char": current_start + len(current_chunk),
            }
        )

    deduped = []
    seen = set()
    for chunk in chunks:
        normalized = normalize_sanskrit_text(chunk["text"])
        if normalized and normalized not in seen:
            deduped.append({**chunk, "text": normalized})
            seen.add(normalized)
    return deduped


def _legacy_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Compatibility helper for old callers that expected text-only chunks."""
    chunks = chunk_text(text, max(1, chunk_size // 4), max(1, overlap // 4))
    return [chunk["text"] for chunk in chunks]


def _deterministic_id(filename: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic chunk ID based on source metadata."""
    raw = f"{filename}::page{page_number}::chunk{chunk_index}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def create_chunks_with_metadata(pages_data: list[dict], filename: str) -> list[dict]:
    all_chunks = []
    global_chunk_idx = 0

    for page in pages_data:
        text_chunks = chunk_text(page["text"])
        for page_chunk_idx, chunk in enumerate(text_chunks):
            chunk_id = _deterministic_id(filename, page["page_number"], global_chunk_idx)
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_idx,
                    "page_chunk_index": page_chunk_idx,
                    "text": chunk["text"],
                    "embedding_text": retrieval_text(chunk["text"]),
                    "filename": filename,
                    "page_number": page["page_number"],
                    "source_type": page.get("source_type", "pdf"),
                    "extraction_mode": page.get("extraction_mode", "native"),
                    "start_char": chunk.get("start_char"),
                    "end_char": chunk.get("end_char"),
                }
            )
            global_chunk_idx += 1

    return all_chunks
