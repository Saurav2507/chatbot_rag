import re
import hashlib

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Chunks text into roughly `chunk_size` characters with `overlap`.
    Tries to split at sentence boundaries if possible.
    Optimized: larger chunks = fewer vectors = faster retrieval.
    """
    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Handle the case where a single sentence is larger than chunk_size
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - overlap):
                    part = sentence[i:i + chunk_size].strip()
                    if part:
                        chunks.append(part)
                current_chunk = chunks[-1][-overlap:] if chunks else ""
            else:
                current_chunk = current_chunk[-overlap:] + " " + sentence + " " if current_chunk else sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def _deterministic_id(filename: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic chunk ID based on source metadata.
    This enables deduplication on re-ingestion of the same file."""
    raw = f"{filename}::page{page_number}::chunk{chunk_index}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:32]
    # Format as UUID: 8-4-4-4-12
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def create_chunks_with_metadata(pages_data: list[dict], filename: str) -> list[dict]:
    all_chunks = []
    global_chunk_idx = 0
    
    for page in pages_data:
        text_chunks = chunk_text(page["text"])
        for chunk in text_chunks:
            chunk_id = _deterministic_id(filename, page["page_number"], global_chunk_idx)
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "filename": filename,
                "page_number": page["page_number"],
                "extraction_mode": page["extraction_mode"]
            })
            global_chunk_idx += 1
            
    return all_chunks
