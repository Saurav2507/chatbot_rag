from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChunkMetadata(BaseModel):
    filename: str
    page_number: int
    chunk_id: str
    extraction_mode: str
    text: str

class Citation(BaseModel):
    filename: str
    page_number: int
    text_snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: Dict[str, float]

class IngestResponse(BaseModel):
    status: str
    total_pages: int
    total_chunks: int
    filename: str
