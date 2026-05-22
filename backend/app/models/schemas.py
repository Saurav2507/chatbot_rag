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
    chunk_id: Optional[str] = None
    source_type: Optional[str] = None
    text_snippet: str
    relevance_score: float

class RetrievedChunk(BaseModel):
    filename: str
    page_number: int
    chunk_id: Optional[str] = None
    source_type: Optional[str] = None
    extraction_mode: Optional[str] = None
    text: str
    score: Optional[float] = None
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk] = []
    latency_ms: Dict[str, float]

class IngestResponse(BaseModel):
    status: str
    total_pages: int
    total_chunks: int
    filename: str
