import os
import logging
import shutil
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from typing import List
from app.models.schemas import QueryRequest, QueryResponse, IngestResponse
from app.services.ingestion import process_pdf
from app.services.chunking import create_chunks_with_metadata
from app.services.embeddings import bge_models
from app.db.qdrant_client import insert_chunks
from app.services.retrieval import retrieve_and_generate, retrieve_and_generate_stream

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def background_ingest(file_path: str, filename: str):
    logger.info(f"Starting ingestion for {filename}")
    # 1. Extract text
    pages_data = process_pdf(file_path)
    
    # 2. Chunk text
    chunks = create_chunks_with_metadata(pages_data, filename)
    
    # 3. Embed chunks in batches to avoid OOM
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = bge_models.embed_corpus(texts)
        # 4. Insert into Qdrant
        insert_chunks(batch, embeddings)
        logger.info(f"  Ingested batch {i//batch_size + 1} ({len(batch)} chunks)")
        
    logger.info(f"Finished ingestion for {filename}. Inserted {len(chunks)} chunks.")

@router.post("/upload", response_model=IngestResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Kick off ingestion in background
    background_tasks.add_task(background_ingest, file_path, file.filename)
    
    return IngestResponse(
        status="processing",
        total_pages=0,
        total_chunks=0,
        filename=file.filename
    )

@router.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Synchronous chat — returns full response at once."""
    result = retrieve_and_generate(request.query, request.top_k)
    return QueryResponse(**result)

@router.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """Streaming chat via SSE — tokens arrive in real-time.
    
    Events:
      - type=citations: source documents used
      - type=token: each generated token
      - type=done: final latency metrics
    """
    async def event_generator():
        for event_data in retrieve_and_generate_stream(request.query, request.top_k):
            yield {"data": event_data}
    
    return EventSourceResponse(event_generator())

@router.get("/status")
async def status():
    return {"status": "online"}
