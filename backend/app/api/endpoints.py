import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.models.schemas import IngestResponse, QueryRequest, QueryResponse
from app.services.ingestion import ingest_document, ingest_folder as ingest_folder_service
from app.services.loaders import SUPPORTED_EXTENSIONS, iter_supported_documents
from app.services.retrieval import retrieve_and_generate, retrieve_and_generate_stream, retrieve_chunks

logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_DIR = settings.data_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def background_ingest(file_path: str, filename: str):
    try:
        ingest_document(file_path, replace_existing=True)
    except Exception as exc:
        logger.exception("Failed ingestion for %s: %s", filename, exc)


@router.post("/upload", response_model=IngestResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and DOCX files are supported.")

    file_path = UPLOAD_DIR / filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(background_ingest, str(file_path), filename)
    return IngestResponse(status="processing", total_pages=0, total_chunks=0, filename=filename)


@router.post("/ingest_folder")
async def ingest_folder(background_tasks: BackgroundTasks):
    files = iter_supported_documents(UPLOAD_DIR)
    if not files:
        return {"message": f"No .pdf, .txt, or .docx documents found in {UPLOAD_DIR}."}

    for file_path in files:
        background_tasks.add_task(background_ingest, str(file_path), file_path.name)

    return {"message": f"Started background ingestion for {len(files)} document(s) from {UPLOAD_DIR}."}


@router.post("/ingest_now")
def ingest_now():
    """Synchronous ingestion, useful for scripts and tests."""
    results = ingest_folder_service(UPLOAD_DIR, replace_existing=True)
    return {"results": results}


@router.post("/retrieve")
def retrieve(request: QueryRequest):
    return retrieve_chunks(request.query, request.top_k)


@router.post("/chat")
def chat(request: QueryRequest):
    try:
        res = retrieve_and_generate(request.query, request.top_k)
        QueryResponse(**res) # Test validation
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
def chat_stream(request: QueryRequest):
    def event_generator():
        for event_data in retrieve_and_generate_stream(request.query, request.top_k):
            yield {"data": event_data}

    return EventSourceResponse(event_generator())


@router.get("/status")
async def status():
    return {
        "status": "online",
        "data_dir": str(UPLOAD_DIR),
        "collection": settings.collection_name,
        "embedding_model": settings.embedding_model_id,
        "llm": f"{settings.llm_repo_id}/{settings.llm_filename}",
        "device": "cpu",
    }
