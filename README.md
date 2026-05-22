# Sanskrit Document Retrieval-Augmented Generation System

CPU-only RAG for Sanskrit `.txt`, `.pdf`, and `.docx` documents. The system ingests local documents, normalizes Sanskrit text, supports Devanagari and transliterated queries, retrieves grounded context chunks, and answers with citations using a local open-source GGUF LLM.

## Architecture

```text
Documents -> Loader -> Preprocessor -> Chunker -> Embedder -> Qdrant
Query -> Query Normalizer -> Embedder -> Retriever -> Lightweight Reranker -> Generator -> Answer + Citations
```

The backend is modular:

```text
backend/app/
  api/endpoints.py          FastAPI routes
  core/config.py            environment-based settings
  db/qdrant_client.py       vector store operations
  models/schemas.py         API schemas
  services/loaders.py       PDF/TXT loading
  services/preprocessing.py Unicode, Sanskrit, transliteration normalization
  services/chunking.py      page-aware chunking and metadata
  services/indexing.py      embeddings and Qdrant indexing
  services/retrieval.py     retrieval and grounded QA pipeline
  services/reranking.py     lightweight lexical score fusion
  services/generation.py    CPU llama.cpp generation
frontend/app.py             Streamlit chat and ingestion UI
```

## Model Choices

- Embeddings: `intfloat/multilingual-e5-small`
  - 384-dimensional, multilingual, CPU-friendly, and suitable for Sanskrit plus transliterated queries.
  - Alternative: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Generator: `Qwen2.5-1.5B-Instruct` GGUF via `llama-cpp-python`
  - Runs locally on CPU with `n_gpu_layers=0`.
  - Configure a larger 3B GGUF model if your CPU/RAM budget allows it.
- Vector store: Qdrant
  - Already compatible with the project and simple to run locally in Docker.

No paid APIs are used.

## Retrieval Strategy

- PDF pages, TXT files, and DOCX files are converted to page-level records.
- Text is Unicode-normalized, whitespace-cleaned, and PDF headers/footers repeated across pages are removed.
- Chunks are page-aware, deterministic, and include metadata:
  - filename
  - page number
  - chunk id
  - source type
  - extraction mode
  - character offsets when available
- Embedding text includes conservative Devanagari/transliteration variants for better cross-script retrieval.
- Dense Qdrant retrieval is followed by a small lexical overlap boost. Heavy rerankers are intentionally avoided for CPU latency.

## Prompt Strategy

The generator receives numbered retrieved chunks with source labels and is instructed to:

- answer only from retrieved context
- say when the documents do not contain enough information
- cite factual claims as `[filename, page X]`
- keep the answer concise

## Setup

Prerequisites:

- Python 3.10+
- Docker and Docker Compose for Qdrant
- Optional: Tesseract OCR if you enable scanned-PDF OCR

Start Qdrant:

```bash
docker-compose up -d
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

Run the backend:

```bash
python backend/main.py
```

Run the frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

Open Streamlit, upload `.pdf`, `.txt`, or `.docx` Sanskrit documents, ingest them, and ask questions.

## Configuration

Environment variables:

```text
DATA_DIR=backend/data
COLLECTION_NAME=sanskrit_documents
EMBEDDING_MODEL_ID=intfloat/multilingual-e5-small
EMBEDDING_DIMENSION=384
RETRIEVAL_TOP_K=5
RETRIEVAL_CANDIDATE_K=12
RETRIEVAL_SCORE_THRESHOLD=0.25
CHUNK_SIZE_TOKENS=500
CHUNK_OVERLAP_TOKENS=80
LLM_REPO_ID=bartowski/Qwen2.5-1.5B-Instruct-GGUF
LLM_FILENAME=Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
LLM_N_CTX=2048
LLM_N_THREADS=0
LLM_MAX_TOKENS=512
ENABLE_OCR=false
```

The first backend answer downloads the GGUF model if it is not already cached. After that, generation runs locally on CPU.

If you change embedding models to one with a different vector size, use a new `COLLECTION_NAME` or recreate the Qdrant collection.

## API

```bash
POST /api/upload
POST /api/ingest_folder
POST /api/ingest_now
POST /api/retrieve
POST /api/chat
POST /api/chat/stream
GET  /api/status
```

Example retrieval request:

```bash
curl -X POST http://localhost:8000/api/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"धर्मः किम्?\",\"top_k\":5}"
```

## Sample Sanskrit QA Questions

- `धर्मः किम्?`
- `कर्मयोगस्य स्वरूपं किम्?`
- `ātman iti kim ucyate?`
- `mokṣasya lakṣaṇam kim?`
- `veda-granthe agneḥ varṇanam kutra asti?`

Answers include citations such as `[filename.pdf, page 12]` when the retrieved context supports the response.
