# Private PDF Knowledge Base RAG Chatbot

This project implements a production-grade Retrieval-Augmented Generation (RAG) chatbot designed to run efficiently on standard hardware (16GB RAM).

## Features
- **Native PDF Extraction + OCR**: Uses PyMuPDF for fast extraction, falling back to Tesseract OCR for scanned pages.
- **Hybrid Search**: Embeddings via BAAI/bge-m3 with Qdrant vector database.
- **Reranking**: BAAI/bge-reranker-v2-m3 for highly accurate top-K retrieval.
- **Local LLM Generation**: Qwen2.5-7B-Instruct running via `llama-cpp-python` (4-bit quantized) to fit in standard RAM without external APIs.
- **Streamlit Frontend**: Clean UI with chat, citations, snippet previews, and latency tracking.

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for Qdrant)
- Tesseract OCR installed on your system.
  - Windows: `winget install UB-Mannheim.TesseractOCR`
  - Linux: `sudo apt-get install tesseract-ocr`

### 1. Start Vector Database (Qdrant)
```bash
docker-compose up -d
```

### 2. Install Dependencies
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend requirements
pip install -r backend/requirements.txt

# Install frontend requirements
pip install -r frontend/requirements.txt
```

### 3. Run the Backend (FastAPI)
The first run will automatically download the LLM and Embedding models (~6GB total). Make sure you have a stable internet connection.
```bash
python backend/main.py
```

### 4. Run the Frontend (Streamlit)
In a new terminal (with the virtual environment activated):
```bash
streamlit run frontend/app.py
```

## Usage
1. Open the Streamlit interface (usually http://localhost:8501).
2. Use the sidebar to upload a PDF. Click **Ingest**.
3. Once ingested, ask questions in the chat.
4. Click on "Sources Used" under each answer to see exact page citations, scores, and text snippets.

## Evaluation
Run the evaluation harness to test Recall@5 and Latency:
```bash
python evaluation/run_eval.py
```
