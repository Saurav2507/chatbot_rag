import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("COLLECTION_NAME", "sanskrit_documents")
    embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-small")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    retrieval_candidate_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "12"))
    retrieval_score_threshold: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.25"))
    lexical_boost_weight: float = float(os.getenv("LEXICAL_BOOST_WEIGHT", "0.08"))

    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))
    max_context_chars_per_chunk: int = int(os.getenv("MAX_CONTEXT_CHARS_PER_CHUNK", "900"))

    llm_repo_id: str = os.getenv("LLM_REPO_ID", "bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    llm_filename: str = os.getenv("LLM_FILENAME", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
    llm_n_ctx: int = int(os.getenv("LLM_N_CTX", "2048"))
    llm_n_threads: int = int(os.getenv("LLM_N_THREADS", "0")) or (os.cpu_count() or 4)
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

    enable_ocr: bool = _bool_env("ENABLE_OCR", False)
    ocr_min_chars: int = int(os.getenv("OCR_MIN_CHARS", "50"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
