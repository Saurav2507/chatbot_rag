import logging
import os
import time

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        repo_id = settings.llm_repo_id
        filename = settings.llm_filename

        logger.info("Loading CPU LLM: %s / %s", repo_id, filename)
        logger.info("LLM runtime: n_ctx=%s, n_threads=%s, n_gpu_layers=0", settings.llm_n_ctx, settings.llm_n_threads)

        try:
            from huggingface_hub import hf_hub_download

            t0 = time.time()
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.info("Model downloaded/cached in %.1fs", time.time() - t0)
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required to load the GGUF model. "
                "Install backend dependencies with: pip install -r backend/requirements.txt"
            ) from exc
        except Exception as exc:
            logger.error("Failed to download/load model metadata: %s", exc)
            raise

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for CPU generation. "
                "Install backend dependencies with: pip install -r backend/requirements.txt"
            ) from exc

        self.llm = Llama(
            model_path=model_path,
            n_ctx=settings.llm_n_ctx,
            n_threads=settings.llm_n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        self.max_tokens = settings.llm_max_tokens
        logger.info("LLM loaded and ready.")

    def generate(self, prompt: str, system_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        t0 = time.time()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,
            top_p=0.9,
        )
        logger.info("LLM generation: %.0fms", (time.time() - t0) * 1000)
        return response["choices"][0]["message"]["content"].strip()

    def generate_stream(self, prompt: str, system_message: str):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,
            top_p=0.9,
            stream=True,
        )

        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token


_llm_generator: LLMGenerator | None = None


def get_llm_generator() -> LLMGenerator:
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMGenerator()
    return _llm_generator
