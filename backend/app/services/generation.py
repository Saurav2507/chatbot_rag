import os
import logging
import time
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self):
        # Configurable via .env — defaults to smaller, faster 3B model
        repo_id = os.getenv("LLM_REPO_ID", "bartowski/Qwen2.5-3B-Instruct-GGUF")
        filename = os.getenv("LLM_FILENAME", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
        n_ctx = int(os.getenv("LLM_N_CTX", "2048"))
        n_threads = int(os.getenv("LLM_N_THREADS", "0")) or os.cpu_count()
        
        logger.info(f"Loading LLM: {repo_id} / {filename}")
        logger.info(f"  n_ctx={n_ctx}, n_threads={n_threads}")
        
        try:
            t0 = time.time()
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.info(f"Model downloaded/cached in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise e

        # Initialize Llama.cpp with optimized settings
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        logger.info("LLM loaded and ready.")

    def generate(self, prompt: str, system_message: str) -> str:
        """Synchronous generation — returns full response."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        t0 = time.time()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,
            top_p=0.9
        )
        elapsed = (time.time() - t0) * 1000
        logger.info(f"LLM generation: {elapsed:.0f}ms")
        return response["choices"][0]["message"]["content"]

    def generate_stream(self, prompt: str, system_message: str):
        """Streaming generation — yields token strings as they are produced.
        This allows the frontend to display tokens in real-time."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,
            top_p=0.9,
            stream=True
        )
        
        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

# Singleton
llm_generator = LLMGenerator()
