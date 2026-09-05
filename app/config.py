"""Centralised configuration.

Single source of truth for every tunable knob in the project. The class is a
`pydantic-settings` `BaseSettings`, which means values are resolved with the
following priority chain (highest first):

    1. Explicit constructor kwargs         -> Settings(top_k=10)
    2. Environment variables               -> TOP_K=10 python app.py
    3. Variables in the .env file          -> TOP_K=10 in ./.env
    4. Class defaults                      -> top_k: int = 5

`extra="ignore"` lets us evolve the .env file (e.g. adding new tuning keys)
without breaking older deployments that haven't pruned their config yet.

`get_settings()` is wrapped in `lru_cache` so the `Settings` instance is built
once per process. FastAPI dependency-injection (`Depends(get_settings)`)
becomes a cheap no-op after the first call.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Storage paths ----------------------------------------------------
    data_dir: str = "data/raw"          # source corpus (.txt / .md / .docx / .pdf)
    chroma_dir: str = "data/chroma"     # chromadb persistent dir
    sqlite_path: str = "logs/app.db"    # request / latency / token log

    # ---- Embeddings -------------------------------------------------------
    embedding_model: str = "BAAI/bge-m3"

    # ---- Retrieval --------------------------------------------------------
    top_k: int = 5
    retrieval_distance_threshold: float = 0.65
    # Cosine distance threshold: results strictly above this are discarded.
    # 0.65 corresponds to cosine similarity ≈ 0.35, which on bge-m3's score
    # distribution empirically separates "in-corpus" from "out-of-domain".
    # Tune via EVAL on a labelled query set.

    reranker_enabled: bool = False
    # When True, the retriever reranks the top_k × 4 candidates with
    # a cross-encoder. Disabled by default because the reranker adds ~200ms
    # per request and is only worthwhile when retrieval precision is the
    # dominant bottleneck.

    # ---- Generation -------------------------------------------------------
    temperature: float = 0.1
    llm_base_url: str | None = None     # OpenAI-compatible endpoint
    llm_api_key: str | None = None
    llm_model: str = "qwen2.5:7b"
    # Local Ollama default; DeepSeek for evaluation (just swap .env values).

    # ---- Chunking ---------------------------------------------------------
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 80

    # ---- App --------------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
