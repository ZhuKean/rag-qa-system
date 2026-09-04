from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    data_dir:str = "data"
    chroma_dir:str = "data/chroma"
    embedding_model:str = "BAAI/bge-m3"
    top_k:int = 5
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model:str = "qwen2.5:7b"

from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()