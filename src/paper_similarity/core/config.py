"""Application configuration.

All settings come from environment variables prefixed with ``PULSE_`` (or a local ``.env``
file). Nothing is hard-coded in the modules that use them: they receive a ``Settings``
instance, which keeps configuration explicit and easy to override in tests.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PULSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Persistence
    database_url: str = "postgresql+asyncpg://pulse:pulse@localhost:5432/pulse"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "arxiv_papers"
    qdrant_timeout_seconds: int = 10

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = None

    # arXiv client
    arxiv_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_page_size: int = Field(default=50, ge=1, le=2000)
    arxiv_request_delay_seconds: float = Field(default=3.0, ge=0)
    arxiv_timeout_seconds: float = Field(default=30.0, gt=0)
    arxiv_max_retries: int = Field(default=4, ge=1)

    # Raw landing zone (Parquet)
    raw_data_dir: Path = Path("data/raw/arxiv")

    # Submissions / similarity
    max_submission_chars: int = Field(default=100_000, ge=1)
    # Capped at the embedding model's own input limit at startup.
    chunk_max_tokens: int = Field(default=512, ge=16)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    @field_validator("embedding_cache_dir", mode="before")
    @classmethod
    def _empty_means_default(cls, value: object) -> object:
        # `PULSE_EMBEDDING_CACHE_DIR=` (as in .env.example) means "library default". Without
        # this it would become Path("."), downloading the model into the working directory.
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        # Each window holds chunk_max_tokens - 2 content tokens ([CLS] and [SEP] are added).
        if self.chunk_overlap_tokens >= self.chunk_max_tokens - 2:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_max_tokens - 2")
        return self


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings, read once from the environment."""
    return Settings()
