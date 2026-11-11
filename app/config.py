
"""
CampusGuide - Central Application Configuration

Loads application settings from environment variables and .env.
Keeps configuration centralized and type-safe.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# APPLICATION SETTINGS
# ---------------------------------------------------------

class Settings(BaseSettings):
    """Central configuration for the CampusGuide application."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "CampusGuide"
    app_env: Literal["development", "testing", "production"] = "development"
    debug: bool = True

    # API
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # Directories
    data_dir: Path = PROJECT_ROOT / "data"
    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"

    # Vector database
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"
    chroma_collection_name: str = "campusguide_documents"

    # NLP and embedding configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Text chunking
    chunk_size: int = Field(default=800, ge=100, le=5000)
    chunk_overlap: int = Field(default=120, ge=0)

    # Retrieval
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    min_retrieval_similarity: float = Field (
        default=0.35,
        ge=-1.0,
        le=1.0
    )

    # Optional LLM configuration
    llm_provider: Literal["none", "openai"] = "none"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # Upload configuration
    max_upload_size_mb: int = Field(default=25, ge=1, le=100)

    # Validation
    @model_validator(mode="after")
    def validate_chunk_settings(self):
        """Ensure chunk overlap is smaller than chunk size."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        return self

    def create_required_directories(self) -> None:
        """Create application directories when required."""

        directories = [
            self.data_dir,
            self.sample_docs_dir,
            self.evaluation_dir,
            self.chroma_persist_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CACHED SETTINGS INSTANCE
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Reusing the same instance avoids repeatedly parsing
    environment configuration.
    """

    return Settings()


# Convenient settings instance for application modules.
settings = get_settings()
