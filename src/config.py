"""Runtime configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_OPENAI_BASE_URL: str | None = None
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_RETRIEVAL_TOP_K = 4
# Retrieval honesty (P0-1): keep only chunks within this cosine distance of the
# query. Chroma cosine distance is 1 - cosine_similarity (range 0..2); lower is
# closer. Chunks above the threshold are treated as "not found" so off-topic
# questions return the grounded no-context answer instead of confident citations.
DEFAULT_RETRIEVAL_MAX_DISTANCE = 0.75
# Indexing limits (P0-2): guard against cost blow-up / memory DoS from oversized
# uploads. A value <= 0 disables that individual limit.
DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_MAX_PAGES_PER_PDF = 500
DEFAULT_MAX_CHUNKS_PER_RUN = 5000
DEFAULT_MAX_FILES_PER_RUN = 20
MISSING_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is missing. Add it to your .env file before using OpenAI features."
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    openai_api_key: str | None
    openai_base_url: str | None
    openai_chat_model: str
    openai_embedding_model: str
    chroma_persist_dir: str
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    retrieval_max_distance: float = DEFAULT_RETRIEVAL_MAX_DISTANCE
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    max_pages_per_pdf: int = DEFAULT_MAX_PAGES_PER_PDF
    max_chunks_per_run: int = DEFAULT_MAX_CHUNKS_PER_RUN
    max_files_per_run: int = DEFAULT_MAX_FILES_PER_RUN
    validation_error: str | None = None

    @property
    def has_openai_api_key(self) -> bool:
        """Return whether OpenAI-backed features can be used."""
        return bool(self.openai_api_key)

    @property
    def is_valid(self) -> bool:
        """Return whether the settings passed validation."""
        return self.validation_error is None


def get_settings() -> Settings:
    """Load settings from `.env` and environment variables."""
    load_dotenv(_project_root() / ".env")

    openai_api_key = _get_optional_env("OPENAI_API_KEY")
    validation_error = None if openai_api_key else MISSING_API_KEY_MESSAGE

    return Settings(
        openai_api_key=openai_api_key,
        openai_base_url=_get_optional_env("OPENAI_BASE_URL"),
        openai_chat_model=_get_env("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        openai_embedding_model=_get_env(
            "OPENAI_EMBEDDING_MODEL",
            DEFAULT_OPENAI_EMBEDDING_MODEL,
        ),
        chroma_persist_dir=_get_env("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_PERSIST_DIR),
        chunk_size=_get_int_env("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_get_int_env("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        retrieval_top_k=_get_int_env("RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K),
        retrieval_max_distance=_get_float_env(
            "RETRIEVAL_MAX_DISTANCE",
            DEFAULT_RETRIEVAL_MAX_DISTANCE,
        ),
        max_upload_mb=_get_int_env("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB),
        max_pages_per_pdf=_get_int_env("MAX_PAGES_PER_PDF", DEFAULT_MAX_PAGES_PER_PDF),
        max_chunks_per_run=_get_int_env(
            "MAX_CHUNKS_PER_RUN",
            DEFAULT_MAX_CHUNKS_PER_RUN,
        ),
        max_files_per_run=_get_int_env(
            "MAX_FILES_PER_RUN",
            DEFAULT_MAX_FILES_PER_RUN,
        ),
        validation_error=validation_error,
    )


def load_config() -> Settings:
    """Backward-compatible alias for settings loading."""
    return get_settings()


def _get_env(name: str, default: str) -> str:
    """Return a stripped environment variable or a default value."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _get_optional_env(name: str) -> str | None:
    """Return a stripped environment variable or None when empty."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _get_int_env(name: str, default: int) -> int:
    """Return an integer environment variable or a safe default."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    """Return a float environment variable or a safe default."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _project_root() -> Path:
    """Return the project root so .env loading does not depend on cwd."""
    return Path(__file__).resolve().parents[1]


AppConfig = Settings
