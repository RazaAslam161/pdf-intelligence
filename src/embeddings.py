"""OpenAI embedding helpers."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from src.observability import get_logger

logger = get_logger("embeddings")

DEFAULT_EMBEDDING_BATCH_SIZE = 256
DEFAULT_EMBED_CONCURRENCY = 8
MISSING_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is missing. Add it to your .env file before creating embeddings."
)


class EmbeddingSettings(Protocol):
    """Settings fields required by the embedding service."""

    openai_api_key: str | None
    openai_base_url: str | None
    openai_embedding_model: str
    request_timeout: float
    max_retries: int
    embed_batch_size: int
    embed_concurrency: int


class EmbeddingServiceError(RuntimeError):
    """Raised when embeddings cannot be created."""


class EmbeddingService:
    """Reusable OpenAI embeddings service."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        client: Any | None = None,
        batch_size: int | None = None,
        concurrency: int | None = None,
    ) -> None:
        """Create an embedding service from settings and an optional client."""
        api_key = settings.openai_api_key
        base_url = getattr(settings, "openai_base_url", None)
        if client is None and not api_key:
            raise EmbeddingServiceError(MISSING_API_KEY_MESSAGE)
        if client is None:
            _validate_openai_compatible_settings(str(api_key), base_url)

        resolved_batch = (
            batch_size
            if batch_size is not None
            else getattr(settings, "embed_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)
        )
        if resolved_batch <= 0:
            raise ValueError("batch_size must be greater than 0.")
        resolved_concurrency = (
            concurrency
            if concurrency is not None
            else getattr(settings, "embed_concurrency", DEFAULT_EMBED_CONCURRENCY)
        )

        self._model = settings.openai_embedding_model
        self._batch_size = resolved_batch
        self._concurrency = max(1, resolved_concurrency)
        self._client = client or _create_openai_client(
            str(api_key),
            base_url=base_url,
            timeout=getattr(settings, "request_timeout", None),
            max_retries=getattr(settings, "max_retries", None),
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for texts while preserving input order.

        Multiple batches are embedded concurrently (embedding round-trips are the
        indexing bottleneck); ThreadPoolExecutor.map preserves batch order.
        """
        _validate_texts(texts)
        if not texts:
            return []

        batches = _batched(texts, self._batch_size)
        if len(batches) == 1:
            return self._embed_batch(batches[0])

        embeddings: list[list[float]] = []
        workers = min(self._concurrency, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batch_embeddings in pool.map(self._embed_batch, batches):
                embeddings.extend(batch_embeddings)
        return embeddings

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Create embeddings for one batch of text."""
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            embeddings = _extract_embeddings(response)
        except Exception as exc:
            logger.exception("embed_failed model=%s n=%d", self._model, len(batch))
            raise EmbeddingServiceError(
                "Failed to create embeddings with the OpenAI API."
            ) from exc

        if len(embeddings) != len(batch):
            raise EmbeddingServiceError(
                "OpenAI returned an unexpected number of embeddings."
            )

        usage = getattr(response, "usage", None)
        logger.debug(
            "embed model=%s n=%d total_tokens=%s",
            self._model,
            len(batch),
            getattr(usage, "total_tokens", None),
        )
        return embeddings


def _create_openai_client(
    api_key: str,
    *,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Create the OpenAI SDK client with a bounded timeout and retry budget."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise EmbeddingServiceError(
            "The openai package is not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return OpenAI(**kwargs)


def _validate_openai_compatible_settings(
    api_key: str,
    base_url: str | None,
) -> None:
    """Prevent provider keys from being sent to the wrong endpoint."""
    if _looks_like_openrouter_key(api_key) and not _is_openrouter_base_url(base_url):
        raise EmbeddingServiceError(
            "OpenRouter API key detected. Set "
            "OPENAI_BASE_URL=https://openrouter.ai/api/v1 and use an OpenRouter "
            "embedding model such as openai/text-embedding-3-small."
        )


def _looks_like_openrouter_key(api_key: str) -> bool:
    """Return whether an API key appears to be an OpenRouter key."""
    return api_key.strip().startswith("sk-or-")


def _is_openrouter_base_url(base_url: str | None) -> bool:
    """Return whether a base URL points at OpenRouter."""
    return bool(base_url and "openrouter.ai/api/v1" in base_url.lower())


def _validate_texts(texts: Sequence[str]) -> None:
    """Validate embedding input before sending it to the API."""
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be a string.")
        if not text.strip():
            raise ValueError("texts must not contain empty strings.")


def _batched(texts: Sequence[str], batch_size: int) -> list[list[str]]:
    """Split texts into request-sized batches."""
    return [
        list(texts[index : index + batch_size])
        for index in range(0, len(texts), batch_size)
    ]


def _extract_embeddings(response: Any) -> list[list[float]]:
    """Extract embeddings from an OpenAI SDK response."""
    data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
    return [list(item.embedding) for item in data]
