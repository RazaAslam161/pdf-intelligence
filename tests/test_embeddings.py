"""Tests for OpenAI embedding service wrapper."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.embeddings import EmbeddingService, EmbeddingServiceError


@dataclass(frozen=True)
class FakeSettings:
    """Minimal settings object for embedding tests."""

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_embedding_model: str = "test-embedding-model"


class FakeEmbeddingsEndpoint:
    """Fake OpenAI embeddings endpoint."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        """Return deterministic fake embeddings for each input text."""
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(len(text)), float(index)])
                for index, text in enumerate(input)
            ]
        )


class FakeClient:
    """Fake OpenAI client with an embeddings endpoint."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsEndpoint()


def test_embed_texts_batches_requests_and_preserves_order() -> None:
    """Embedding output should match input order across batches."""
    client = FakeClient()
    service = EmbeddingService(FakeSettings(), client=client, batch_size=2)

    embeddings = service.embed_texts(["one", "two", "three"])

    assert embeddings == [[3.0, 0.0], [3.0, 1.0], [5.0, 0.0]]
    assert client.embeddings.calls == [
        {"model": "test-embedding-model", "input": ["one", "two"]},
        {"model": "test-embedding-model", "input": ["three"]},
    ]


def test_embed_texts_allows_injected_client_without_api_key() -> None:
    """Tests and preconfigured clients should not need placeholder API keys."""
    service = EmbeddingService(FakeSettings(), client=FakeClient())

    assert service.embed_texts(["hello"]) == [[5.0, 0.0]]


def test_embed_texts_rejects_empty_strings() -> None:
    """Empty strings should not be sent to the embedding API."""
    service = EmbeddingService(FakeSettings(), client=FakeClient())

    with pytest.raises(ValueError, match="texts must not contain empty strings"):
        service.embed_texts(["valid", "   "])


def test_missing_api_key_raises_clear_error() -> None:
    """Missing API keys should produce a clear error message."""
    with pytest.raises(EmbeddingServiceError, match="OPENAI_API_KEY is missing"):
        EmbeddingService(FakeSettings(openai_api_key=None))


def test_openrouter_key_without_base_url_raises_clear_error() -> None:
    """OpenRouter keys should not be sent to the default OpenAI endpoint."""
    with pytest.raises(EmbeddingServiceError, match="OPENAI_BASE_URL"):
        EmbeddingService(FakeSettings(openai_api_key="sk-or-v1-example"))


def test_api_failure_raises_clear_error() -> None:
    """OpenAI API failures should be wrapped in an embedding-specific error."""

    class FailingEmbeddingsEndpoint:
        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            raise RuntimeError("upstream failed")

    class FailingClient:
        embeddings = FailingEmbeddingsEndpoint()

    service = EmbeddingService(FakeSettings(), client=FailingClient())

    with pytest.raises(EmbeddingServiceError, match="Failed to create embeddings"):
        service.embed_texts(["hello"])


def test_embed_texts_rejects_mismatched_api_response_count() -> None:
    """A malformed embedding response should fail instead of losing chunks."""

    class ShortEmbeddingsEndpoint:
        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])

    class ShortClient:
        embeddings = ShortEmbeddingsEndpoint()

    service = EmbeddingService(FakeSettings(), client=ShortClient())

    with pytest.raises(EmbeddingServiceError, match="unexpected number"):
        service.embed_texts(["one", "two"])
