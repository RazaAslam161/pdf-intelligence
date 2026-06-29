"""Tests for the high-level RAG service."""

import logging
from types import SimpleNamespace

import pytest

from src.models import RagAnswer, RetrievedChunk, TextChunk
from src.rag_service import (
    NO_CONTEXT_ANSWER,
    IndexingLimitError,
    RAGService,
    RAGServiceError,
    _create_openai_client,
    build_grounded_prompt,
    filter_relevant_chunks,
    validate_chunk_count,
    validate_page_count,
    validate_pdf_size,
)


class FakeSettings:
    """Minimal settings for RAG service tests.

    Indexing limits and the relevance threshold default to 0 (disabled) so the
    orchestration tests are unaffected; per-test overrides exercise them.
    """

    openai_api_key = None
    openai_base_url = None
    openai_chat_model = "test-chat-model"
    openai_embedding_model = "test-embedding-model"
    chroma_persist_dir = "./chroma_db"
    chunk_size = 100
    chunk_overlap = 20
    retrieval_top_k = 2
    retrieval_max_distance = 0.0
    max_upload_mb = 0
    max_pages_per_pdf = 0
    max_chunks_per_run = 0
    max_files_per_run = 0
    request_timeout = 30.0
    max_retries = 2
    max_output_tokens = 256


class FakeEmbeddingService:
    """Fake embedding service that records requested texts."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    """Fake vector store for orchestration tests."""

    def __init__(self, search_results: list[RetrievedChunk] | None = None) -> None:
        self.added_chunks: list[TextChunk] = []
        self.added_embeddings: list[list[float]] = []
        self.search_calls: list[tuple[list[float], int]] = []
        self.search_results = search_results or []

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        *,
        tenant_id: str = "default",
    ) -> None:
        self.added_chunks = chunks
        self.added_embeddings = embeddings

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        tenant_id: str = "default",
    ) -> list[RetrievedChunk]:
        self.search_calls.append((query_embedding, top_k))
        return self.search_results


class FakeCompletionsEndpoint:
    """Fake Chat Completions API endpoint."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {"model": model, "messages": messages, "max_tokens": max_tokens}
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="The answer is in the PDF.")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11, completion_tokens=7, total_tokens=18
            ),
        )


class FakeChatNamespace:
    """Fake chat namespace with a completions endpoint."""

    def __init__(self) -> None:
        self.completions = FakeCompletionsEndpoint()


class FakeOpenAIClient:
    """Fake OpenAI client with a Chat Completions API endpoint."""

    def __init__(self) -> None:
        self.chat = FakeChatNamespace()


def test_index_pdf_extracts_chunks_embeds_and_stores() -> None:
    """index_pdf should run the full indexing pipeline."""
    embeddings = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    service = RAGService(
        FakeSettings(),
        embedding_service=embeddings,
        vector_store=vector_store,
        openai_client=FakeOpenAIClient(),
    )

    result = service.index_pdf(_simple_pdf_bytes(), "sample.pdf")

    assert result.file_name == "sample.pdf"
    assert result.page_count == 1
    assert result.chunk_count == 1
    assert isinstance(result.document_id, str)
    assert embeddings.calls == [["Hello from PDF"]]
    assert vector_store.added_chunks[0].file_name == "sample.pdf"
    assert vector_store.added_chunks[0].page_number == 1
    assert vector_store.added_embeddings == [[1.0, 0.0]]


def test_answer_question_retrieves_context_and_calls_openai() -> None:
    """answer_question should generate an answer from retrieved chunks."""
    retrieved = [
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                file_name="source.pdf",
                page_number=3,
                text="The refund window is 30 days.",
            ),
            score=0.12,
        )
    ]
    embeddings = FakeEmbeddingService()
    vector_store = FakeVectorStore(search_results=retrieved)
    openai_client = FakeOpenAIClient()
    service = RAGService(
        FakeSettings(),
        embedding_service=embeddings,
        vector_store=vector_store,
        openai_client=openai_client,
    )

    result = service.answer_question("What is the refund window?")

    assert result.answer == "The answer is in the PDF."
    assert len(result.sources) == 1
    assert result.sources[0].chunk.file_name == "source.pdf"
    assert result.sources[0].chunk.page_number == 3
    assert result.sources[0].chunk.text == "The refund window is 30 days."
    assert embeddings.calls == [["What is the refund window?"]]
    assert vector_store.search_calls == [([1.0, 0.0], 2)]
    assert openai_client.chat.completions.calls[0]["model"] == "test-chat-model"
    assert openai_client.chat.completions.calls[0]["max_tokens"] == 256
    user_message = openai_client.chat.completions.calls[0]["messages"][1]["content"]
    assert "The refund window is 30 days." in user_message


def test_build_grounded_prompt_contains_answer_rules_and_context() -> None:
    """Prompt construction should keep grounding instructions near context."""
    retrieved = [
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                file_name="handbook.pdf",
                page_number=4,
                text="Use vacation time with manager approval.",
            )
        )
    ]

    prompt = build_grounded_prompt("How do I use vacation time?", retrieved)

    assert "Answer only from the provided PDF context" in prompt
    assert "could not find the answer in the uploaded documents" in prompt
    assert "handbook.pdf, page 4" in prompt
    assert "Use vacation time with manager approval." in prompt


def test_build_grounded_prompt_lists_multiple_sources() -> None:
    """Prompt context should identify each retrieved source separately."""
    retrieved = [
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                file_name="first.pdf",
                page_number=1,
                text="First context.",
            )
        ),
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-2",
                document_id="doc-2",
                file_name="second.pdf",
                page_number=5,
                text="Second context.",
            )
        ),
    ]

    prompt = build_grounded_prompt("Compare them.", retrieved)

    assert "[Source 1: first.pdf, page 1]" in prompt
    assert "[Source 2: second.pdf, page 5]" in prompt


def test_build_grounded_prompt_requests_inline_citations() -> None:
    """The prompt instructs the model to cite supporting sentences inline as [n]."""
    retrieved = [RetrievedChunk(chunk=_text_chunk("a"))]

    prompt = build_grounded_prompt("How does it work?", retrieved)

    assert "cite" in prompt.lower()
    assert "[1]" in prompt


def test_answer_question_returns_no_context_message_without_results() -> None:
    """No retrieval results should return the grounded fallback answer."""
    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(search_results=[]),
        openai_client=FakeOpenAIClient(),
    )

    result = service.answer_question("Unknown?")

    assert result == RagAnswer(answer=NO_CONTEXT_ANSWER, sources=[])


def test_answer_question_wraps_generation_failures() -> None:
    """Generation failures should be readable for Streamlit."""

    class FailingCompletionsEndpoint:
        def create(
            self,
            *,
            model: str,
            messages: list[dict[str, str]],
            max_tokens: int | None = None,
        ) -> None:
            raise RuntimeError("upstream failed")

    class FailingChatNamespace:
        completions = FailingCompletionsEndpoint()

    class FailingOpenAIClient:
        chat = FailingChatNamespace()

    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(
            search_results=[
                RetrievedChunk(
                    chunk=TextChunk(
                        chunk_id="chunk-1",
                        document_id="doc-1",
                        file_name="source.pdf",
                        page_number=1,
                        text="Some context.",
                    )
                )
            ]
        ),
        openai_client=FailingOpenAIClient(),
    )

    with pytest.raises(RAGServiceError, match="Failed to generate an answer"):
        service.answer_question("Question?")


def test_answer_question_drops_chunks_beyond_relevance_threshold() -> None:
    """Off-topic hits past the distance cutoff yield the grounded no-context answer."""
    retrieved = [
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                file_name="source.pdf",
                page_number=1,
                text="Unrelated content.",
            ),
            score=0.95,
        )
    ]
    settings = FakeSettings()
    settings.retrieval_max_distance = 0.5
    service = RAGService(
        settings,
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(search_results=retrieved),
        openai_client=FakeOpenAIClient(),
    )

    result = service.answer_question("Something off topic?")

    assert result == RagAnswer(answer=NO_CONTEXT_ANSWER, sources=[])


def test_answer_question_keeps_chunks_within_threshold() -> None:
    """A close-enough hit is kept, answered, and cited."""
    retrieved = [
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                file_name="source.pdf",
                page_number=3,
                text="The refund window is 30 days.",
            ),
            score=0.2,
        )
    ]
    settings = FakeSettings()
    settings.retrieval_max_distance = 0.5
    service = RAGService(
        settings,
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(search_results=retrieved),
        openai_client=FakeOpenAIClient(),
    )

    result = service.answer_question("What is the refund window?")

    assert result.answer == "The answer is in the PDF."
    assert len(result.sources) == 1


def test_filter_relevant_chunks_keeps_within_threshold_and_unscored() -> None:
    """Chunks within the cutoff (or lacking a score) survive; distant ones drop."""
    chunks = [
        RetrievedChunk(chunk=_text_chunk("a"), score=0.2),
        RetrievedChunk(chunk=_text_chunk("b"), score=0.9),
        RetrievedChunk(chunk=_text_chunk("c"), score=None),
    ]

    kept = filter_relevant_chunks(chunks, 0.5)

    assert [retrieved.score for retrieved in kept] == [0.2, None]


def test_filter_relevant_chunks_disabled_returns_all() -> None:
    """A threshold of 0 disables filtering."""
    chunks = [
        RetrievedChunk(chunk=_text_chunk("a"), score=0.2),
        RetrievedChunk(chunk=_text_chunk("b"), score=1.5),
    ]

    assert filter_relevant_chunks(chunks, 0) == chunks


def test_validate_pdf_size_enforces_and_disables_cap() -> None:
    """Oversized uploads raise; a 0 cap disables the check."""
    settings = SimpleNamespace(max_upload_mb=1)

    validate_pdf_size(500_000, "ok.pdf", settings)
    with pytest.raises(IndexingLimitError, match="upload limit"):
        validate_pdf_size(2 * 1024 * 1024, "big.pdf", settings)
    validate_pdf_size(10**9, "huge.pdf", SimpleNamespace(max_upload_mb=0))


def test_validate_page_and_chunk_counts_enforce_caps() -> None:
    """Page and chunk caps raise only above the limit; equal is allowed."""
    settings = SimpleNamespace(max_pages_per_pdf=2, max_chunks_per_run=3)

    validate_page_count(2, "f.pdf", settings)
    with pytest.raises(IndexingLimitError, match="page limit"):
        validate_page_count(3, "f.pdf", settings)

    validate_chunk_count(3, "f.pdf", settings)
    with pytest.raises(IndexingLimitError, match="chunk limit"):
        validate_chunk_count(4, "f.pdf", settings)


def test_answer_question_logs_correlated_line(caplog) -> None:
    """A successful answer emits one correlated line with timings and tokens."""
    retrieved = [
        RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                file_name="s.pdf",
                page_number=1,
                text="Body.",
            ),
            score=0.1,
        )
    ]
    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(search_results=retrieved),
        openai_client=FakeOpenAIClient(),
    )

    with caplog.at_level(logging.INFO, logger="ragpdf"):
        service.answer_question("What?")

    text = caplog.text
    assert "answer request_id=" in text
    assert "embed_ms=" in text
    assert "search_ms=" in text
    assert "llm_ms=" in text
    assert "total_tokens=18" in text
    assert "sources=1" in text


def test_index_pdf_logs_outcome(caplog) -> None:
    """Indexing emits a per-document outcome line."""
    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        openai_client=FakeOpenAIClient(),
    )

    with caplog.at_level(logging.INFO, logger="ragpdf"):
        service.index_pdf(_simple_pdf_bytes(), "sample.pdf")

    text = caplog.text
    assert "index request_id=" in text
    assert "file=sample.pdf" in text
    assert "pages=1" in text
    assert "chunks=1" in text


def test_create_openai_client_passes_timeout_and_retries() -> None:
    """The per-request deadline and retry budget are configured on the client."""
    from unittest.mock import patch

    with patch("openai.OpenAI") as mock_openai:
        _create_openai_client(
            "key",
            base_url="https://example.test/v1",
            timeout=12.5,
            max_retries=4,
        )

    mock_openai.assert_called_once_with(
        api_key="key",
        base_url="https://example.test/v1",
        timeout=12.5,
        max_retries=4,
    )


def test_generation_aborts_when_openai_call_times_out() -> None:
    """A timed-out call aborts with a clean error instead of hanging."""

    class TimingOutCompletions:
        def create(
            self,
            *,
            model: str,
            messages: list[dict[str, str]],
            max_tokens: int | None = None,
        ) -> None:
            raise TimeoutError("Request timed out.")

    class TimingOutChat:
        completions = TimingOutCompletions()

    class TimingOutClient:
        chat = TimingOutChat()

    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(
            search_results=[RetrievedChunk(chunk=_text_chunk("a"))]
        ),
        openai_client=TimingOutClient(),
    )

    with pytest.raises(RAGServiceError, match="Failed to generate an answer"):
        service.answer_question("Question?")


def test_index_pdf_rejects_oversized_upload_before_parsing() -> None:
    """The size cap fires before pypdf/embedding, so junk bytes never get parsed."""
    settings = FakeSettings()
    settings.max_upload_mb = 1
    embeddings = FakeEmbeddingService()
    service = RAGService(
        settings,
        embedding_service=embeddings,
        vector_store=FakeVectorStore(),
        openai_client=FakeOpenAIClient(),
    )

    oversized = b"%PDF-" + b"0" * (2 * 1024 * 1024)
    with pytest.raises(IndexingLimitError, match="upload limit"):
        service.index_pdf(oversized, "big.pdf")
    assert embeddings.calls == []


class FakeStreamingClient:
    """Fake OpenAI client that streams content deltas for stream_answer tests."""

    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas
        self.chat = SimpleNamespace(completions=self)

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        max_tokens: int | None = None,
    ):
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
            for d in self._deltas
        )


def test_stream_answer_yields_tokens_then_sources() -> None:
    """stream_answer emits token events, then one sources event."""
    retrieved = [RetrievedChunk(chunk=_text_chunk("a"), score=0.1)]
    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(search_results=retrieved),
        openai_client=FakeStreamingClient(["The ", "answer ", "is here [1]."]),
    )

    events = list(service.stream_answer("What?"))

    assert [kind for kind, _ in events] == ["token", "token", "token", "sources"]
    text = "".join(value for kind, value in events if kind == "token")
    assert text == "The answer is here [1]."
    sources = [value for kind, value in events if kind == "sources"][0]
    assert len(sources) == 1
    assert sources[0].chunk.chunk_id == "chunk-a"


def test_stream_answer_no_context_yields_fallback() -> None:
    """With nothing relevant, stream_answer yields the no-context fallback."""
    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(search_results=[]),
        openai_client=FakeStreamingClient([]),
    )

    events = list(service.stream_answer("?"))

    assert events == [("token", NO_CONTEXT_ANSWER), ("sources", [])]


def _text_chunk(suffix: str) -> TextChunk:
    """Build a minimal TextChunk fixture for filter tests."""
    return TextChunk(
        chunk_id=f"chunk-{suffix}",
        document_id="doc-1",
        file_name="source.pdf",
        page_number=1,
        text=f"Context {suffix}.",
    )


def _simple_pdf_bytes() -> bytes:
    """Return a tiny valid one-page PDF containing text."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
        b"endobj\n"
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
        b"5 0 obj\n<< /Length 43 >>\nstream\n"
        b"BT /F1 24 Tf 100 700 Td (Hello from PDF) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000058 00000 n \n0000000115 00000 n \n"
        b"0000000241 00000 n \n0000000311 00000 n \n"
        b"trailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n405\n%%EOF\n"
    )
