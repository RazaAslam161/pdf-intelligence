"""Tests for the high-level RAG service."""

from types import SimpleNamespace

import pytest

from src.models import RetrievedChunk, TextChunk
from src.rag_service import RAGService, RAGServiceError, build_grounded_prompt


class FakeSettings:
    """Minimal settings for RAG service tests."""

    openai_api_key = None
    openai_base_url = None
    openai_chat_model = "test-chat-model"
    openai_embedding_model = "test-embedding-model"
    chroma_persist_dir = "./chroma_db"
    chunk_size = 100
    chunk_overlap = 20
    retrieval_top_k = 2


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
    ) -> None:
        self.added_chunks = chunks
        self.added_embeddings = embeddings

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        self.search_calls.append((query_embedding, top_k))
        return self.search_results


class FakeCompletionsEndpoint:
    """Fake Chat Completions API endpoint."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:
        self.calls.append({"model": model, "messages": messages})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="The answer is in the PDF.")
                )
            ]
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

    summary = service.index_pdf(_simple_pdf_bytes(), "sample.pdf")

    assert summary["file_name"] == "sample.pdf"
    assert summary["page_count"] == 1
    assert summary["chunk_count"] == 1
    assert isinstance(summary["document_id"], str)
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

    assert result["answer"] == "The answer is in the PDF."
    assert result["sources"] == [
        {
            "label": "[Source 1] source.pdf, page 3",
            "file_name": "source.pdf",
            "page_number": 3,
            "preview": "The refund window is 30 days.",
        }
    ]
    assert embeddings.calls == [["What is the refund window?"]]
    assert vector_store.search_calls == [([1.0, 0.0], 2)]
    assert openai_client.chat.completions.calls[0]["model"] == "test-chat-model"
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


def test_answer_question_returns_no_context_message_without_results() -> None:
    """No retrieval results should return the grounded fallback answer."""
    service = RAGService(
        FakeSettings(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(search_results=[]),
        openai_client=FakeOpenAIClient(),
    )

    result = service.answer_question("Unknown?")

    assert result == {
        "answer": "I could not find the answer in the uploaded documents.",
        "sources": [],
    }


def test_answer_question_wraps_generation_failures() -> None:
    """Generation failures should be readable for Streamlit."""

    class FailingCompletionsEndpoint:
        def create(self, *, model: str, messages: list[dict[str, str]]) -> None:
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
