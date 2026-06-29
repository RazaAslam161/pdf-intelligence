"""Tests for the FastAPI transport layer (M2).

The RAG service is replaced with a fake via dependency override, so these tests
exercise the HTTP contract without touching OpenAI or Chroma.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api import deps
from api.deps import get_rag_service
from api.main import app, to_sources
from src.models import IndexResult, RagAnswer, RetrievedChunk, TextChunk


class FakeRagService:
    """Minimal stand-in for RAGService returning typed objects."""

    def answer_question(self, question: str) -> RagAnswer:
        return RagAnswer(
            answer="The refund window is 30 days.",
            sources=[
                RetrievedChunk(
                    chunk=TextChunk(
                        chunk_id="doc-1-page-3-chunk-1",
                        document_id="doc-1",
                        file_name="policy.pdf",
                        page_number=3,
                        text="Customers may request a refund within 30 days.",
                    ),
                    score=0.21,
                )
            ],
        )

    def index_pdf(self, pdf_bytes: bytes, file_name: str) -> IndexResult:
        return IndexResult(
            file_name=file_name,
            document_id="doc-1",
            page_count=2,
            chunk_count=5,
        )


# Markup/markdown metacharacters that must survive as inert string data.
DANGEROUS = "</div><script>x</script> [x](http://evil) & <b>"


class DangerousRagService:
    """Fake service whose source file name and preview contain markup."""

    def answer_question(self, question: str) -> RagAnswer:
        return RagAnswer(
            answer="ok",
            sources=[
                RetrievedChunk(
                    chunk=TextChunk(
                        chunk_id="c1",
                        document_id="d1",
                        file_name=DANGEROUS,
                        page_number=1,
                        text=DANGEROUS,
                    ),
                    score=0.1,
                )
            ],
        )

    def index_pdf(self, pdf_bytes: bytes, file_name: str) -> IndexResult:
        raise NotImplementedError


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_health_returns_ok() -> None:
    """Health check should not require the RAG service or OpenAI config."""
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_typed_answer_and_sources(client: TestClient) -> None:
    """POST /ask returns the answer plus citation DTOs with score and id."""
    response = client.post("/ask", json={"question": "What is the refund window?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The refund window is 30 days."
    assert body["sources"] == [
        {
            "id": "doc-1-page-3-chunk-1",
            "file_name": "policy.pdf",
            "page_number": 3,
            "preview": "Customers may request a refund within 30 days.",
            "score": 0.21,
        }
    ]


def test_ask_rejects_blank_question(client: TestClient) -> None:
    """A blank question fails validation before reaching the service."""
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_index_returns_per_file_results(client: TestClient) -> None:
    """POST /index returns a typed per-file result list."""
    files = [("files", ("report.pdf", b"%PDF-1.4 fake", "application/pdf"))]
    response = client.post("/index", files=files)

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "file_name": "report.pdf",
                "page_count": 2,
                "chunk_count": 5,
                "status": "indexed",
            }
        ]
    }


def test_index_marks_non_pdf_unsupported(client: TestClient) -> None:
    """A non-PDF upload is reported as unsupported, not indexed."""
    files = [("files", ("notes.txt", b"hello", "text/plain"))]
    response = client.post("/index", files=files)

    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "unsupported"


def test_ask_returns_special_chars_as_inert_json() -> None:
    """A3: markup in file_name/preview round-trips as inert JSON, never HTML."""
    app.dependency_overrides[get_rag_service] = lambda: DangerousRagService()
    try:
        response = TestClient(app).post("/ask", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    source = response.json()["sources"][0]
    # Special chars survive verbatim as JSON string data — not HTML-encoded,
    # not stripped — so the API emits inert structured data, not markup.
    assert source["file_name"] == DANGEROUS
    assert source["preview"] == DANGEROUS
    assert "&lt;" not in response.text
    assert "&amp;" not in response.text


def test_index_filename_is_inert_in_json(client: TestClient) -> None:
    """A3: a markup-laden filename comes back as inert data, not HTML entities."""
    name = "</div>[x](y)<b>.pdf"
    files = [("files", (name, b"%PDF-1.4 fake", "application/pdf"))]
    response = client.post("/index", files=files)

    assert response.status_code == 200
    returned = response.json()["results"][0]["file_name"]
    assert "<" in returned
    assert "[x](y)" in returned
    assert "&lt;" not in returned


def test_openapi_schema_exposes_endpoints() -> None:
    """The OpenAPI schema (served at /docs) lists the three endpoints."""
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {"/health", "/ask", "/index"} <= set(paths)


def test_to_sources_preserves_order_for_citation_alignment() -> None:
    """B4: sources are 1:1 and ordered so positions match inline [n] markers."""
    chunks = [
        RetrievedChunk(
            chunk=TextChunk("c1", "d1", "f.pdf", 1, "First passage."),
            score=0.10,
        ),
        # Same page as c1 — must NOT be deduped, so [2] still maps to it.
        RetrievedChunk(
            chunk=TextChunk("c2", "d1", "f.pdf", 1, "Second passage."),
            score=0.42,
        ),
    ]

    sources = to_sources(chunks)

    assert [s.id for s in sources] == ["c1", "c2"]
    assert [s.page_number for s in sources] == [1, 1]
    assert [s.score for s in sources] == [0.10, 0.42]


def test_get_rag_service_caches_a_single_instance(monkeypatch) -> None:
    """A4: the service provider returns the same cached instance."""
    deps.get_rag_service.cache_clear()
    monkeypatch.setattr(deps, "RAGService", lambda settings: object())
    try:
        assert deps.get_rag_service() is deps.get_rag_service()
    finally:
        deps.get_rag_service.cache_clear()


def test_service_constructed_once_across_requests(monkeypatch) -> None:
    """A4: no per-request construction — the Chroma/OpenAI clients build once."""
    deps.get_rag_service.cache_clear()
    constructions = {"count": 0}

    class CountingService:
        def __init__(self) -> None:
            constructions["count"] += 1

        def answer_question(self, question: str) -> RagAnswer:
            return RagAnswer(answer="ok", sources=[])

    monkeypatch.setattr(deps, "RAGService", lambda settings: CountingService())
    try:
        http = TestClient(app)
        for _ in range(3):
            assert http.post("/ask", json={"question": "ping"}).status_code == 200
        assert constructions["count"] == 1
    finally:
        deps.get_rag_service.cache_clear()


def test_cors_preflight_allows_configured_origin() -> None:
    """A CORS preflight from an allowed origin returns the allow-origin header."""
    response = TestClient(app).options(
        "/ask",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
