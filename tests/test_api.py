"""Tests for the FastAPI transport layer (M2).

The RAG service is replaced with a fake via dependency override, so these tests
exercise the HTTP contract without touching OpenAI or Chroma.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import deps
from api.deps import get_rag_service, get_vector_store
from api.main import app, to_sources
from src.models import (
    DocumentSummary,
    IndexResult,
    RagAnswer,
    RetrievedChunk,
    TextChunk,
)
from src.rag_service import RAGService
from src.vector_store import VectorStore

# A minimal valid one-page PDF whose extractable text is "Hello from PDF".
_VALID_PDF = (
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


class _FakeEmbeddings:
    """Deterministic embeddings for the integration test (no OpenAI)."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE response body into its decoded `data:` event payloads."""
    events: list[dict] = []
    for block in text.split("\n\n"):
        line = block.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:") :].strip()))
    return events


class FakeRagService:
    """Minimal stand-in for RAGService returning typed objects."""

    def answer_question(
        self,
        question: str,
        *,
        tenant_id: str = "default",
    ) -> RagAnswer:
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

    def stream_answer(self, question: str, *, tenant_id: str = "default"):
        yield ("token", "The refund window ")
        yield ("token", "is 30 days [1].")
        yield (
            "sources",
            [
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

    def index_pdf(
        self,
        pdf_bytes: bytes,
        file_name: str,
        *,
        tenant_id: str = "default",
    ) -> IndexResult:
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

    def answer_question(
        self,
        question: str,
        *,
        tenant_id: str = "default",
    ) -> RagAnswer:
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

    def stream_answer(self, question: str, *, tenant_id: str = "default"):
        yield ("token", DANGEROUS)
        yield (
            "sources",
            [
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

    def index_pdf(
        self,
        pdf_bytes: bytes,
        file_name: str,
        *,
        tenant_id: str = "default",
    ) -> IndexResult:
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


def test_ask_streams_tokens_then_sources(client: TestClient) -> None:
    """POST /ask streams token events, then one sources event, then done."""
    response = client.post("/ask", json={"question": "What is the refund window?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == "The refund window is 30 days [1]."

    sources_events = [e for e in events if e["type"] == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["sources"] == [
        {
            "id": "doc-1-page-3-chunk-1",
            "file_name": "policy.pdf",
            "page_number": 3,
            "preview": "Customers may request a refund within 30 days.",
            "score": 0.21,
        }
    ]
    # Tokens arrive before the sources event, which arrives before done.
    types = [e["type"] for e in events]
    assert types.index("token") < types.index("sources") < types.index("done")


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


def test_ask_streams_special_chars_as_inert_json() -> None:
    """A3: markup in streamed tokens + source fields stays inert JSON, never HTML."""
    app.dependency_overrides[get_rag_service] = lambda: DangerousRagService()
    try:
        response = TestClient(app).post("/ask", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = _parse_sse(response.text)
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == DANGEROUS
    source = [e for e in events if e["type"] == "sources"][0]["sources"][0]
    # Special chars survive verbatim as JSON string data — never HTML-encoded.
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


def test_config_exposes_upload_limits() -> None:
    """B5: the client can read the P0-2 caps for pre-flight feedback."""
    response = TestClient(app).get("/config")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"max_upload_mb", "max_pages_per_pdf", "max_files_per_run"}
    assert all(isinstance(value, int) for value in body.values())


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
    monkeypatch.setattr(deps, "get_vector_store", lambda: object())
    monkeypatch.setattr(deps, "RAGService", lambda *args, **kwargs: object())
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

        def stream_answer(self, question: str, *, tenant_id: str = "default"):
            yield ("token", "ok")
            yield ("sources", [])

    monkeypatch.setattr(deps, "get_vector_store", lambda: object())
    monkeypatch.setattr(deps, "RAGService", lambda *args, **kwargs: CountingService())
    try:
        http = TestClient(app)
        for _ in range(3):
            assert http.post("/ask", json={"question": "ping"}).status_code == 200
        assert constructions["count"] == 1
    finally:
        deps.get_rag_service.cache_clear()


class FakeStore:
    """Fake vector store for the workspace endpoints."""

    def __init__(self) -> None:
        self.cleared = False

    def list_documents(self, *, tenant_id: str = "default") -> list[DocumentSummary]:
        if self.cleared:
            return []
        return [
            DocumentSummary(
                document_id="docA",
                file_name="a.pdf",
                page_count=2,
                chunk_count=5,
            )
        ]

    def count(self, *, tenant_id: str | None = None) -> int:
        return 0 if self.cleared else 5

    def clear(self, *, tenant_id: str = "default") -> None:
        self.cleared = True


def test_documents_returns_live_store_state() -> None:
    """B6: GET /documents reflects the live store, not a session."""
    store = FakeStore()
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        response = TestClient(app).get("/documents")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] == 5
    assert body["ready"] is True
    assert body["documents"] == [
        {
            "document_id": "docA",
            "file_name": "a.pdf",
            "page_count": 2,
            "chunk_count": 5,
        }
    ]


def test_delete_documents_clears_and_returns_empty_state() -> None:
    """B6: DELETE /documents wipes the store and returns the empty state."""
    store = FakeStore()
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        response = TestClient(app).delete("/documents")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"documents": [], "total_chunks": 0, "ready": False}
    assert store.cleared is True


def test_index_isolates_per_file_failures(tmp_path) -> None:
    """A6: one bad file fails alone; the others index, persist, and stay queryable."""
    store = VectorStore(str(tmp_path / "chroma"))
    settings = SimpleNamespace(
        chunk_size=1000,
        chunk_overlap=150,
        max_upload_mb=0,
        max_pages_per_pdf=0,
        max_chunks_per_run=0,
    )
    service = RAGService(
        settings,
        embedding_service=_FakeEmbeddings(),
        vector_store=store,
    )

    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        client = TestClient(app)
        files = [
            ("files", ("a.pdf", _VALID_PDF, "application/pdf")),
            ("files", ("b.pdf", b"this is not a pdf", "application/pdf")),
            ("files", ("c.pdf", _VALID_PDF, "application/pdf")),
        ]
        index_response = client.post("/index", files=files)
        documents_response = client.get("/documents")
    finally:
        app.dependency_overrides.clear()

    statuses = {
        result["file_name"]: result["status"]
        for result in index_response.json()["results"]
    }
    assert statuses == {"a.pdf": "indexed", "b.pdf": "failed", "c.pdf": "indexed"}

    # Partial success is persisted and queryable: the two good files are in the
    # store; the failed one is not.
    store_state = documents_response.json()
    assert store_state["ready"] is True
    assert store_state["total_chunks"] >= 2
    assert {doc["file_name"] for doc in store_state["documents"]} == {"a.pdf", "c.pdf"}


def test_tenant_isolation_via_api(tmp_path) -> None:
    """A8: a tenant cannot retrieve or clear another tenant's documents."""
    store = VectorStore(str(tmp_path / "chroma"))
    settings = SimpleNamespace(
        chunk_size=1000,
        chunk_overlap=150,
        max_upload_mb=0,
        max_pages_per_pdf=0,
        max_chunks_per_run=0,
    )
    service = RAGService(
        settings,
        embedding_service=_FakeEmbeddings(),
        vector_store=store,
    )
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_vector_store] = lambda: store

    alice = {"X-Tenant-Id": "alice"}
    bob = {"X-Tenant-Id": "bob"}
    try:
        client = TestClient(app)
        client.post(
            "/index",
            files=[("files", ("alice.pdf", _VALID_PDF, "application/pdf"))],
            headers=alice,
        )
        client.post(
            "/index",
            files=[("files", ("bob.pdf", _VALID_PDF, "application/pdf"))],
            headers=bob,
        )
        alice_docs = client.get("/documents", headers=alice).json()
        bob_docs = client.get("/documents", headers=bob).json()
        client.delete("/documents", headers=alice)
        alice_after = client.get("/documents", headers=alice).json()
        bob_after = client.get("/documents", headers=bob).json()
    finally:
        app.dependency_overrides.clear()

    # Each tenant retrieves only their own document.
    assert {d["file_name"] for d in alice_docs["documents"]} == {"alice.pdf"}
    assert {d["file_name"] for d in bob_docs["documents"]} == {"bob.pdf"}
    # Alice's clear wipes only her store; bob's documents are untouched.
    assert alice_after["documents"] == []
    assert alice_after["ready"] is False
    assert {d["file_name"] for d in bob_after["documents"]} == {"bob.pdf"}
    assert bob_after["ready"] is True


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
