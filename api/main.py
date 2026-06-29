"""FastAPI app exposing the RAG service over HTTP.

Thin transport layer: adapts RAGService to the JSON contract in ``schemas``
without reaching into ``src`` internals beyond the public service.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.citations import preview_text
from src.config import Settings, get_settings
from src.models import RetrievedChunk
from src.observability import get_logger
from src.rag_service import IndexingLimitError, RAGService, RAGServiceError
from src.vector_store import VectorStore, VectorStoreError

from .deps import get_rag_service, get_tenant_id, get_vector_store
from .schemas import (
    AskRequest,
    ConfigResponse,
    DocumentSummary,
    HealthResponse,
    IndexFileResult,
    IndexResponse,
    Source,
    StoreState,
)

logger = get_logger("api")

DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:3000"
PREVIEW_CHARS = 220


def _allowed_origins() -> list[str]:
    """Read allowed CORS origins (comma-separated) from the environment."""
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def to_sources(chunks: Sequence[RetrievedChunk]) -> list[Source]:
    """Map retrieved chunks to citation DTOs, preserving order.

    Positions stay 1:1 with the chunks so they line up with the inline [n]
    markers the model emits (sources[0] is [1]).
    """
    sources: list[Source] = []
    for retrieved in chunks:
        chunk = retrieved.chunk
        sources.append(
            Source(
                id=chunk.chunk_id,
                file_name=chunk.file_name,
                page_number=chunk.page_number,
                preview=preview_text(chunk.text, PREVIEW_CHARS),
                score=retrieved.score,
            )
        )
    return sources


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm the singletons on boot so the first request doesn't pay cold-start.

    Best-effort: a missing or invalid key just logs a warning and never blocks
    startup, so health stays up.
    """
    try:
        get_vector_store()
        get_rag_service().embedding_service.embed_texts(["warmup"])
        logger.info("startup_warmup ok")
    except Exception as exc:
        logger.warning("startup_warmup skipped: %s", exc)
    yield


app = FastAPI(
    title="PDF Intelligence API",
    version="0.1.0",
    description="Grounded question answering over uploaded PDFs with citations.",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check that does not require OpenAI configuration."""
    return HealthResponse(status="ok")


@app.get("/config", response_model=ConfigResponse)
def config(settings: Settings = Depends(get_settings)) -> ConfigResponse:
    """Expose upload limits so the client can reject oversized files pre-flight."""
    return ConfigResponse(
        max_upload_mb=settings.max_upload_mb,
        max_pages_per_pdf=settings.max_pages_per_pdf,
        max_files_per_run=settings.max_files_per_run,
    )


def _store_state(store: VectorStore, tenant_id: str) -> StoreState:
    """Build the live store state DTO for one tenant."""
    documents = [
        DocumentSummary(
            document_id=doc.document_id,
            file_name=doc.file_name,
            page_count=doc.page_count,
            chunk_count=doc.chunk_count,
        )
        for doc in store.list_documents(tenant_id=tenant_id)
    ]
    total = store.count(tenant_id=tenant_id)
    return StoreState(documents=documents, total_chunks=total, ready=total > 0)


@app.get("/documents", response_model=StoreState)
def documents(
    store: VectorStore = Depends(get_vector_store),
    tenant_id: str = Depends(get_tenant_id),
) -> StoreState:
    """Return the caller's live document store (scoped to their tenant)."""
    try:
        return _store_state(store, tenant_id)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/documents", response_model=StoreState)
def clear_documents(
    store: VectorStore = Depends(get_vector_store),
    tenant_id: str = Depends(get_tenant_id),
) -> StoreState:
    """Wipe only the caller's documents and return their (now empty) state."""
    try:
        store.clear(tenant_id=tenant_id)
        return _store_state(store, tenant_id)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _sse(data: dict[str, object]) -> str:
    """Format one server-sent event."""
    return f"data: {json.dumps(data)}\n\n"


@app.post("/ask")
def ask(
    payload: AskRequest,
    service: RAGService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_id),
) -> StreamingResponse:
    """Stream a grounded answer token-by-token (SSE); sources arrive at the end."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty.")

    def event_stream() -> Iterator[str]:
        try:
            for kind, value in service.stream_answer(question, tenant_id=tenant_id):
                if kind == "token":
                    yield _sse({"type": "token", "text": value})
                elif kind == "sources":
                    sources = [source.model_dump() for source in to_sources(value)]
                    yield _sse({"type": "sources", "sources": sources})
            yield _sse({"type": "done"})
        except RAGServiceError as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/index", response_model=IndexResponse)
def index(
    files: list[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_id),
) -> IndexResponse:
    """Index uploaded PDFs for the caller's tenant; each file is isolated.

    Declared sync so FastAPI runs it in a worker thread, since index_pdf does
    blocking work (pypdf, embedding HTTP, Chroma) that must not block the event loop.
    """
    results: list[IndexFileResult] = []
    for upload in files:
        file_name = upload.filename or "upload.pdf"
        if not file_name.lower().endswith(".pdf"):
            results.append(_failed_result(file_name, "unsupported"))
            continue

        data = upload.file.read()
        # Each file gets its own try/except so one bad file never aborts the batch.
        try:
            result = service.index_pdf(data, file_name, tenant_id=tenant_id)
        except IndexingLimitError as exc:
            logger.info("index_rejected file=%s reason=%s", file_name, exc)
            results.append(_failed_result(file_name, "rejected"))
            continue
        except Exception:
            logger.exception("index_failed file=%s", file_name)
            results.append(_failed_result(file_name, "failed"))
            continue

        results.append(
            IndexFileResult(
                file_name=result.file_name,
                page_count=result.page_count,
                chunk_count=result.chunk_count,
                status="indexed",
            )
        )
    return IndexResponse(results=results)


def _failed_result(file_name: str, status: str) -> IndexFileResult:
    """Build a zero-count result entry for a skipped or failed file."""
    return IndexFileResult(
        file_name=file_name,
        page_count=0,
        chunk_count=0,
        status=status,
    )
