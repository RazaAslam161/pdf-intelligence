"""FastAPI app exposing the RAG service over HTTP (M2).

Thin transport only: it adapts the typed RAGService boundary (RagAnswer /
IndexResult) to the JSON contract in ``schemas`` and never reaches into ``src``
internals beyond the public service. The ``src`` backbone is unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.citations import preview_text
from src.models import RetrievedChunk
from src.rag_service import IndexingLimitError, RAGService, RAGServiceError

from .deps import get_rag_service
from .schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IndexFileResult,
    IndexResponse,
    Source,
)

DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:3000"
PREVIEW_CHARS = 220


def _allowed_origins() -> list[str]:
    """Read allowed CORS origins (comma-separated) from the environment."""
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def to_sources(chunks: Sequence[RetrievedChunk]) -> list[Source]:
    """Map retrieved chunks to citation DTOs, preserving order.

    The list is 1:1 with the chunks (no dedup) so its positions line up with the
    inline [n] markers the model emits — sources[0] is [1], and so on. Each DTO
    keeps the relevance score so the client can show confidence.
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


app = FastAPI(
    title="PDF Intelligence API",
    version="0.1.0",
    description="Grounded question answering over uploaded PDFs with citations.",
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


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """Answer a question from indexed PDF context with grounded citations."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty.")
    try:
        answer = service.answer_question(question)
    except RAGServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return AskResponse(answer=answer.answer, sources=to_sources(answer.sources))


@app.post("/index", response_model=IndexResponse)
async def index(
    files: list[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service),
) -> IndexResponse:
    """Index one or more uploaded PDFs; each file succeeds or fails on its own."""
    results: list[IndexFileResult] = []
    for upload in files:
        file_name = upload.filename or "upload.pdf"
        if not file_name.lower().endswith(".pdf"):
            results.append(_failed_result(file_name, "unsupported"))
            continue

        data = await upload.read()
        try:
            result = service.index_pdf(data, file_name)
        except IndexingLimitError:
            results.append(_failed_result(file_name, "rejected"))
            continue
        except RAGServiceError:
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
