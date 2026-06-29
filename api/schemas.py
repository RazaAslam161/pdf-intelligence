"""Pydantic request/response models for the JSON API.

Kept in sync with the frontend types in ``frontend/src/lib/types.ts``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A grounding citation for an answer."""

    id: str
    file_name: str
    page_number: int
    preview: str
    score: float | None = None


class AskRequest(BaseModel):
    """Question payload for POST /ask."""

    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    """Grounded answer plus its supporting sources."""

    answer: str
    sources: list[Source]


class IndexFileResult(BaseModel):
    """Per-file outcome of an indexing request."""

    file_name: str
    page_count: int
    chunk_count: int
    status: str


class IndexResponse(BaseModel):
    """Result of POST /index for one or more uploaded PDFs."""

    results: list[IndexFileResult]


class HealthResponse(BaseModel):
    """Liveness payload for GET /health."""

    status: str


class ConfigResponse(BaseModel):
    """Upload limits the client checks before sending bytes."""

    max_upload_mb: int
    max_pages_per_pdf: int
    max_files_per_run: int


class DocumentSummary(BaseModel):
    """One document currently held in the vector store."""

    document_id: str
    file_name: str
    page_count: int
    chunk_count: int


class StoreState(BaseModel):
    """Live state of the document store for the workspace view."""

    documents: list[DocumentSummary]
    total_chunks: int
    ready: bool
