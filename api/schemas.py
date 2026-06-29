"""Pydantic request/response models — the public JSON contract.

These intentionally mirror the frontend types in ``frontend/src/lib/types.ts`` so
the React client and the API never drift.
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
