"""Shared data models for PDF pages, chunks, and retrieval results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page with source metadata."""

    document_id: str
    file_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    """A searchable chunk of PDF text with source metadata."""

    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    text: str

    @property
    def id(self) -> str:
        """Backward-compatible alias for chunk_id."""
        return self.chunk_id

    @property
    def source_name(self) -> str:
        """Backward-compatible alias for file_name."""
        return self.file_name


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by vector search."""

    chunk: TextChunk
    score: float | None = None


@dataclass(frozen=True)
class RagAnswer:
    """Generated answer and the chunks used to support it."""

    answer: str
    sources: list[RetrievedChunk]


PdfPage = PageText
