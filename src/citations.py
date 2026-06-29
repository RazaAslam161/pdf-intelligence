"""Helpers for formatting source citations."""

from __future__ import annotations

from collections.abc import Sequence

from src.models import RetrievedChunk


def build_source_citations(
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    preview_chars: int = 220,
) -> list[dict[str, int | str]]:
    """Build deduplicated source citation dictionaries for UI display."""
    citations: list[dict[str, int | str]] = []
    seen: set[tuple[str, int]] = set()

    for retrieved in retrieved_chunks:
        chunk = retrieved.chunk
        key = (chunk.file_name, chunk.page_number)
        if key in seen:
            continue

        seen.add(key)
        citations.append(
            {
                "label": format_source_label(
                    len(citations) + 1,
                    chunk.file_name,
                    chunk.page_number,
                ),
                "file_name": chunk.file_name,
                "page_number": chunk.page_number,
                "preview": preview_text(chunk.text, preview_chars),
            }
        )

    return citations


def format_source_label(index: int, file_name: str, page_number: int) -> str:
    """Format a source label such as [Source 1] file.pdf, page 3."""
    return f"[Source {index}] {file_name}, page {page_number}"


def preview_text(text: str, max_chars: int) -> str:
    """Return a compact single-line context preview."""
    clean_text = " ".join(text.split())
    if len(clean_text) <= max_chars:
        return clean_text
    return f"{clean_text[: max_chars - 3].rstrip()}..."
