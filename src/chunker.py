"""Text chunking utilities for extracted PDF pages."""

from collections.abc import Sequence

from src.models import PageText, TextChunk


def chunk_pages(
    pages: Sequence[PageText],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Split page text into overlapping chunks for retrieval.

    Chunks preserve source metadata and prefer paragraph or sentence boundaries
    when a clean boundary appears near the chunk-size limit.
    """
    _validate_chunk_settings(chunk_size, chunk_overlap)

    chunks: list[TextChunk] = []
    chunk_counts_by_page: dict[tuple[str, int], int] = {}

    for page in pages:
        text = _clean_text(page.text)
        if not text:
            continue

        page_key = (page.document_id, page.page_number)
        for chunk_text_value in _split_text(text, chunk_size, chunk_overlap):
            if not chunk_text_value:
                continue

            chunk_counts_by_page[page_key] = chunk_counts_by_page.get(page_key, 0) + 1
            chunk_number = chunk_counts_by_page[page_key]
            chunks.append(
                TextChunk(
                    chunk_id=(
                        f"{page.document_id}-page-{page.page_number}"
                        f"-chunk-{chunk_number}"
                    ),
                    document_id=page.document_id,
                    file_name=page.file_name,
                    page_number=page.page_number,
                    text=chunk_text_value,
                )
            )

    return chunks


def chunk_text(
    pages: Sequence[PageText],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Backward-compatible wrapper for chunk_pages."""
    return chunk_pages(pages, chunk_size, chunk_overlap)


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk sizing inputs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks no larger than chunk_size."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        max_end = min(start + chunk_size, len(text))
        end = (
            len(text)
            if max_end == len(text)
            else _find_split_end(text, start, max_end)
        )
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(0, end - chunk_overlap)
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start

    return chunks


def _find_split_end(text: str, start: int, max_end: int) -> int:
    """Find a preferred split point at or before max_end."""
    min_end = start + max(1, (max_end - start) // 2)

    for delimiter in ("\n\n", ". ", "? ", "! ", "\n", " "):
        index = text.rfind(delimiter, start, max_end + 1)
        if index == -1:
            continue

        end = index + (1 if delimiter in {". ", "? ", "! "} else len(delimiter))
        if end > min_end:
            return end

    return max_end


def _clean_text(text: str) -> str:
    """Normalize whitespace around extracted page text."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
