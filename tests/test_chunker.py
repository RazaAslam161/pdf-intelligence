"""Tests for text chunking."""

import pytest

from src.chunker import chunk_pages
from src.models import PageText


def test_short_text_creates_one_chunk() -> None:
    """A page shorter than the chunk size should stay as one chunk."""
    pages = [_page("Short page text.")]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0].text == "Short page text."
    assert chunks[0].chunk_id == "doc-1-page-1-chunk-1"


def test_long_text_creates_multiple_chunks() -> None:
    pages = [_page("Sentence one. Sentence two. Sentence three. Sentence four.")]

    chunks = chunk_pages(pages, chunk_size=28, chunk_overlap=8)

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 28 for chunk in chunks)


def test_overlap_works() -> None:
    """Adjacent chunks should include the requested character overlap."""
    pages = [_page("abcdefghijklmnopqrstuvwxyz")]

    chunks = chunk_pages(pages, chunk_size=10, chunk_overlap=3)

    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text.startswith("hij")


def test_metadata_is_preserved() -> None:
    """Chunks should carry over the page's document, file, and page metadata."""
    pages = [
        PageText(
            document_id="doc-2",
            file_name="source.pdf",
            page_number=7,
            text="Metadata stays attached.",
        )
    ]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)

    assert chunks[0].document_id == "doc-2"
    assert chunks[0].file_name == "source.pdf"
    assert chunks[0].page_number == 7


def test_empty_page_text_is_skipped() -> None:
    """Whitespace-only page text should not create empty chunks."""
    pages = [_page("   \n\t  ")]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)

    assert chunks == []


def test_sentence_boundary_is_preferred_when_available() -> None:
    """Long text should split on a nearby sentence boundary when practical."""
    pages = [_page("First sentence. Second sentence continues with more words.")]

    chunks = chunk_pages(pages, chunk_size=26, chunk_overlap=5)

    assert chunks[0].text == "First sentence."


def test_invalid_chunk_settings_raise_clear_error() -> None:
    """chunk_overlap must be smaller than chunk_size."""
    pages = [_page("Some text.")]

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        chunk_pages(pages, chunk_size=100, chunk_overlap=100)


def _page(text: str) -> PageText:
    return PageText(
        document_id="doc-1",
        file_name="sample.pdf",
        page_number=1,
        text=text,
    )
