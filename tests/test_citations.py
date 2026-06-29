"""Tests for source citation formatting."""

from src.citations import build_source_citations
from src.models import RetrievedChunk, TextChunk


def test_build_source_citations_deduplicates_same_file_and_page() -> None:
    """Duplicate-looking sources from the same file/page should collapse."""
    citations = build_source_citations(
        [
            _retrieved("chunk-1", "handbook.pdf", 3, "First relevant passage."),
            _retrieved("chunk-2", "handbook.pdf", 3, "Second relevant passage."),
            _retrieved("chunk-3", "policy.pdf", 8, "Another document."),
        ]
    )

    assert citations == [
        {
            "label": "[Source 1] handbook.pdf, page 3",
            "file_name": "handbook.pdf",
            "page_number": 3,
            "preview": "First relevant passage.",
        },
        {
            "label": "[Source 2] policy.pdf, page 8",
            "file_name": "policy.pdf",
            "page_number": 8,
            "preview": "Another document.",
        },
    ]


def test_build_source_citations_truncates_preview() -> None:
    """Long previews should be compact and readable."""
    citations = build_source_citations(
        [_retrieved("chunk-1", "long.pdf", 1, "a " * 100)],
        preview_chars=20,
    )

    assert citations[0]["preview"].endswith("...")
    assert len(citations[0]["preview"]) <= 20


def test_build_source_citations_keeps_distinct_pages_from_same_file() -> None:
    """Different pages in the same PDF should remain separate citations."""
    citations = build_source_citations(
        [
            _retrieved("chunk-1", "guide.pdf", 1, "Page one."),
            _retrieved("chunk-2", "guide.pdf", 2, "Page two."),
        ]
    )

    assert [citation["label"] for citation in citations] == [
        "[Source 1] guide.pdf, page 1",
        "[Source 2] guide.pdf, page 2",
    ]


def _retrieved(
    chunk_id: str,
    file_name: str,
    page_number: int,
    text: str,
) -> RetrievedChunk:
    """Build a retrieved chunk fixture."""
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            file_name=file_name,
            page_number=page_number,
            text=text,
        )
    )
