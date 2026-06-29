"""Tests for PDF text extraction."""

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.models import PageText
from src.pdf_loader import (
    PdfLoadError,
    extract_pdf_pages,
    generate_document_id,
    load_pdf_pages,
)


def test_generate_document_id_is_stable_and_content_based() -> None:
    """Document IDs should be stable for the same file name and bytes."""
    first = generate_document_id("report.pdf", b"same content")
    second = generate_document_id("report.pdf", b"same content")
    changed_content = generate_document_id("report.pdf", b"different content")
    changed_name = generate_document_id("other.pdf", b"same content")

    assert first == second
    assert first != changed_content
    assert first != changed_name


def test_load_pdf_pages_extracts_text_and_metadata() -> None:
    """PDF extraction should preserve file, document, and page metadata."""
    pdf_bytes = _build_pdf(["Hello PDF text"])
    document_id = generate_document_id("sample.pdf", pdf_bytes)

    pages = load_pdf_pages(pdf_bytes, "sample.pdf")

    assert pages == [
        PageText(
            document_id=document_id,
            file_name="sample.pdf",
            page_number=1,
            text="Hello PDF text",
        )
    ]


def test_load_pdf_pages_skips_empty_pages() -> None:
    pdf_bytes = _build_pdf([None, "Second page text"])

    pages = load_pdf_pages(pdf_bytes, "mixed.pdf")

    assert len(pages) == 1
    assert pages[0].page_number == 2
    assert pages[0].text == "Second page text"


def test_load_pdf_pages_raises_clear_error_for_invalid_pdf() -> None:
    """Invalid PDF bytes should raise a user-facing loader error."""
    with pytest.raises(PdfLoadError, match="Unable to read PDF 'broken.pdf'"):
        load_pdf_pages(b"not a real pdf", "broken.pdf")


def test_extract_pdf_pages_accepts_file_like_object() -> None:
    """The compatibility wrapper should accept Streamlit-like file objects."""
    pdf_bytes = _build_pdf(["Wrapped PDF text"])

    pages = extract_pdf_pages(BytesIO(pdf_bytes), "wrapped.pdf")

    assert len(pages) == 1
    assert pages[0].file_name == "wrapped.pdf"
    assert pages[0].page_number == 1
    assert pages[0].text == "Wrapped PDF text"


def _build_pdf(page_texts: list[str | None]) -> bytes:
    """Build an in-memory PDF where None means a blank page."""
    writer = PdfWriter()

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue

        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )

        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()
