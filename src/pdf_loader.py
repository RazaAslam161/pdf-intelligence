"""PDF text extraction utilities."""

import re
from hashlib import sha256
from io import BytesIO
from typing import BinaryIO

from pypdf import PdfReader

from src.models import PageText


class PdfLoadError(ValueError):
    """Raised when a PDF cannot be read or processed."""


def generate_document_id(file_name: str, pdf_bytes: bytes) -> str:
    """Generate a stable ID from a file name and PDF bytes."""
    safe_name = _slugify(file_name)
    digest = sha256()
    digest.update(file_name.strip().lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(pdf_bytes)
    return f"{safe_name}-{digest.hexdigest()[:16]}"


def load_pdf_pages(pdf_bytes: bytes, file_name: str) -> list[PageText]:
    """Extract non-empty page text from PDF bytes.

    Page numbers are 1-based and empty pages are skipped.
    """
    if not pdf_bytes:
        raise PdfLoadError(f"Unable to read PDF '{file_name}': the file is empty.")

    reader = _read_pdf(pdf_bytes, file_name)
    document_id = generate_document_id(file_name, pdf_bytes)
    pages: list[PageText] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception as exc:
            raise PdfLoadError(
                f"Unable to extract text from PDF '{file_name}' on page {index}."
            ) from exc

        if not text:
            continue

        pages.append(
            PageText(
                document_id=document_id,
                file_name=file_name,
                page_number=index,
                text=text,
            )
        )

    return pages


def extract_pdf_pages(file: BinaryIO | bytes, source_name: str) -> list[PageText]:
    """Compatibility wrapper for older callers using a file-like object."""
    pdf_bytes = file if isinstance(file, bytes) else file.read()
    return load_pdf_pages(pdf_bytes, source_name)


def _read_pdf(pdf_bytes: bytes, file_name: str) -> PdfReader:
    """Create a PdfReader with clear user-facing errors."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise PdfLoadError(
            f"Unable to read PDF '{file_name}': the file is invalid or unreadable."
        ) from exc

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise PdfLoadError(
                f"Unable to read PDF '{file_name}': the file is encrypted."
            ) from exc

        if decrypted == 0:
            raise PdfLoadError(
                f"Unable to read PDF '{file_name}': the file is encrypted."
            )

    return reader


def _clean_text(text: str) -> str:
    """Normalize extracted page text enough to detect empty pages."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _slugify(file_name: str) -> str:
    """Return a compact identifier-safe file-name prefix."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", file_name.strip().lower()).strip("-")
    return slug or "document"
