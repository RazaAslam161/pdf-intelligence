"""Tests for Streamlit app helper logic."""

from dataclasses import dataclass

import pytest

from app import (
    aggregate_indexing_stats,
    build_chat_message,
    build_indexing_feedback,
    build_sidebar_status,
    can_ask_questions,
    can_index_documents,
    classify_user_error,
    format_source_preview,
    has_indexed_documents,
    is_supported_pdf_file,
    normalize_question,
)


@dataclass(frozen=True)
class FakeSettings:
    """Minimal settings for app helper tests."""

    openai_api_key: str | None = "configured"
    openai_base_url: str | None = None
    openai_chat_model: str = "chat-model"
    openai_embedding_model: str = "embedding-model"
    chroma_persist_dir: str = "./chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4


def test_build_sidebar_status_excludes_raw_config_and_api_key() -> None:
    """The sidebar status should not expose raw internal configuration."""
    summary = build_sidebar_status(
        FakeSettings(),
        indexed_chunks=12,
        session_chunks=5,
    )

    assert "openai_api_key" not in summary
    assert "Embedding model" not in summary
    assert "configured" not in summary.values()
    assert summary["API status"] == "Connected"
    assert summary["Indexed chunks"] == "12"
    assert summary["Session chunks"] == "5"


def test_aggregate_indexing_stats_counts_pages_and_chunks() -> None:
    """Indexing summaries should aggregate across uploaded PDFs."""
    stats = aggregate_indexing_stats(
        [
            {"file_name": "a.pdf", "page_count": 2, "chunk_count": 5},
            {"file_name": "b.pdf", "page_count": 3, "chunk_count": 7},
        ]
    )

    assert stats == {"files": 2, "pages": 5, "chunks": 12}


def test_has_indexed_documents_requires_chunks() -> None:
    """Question answering should only enable after searchable chunks exist."""
    assert not has_indexed_documents([])
    assert not has_indexed_documents([{"file_name": "a.pdf", "chunk_count": 0}])
    assert has_indexed_documents([{"file_name": "a.pdf", "chunk_count": 1}])


def test_can_index_documents_requires_files_and_api_key_only() -> None:
    """Indexing should not be blocked by chat model setup."""
    missing_model = FakeSettings(openai_chat_model="")

    assert can_index_documents(missing_model, uploaded_file_count=1)
    assert not can_index_documents(
        FakeSettings(openai_api_key=None),
        uploaded_file_count=1,
    )
    assert not can_index_documents(FakeSettings(), uploaded_file_count=0)


def test_can_index_documents_rejects_openrouter_key_without_base_url() -> None:
    """OpenRouter keys need the OpenRouter base URL before indexing."""
    settings = FakeSettings(openai_api_key="sk-or-v1-example")

    assert not can_index_documents(settings, uploaded_file_count=1)
    assert can_index_documents(
        FakeSettings(
            openai_api_key="sk-or-v1-example",
            openai_base_url="https://openrouter.ai/api/v1",
        ),
        uploaded_file_count=1,
    )


def test_can_ask_questions_requires_model_api_and_indexed_chunks() -> None:
    """Questions should enable after either session or persisted chunks exist."""
    assert can_ask_questions(FakeSettings(), session_chunks=0, vector_chunks=3)
    assert can_ask_questions(FakeSettings(), session_chunks=2, vector_chunks=0)
    assert not can_ask_questions(FakeSettings(), session_chunks=0, vector_chunks=0)
    assert not can_ask_questions(
        FakeSettings(openai_api_key=None),
        session_chunks=1,
        vector_chunks=0,
    )
    assert not can_ask_questions(
        FakeSettings(openai_chat_model=""),
        session_chunks=1,
        vector_chunks=0,
    )


def test_normalize_question_strips_text_and_rejects_empty() -> None:
    """Questions should be stripped and validated before asking RAG."""
    assert normalize_question("  What is covered?  ") == "What is covered?"

    with pytest.raises(ValueError, match="Enter a question before asking."):
        normalize_question("   ")


def test_build_chat_message_has_predictable_shape() -> None:
    """Chat history messages should have a stable format."""
    message = build_chat_message(
        "assistant",
        "Answer text",
        sources=[{"file_name": "a.pdf", "page_number": 1, "preview": "Preview"}],
    )

    assert message == {
        "role": "assistant",
        "content": "Answer text",
        "sources": [{"file_name": "a.pdf", "page_number": 1, "preview": "Preview"}],
    }


def test_format_source_preview_compacts_and_truncates_text() -> None:
    """Source previews should be compact enough for citation cards."""
    preview = format_source_preview("First line.\n\nSecond line with extra text.", 22)

    assert preview == "First line. Second..."


def test_is_supported_pdf_file_accepts_pdf_extension_only() -> None:
    """Unsupported file types should be detected before indexing."""
    assert is_supported_pdf_file("report.PDF")
    assert not is_supported_pdf_file("notes.txt")


def test_build_indexing_feedback_detects_scanned_pdf_case() -> None:
    """Zero chunks should produce scanned/no-text guidance."""
    feedback = build_indexing_feedback(
        [{"file_name": "scan.pdf", "page_count": 0, "chunk_count": 0}]
    )

    assert feedback["level"] == "warning"
    assert "No extractable text" in feedback["message"]
    assert "scanned PDF" in feedback["message"]


def test_classify_user_error_handles_api_errors() -> None:
    """API failures should get a user-friendly message."""
    message = classify_user_error(RuntimeError("OpenAI API rate limit reached"))

    assert message == (
        "The OpenAI request could not be completed. Check your API key, quota, "
        "model name, and network connection."
    )


def test_classify_user_error_checks_wrapped_causes() -> None:
    """Wrapped service errors should still expose useful root-cause guidance."""
    try:
        raise RuntimeError("Unable to read PDF 'broken.pdf': invalid or unreadable.")
    except RuntimeError as exc:
        wrapped = RuntimeError("Failed to index PDF 'broken.pdf'.")
        wrapped.__cause__ = exc

    assert classify_user_error(wrapped) == (
        "This PDF could not be read. Try a different or unprotected PDF."
    )
