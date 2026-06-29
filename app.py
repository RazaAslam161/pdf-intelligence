"""Streamlit UI for the RAG PDF chatbot."""

from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import streamlit as st

from src.config import DEFAULT_OPENAI_CHAT_MODEL, Settings, get_settings
from src.rag_service import (
    NO_CONTEXT_ANSWER,
    IndexingLimitError,
    RAGService,
    RAGServiceError,
    build_sources,
)
from src.vector_store import VectorStore, VectorStoreError

DEBUG_MODE = os.getenv("APP_DEBUG", "").lower() in {"1", "true", "yes"}


class SettingsLike(Protocol):
    """Settings fields displayed by the Streamlit app."""

    openai_api_key: str | None
    openai_chat_model: str
    openai_embedding_model: str
    chroma_persist_dir: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    max_upload_mb: int
    max_files_per_run: int


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="PDF Intelligence Assistant",
        page_icon=":material/quick_reference_all:",
        layout="wide",
    )
    _inject_css()
    settings = get_settings()
    _init_session_state()
    vector_chunks = get_vector_chunk_count(settings)

    render_sidebar(settings, vector_chunks)
    render_header(settings, vector_chunks)

    upload_tab, ask_tab, sources_tab = st.tabs(
        ["Upload & Index", "Ask Questions", "Sources / Session Info"]
    )
    with upload_tab:
        render_upload_section(settings, vector_chunks)
    with ask_tab:
        render_chat_section(settings, vector_chunks)
    with sources_tab:
        render_session_info(settings, vector_chunks)


def aggregate_indexing_stats(
    summaries: list[dict[str, Any]],
) -> dict[str, int]:
    """Aggregate per-file indexing summaries."""
    return {
        "files": len(summaries),
        "pages": sum(int(summary.get("page_count", 0)) for summary in summaries),
        "chunks": sum(int(summary.get("chunk_count", 0)) for summary in summaries),
    }


def build_sidebar_status(
    settings: SettingsLike,
    *,
    indexed_chunks: int | None,
    session_chunks: int,
) -> dict[str, str]:
    """Build a compact status summary without exposing raw configuration."""
    api_status = "Connected" if settings.openai_api_key else "Not configured"
    model_status = settings.openai_chat_model or "Not configured"
    app_status = (
        "Ready"
        if settings.openai_api_key and settings.openai_chat_model
        else "Setup needed"
    )
    indexed_value = "Unavailable" if indexed_chunks is None else str(indexed_chunks)

    return {
        "App status": app_status,
        "API status": api_status,
        "Answer model": model_status,
        "Indexed chunks": indexed_value,
        "Session chunks": str(session_chunks),
    }


def can_index_documents(settings: SettingsLike, uploaded_file_count: int) -> bool:
    """Return whether uploaded files can be indexed now."""
    if not settings.openai_api_key or uploaded_file_count <= 0:
        return False
    # OpenRouter keys need the OpenRouter base URL to work.
    if _is_openrouter_key(settings.openai_api_key) and not getattr(
        settings, "openai_base_url", None
    ):
        return False
    return True


def _is_openrouter_key(api_key: str) -> bool:
    """Return whether an API key looks like an OpenRouter key."""
    return api_key.startswith("sk-or-")


def can_ask_questions(
    settings: SettingsLike,
    *,
    session_chunks: int,
    vector_chunks: int | None,
) -> bool:
    """Return whether the chat input should be enabled."""
    searchable_chunks = session_chunks > 0 or (vector_chunks or 0) > 0
    return bool(
        settings.openai_api_key
        and settings.openai_chat_model
        and searchable_chunks
    )


def build_indexing_feedback(summaries: list[dict[str, Any]]) -> dict[str, str]:
    """Build a user-facing indexing status message."""
    totals = aggregate_indexing_stats(summaries)
    if totals["chunks"] == 0:
        return {
            "level": "warning",
            "message": (
                "No extractable text was found. This PDF may be a scanned PDF "
                "or image-based document. Try a text-based PDF or add OCR later."
            ),
        }

    skip_statuses = {"unsupported", "too_large", "rejected"}
    zero_chunk_files = [
        str(summary.get("file_name", "a PDF"))
        for summary in summaries
        if summary.get("status") not in skip_statuses
        and int(summary.get("chunk_count", 0)) == 0
    ]
    if zero_chunk_files:
        return {
            "level": "warning",
            "message": (
                "Some files were indexed, but no extractable text was found in "
                f"{', '.join(zero_chunk_files)}. Those PDFs may be scanned or "
                "image-based."
            ),
        }

    return {
        "level": "success",
        "message": (
            f"Indexed {totals['files']} document(s): "
            f"{totals['pages']} page(s), {totals['chunks']} chunk(s)."
        ),
    }


def classify_user_error(exc: Exception) -> str:
    """Map common failures to concise user-facing messages."""
    messages = _exception_messages(exc)
    message = messages[0] if messages else str(exc)
    lower_message = " ".join(messages).lower()

    if "openai" in lower_message or "api" in lower_message:
        return (
            "The OpenAI request could not be completed. Check your API key, "
            "quota, model name, and network connection."
        )
    if (
        "invalid or unreadable" in lower_message
        or "unable to read pdf" in lower_message
    ):
        return "This PDF could not be read. Try a different or unprotected PDF."
    if "encrypted" in lower_message:
        return "This PDF is encrypted. Upload an unlocked PDF and try again."
    if "not supported" in lower_message or "unsupported" in lower_message:
        return "This file type is not supported. Upload PDF files only."
    if "no extractable text" in lower_message:
        return "This PDF may be scanned or image-based."
    return message


def has_indexed_documents(summaries: list[dict[str, Any]]) -> bool:
    """Return whether the current session has searchable indexed chunks."""
    return aggregate_indexing_stats(summaries)["chunks"] > 0


def is_supported_pdf_file(file_name: str) -> bool:
    """Return whether a user-uploaded file is a supported PDF."""
    return file_name.lower().endswith(".pdf")


def normalize_question(question: str) -> str:
    """Validate and normalize a user question."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Enter a question before asking.")
    return clean_question


def build_chat_message(
    role: str,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a stable chat-history message dictionary."""
    return {
        "role": role,
        "content": content,
        "sources": sources or [],
    }


def _esc(value: object) -> str:
    """HTML-escape a dynamic value before interpolating into raw HTML."""
    return html.escape(str(value))


def format_source_preview(text: str, max_chars: int = 220) -> str:
    """Return a compact preview for source citation cards."""
    clean_text = " ".join(str(text).split())
    if max_chars <= 0:
        return ""
    if len(clean_text) <= max_chars:
        return clean_text
    if max_chars <= 3:
        return clean_text[:max_chars]
    return f"{clean_text[: max_chars - 3].rstrip()}..."


def get_vector_chunk_count(settings: Settings) -> int | None:
    """Return stored vector count without creating raw UI errors."""
    persist_dir = Path(settings.chroma_persist_dir)
    if not persist_dir.exists():
        return 0

    try:
        return VectorStore(settings).count()
    except Exception:
        return None


def render_header(settings: Settings, vector_chunks: int | None) -> None:
    """Render the top hero section."""
    session_chunks = aggregate_indexing_stats(
        st.session_state["indexed_summaries"]
    )["chunks"]
    ready_for_indexing = bool(settings.openai_api_key)
    ready_for_questions = can_ask_questions(
        settings,
        session_chunks=session_chunks,
        vector_chunks=vector_chunks,
    )

    st.markdown(
        f"""
        <section class="pia-hero">
          <div>
            <h1>PDF Intelligence Assistant</h1>
            <p>
              Ask grounded questions across your uploaded documents with
              concise answers and page-level citations.
            </p>
          </div>
          <div class="pia-hero-status">
            <span class="pia-pill {'ok' if ready_for_indexing else 'warn'}">
              {'API connected' if ready_for_indexing else 'Setup required'}
            </span>
            <span class="pia-pill {'ok' if ready_for_questions else 'neutral'}">
              {'Ready to answer' if ready_for_questions else 'Index documents first'}
            </span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(settings: Settings, vector_chunks: int | None) -> None:
    """Render polished app status and controls in the sidebar."""
    session_chunks = aggregate_indexing_stats(
        st.session_state["indexed_summaries"]
    )["chunks"]
    status = build_sidebar_status(
        settings,
        indexed_chunks=vector_chunks,
        session_chunks=session_chunks,
    )

    with st.sidebar:
        st.markdown("### PDF Intelligence")
        st.markdown(
            _status_card_html(
                title=status["App status"],
                api_status=status["API status"],
                model=status["Answer model"],
                indexed_chunks=status["Indexed chunks"],
                session_chunks=status["Session chunks"],
            ),
            unsafe_allow_html=True,
        )

        st.markdown("### Workspace")
        clear_docs = st.button(
            "Clear indexed documents",
            type="secondary",
            use_container_width=True,
        )
        clear_chat = st.button(
            "Clear chat",
            type="secondary",
            use_container_width=True,
        )

        if clear_docs:
            _clear_vector_database(settings)
        if clear_chat:
            st.session_state["chat_messages"] = []
            st.success("Chat cleared.")


def render_upload_section(settings: Settings, vector_chunks: int | None) -> None:
    """Render upload and indexing workflow."""
    st.markdown("### Upload documents")
    st.caption(
        "Add text-based PDFs, index them into a local vector store, then ask "
        "questions from the next tab."
    )

    if not settings.openai_api_key:
        render_setup_warning(settings)

    if settings.openai_api_key and not settings.openai_chat_model:
        render_model_warning()

    uploaded_files = st.file_uploader(
        "PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    upload_count = len(uploaded_files or [])

    col_a, col_b = st.columns([1, 2], vertical_alignment="center")
    with col_a:
        index_clicked = st.button(
            "Index documents",
            type="primary",
            disabled=not can_index_documents(settings, upload_count),
            use_container_width=True,
        )
    with col_b:
        _render_upload_empty_state(settings, upload_count)

    if index_clicked:
        _index_uploaded_files(settings, uploaded_files or [])

    summaries = st.session_state["indexed_summaries"]
    if summaries:
        _render_indexing_result_cards(summaries)
    elif vector_chunks:
        st.info(
            "There are indexed chunks in the local vector database. You can ask "
            "questions from the Ask Questions tab or clear the database from "
            "the sidebar."
        )


def render_chat_section(settings: Settings, vector_chunks: int | None) -> None:
    """Render the question-answering chat UI."""
    summaries = st.session_state["indexed_summaries"]
    session_chunks = aggregate_indexing_stats(summaries)["chunks"]
    chat_ready = can_ask_questions(
        settings,
        session_chunks=session_chunks,
        vector_chunks=vector_chunks,
    )

    st.markdown("### Ask grounded questions")
    st.caption(
        "Answers are generated only from indexed PDF context and include "
        "source citations when relevant."
    )

    if not settings.openai_api_key:
        _render_empty_state(
            "Setup required",
            "Add `OPENAI_API_KEY` to `.env`, restart the app, and index PDFs.",
        )
    elif not settings.openai_chat_model:
        _render_empty_state(
            "Answer model missing",
            "Set `OPENAI_CHAT_MODEL` in `.env` before asking questions.",
        )
    elif not chat_ready:
        _render_empty_state(
            "No indexed documents yet",
            "Upload and index at least one PDF before asking a question.",
        )

    _render_chat_history()

    question = st.chat_input(
        "Ask a question about your indexed PDFs",
        disabled=not chat_ready,
    )
    if question is not None:
        _answer_chat_question(settings, question)


def render_session_info(settings: Settings, vector_chunks: int | None) -> None:
    """Render source and session information without raw backend objects."""
    summaries = st.session_state["indexed_summaries"]
    totals = aggregate_indexing_stats(summaries)

    st.markdown("### Session overview")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Files indexed", totals["files"])
    col_b.metric("Pages read", totals["pages"])
    col_c.metric("Session chunks", totals["chunks"])
    col_d.metric(
        "Stored chunks",
        "Unavailable" if vector_chunks is None else vector_chunks,
    )

    last_indexed_at = st.session_state.get("last_indexed_at")
    if last_indexed_at:
        st.caption(f"Last indexing run: {last_indexed_at}")

    if summaries:
        st.markdown("#### Indexed documents")
        for summary in summaries:
            _render_indexed_document_card(summary)
    else:
        _render_empty_state(
            "No session documents",
            "Indexed document summaries will appear here after a successful run.",
        )

    sources = _latest_sources()
    if sources:
        st.markdown("#### Sources from latest answer")
        render_sources(sources)


def render_setup_warning(settings: Settings) -> None:
    """Render a polished setup card when OpenAI configuration is missing."""
    st.markdown(
        """
        <div class="pia-card setup-card">
          <div class="pia-card-title">Connect OpenAI to start indexing</div>
          <p>
            This assistant needs an OpenAI API key to create embeddings and
            generate answers. Secrets stay in your local <code>.env</code> file.
          </p>
          <ol>
            <li>Copy <code>.env.example</code> to <code>.env</code>.</li>
            <li>Add your OpenAI API key to <code>OPENAI_API_KEY</code>.</li>
            <li>Restart Streamlit after saving the file.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "\n".join(
            [
                "cp .env.example .env",
                "",
                "OPENAI_API_KEY=",
                (
                    "OPENAI_CHAT_MODEL="
                    f"{settings.openai_chat_model or DEFAULT_OPENAI_CHAT_MODEL}"
                ),
                "OPENAI_EMBEDDING_MODEL=text-embedding-3-small",
                "CHROMA_PERSIST_DIR=./chroma_db",
            ]
        ),
        language="dotenv",
    )


def render_model_warning() -> None:
    """Render a clear warning for missing answer model settings."""
    st.warning(
        "OPENAI_CHAT_MODEL is not configured. Set it in `.env`, or use the "
        f"default `{DEFAULT_OPENAI_CHAT_MODEL}` from `src.config`."
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render citation sources as compact expandable cards."""
    if not sources:
        return

    for index, source in enumerate(sources, start=1):
        label = str(source.get("label") or f"Source {index}")
        file_name = str(source.get("file_name") or "Unknown PDF")
        page_number = source.get("page_number", "Unknown")
        preview = format_source_preview(str(source.get("preview", "")), 260)

        with st.expander(label, expanded=False):
            col_a, col_b = st.columns([2, 1])
            col_a.markdown(f"**File:** {file_name}")
            col_b.markdown(f"**Page:** {page_number}")
            st.markdown(preview or "_No preview available._")


def _render_chat_history() -> None:
    """Render all chat messages stored in the current session."""
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_answer_text(str(message["content"]))
                render_sources(list(message.get("sources", [])))
            else:
                st.markdown(str(message["content"]))


def _answer_chat_question(settings: Settings, raw_question: str) -> None:
    """Answer one user question and append the exchange to chat history."""
    try:
        question = normalize_question(raw_question)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.session_state["chat_messages"].append(build_chat_message("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching indexed PDFs..."):
            try:
                result = RAGService(settings).answer_question(question)
            except (RAGServiceError, ValueError) as exc:
                _show_error("I could not answer that question.", exc)
                return
            except Exception as exc:
                _show_error("Something went wrong while answering.", exc)
                return

        answer = result.answer.strip() or NO_CONTEXT_ANSWER
        sources = build_sources(result.sources)
        st.session_state["chat_messages"].append(
            build_chat_message("assistant", answer, sources=sources)
        )
        _render_answer_text(answer)
        render_sources(sources)


def _rejected_summary(file_name: str, status: str) -> dict[str, Any]:
    """Build a zero-chunk summary entry for a skipped or rejected file."""
    return {
        "file_name": file_name,
        "page_count": 0,
        "chunk_count": 0,
        "status": status,
    }


def _index_uploaded_files(settings: Settings, uploaded_files: list[Any]) -> None:
    """Index uploaded PDFs and render progress."""
    if not can_index_documents(settings, len(uploaded_files)):
        st.warning("Add PDFs and configure `OPENAI_API_KEY` before indexing.")
        return

    files_to_index = list(uploaded_files)
    if (
        settings.max_files_per_run > 0
        and len(files_to_index) > settings.max_files_per_run
    ):
        st.warning(
            f"You added {len(files_to_index)} files; indexing the first "
            f"{settings.max_files_per_run} (the per-run limit)."
        )
        files_to_index = files_to_index[: settings.max_files_per_run]

    progress_bar = st.progress(0, text="Preparing documents...")
    status = st.empty()
    summaries: list[dict[str, Any]] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        rag_service = RAGService(settings)
        total_files = len(files_to_index)
        for index, uploaded_file in enumerate(files_to_index, start=1):
            file_name = str(uploaded_file.name)
            if not is_supported_pdf_file(file_name):
                summaries.append(_rejected_summary(file_name, "unsupported"))
                continue

            size_bytes = getattr(uploaded_file, "size", None)
            if size_bytes is None:
                size_bytes = len(uploaded_file.getvalue())
            if settings.max_upload_mb > 0 and size_bytes > max_bytes:
                summaries.append(_rejected_summary(file_name, "too_large"))
                st.warning(
                    f"{file_name} is over the {settings.max_upload_mb} MB upload "
                    "limit and was skipped."
                )
                continue

            status.info(f"Indexing {file_name}")
            try:
                result = rag_service.index_pdf(uploaded_file.getvalue(), file_name)
            except IndexingLimitError as exc:
                summaries.append(_rejected_summary(file_name, "rejected"))
                st.warning(str(exc))
                continue
            summaries.append(
                {
                    "file_name": result.file_name,
                    "document_id": result.document_id,
                    "page_count": result.page_count,
                    "chunk_count": result.chunk_count,
                    "status": "indexed",
                }
            )
            progress_bar.progress(
                index / total_files,
                text=f"Indexed {index} of {total_files} document(s)",
            )
    except (RAGServiceError, VectorStoreError, ValueError) as exc:
        _show_error("Indexing could not be completed.", exc)
        return
    except Exception as exc:
        _show_error("An unexpected indexing error occurred.", exc)
        return

    st.session_state["indexed_summaries"] = summaries
    st.session_state["last_indexed_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    status.success("Indexing complete.")

    feedback = build_indexing_feedback(summaries)
    if feedback["level"] == "warning":
        st.warning(feedback["message"])
    else:
        st.success(feedback["message"])


def _render_answer_text(answer: str) -> None:
    """Render an answer, highlighting the no-context fallback."""
    if answer.strip() == NO_CONTEXT_ANSWER:
        st.warning(
            "I could not find the answer in the indexed documents. Try asking "
            "about content that appears in the uploaded PDFs."
        )
        return
    st.markdown(answer)


def _render_indexing_result_cards(summaries: list[dict[str, Any]]) -> None:
    """Render indexing metrics in polished cards."""
    totals = aggregate_indexing_stats(summaries)
    st.markdown("#### Indexing summary")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Documents", totals["files"])
    col_b.metric("Pages", totals["pages"])
    col_c.metric("Chunks", totals["chunks"])

    for summary in summaries:
        _render_indexed_document_card(summary)


def _render_indexed_document_card(summary: dict[str, Any]) -> None:
    """Render one indexed document summary without dumping raw data."""
    file_name = str(summary.get("file_name", "Document"))
    pages = int(summary.get("page_count", 0))
    chunks = int(summary.get("chunk_count", 0))
    status = str(summary.get("status", "indexed"))
    status_label = "Indexed" if chunks > 0 else "Needs review"
    if status == "unsupported":
        status_label = "Unsupported"
    elif status == "too_large":
        status_label = "Too large"
    elif status == "rejected":
        status_label = "Rejected"

    st.markdown(
        f"""
        <div class="pia-card document-card">
          <div>
            <div class="pia-card-title">{_esc(file_name)}</div>
            <div class="pia-muted">
              {pages} page(s) processed · {chunks} searchable chunk(s)
            </div>
          </div>
          <span class="pia-pill {'ok' if chunks > 0 else 'warn'}">
            {_esc(status_label)}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if status not in {"unsupported", "too_large", "rejected"} and (
        pages == 0 or chunks == 0
    ):
        st.warning("This PDF may be scanned or image-based.")


def _render_upload_empty_state(settings: Settings, upload_count: int) -> None:
    """Render contextual upload guidance."""
    if not settings.openai_api_key:
        st.caption("Indexing is disabled until OpenAI is connected.")
    elif upload_count == 0:
        st.caption("Choose one or more PDFs to enable indexing.")
    else:
        st.caption(f"{upload_count} PDF(s) ready to index.")


def _render_empty_state(title: str, body: str) -> None:
    """Render a small empty-state card."""
    st.markdown(
        f"""
        <div class="pia-card empty-state">
          <div class="pia-card-title">{_esc(title)}</div>
          <p>{_esc(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_vector_database(settings: Settings) -> None:
    """Clear the local Chroma vector database."""
    try:
        vector_store = VectorStore(settings)
        vector_store.clear()
    except VectorStoreError as exc:
        _show_error("Could not clear indexed documents.", exc)
        return
    except Exception as exc:
        _show_error("Unexpected error while clearing indexed documents.", exc)
        return

    st.session_state["indexed_summaries"] = []
    st.session_state["chat_messages"] = []
    st.session_state["last_indexed_at"] = None
    st.success("Indexed documents cleared.")


def _show_error(message: str, exc: Exception) -> None:
    """Show a user-safe error with technical details hidden by default."""
    st.error(f"{message} {classify_user_error(exc)}")
    with st.expander("Developer details"):
        st.code(_technical_error_details(exc), language="text")
        if DEBUG_MODE:
            st.exception(exc)


def _technical_error_details(exc: BaseException) -> str:
    """Return a compact exception cause chain for developer details."""
    messages = _exception_messages(exc)
    if not messages:
        return exc.__class__.__name__
    return "\nCaused by: ".join(messages)


def _exception_messages(exc: BaseException) -> list[str]:
    """Return messages from an exception and its direct cause chain."""
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        message = str(current).strip() or current.__class__.__name__
        messages.append(message)
        current = current.__cause__
    return messages


def _latest_sources() -> list[dict[str, Any]]:
    """Return sources from the latest assistant message that has citations."""
    for message in reversed(st.session_state.get("chat_messages", [])):
        if message.get("role") == "assistant" and message.get("sources"):
            return list(message["sources"])
    return []


def _init_session_state() -> None:
    """Initialize Streamlit session state used by the app."""
    st.session_state.setdefault("indexed_summaries", [])
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("last_indexed_at", None)


def _status_card_html(
    *,
    title: str,
    api_status: str,
    model: str,
    indexed_chunks: str,
    session_chunks: str,
) -> str:
    """Build sidebar status card markup."""
    return f"""
    <div class="pia-card sidebar-card">
      <div class="pia-card-title">{_esc(title)}</div>
      <div class="status-row">
        <span>API</span><strong>{_esc(api_status)}</strong>
      </div>
      <div class="status-row">
        <span>Answer model</span><strong>{_esc(model)}</strong>
      </div>
      <div class="status-row">
        <span>Stored chunks</span><strong>{_esc(indexed_chunks)}</strong>
      </div>
      <div class="status-row">
        <span>This session</span><strong>{_esc(session_chunks)}</strong>
      </div>
    </div>
    """


def _inject_css() -> None:
    """Inject a small Streamlit design system."""
    st.markdown(
        """
        <style>
        .block-container {
          padding-top: 2rem;
          max-width: 1180px;
        }

        .pia-hero {
          display: flex;
          justify-content: space-between;
          gap: 2rem;
          align-items: flex-end;
          padding: 2rem;
          margin-bottom: 1.4rem;
          border: 1px solid rgba(128, 128, 128, 0.18);
          border-radius: 18px;
          background:
            radial-gradient(
              circle at top left,
              color-mix(in srgb, var(--primary-color) 18%, transparent),
              transparent 36%
            ),
            var(--secondary-background-color);
        }

        .pia-hero h1 {
          margin: 0 0 0.5rem;
          font-size: 2.4rem;
          line-height: 1.08;
          letter-spacing: 0;
        }

        .pia-hero p {
          margin: 0;
          max-width: 680px;
          color: color-mix(in srgb, var(--text-color) 72%, transparent);
          font-size: 1.02rem;
        }

        .pia-hero-status {
          display: flex;
          flex-wrap: wrap;
          justify-content: flex-end;
          gap: 0.5rem;
          min-width: 220px;
        }

        .pia-card {
          border: 1px solid rgba(128, 128, 128, 0.18);
          border-radius: 14px;
          background: var(--secondary-background-color);
          padding: 1rem;
          margin: 0.5rem 0 1rem;
        }

        .sidebar-card {
          padding: 1rem;
        }

        .document-card {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          align-items: center;
        }

        .empty-state {
          border-style: dashed;
        }

        .setup-card {
          border-color: color-mix(in srgb, var(--primary-color) 45%, transparent);
        }

        .pia-card-title {
          font-weight: 700;
          font-size: 1rem;
          margin-bottom: 0.35rem;
        }

        .pia-muted,
        .pia-card p,
        .pia-card li,
        .status-row span {
          color: color-mix(in srgb, var(--text-color) 68%, transparent);
        }

        .pia-pill {
          display: inline-flex;
          align-items: center;
          border: 1px solid rgba(128, 128, 128, 0.22);
          border-radius: 999px;
          padding: 0.25rem 0.65rem;
          font-size: 0.82rem;
          font-weight: 650;
          white-space: nowrap;
        }

        .pia-pill.ok {
          color: #087f5b;
          background: rgba(8, 127, 91, 0.12);
          border-color: rgba(8, 127, 91, 0.24);
        }

        .pia-pill.warn {
          color: #b7791f;
          background: rgba(183, 121, 31, 0.12);
          border-color: rgba(183, 121, 31, 0.24);
        }

        .pia-pill.neutral {
          color: color-mix(in srgb, var(--text-color) 78%, transparent);
          background: rgba(128, 128, 128, 0.10);
        }

        .status-row {
          display: flex;
          justify-content: space-between;
          gap: 0.75rem;
          padding: 0.5rem 0;
          border-top: 1px solid rgba(128, 128, 128, 0.14);
          font-size: 0.9rem;
        }

        .status-row:first-of-type {
          border-top: 0;
        }

        @media (max-width: 760px) {
          .pia-hero {
            display: block;
            padding: 1.25rem;
          }

          .pia-hero h1 {
            font-size: 2rem;
          }

          .pia-hero-status {
            justify-content: flex-start;
            margin-top: 1rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
