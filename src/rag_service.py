"""High-level RAG orchestration service."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from src.chunker import chunk_pages
from src.citations import build_source_citations
from src.embeddings import EmbeddingService
from src.models import RetrievedChunk, TextChunk
from src.pdf_loader import generate_document_id, load_pdf_pages
from src.vector_store import VectorStore

GROUNDING_INSTRUCTIONS = (
    "You are a careful PDF question-answering assistant. "
    "Use only the provided PDF context. "
    "If the answer is not present in the context, say you could not find the "
    "answer in the uploaded documents."
)
NO_CONTEXT_ANSWER = "I could not find the answer in the uploaded documents."


class RAGSettings(Protocol):
    """Settings fields required by the RAG service."""

    openai_api_key: str | None
    openai_base_url: str | None
    openai_chat_model: str
    openai_embedding_model: str
    chroma_persist_dir: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    retrieval_max_distance: float
    max_upload_mb: int
    max_pages_per_pdf: int
    max_chunks_per_run: int


class EmbeddingServiceLike(Protocol):
    """Embedding service interface used by RAGService."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for text strings."""


class VectorStoreLike(Protocol):
    """Vector store interface used by RAGService."""

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store embedded chunks."""

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Search chunks by embedding."""


class RAGServiceError(RuntimeError):
    """Raised when indexing or answering fails."""


class IndexingLimitError(ValueError):
    """Raised when an upload exceeds a configured indexing limit (P0-2)."""


class RAGService:
    """Main service for indexing PDFs and answering grounded questions."""

    def __init__(
        self,
        settings: RAGSettings,
        *,
        embedding_service: EmbeddingServiceLike | None = None,
        vector_store: VectorStoreLike | None = None,
        openai_client: Any | None = None,
    ) -> None:
        """Create a RAG service from settings and optional test doubles."""
        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService(settings)
        self.vector_store = vector_store or VectorStore(settings)
        self._openai_client = openai_client

    def index_pdf(self, pdf_bytes: bytes, file_name: str) -> dict[str, int | str]:
        """Extract, chunk, embed, and store one uploaded PDF."""
        # Enforce the size cap before parsing so an oversized upload never
        # reaches pypdf or the paid embedding API.
        validate_pdf_size(len(pdf_bytes), file_name, self.settings)
        try:
            pages = load_pdf_pages(pdf_bytes, file_name)
            validate_page_count(len(pages), file_name, self.settings)
            chunks = chunk_pages(
                pages,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            validate_chunk_count(len(chunks), file_name, self.settings)
            embeddings = self.embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            self.vector_store.add_chunks(chunks, embeddings)
        except IndexingLimitError:
            raise
        except Exception as exc:
            raise RAGServiceError(f"Failed to index PDF '{file_name}'.") from exc

        document_id = pages[0].document_id if pages else generate_document_id(
            file_name,
            pdf_bytes,
        )
        return {
            "file_name": file_name,
            "document_id": document_id,
            "page_count": len(pages),
            "chunk_count": len(chunks),
        }

    def answer_question(self, question: str) -> dict[str, Any]:
        """Answer a user question using retrieved PDF context."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question must not be empty.")

        try:
            query_embedding = self.embedding_service.embed_texts([clean_question])[0]
            retrieved_chunks = self.vector_store.search(
                query_embedding,
                top_k=self.settings.retrieval_top_k,
            )
        except Exception as exc:
            raise RAGServiceError("Failed to retrieve relevant PDF context.") from exc

        # Retrieval honesty (P0-1): Chroma always returns the top_k nearest
        # neighbors, however distant. Drop chunks past the relevance threshold so
        # an off-topic question yields the grounded no-context answer with no
        # citations, instead of confidently citing irrelevant passages.
        relevant_chunks = filter_relevant_chunks(
            retrieved_chunks,
            self.settings.retrieval_max_distance,
        )
        if not relevant_chunks:
            return {"answer": NO_CONTEXT_ANSWER, "sources": []}

        prompt = build_grounded_prompt(clean_question, relevant_chunks)
        answer = self._generate_answer(prompt)

        return {
            "answer": answer,
            "sources": build_sources(relevant_chunks),
        }

    def _generate_answer(self, prompt: str) -> str:
        """Generate an answer with the OpenAI Chat Completions API."""
        if not self.settings.openai_chat_model:
            raise RAGServiceError("OPENAI_CHAT_MODEL is missing.")

        if self._openai_client is None:
            self._openai_client = _create_openai_client(
                self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )

        try:
            response = self._openai_client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": GROUNDING_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise RAGServiceError("Failed to generate an answer with OpenAI.") from exc

        answer = str(response.choices[0].message.content or "").strip()
        if not answer:
            raise RAGServiceError("OpenAI returned an empty answer.")
        return answer


def build_grounded_prompt(
    question: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> str:
    """Build the grounded prompt sent to the answer model."""
    context_blocks = []
    for index, retrieved in enumerate(retrieved_chunks, start=1):
        chunk = retrieved.chunk
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}: {chunk.file_name}, page {chunk.page_number}]",
                    chunk.text,
                ]
            )
        )

    context = "\n\n".join(context_blocks)
    return (
        "Answer only from the provided PDF context.\n"
        "If the answer is not in the context, say: "
        '"I could not find the answer in the uploaded documents."\n\n'
        f"Question:\n{question.strip()}\n\n"
        f"PDF context:\n{context}"
    )


def build_sources(
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    preview_chars: int = 220,
) -> list[dict[str, int | str]]:
    """Build simple citation dictionaries for UI display."""
    return build_source_citations(retrieved_chunks, preview_chars=preview_chars)


def filter_relevant_chunks(
    retrieved_chunks: Sequence[RetrievedChunk],
    max_distance: float,
) -> list[RetrievedChunk]:
    """Keep only chunks within the cosine-distance relevance threshold (P0-1).

    Chunks without a score are kept (their relevance cannot be disproven). A
    ``max_distance`` of 0 or less disables filtering.
    """
    if max_distance <= 0:
        return list(retrieved_chunks)
    return [
        retrieved
        for retrieved in retrieved_chunks
        if retrieved.score is None or retrieved.score <= max_distance
    ]


def validate_pdf_size(num_bytes: int, file_name: str, settings: RAGSettings) -> None:
    """Reject an upload larger than the configured size cap (P0-2)."""
    max_mb = settings.max_upload_mb
    if max_mb > 0 and num_bytes > max_mb * 1024 * 1024:
        size_mb = num_bytes / (1024 * 1024)
        raise IndexingLimitError(
            f"'{file_name}' is {size_mb:.1f} MB, over the {max_mb} MB upload limit."
        )


def validate_page_count(page_count: int, file_name: str, settings: RAGSettings) -> None:
    """Reject a document with more pages than the configured cap (P0-2)."""
    max_pages = settings.max_pages_per_pdf
    if max_pages > 0 and page_count > max_pages:
        raise IndexingLimitError(
            f"'{file_name}' has {page_count} pages, over the {max_pages}-page limit."
        )


def validate_chunk_count(
    chunk_count: int,
    file_name: str,
    settings: RAGSettings,
) -> None:
    """Reject a document producing more chunks than the cap before embedding (P0-2)."""
    max_chunks = settings.max_chunks_per_run
    if max_chunks > 0 and chunk_count > max_chunks:
        raise IndexingLimitError(
            f"'{file_name}' produced {chunk_count} chunks, over the "
            f"{max_chunks}-chunk limit."
        )


def _create_openai_client(api_key: str | None, *, base_url: str | None = None) -> Any:
    """Create the OpenAI SDK client only when needed."""
    if not api_key:
        raise RAGServiceError("OPENAI_API_KEY is missing.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RAGServiceError(
            "The openai package is not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


RagService = RAGService
