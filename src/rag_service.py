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
        try:
            pages = load_pdf_pages(pdf_bytes, file_name)
            chunks = chunk_pages(
                pages,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            embeddings = self.embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            self.vector_store.add_chunks(chunks, embeddings)
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

        if not retrieved_chunks:
            return {"answer": NO_CONTEXT_ANSWER, "sources": []}

        prompt = build_grounded_prompt(clean_question, retrieved_chunks)
        answer = self._generate_answer(prompt)

        return {
            "answer": answer,
            "sources": build_sources(retrieved_chunks),
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
