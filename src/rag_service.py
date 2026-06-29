"""High-level RAG orchestration service."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from time import perf_counter
from typing import Any, Protocol

from src.chunker import chunk_pages
from src.citations import build_source_citations
from src.embeddings import EmbeddingService
from src.models import IndexResult, RagAnswer, RetrievedChunk, TextChunk
from src.observability import get_logger
from src.pdf_loader import generate_document_id, load_pdf_pages
from src.vector_store import DEFAULT_TENANT, VectorStore

logger = get_logger("rag_service")

GROUNDING_INSTRUCTIONS = (
    "You are a careful PDF question-answering assistant. "
    "Use only the provided PDF context. "
    "If the answer is not present in the context, say you could not find the "
    "answer in the uploaded documents. "
    "When a sentence is supported by the context, cite the source inline using a "
    "single bracketed number matching the numbered sources, like [1]. For multiple "
    "sources use separate brackets, like [1][2]. Do not write the word 'Source' and "
    "do not put more than one number in a bracket (write [1][2], never [1, 2]). "
    "Only cite the numbered sources provided. "
    "Write in plain prose. Do not use any markdown formatting: no asterisks for "
    "bold or italic, no underscores, no headings, and no bullet or numbered-list "
    "syntax."
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
    request_timeout: float
    max_retries: int
    max_output_tokens: int
    embed_batch_size: int
    embed_concurrency: int


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
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        """Store embedded chunks for a tenant."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> list[RetrievedChunk]:
        """Search a tenant's chunks by embedding."""


class RAGServiceError(RuntimeError):
    """Raised when indexing or answering fails."""


class IndexingLimitError(ValueError):
    """Raised when an upload exceeds a configured indexing limit."""


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

    def index_pdf(
        self,
        pdf_bytes: bytes,
        file_name: str,
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> IndexResult:
        """Extract, chunk, embed, and store one uploaded PDF for a tenant."""
        request_id = _new_request_id()
        # Enforce the size cap before parsing so an oversized upload never
        # reaches pypdf or the paid embedding API.
        validate_pdf_size(len(pdf_bytes), file_name, self.settings)
        embed_ms = 0.0
        try:
            pages = load_pdf_pages(pdf_bytes, file_name)
            validate_page_count(len(pages), file_name, self.settings)
            chunks = chunk_pages(
                pages,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            validate_chunk_count(len(chunks), file_name, self.settings)
            embed_start = perf_counter()
            embeddings = self.embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            embed_ms = (perf_counter() - embed_start) * 1000
            self.vector_store.add_chunks(chunks, embeddings, tenant_id=tenant_id)
        except IndexingLimitError:
            raise
        except Exception as exc:
            # Capture the root cause to a durable sink before sanitizing it.
            logger.exception(
                "index_failed request_id=%s file=%s", request_id, file_name
            )
            raise RAGServiceError(f"Failed to index PDF '{file_name}'.") from exc

        document_id = pages[0].document_id if pages else generate_document_id(
            file_name,
            pdf_bytes,
        )
        logger.info(
            "index request_id=%s file=%s pages=%d chunks=%d embed_ms=%.1f",
            request_id,
            file_name,
            len(pages),
            len(chunks),
            embed_ms,
        )
        return IndexResult(
            file_name=file_name,
            document_id=document_id,
            page_count=len(pages),
            chunk_count=len(chunks),
        )

    def answer_question(
        self,
        question: str,
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> RagAnswer:
        """Answer a user question using the tenant's retrieved PDF context."""
        request_id = _new_request_id()
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question must not be empty.")

        relevant_chunks, embed_ms, search_ms = self._retrieve_relevant(
            clean_question, tenant_id, request_id
        )
        if not relevant_chunks:
            logger.info(
                "answer request_id=%s embed_ms=%.1f search_ms=%.1f llm_ms=0.0 "
                "total_tokens=0 sources=0 grounded=false",
                request_id,
                embed_ms,
                search_ms,
            )
            return RagAnswer(answer=NO_CONTEXT_ANSWER, sources=[])

        prompt = build_grounded_prompt(clean_question, relevant_chunks)
        llm_start = perf_counter()
        answer, usage = self._generate_answer(prompt)
        llm_ms = (perf_counter() - llm_start) * 1000

        logger.info(
            "answer request_id=%s embed_ms=%.1f search_ms=%.1f llm_ms=%.1f "
            "prompt_tokens=%s completion_tokens=%s total_tokens=%s sources=%d "
            "grounded=true",
            request_id,
            embed_ms,
            search_ms,
            llm_ms,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            len(relevant_chunks),
        )
        return RagAnswer(answer=answer, sources=list(relevant_chunks))

    def stream_answer(
        self,
        question: str,
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> Iterator[tuple[str, Any]]:
        """Stream the answer as events, mirroring answer_question's grounding.

        Yields ("token", delta) for each streamed token, then exactly one
        ("sources", list[RetrievedChunk]) event once generation completes.
        """
        request_id = _new_request_id()
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question must not be empty.")

        relevant_chunks, embed_ms, search_ms = self._retrieve_relevant(
            clean_question, tenant_id, request_id
        )
        if not relevant_chunks:
            yield ("token", NO_CONTEXT_ANSWER)
            yield ("sources", [])
            logger.info(
                "answer request_id=%s embed_ms=%.1f search_ms=%.1f llm_ms=0.0 "
                "sources=0 grounded=false stream=true",
                request_id,
                embed_ms,
                search_ms,
            )
            return

        prompt = build_grounded_prompt(clean_question, relevant_chunks)
        llm_start = perf_counter()
        for delta in self._stream_generation(prompt):
            yield ("token", delta)
        llm_ms = (perf_counter() - llm_start) * 1000
        yield ("sources", list(relevant_chunks))
        logger.info(
            "answer request_id=%s embed_ms=%.1f search_ms=%.1f llm_ms=%.1f "
            "sources=%d grounded=true stream=true",
            request_id,
            embed_ms,
            search_ms,
            llm_ms,
            len(relevant_chunks),
        )

    def _retrieve_relevant(
        self,
        clean_question: str,
        tenant_id: str,
        request_id: str,
    ) -> tuple[list[RetrievedChunk], float, float]:
        """Embed the question, search the tenant's store, and drop chunks past the
        relevance threshold. Returns (relevant_chunks, embed_ms, search_ms)."""
        try:
            embed_start = perf_counter()
            query_embedding = self.embedding_service.embed_texts([clean_question])[0]
            embed_ms = (perf_counter() - embed_start) * 1000
            search_start = perf_counter()
            retrieved_chunks = self.vector_store.search(
                query_embedding,
                top_k=self.settings.retrieval_top_k,
                tenant_id=tenant_id,
            )
            search_ms = (perf_counter() - search_start) * 1000
        except Exception as exc:
            logger.exception("answer_failed request_id=%s stage=retrieve", request_id)
            raise RAGServiceError("Failed to retrieve relevant PDF context.") from exc

        # Drop chunks past the threshold so an off-topic question returns the
        # no-context answer instead of citing irrelevant passages.
        relevant_chunks = filter_relevant_chunks(
            retrieved_chunks,
            self.settings.retrieval_max_distance,
        )
        return relevant_chunks, embed_ms, search_ms

    def _ensure_client(self) -> Any:
        """Lazily create the OpenAI client (with timeout/retries) on first use."""
        if not self.settings.openai_chat_model:
            raise RAGServiceError("OPENAI_CHAT_MODEL is missing.")
        if self._openai_client is None:
            self._openai_client = _create_openai_client(
                self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                timeout=getattr(self.settings, "request_timeout", None),
                max_retries=getattr(self.settings, "max_retries", None),
            )
        return self._openai_client

    def _chat_kwargs(self, prompt: str) -> dict[str, Any]:
        """Build the chat.completions.create kwargs shared by both answer paths."""
        kwargs: dict[str, Any] = {
            "model": self.settings.openai_chat_model,
            "messages": [
                {"role": "system", "content": GROUNDING_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        }
        max_output_tokens = getattr(self.settings, "max_output_tokens", None)
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens
        return kwargs

    def _generate_answer(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        """Generate a full answer with the OpenAI Chat Completions API."""
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(**self._chat_kwargs(prompt))
        except Exception as exc:
            logger.exception("generation_failed")
            raise RAGServiceError("Failed to generate an answer with OpenAI.") from exc

        answer = str(response.choices[0].message.content or "").strip()
        if not answer:
            raise RAGServiceError("OpenAI returned an empty answer.")
        return answer, _extract_usage(response)

    def _stream_generation(self, prompt: str) -> Iterator[str]:
        """Yield answer text deltas from a streaming chat completion."""
        client = self._ensure_client()
        try:
            stream = client.chat.completions.create(
                stream=True, **self._chat_kwargs(prompt)
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0].delta, "content", None)
                if delta:
                    yield delta
        except Exception as exc:
            logger.exception("generation_failed stream=true")
            raise RAGServiceError("Failed to generate an answer with OpenAI.") from exc


def _new_request_id() -> str:
    """Return a short correlation id for one logical request."""
    return uuid.uuid4().hex[:12]


def _extract_usage(response: Any) -> dict[str, int | None]:
    """Pull token counts from an OpenAI response.usage, tolerating absence."""
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


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
        '"I could not find the answer in the uploaded documents."\n'
        "Cite each supported sentence inline with its source's bracketed number "
        "(for example [1] for Source 1), using the numbers shown below.\n\n"
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
    """Keep only chunks within the cosine-distance relevance threshold.

    Chunks without a score are kept. A ``max_distance`` of 0 or less disables
    filtering.
    """
    if max_distance <= 0:
        return list(retrieved_chunks)
    return [
        retrieved
        for retrieved in retrieved_chunks
        if retrieved.score is None or retrieved.score <= max_distance
    ]


def validate_pdf_size(num_bytes: int, file_name: str, settings: RAGSettings) -> None:
    """Reject an upload larger than the configured size cap."""
    max_mb = settings.max_upload_mb
    if max_mb > 0 and num_bytes > max_mb * 1024 * 1024:
        size_mb = num_bytes / (1024 * 1024)
        raise IndexingLimitError(
            f"'{file_name}' is {size_mb:.1f} MB, over the {max_mb} MB upload limit."
        )


def validate_page_count(page_count: int, file_name: str, settings: RAGSettings) -> None:
    """Reject a document with more pages than the configured cap."""
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
    """Reject a document producing more chunks than the cap before embedding."""
    max_chunks = settings.max_chunks_per_run
    if max_chunks > 0 and chunk_count > max_chunks:
        raise IndexingLimitError(
            f"'{file_name}' produced {chunk_count} chunks, over the "
            f"{max_chunks}-chunk limit."
        )


def _create_openai_client(
    api_key: str | None,
    *,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Create the OpenAI SDK client with a bounded timeout and retry budget."""
    if not api_key:
        raise RAGServiceError("OPENAI_API_KEY is missing.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RAGServiceError(
            "The openai package is not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return OpenAI(**kwargs)


RagService = RAGService
