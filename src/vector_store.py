"""ChromaDB vector store integration."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

from src.models import DocumentSummary, RetrievedChunk, TextChunk

DEFAULT_COLLECTION_NAME = "pdf_chunks"
# Tenant isolation (A8): every chunk carries a tenant_id in its metadata and all
# reads/writes/deletes are scoped to it via a Chroma `where` filter. The default
# tenant is the single-tenant namespace used when no caller identity is supplied.
DEFAULT_TENANT = "default"


class VectorStoreSettings(Protocol):
    """Settings fields required by the vector store."""

    chroma_persist_dir: str


class VectorStoreError(RuntimeError):
    """Raised when local vector storage fails."""


class VectorStore:
    """ChromaDB-backed persistent vector store for PDF chunks."""

    def __init__(
        self,
        settings_or_path: VectorStoreSettings | str | PathLike[str],
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Create a persistent local Chroma vector store."""
        self.persist_dir = _resolve_persist_dir(settings_or_path)
        self.collection_name = collection_name
        self._client = _create_chroma_client(self.persist_dir)
        self._collection = self._get_or_create_collection()

    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        """Store chunks under the tenant, using chunk_id to avoid duplicates."""
        chunk_list = list(chunks)
        embedding_list = [list(embedding) for embedding in embeddings]
        _validate_chunks_and_embeddings(chunk_list, embedding_list)

        if not chunk_list:
            return

        try:
            self._collection.upsert(
                ids=[_scoped_id(chunk.chunk_id, tenant_id) for chunk in chunk_list],
                embeddings=embedding_list,
                documents=[chunk.text for chunk in chunk_list],
                metadatas=[
                    _metadata_from_chunk(chunk, tenant_id) for chunk in chunk_list
                ],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to store chunks in ChromaDB.") from exc

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a query, scoped to the tenant."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not query_embedding:
            raise ValueError("query_embedding must not be empty.")
        stored_count = self.count()
        if stored_count == 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(top_k, stored_count),
                where={"tenant_id": tenant_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to search chunks in ChromaDB.") from exc

        return _search_results_to_chunks(results)

    def clear(self, *, tenant_id: str = DEFAULT_TENANT) -> None:
        """Delete only the given tenant's chunks (other tenants are untouched)."""
        try:
            self._collection.delete(where={"tenant_id": tenant_id})
        except Exception as exc:
            raise VectorStoreError("Failed to clear ChromaDB documents.") from exc

    def count(self, *, tenant_id: str | None = None) -> int:
        """Count stored chunks (globally, or scoped to a tenant)."""
        try:
            if tenant_id is None:
                return int(self._collection.count())
            result = self._collection.get(where={"tenant_id": tenant_id}, include=[])
            return len(result.get("ids") or [])
        except Exception as exc:
            raise VectorStoreError("Failed to count chunks in ChromaDB.") from exc

    def list_documents(
        self,
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> list[DocumentSummary]:
        """Summarize the tenant's documents, aggregated from chunk metadata."""
        try:
            result = self._collection.get(
                where={"tenant_id": tenant_id},
                include=["metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to list documents in ChromaDB.") from exc
        return _aggregate_documents(result.get("metadatas") or [])

    def _get_or_create_collection(self) -> Any:
        """Return the configured Chroma collection."""
        try:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError("Failed to initialize ChromaDB collection.") from exc


def _create_chroma_client(persist_dir: str) -> Any:
    """Create a persistent Chroma client."""
    try:
        import chromadb
    except ImportError as exc:
        raise VectorStoreError(
            "The chromadb package is not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    try:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=persist_dir)
    except Exception as exc:
        raise VectorStoreError("Failed to initialize persistent ChromaDB.") from exc


def _resolve_persist_dir(
    settings_or_path: VectorStoreSettings | str | PathLike[str],
) -> str:
    """Resolve a settings object or direct path into a Chroma directory."""
    if isinstance(settings_or_path, (str, PathLike)):
        return str(settings_or_path)

    persist_dir = getattr(settings_or_path, "chroma_persist_dir", None)
    if not persist_dir:
        persist_dir = getattr(settings_or_path, "CHROMA_PERSIST_DIR", None)
    if not persist_dir:
        raise ValueError("CHROMA_PERSIST_DIR is required for VectorStore.")
    return str(persist_dir)


def _validate_chunks_and_embeddings(
    chunks: Sequence[TextChunk],
    embeddings: Sequence[Sequence[float]],
) -> None:
    """Validate chunk and embedding inputs before writing to Chroma."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length.")

    for index, chunk in enumerate(chunks):
        if not chunk.chunk_id.strip():
            raise ValueError(f"chunks[{index}].chunk_id must not be empty.")
        if not chunk.text.strip():
            raise ValueError(f"chunks[{index}].text must not be empty.")

    for index, embedding in enumerate(embeddings):
        if not embedding:
            raise ValueError(f"embeddings[{index}] must not be empty.")


def _scoped_id(chunk_id: str, tenant_id: str) -> str:
    """Namespace a chunk's storage id by tenant, injectively.

    chunk_id is content-addressed, so two tenants uploading the same file would
    otherwise collide on one Chroma row. The encoding is length-prefixed so the
    (tenant_id, chunk_id) -> id mapping stays unambiguous even when either part
    contains the ':' delimiter (e.g. tenant 'a:b'+'c' vs tenant 'a'+'b:c').
    """
    return f"{len(tenant_id)}:{tenant_id}:{chunk_id}"


def _metadata_from_chunk(chunk: TextChunk, tenant_id: str) -> dict[str, str | int]:
    """Build Chroma metadata for a text chunk, tagged with its tenant."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "file_name": chunk.file_name,
        "page_number": chunk.page_number,
        "tenant_id": tenant_id,
    }


def _search_results_to_chunks(results: dict[str, Any]) -> list[RetrievedChunk]:
    """Convert Chroma query output into retrieved chunks."""
    ids = _first_result_list(results.get("ids"))
    documents = _first_result_list(results.get("documents"))
    metadatas = _first_result_list(results.get("metadatas"))
    distances = _first_result_list(results.get("distances"))

    retrieved: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        text = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else None
        retrieved.append(
            RetrievedChunk(
                chunk=TextChunk(
                    chunk_id=str(metadata.get("chunk_id") or chunk_id),
                    document_id=str(metadata.get("document_id", "")),
                    file_name=str(metadata.get("file_name", "")),
                    page_number=int(metadata.get("page_number", 0)),
                    text=str(text or ""),
                ),
                score=float(distance) if distance is not None else None,
            )
        )

    return retrieved


def _first_result_list(value: Any) -> list[Any]:
    """Return the first nested Chroma result list safely."""
    if not value:
        return []
    return list(value[0])


def _aggregate_documents(metadatas: Sequence[dict[str, Any]]) -> list[DocumentSummary]:
    """Group chunk metadata into per-document summaries, sorted by file name."""
    docs: dict[str, dict[str, Any]] = {}
    for metadata in metadatas:
        document_id = str(metadata.get("document_id", ""))
        entry = docs.setdefault(
            document_id,
            {
                "file_name": str(metadata.get("file_name", "")),
                "pages": set(),
                "chunk_count": 0,
            },
        )
        entry["chunk_count"] += 1
        entry["pages"].add(int(metadata.get("page_number", 0)))

    return [
        DocumentSummary(
            document_id=document_id,
            file_name=entry["file_name"],
            page_count=len(entry["pages"]),
            chunk_count=entry["chunk_count"],
        )
        for document_id, entry in sorted(
            docs.items(), key=lambda item: item[1]["file_name"]
        )
    ]
