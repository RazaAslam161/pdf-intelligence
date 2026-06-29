"""ChromaDB vector store integration."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

from src.models import RetrievedChunk, TextChunk

DEFAULT_COLLECTION_NAME = "pdf_chunks"


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
    ) -> None:
        """Store chunks and embeddings, using chunk_id to avoid duplicates."""
        chunk_list = list(chunks)
        embedding_list = [list(embedding) for embedding in embeddings]
        _validate_chunks_and_embeddings(chunk_list, embedding_list)

        if not chunk_list:
            return

        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunk_list],
                embeddings=embedding_list,
                documents=[chunk.text for chunk in chunk_list],
                metadatas=[_metadata_from_chunk(chunk) for chunk in chunk_list],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to store chunks in ChromaDB.") from exc

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the most relevant stored chunks for a query embedding."""
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
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to search chunks in ChromaDB.") from exc

        return _search_results_to_chunks(results)

    def clear(self) -> None:
        """Reset the local Chroma collection."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as exc:
            message = str(exc).lower()
            if "does not exist" not in message and "not found" not in message:
                raise VectorStoreError(
                    "Failed to clear ChromaDB collection."
                ) from exc

        self._collection = self._get_or_create_collection()

    def count(self) -> int:
        """Return the number of stored chunks."""
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError("Failed to count chunks in ChromaDB.") from exc

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


def _metadata_from_chunk(chunk: TextChunk) -> dict[str, str | int]:
    """Build Chroma metadata for a text chunk."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "file_name": chunk.file_name,
        "page_number": chunk.page_number,
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
