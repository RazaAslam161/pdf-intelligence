"""Tests for the Chroma-backed vector store."""

from src.models import TextChunk
from src.vector_store import VectorStore


def test_add_search_count_and_clear_chunks(tmp_path) -> None:
    """Chunks should persist, search, count, and clear through Chroma."""
    store = VectorStore(str(tmp_path / "chroma"))
    chunks = [
        _chunk("chunk-1", "Python testing basics", page_number=1),
        _chunk("chunk-2", "Cooking pasta guide", page_number=2),
    ]

    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

    assert store.count() == 2

    results = store.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].chunk.document_id == "doc-1"
    assert results[0].chunk.file_name == "sample.pdf"
    assert results[0].chunk.page_number == 1
    assert results[0].chunk.text == "Python testing basics"

    store.clear()

    assert store.count() == 0


def test_add_chunks_uses_chunk_id_to_avoid_duplicates(tmp_path) -> None:
    """Adding the same chunk ID should update rather than duplicate it."""
    store = VectorStore(str(tmp_path / "chroma"))

    store.add_chunks([_chunk("chunk-1", "Original text")], [[1.0, 0.0]])
    store.add_chunks([_chunk("chunk-1", "Updated text")], [[0.0, 1.0]])

    assert store.count() == 1
    result = store.search([0.0, 1.0], top_k=1)[0]
    assert result.chunk.text == "Updated text"


def test_add_chunks_validates_embedding_count(tmp_path) -> None:
    """Each chunk must have exactly one embedding."""
    store = VectorStore(str(tmp_path / "chroma"))

    try:
        store.add_chunks([_chunk("chunk-1", "Only chunk")], [])
    except ValueError as exc:
        assert "chunks and embeddings must have the same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_search_caps_top_k_to_available_chunks(tmp_path) -> None:
    """Asking for more results than stored chunks should remain safe."""
    store = VectorStore(str(tmp_path / "chroma"))
    store.add_chunks([_chunk("chunk-1", "Only stored chunk")], [[1.0, 0.0]])

    results = store.search([1.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0].chunk.page_number == 1


def _chunk(chunk_id: str, text: str, page_number: int = 1) -> TextChunk:
    """Build a TextChunk fixture."""
    return TextChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        file_name="sample.pdf",
        page_number=page_number,
        text=text,
    )
