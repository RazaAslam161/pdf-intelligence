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


def test_list_documents_aggregates_by_document(tmp_path) -> None:
    """list_documents groups stored chunks into per-document summaries."""
    store = VectorStore(str(tmp_path / "chroma"))
    store.add_chunks(
        [
            TextChunk("a-1", "docA", "a.pdf", 1, "Alpha one"),
            TextChunk("a-2", "docA", "a.pdf", 1, "Alpha two"),
            TextChunk("a-3", "docA", "a.pdf", 2, "Alpha three"),
            TextChunk("b-1", "docB", "b.pdf", 1, "Beta one"),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5]],
    )

    docs = store.list_documents()

    assert [d.file_name for d in docs] == ["a.pdf", "b.pdf"]
    assert docs[0].document_id == "docA"
    assert docs[0].chunk_count == 3
    assert docs[0].page_count == 2
    assert docs[1].chunk_count == 1
    assert docs[1].page_count == 1


def test_list_documents_empty_store_returns_empty(tmp_path) -> None:
    """An empty store lists no documents."""
    store = VectorStore(str(tmp_path / "chroma"))

    assert store.list_documents() == []


def test_tenant_isolation_for_search_count_and_clear(tmp_path) -> None:
    """search/count/clear/list are scoped per tenant; no cross-tenant leak."""
    store = VectorStore(str(tmp_path / "chroma"))
    store.add_chunks([_chunk("a-1", "Alpha")], [[1.0, 0.0]], tenant_id="alice")
    store.add_chunks([_chunk("b-1", "Beta")], [[0.0, 1.0]], tenant_id="bob")

    alice_hits = store.search([1.0, 0.0], top_k=5, tenant_id="alice")
    bob_hits = store.search([0.0, 1.0], top_k=5, tenant_id="bob")
    assert [r.chunk.chunk_id for r in alice_hits] == ["a-1"]
    assert [r.chunk.chunk_id for r in bob_hits] == ["b-1"]

    # Querying bob's vector as alice never returns bob's chunk.
    alice_for_bob = store.search([0.0, 1.0], top_k=5, tenant_id="alice")
    assert all(r.chunk.chunk_id != "b-1" for r in alice_for_bob)

    assert store.count(tenant_id="alice") == 1
    assert store.count(tenant_id="bob") == 1
    assert {d.file_name for d in store.list_documents(tenant_id="alice")} == {
        "sample.pdf"
    }

    # Clearing alice leaves bob's data intact.
    store.clear(tenant_id="alice")
    assert store.count(tenant_id="alice") == 0
    assert store.count(tenant_id="bob") == 1
    survivors = store.search([0.0, 1.0], top_k=5, tenant_id="bob")
    assert [r.chunk.chunk_id for r in survivors] == ["b-1"]


def test_scoped_ids_do_not_collide_across_tenants(tmp_path) -> None:
    """tenant+chunk_id namespacing stays injective even with ':' in the ids."""
    store = VectorStore(str(tmp_path / "chroma"))
    # A naive "tenant:chunk_id" scheme would map both of these to "a:b:c" and
    # silently overwrite one tenant's row with the other's.
    store.add_chunks([_chunk("b:c", "First")], [[1.0, 0.0]], tenant_id="a")
    store.add_chunks([_chunk("c", "Second")], [[0.0, 1.0]], tenant_id="a:b")

    assert store.count(tenant_id="a") == 1
    assert store.count(tenant_id="a:b") == 1


def _chunk(chunk_id: str, text: str, page_number: int = 1) -> TextChunk:
    """Build a TextChunk fixture."""
    return TextChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        file_name="sample.pdf",
        page_number=page_number,
        text=text,
    )
