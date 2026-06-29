"""Shared FastAPI dependencies.

The RAG service (and its Chroma + OpenAI clients) is built once and reused across
requests, rather than per request — this is the API-tier form of P1-8. Tests
override ``get_rag_service`` via ``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Header

from src.config import get_settings
from src.rag_service import RAGService
from src.vector_store import DEFAULT_TENANT, VectorStore


def get_tenant_id(x_tenant_id: str | None = Header(default=None)) -> str:
    """AUTH STUB (A8): derive the caller's tenant from the X-Tenant-Id header.

    Defaults to the shared single-tenant namespace when absent. This is a
    placeholder: before any multi-user deployment, replace it with real
    authentication that derives identity server-side (session / JWT), never from
    a client-supplied header.
    """
    tenant = (x_tenant_id or "").strip()
    return tenant or DEFAULT_TENANT


# Settings is a frozen dataclass; its read-only fields satisfy the settings
# Protocols structurally, but mypy treats Protocol attributes as mutable, so the
# concrete-frozen -> Protocol construction needs an explicit nudge (read-only by
# design — nothing writes back through the protocol).
@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the process-wide singleton vector store.

    Shared by the RAG service and the workspace endpoints so a clear from one
    path is seen by the other (no stale collection handle), and so reading or
    clearing the store needs no OpenAI key.
    """
    return VectorStore(get_settings())  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return a process-wide singleton RAG service over the shared vector store."""
    return RAGService(get_settings(), vector_store=get_vector_store())  # type: ignore[arg-type]
