"""Shared FastAPI dependencies.

The RAG service (and its Chroma + OpenAI clients) is built once and reused across
requests, rather than per request — this is the API-tier form of P1-8. Tests
override ``get_rag_service`` via ``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import get_settings
from src.rag_service import RAGService


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return a process-wide singleton RAG service."""
    return RAGService(get_settings())
