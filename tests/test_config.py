"""Tests for application settings loading."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config import get_settings

CONFIG_ENV_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "CHROMA_PERSIST_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
}


class SettingsTests(unittest.TestCase):
    """Settings loading behavior."""

    def test_get_settings_uses_safe_defaults(self) -> None:
        """Unset optional values should fall back to safe defaults."""
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in CONFIG_ENV_KEYS
        }
        env["OPENAI_API_KEY"] = "configured"

        with patch("src.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            settings = get_settings()

        self.assertEqual(settings.openai_api_key, "configured")
        self.assertIsNone(settings.openai_base_url)
        self.assertEqual(settings.openai_chat_model, "gpt-4.1-mini")
        self.assertEqual(settings.openai_embedding_model, "text-embedding-3-small")
        self.assertEqual(settings.chroma_persist_dir, "./chroma_db")
        self.assertEqual(settings.chunk_size, 1000)
        self.assertEqual(settings.chunk_overlap, 150)
        self.assertEqual(settings.retrieval_top_k, 4)
        self.assertIsNone(settings.validation_error)
        self.assertTrue(settings.has_openai_api_key)

    def test_get_settings_reads_environment_overrides(self) -> None:
        """Configured environment values should override defaults."""
        env = {
            "OPENAI_API_KEY": "configured",
            "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENAI_CHAT_MODEL": "gpt-example",
            "OPENAI_EMBEDDING_MODEL": "embedding-example",
            "CHROMA_PERSIST_DIR": "custom/chroma",
            "CHUNK_SIZE": "1200",
            "CHUNK_OVERLAP": "180",
            "RETRIEVAL_TOP_K": "6",
        }

        with patch("src.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            settings = get_settings()

        self.assertEqual(settings.openai_chat_model, "gpt-example")
        self.assertEqual(settings.openai_base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(settings.openai_embedding_model, "embedding-example")
        self.assertEqual(settings.chroma_persist_dir, "custom/chroma")
        self.assertEqual(settings.chunk_size, 1200)
        self.assertEqual(settings.chunk_overlap, 180)
        self.assertEqual(settings.retrieval_top_k, 6)

    def test_missing_api_key_sets_validation_error(self) -> None:
        """A missing API key should be exposed as a user-facing message."""
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in CONFIG_ENV_KEYS
        }

        with patch("src.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            settings = get_settings()

        self.assertIsNone(settings.openai_api_key)
        self.assertFalse(settings.has_openai_api_key)
        self.assertEqual(
            settings.validation_error,
            "OPENAI_API_KEY is missing. Add it to your .env file before using "
            "OpenAI features.",
        )

    def test_invalid_integer_settings_fall_back_to_defaults(self) -> None:
        """Invalid numeric environment values should not crash settings loading."""
        env = {
            "OPENAI_API_KEY": "configured",
            "CHUNK_SIZE": "not-a-number",
            "CHUNK_OVERLAP": "",
            "RETRIEVAL_TOP_K": "also-bad",
        }

        with patch("src.config.load_dotenv"), patch.dict(os.environ, env, clear=True):
            settings = get_settings()

        self.assertEqual(settings.chunk_size, 1000)
        self.assertEqual(settings.chunk_overlap, 150)
        self.assertEqual(settings.retrieval_top_k, 4)


if __name__ == "__main__":
    unittest.main()
