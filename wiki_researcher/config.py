"""Central configuration for the Autonomous Wikipedia Researcher.

All values are overridable via environment variables so that node logic never
embeds magic numbers. This keeps tunables in one place (Separation of Concerns).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-typed settings snapshot."""

    # Ollama / LLM
    ollama_model: str = _get_str("OLLAMA_MODEL", "llama3")
    ollama_base_url: str = _get_str("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = _get_float("TEMPERATURE", 0.0)
    ollama_timeout: float = _get_float("OLLAMA_TIMEOUT", 120.0)

    # Research loop
    max_iterations: int = _get_int("MAX_ITERATIONS", 4)

    # Wikipedia retrieval
    wiki_top_k: int = _get_int("WIKI_TOP_K", 3)
    wiki_doc_chars_max: int = _get_int("WIKI_DOC_CHARS_MAX", 4000)

    # Resiliency
    max_retries: int = _get_int("MAX_RETRIES", 3)
    retry_backoff_seconds: float = _get_float("RETRY_BACKOFF_SECONDS", 1.5)


settings = Settings()
