"""Local LLM access via Ollama.

A single factory builds the configured `ChatOllama` client, and a lightweight
health check lets the CLI fail fast (with a friendly message) when Ollama is
down or unreachable, rather than blowing up mid-graph on a connection timeout.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Return the configured local Ollama chat model."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        timeout=settings.ollama_timeout,
    )


def check_ollama() -> bool:
    """Ping Ollama once so startup failures are obvious and actionable.

    Returns True if the model responds, False on any connection/timeout error.
    """
    try:
        llm = get_llm()
        llm.invoke([HumanMessage(content="ping")])
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        logger.error("Ollama health check failed: %s", exc)
        return False
