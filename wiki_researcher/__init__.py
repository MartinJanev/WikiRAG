"""Autonomous Wikipedia Researcher.

A resilient, self-correcting RAG agent built on LangGraph with a local Ollama
LLM and Wikipedia retrieval.
"""

from __future__ import annotations

from .config import settings
from .graph import build_graph
from .state import ResearchState

__all__ = ["settings", "build_graph", "ResearchState"]
