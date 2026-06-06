"""Shared graph state (the single source of truth).

`ResearchState` is the typed contract every node reads from and writes to.
Nodes return *partial* updates (only the keys they change); LangGraph merges
them into the shared state before the next node runs.
"""

from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """Everything the agent knows across the research loop."""

    # Conversation history. The `add_messages` reducer appends rather than
    # overwrites, preserving the full trace across cycles.
    messages: Annotated[List[BaseMessage], add_messages]

    # The original user question (never mutated after init).
    question: str

    # Every query we have already tried. Used both as a loop guard and to tell
    # the evaluator which searches to avoid repeating.
    search_queries: List[str]

    # The query the next `search_node` run will execute.
    current_query: str

    # Latest context retrieved from Wikipedia.
    context: str

    # The draft answer produced by `rag_node`.
    draft_answer: str

    # Whether the evaluator judged the draft as sufficient.
    answer_found: bool

    # The evaluator's reasoning / description of the remaining gap.
    critique: str

    # Number of completed research cycles; bounded by MAX_ITERATIONS.
    iterations: int
