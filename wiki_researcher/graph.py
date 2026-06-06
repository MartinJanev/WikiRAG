"""Graph wiring and conditional routing.

Research cycle:

    START -> query_rewriter_node -> search_node -> rag_node
          -> evaluator_node -> (query_rewriter_node | END)

`query_rewriter_node` converts the raw question (or evaluator hint) into a
clean Wikipedia search term before every retrieval attempt. This is the key
fix for the "Expecting value: line 1 column 1" errors caused by passing
natural-language questions directly to the Wikipedia API.

The conditional edge after `evaluator_node` loops back while the answer is
insufficient, and routes to END once found or the iteration cap is reached.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import settings
from .nodes import evaluator_node, query_rewriter_node, rag_node, search_node
from .state import ResearchState


def route_after_eval(state: ResearchState) -> str:
    """Finish when satisfied or when the hard iteration cap is hit."""
    if state.get("answer_found", False):
        return "end"
    if state.get("iterations", 0) >= settings.max_iterations:
        return "end"
    return "rewrite"


def build_graph():
    """Construct and compile the research graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("query_rewriter_node", query_rewriter_node)
    graph.add_node("search_node", search_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("evaluator_node", evaluator_node)

    graph.add_edge(START, "query_rewriter_node")
    graph.add_edge("query_rewriter_node", "search_node")
    graph.add_edge("search_node", "rag_node")
    graph.add_edge("rag_node", "evaluator_node")
    graph.add_conditional_edges(
        "evaluator_node",
        route_after_eval,
        {"rewrite": "query_rewriter_node", "end": END},
    )

    return graph.compile()
