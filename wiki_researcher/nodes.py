"""Graph nodes - one responsibility each (Separation of Concerns).

- `query_rewriter_node`: converts the raw question into a clean Wikipedia
  search term before the first (and each subsequent) retrieval attempt.
- `search_node`        : tool I/O only (fetch Wikipedia context).
- `rag_node`           : synthesis only (draft an answer from context).
- `evaluator_node`     : judgment + self-correction (no tools, no synthesis).

Each node has the signature `(state: ResearchState) -> dict` and returns only
the keys it changes. Cross-cutting concerns never bleed between nodes.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import get_llm
from .state import ResearchState
from .tools import search_wikipedia

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# query_rewriter_node                                                          #
# --------------------------------------------------------------------------- #
_REWRITER_SYSTEM = (
    "You are a Wikipedia search query optimizer. "
    "Convert the user's natural-language question into the shortest, most precise "
    "Wikipedia article title or search phrase that would retrieve the answer. "
    "Output ONLY the search query - no punctuation at the end, no quotes, "
    "no explanation, no full sentences. "
    "Examples:\n"
    "  Q: Which country won the 2022 FIFA World Cup?\n"
    "  A: 2022 FIFA World Cup\n"
    "  Q: Who invented the telephone?\n"
    "  A: Alexander Graham Bell\n"
    "  Q: What is the boiling point of water in Celsius?\n"
    "  A: Properties of water"
)


def query_rewriter_node(state: ResearchState) -> Dict[str, Any]:
    """Rewrite the current question/hint into a clean Wikipedia search term.

    On the first cycle we rewrite the original question.
    On subsequent cycles the evaluator has already set `current_query` to a
    refined hint; we polish that hint into a proper Wikipedia title.
    """
    question = state["question"]
    # After the first cycle the evaluator sets current_query to a new hint.
    raw_query = (state.get("current_query") or "").strip() or question
    already_tried = list(state.get("search_queries", []))

    prompt = (
        f"Original question: {question}\n"
        f"Search hint: {raw_query}\n"
        f"Already tried (do NOT repeat): {already_tried}\n\n"
        "Output the Wikipedia search query:"
    )

    llm = get_llm()
    try:
        response = llm.invoke(
            [SystemMessage(content=_REWRITER_SYSTEM), HumanMessage(content=prompt)]
        )
        rewritten = (
            response.content if isinstance(response.content, str) else str(response.content)
        ).strip().strip('"\'').split("\n")[0].strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("query_rewriter_node LLM call failed: %s", exc)
        rewritten = raw_query  # fall back to the hint as-is

    # If the LLM repeated an already-tried query, append a differentiator.
    if rewritten in already_tried:
        rewritten = f"{rewritten} history"

    logger.info("Query rewriter: %r -> %r", raw_query, rewritten)

    return {
        "current_query": rewritten,
        "messages": [
            SystemMessage(content=f"Rewrote query: {raw_query!r} -> {rewritten!r}")
        ],
    }


# --------------------------------------------------------------------------- #
# search_node                                                                  #
# --------------------------------------------------------------------------- #
def search_node(state: ResearchState) -> Dict[str, Any]:
    """Fetch Wikipedia context for the current (already-rewritten) query."""
    query = (state.get("current_query") or "").strip() or state["question"]

    context = search_wikipedia(query)

    already_tried = list(state.get("search_queries", []))
    if query not in already_tried:
        already_tried.append(query)

    if context:
        note = f"Retrieved Wikipedia context for query: {query!r}"
    else:
        note = f"No Wikipedia content found for query: {query!r}"
    logger.info(note)

    return {
        "context": context,
        "search_queries": already_tried,
        "current_query": query,
        "messages": [SystemMessage(content=note)],
    }


# --------------------------------------------------------------------------- #
# rag_node                                                                     #
# --------------------------------------------------------------------------- #
_RAG_SYSTEM = (
    "You are a careful research assistant. Answer the user's QUESTION using ONLY "
    "the provided CONTEXT. Do not use outside knowledge.\n\n"
    "Rules:\n"
    "- If the context contains the answer, state it clearly and concisely.\n"
    "- Extract the specific fact asked for - do not pad with unrelated content.\n"
    "- If the context does not contain enough information, say exactly what is "
    "missing (one sentence) instead of guessing or hallucinating.\n"
    "- Never say 'I don't have context' if the context DOES mention the answer.\n"
    "Be concise and factual."
)


def rag_node(state: ResearchState) -> Dict[str, Any]:
    """Synthesize a draft answer grounded strictly in the retrieved context."""
    question = state["question"]
    context = state.get("context", "") or "(no context retrieved)"

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write the best possible answer grounded only in the context above."
    )

    llm = get_llm()
    try:
        response = llm.invoke(
            [SystemMessage(content=_RAG_SYSTEM), HumanMessage(content=prompt)]
        )
        draft = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("rag_node LLM call failed: %s", exc)
        draft = ""

    draft = (draft or "").strip()

    return {
        "draft_answer": draft,
        "messages": [AIMessage(content=draft or "(failed to generate a draft answer)")],
    }


# --------------------------------------------------------------------------- #
# evaluator_node                                                               #
# --------------------------------------------------------------------------- #
_EVALUATOR_SYSTEM = (
    "You are a strict answer-quality reviewer for a Wikipedia research agent.\n\n"
    "Your job: decide whether the DRAFT ANSWER fully and correctly answers the "
    "QUESTION given the CONTEXT retrieved.\n\n"
    "Decision rules:\n"
    "- Set answer_found=true if the draft contains a specific, factual answer "
    "that is clearly supported by the context - even if the phrasing is imperfect.\n"
    "- Set answer_found=true if the CONTEXT contains the answer even if the draft "
    "missed or understated it. Do not penalise the draft for being terse.\n"
    "- Set answer_found=false ONLY when the context genuinely does not contain the "
    "information needed to answer the question.\n\n"
    "Rules for next_query when answer_found=false:\n"
    "- Must be a SHORT Wikipedia-style title or search phrase (2-6 words).\n"
    "- Must be different from every query in ALREADY_TRIED.\n"
    "- Approach from a new angle: sub-topic, related entity, or narrower term.\n\n"
    "Respond with a SINGLE JSON object and nothing else:\n"
    '{"answer_found": <bool>, "critique": "<one-sentence reason>", '
    '"next_query": "<Wikipedia search term or empty string>"}\n'
    "Output JSON only - no markdown, no prose."
)


def _parse_evaluation(raw: str) -> Dict[str, Any]:
    """Defensively parse the evaluator's JSON, with a safe fallback."""
    text = (raw or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    candidate = text
    if not candidate.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)

    try:
        data = json.loads(candidate)
        return {
            "answer_found": bool(data.get("answer_found", False)),
            "critique": str(data.get("critique", "")).strip(),
            "next_query": str(data.get("next_query", "")).strip(),
        }
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Could not parse evaluator JSON (%s); treating as not found.", exc)
        return {
            "answer_found": False,
            "critique": "Evaluator output was unparseable.",
            "next_query": "",
        }


def _fallback_query(question: str, already_tried: list[str], iterations: int) -> str:
    """Deterministic refinement when the evaluator fails to supply a new query."""
    suffixes = ["winner", "result", "history", "overview", "background", "details"]
    for suffix in suffixes:
        candidate = f"{question} {suffix}".strip()
        if candidate not in already_tried:
            return candidate
    return f"{question} (refinement {iterations + 1})"


def evaluator_node(state: ResearchState) -> Dict[str, Any]:
    """Judge the draft and, if insufficient, produce a fresh search hint."""
    question = state["question"]
    context = state.get("context", "") or "(no context retrieved)"
    draft = state.get("draft_answer", "") or "(no draft answer)"
    already_tried = list(state.get("search_queries", []))
    iterations = state.get("iterations", 0)

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"ALREADY_TRIED (do not repeat these):\n{already_tried}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        "Evaluate the draft and respond with the JSON object."
    )

    llm = get_llm()
    try:
        response = llm.invoke(
            [SystemMessage(content=_EVALUATOR_SYSTEM), HumanMessage(content=prompt)]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        evaluation = _parse_evaluation(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("evaluator_node LLM call failed: %s", exc)
        evaluation = {
            "answer_found": False,
            "critique": f"Evaluation failed: {exc}",
            "next_query": "",
        }

    answer_found = evaluation["answer_found"]
    critique = evaluation["critique"]
    next_query = evaluation["next_query"]

    if not answer_found:
        if not next_query or next_query in already_tried:
            next_query = _fallback_query(question, already_tried, iterations)

    logger.info(
        "Evaluation: answer_found=%s | next_query=%r | critique=%s",
        answer_found,
        next_query,
        critique,
    )

    return {
        "answer_found": answer_found,
        "critique": critique,
        # This is the *hint* - query_rewriter_node will polish it next cycle.
        "current_query": next_query,
        "iterations": iterations + 1,
        "messages": [
            SystemMessage(
                content=(
                    f"Evaluation -> answer_found={answer_found}; "
                    f"critique={critique or 'n/a'}; "
                    f"next_query={next_query or 'n/a'}"
                )
            )
        ],
    }
