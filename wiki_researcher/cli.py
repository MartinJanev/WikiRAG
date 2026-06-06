"""Interactive CLI for the Autonomous Wikipedia Researcher.

Performs a fast Ollama health check at startup (so connection problems surface
immediately with a clear remediation hint), then runs an interactive loop:
read a question, run the research graph, print the answer.
"""

from __future__ import annotations

import logging

from .config import settings
from .graph import build_graph
from .llm import check_ollama
from .state import ResearchState

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    # The graph logs its own progress at INFO; keep third-party noise down.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _initial_state(question: str) -> ResearchState:
    return {
        "messages": [],
        "question": question,
        "search_queries": [],
        "current_query": "",
        "context": "",
        "draft_answer": "",
        "answer_found": False,
        "critique": "",
        "iterations": 0,
    }


def _print_result(final_state: dict) -> None:
    answer = (final_state.get("draft_answer") or "").strip()
    answer_found = final_state.get("answer_found", False)
    iterations = final_state.get("iterations", 0)
    critique = (final_state.get("critique") or "").strip()

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer or "(no answer produced)")

    if not answer_found:
        print("\n[Best-effort answer - the evaluator was not fully satisfied after "
              f"{iterations} research cycle(s).]")
        if critique:
            print(f"Remaining gap: {critique}")
    queries = final_state.get("search_queries", [])
    if queries:
        print(f"\nSearches performed ({len(queries)}): " + "; ".join(queries))
    print("=" * 70 + "\n")


def main() -> None:
    _configure_logging()

    print("Autonomous Wikipedia Researcher")
    print(f"Model: {settings.ollama_model} @ {settings.ollama_base_url}")
    print("Checking Ollama connection...", flush=True)

    if not check_ollama():
        print(
            "\nERROR: Could not reach Ollama.\n"
            f"  - Make sure the server is running:  ollama serve\n"
            f"  - Make sure the model is pulled:    ollama pull {settings.ollama_model}\n"
            f"  - Check OLLAMA_BASE_URL (currently {settings.ollama_base_url}).\n"
        )
        return

    print("Ollama is reachable. Type a question, or 'quit'/'exit' to leave.\n")

    app = build_graph()

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            return

        try:
            final_state = app.invoke(
                _initial_state(question),
                # Generous recursion limit; the MAX_ITERATIONS router is the real guard.
                config={"recursion_limit": settings.max_iterations * 3 + 10},
            )
            _print_result(final_state)
        except Exception as exc:  # noqa: BLE001 - keep the CLI alive on per-query errors
            print(f"\nSomething went wrong while researching: {exc}\n")


if __name__ == "__main__":
    main()
