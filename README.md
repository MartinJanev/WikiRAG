# Autonomous Wikipedia Researcher

A resilient, self-correcting Retrieval-Augmented Generation (RAG) agent built with
[LangGraph](https://www.langchain.com/langgraph). It searches Wikipedia, drafts an
answer with a local [Ollama](https://ollama.com) LLM, evaluates its own work, and
loops back with a refined search query until the answer is sufficient (or a safety
cap is hit).

## How it works

```
START -> query_rewriter_node -> search_node -> rag_node -> evaluator_node -> (loop back | END)
```

- **`query_rewriter_node`** turns the natural-language question (or the evaluator's
  refinement hint) into a clean Wikipedia search term.
- **`search_node`** fetches context from Wikipedia (tool I/O only).
- **`rag_node`** drafts an answer grounded strictly in the retrieved context.
- **`evaluator_node`** acts as a strict reviewer: it decides whether the answer is
  good enough and, if not, emits a *new, more specific* search query.
- A **conditional edge** routes back to `query_rewriter_node` while `answer_found`
  is `False`, or to `END` once the answer is found or `MAX_ITERATIONS` is reached.

## Requirements

- Python 3.10+
- A running Ollama server with a model pulled (default `llama3`):

```bash
ollama serve          # if not already running
ollama pull llama3
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optionally copy `.env.example` and tweak settings:

```bash
cp .env.example .env   # then export the vars, or set them in your shell
```

## Run

```bash
python main.py
# or, equivalently:
python -m wiki_researcher
```

Type a question at the prompt. Use `quit` or `exit` to leave.

## Configuration

All settings live in [`wiki_researcher/config.py`](wiki_researcher/config.py) and are
overridable via environment variables (see [`.env.example`](.env.example)): model
name, Ollama base URL, temperature/timeout, loop cap, Wikipedia retrieval limits,
and retry/backoff.

## Project layout

```
WikiRAG/
  main.py                     # thin entry point -> wiki_researcher.cli:main
  requirements.txt
  README.md
  .env.example
  wiki_researcher/            # the package
    __init__.py               # exposes build_graph, settings, ResearchState
    __main__.py               # enables `python -m wiki_researcher`
    config.py                 # env-overridable settings (single source of tunables)
    state.py                  # ResearchState typed shared state
    tools.py                  # resilient Wikipedia search (two-step REST retrieval)
    llm.py                    # ChatOllama factory + startup health check
    nodes.py                  # query_rewriter_node, search_node, rag_node, evaluator_node
    graph.py                  # graph wiring + conditional routing
    cli.py                    # interactive CLI loop
```
