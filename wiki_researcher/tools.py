"""Wikipedia retrieval tool (tool I/O only).

This module is the *only* place that talks to the Wikipedia API.
It uses direct `requests` calls instead of langchain_community's
WikipediaQueryRun, which omits the User-Agent header and causes
Wikipedia to return empty responses (JSON parse error).

Two-step retrieval:
  1. Wikipedia search API  -> find the best matching page title
  2. Wikipedia REST summary API -> fetch the actual page content

Resiliency: transient failures are retried with exponential backoff.
On hard failure we return "" so the graph degrades gracefully.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import quote

import requests

from .config import settings

logger = logging.getLogger(__name__)

# Wikipedia requires a descriptive User-Agent or it returns empty responses.
_HEADERS = {
    "User-Agent": "WikiRAG/2.0 (autonomous-research-agent; python-requests)"
}
_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
_TIMEOUT = 15  # seconds per HTTP call


def _search_titles(query: str) -> list[str]:
    """Return up to `wiki_top_k` Wikipedia page titles for `query`."""
    resp = requests.get(
        _SEARCH_URL,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": settings.wiki_top_k,
            "format": "json",
            "utf8": 1,
        },
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("query", {}).get("search", [])
    return [r["title"] for r in results]


def _fetch_summary(title: str) -> str:
    """Fetch the plain-text extract for a Wikipedia page title."""
    resp = requests.get(
        f"{_SUMMARY_URL}/{quote(title, safe='')}",
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("extract", "").strip()


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for `query`, returning page text or "" on failure.

    Step 1 - call the search API to resolve the query to real page titles.
    Step 2 - fetch the summary/extract for each title until we get content.

    Retries the whole sequence with exponential backoff on transient errors.
    Always returns a string so callers never handle exceptions themselves.
    """
    query = (query or "").strip()
    if not query:
        return ""

    attempts = max(1, settings.max_retries)

    for attempt in range(1, attempts + 1):
        try:
            titles = _search_titles(query)
            if not titles:
                logger.warning("No Wikipedia search results for %r", query)
                return ""

            # Try each title; return the first non-empty extract.
            for title in titles:
                try:
                    extract = _fetch_summary(title)
                    if extract:
                        # Trim to configured character limit.
                        extract = extract[: settings.wiki_doc_chars_max]
                        logger.info(
                            "Wikipedia: resolved %r -> %r (%d chars)",
                            query,
                            title,
                            len(extract),
                        )
                        return extract
                except Exception as inner_exc:  # noqa: BLE001
                    logger.warning(
                        "Could not fetch summary for %r: %s", title, inner_exc
                    )

            logger.warning("All title summaries empty for query %r", query)
            return ""

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Wikipedia search failed (attempt %d/%d) for %r: %s",
                attempt,
                attempts,
                query,
                exc,
            )
            if attempt < attempts:
                backoff = settings.retry_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(backoff)

    logger.error(
        "Wikipedia search gave up after %d attempts for %r", attempts, query
    )
    return ""
