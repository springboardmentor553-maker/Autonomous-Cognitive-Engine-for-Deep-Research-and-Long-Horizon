"""
Tavily Web Search tool.

Wraps the Tavily Python client as a LangChain tool so the supervisor
agent can search the web during task execution.
"""

from __future__ import annotations

import os
import json

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily and return structured results.

    Parameters
    ----------
    query : str
        The search query string.
    max_results : int, optional
        Maximum number of results to return (default 5, max 10).

    Returns
    -------
    str
        A formatted string containing titles, URLs and content snippets
        for each result, suitable for the agent to read and summarise.

    Raises
    ------
    ImportError
        If the ``tavily-python`` package is not installed.
    ValueError
        If TAVILY_API_KEY is not set.
    """
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise ImportError(
            "tavily-python is not installed. Run: pip install tavily-python"
        ) from exc

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )

    if not query or not query.strip():
        return "ERROR: query must not be empty."

    max_results = min(max(1, int(max_results)), 10)

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query.strip(),
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )

    # Build a human-readable (and agent-readable) block
    lines: list[str] = []

    answer = response.get("answer", "")
    if answer:
        lines.append(f"SUMMARY ANSWER:\n{answer}\n")

    results: list[dict] = response.get("results", [])
    if results:
        lines.append(f"TOP {len(results)} RESULTS:")
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "").strip()
            score = r.get("score", 0.0)
            lines.append(
                f"\n[{i}] {title}\n"
                f"    URL: {url}\n"
                f"    Relevance: {score:.2f}\n"
                f"    Excerpt: {content[:400]}{'...' if len(content) > 400 else ''}"
            )

    if not lines:
        return f"No results found for query: '{query}'"

    return "\n".join(lines)
