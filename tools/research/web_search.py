"""
tools/research/web_search.py — Tavily web search tool.
"""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information using Tavily.

    Use this to gather facts, news, or background knowledge on any topic.
    Always save important findings to the virtual file system with write_file.

    Args:
        query:       The search query string.
        max_results: Number of results to return (1-10, default 5).

    Returns:
        JSON string with search results including titles, URLs, and snippets.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query, max_results=max_results)

        formatted = []
        for r in results.get("results", []):
            formatted.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    # Truncate content to 300 chars to stay within token limits
                    "content": r.get("content", "")[:300],
                    "score": r.get("score", 0),
                }
            )

        logger.info(f"web_search '{query}' → {len(formatted)} results")
        return json.dumps({"query": query, "results": formatted})

    except Exception as e:
        logger.error(f"web_search error: {e}")
        return json.dumps({"error": str(e), "query": query})