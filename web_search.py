"""
Web Search Module — Tavily API integration for real-time web data.
Provides search_web() and formatting helpers for injecting web results into LLM prompts.
"""
import os
import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using Tavily API.
    Returns a list of results: [{title, url, snippet, content}]
    Falls back gracefully if Tavily is unavailable.
    """
    if not TAVILY_API_KEY:
        logger.warning("[WEB SEARCH] No TAVILY_API_KEY set — skipping web search.")
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        logger.info(f"[WEB SEARCH] Searching: {query[:80]}...")
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:300],
                "content": item.get("content", "")
            })

        logger.info(f"[WEB SEARCH] Found {len(results)} results.")
        return results

    except ImportError:
        logger.error("[WEB SEARCH] tavily-python not installed. Run: pip install tavily-python")
        return []
    except Exception as e:
        logger.error(f"[WEB SEARCH] Tavily API error: {e}")
        # Retry once after 2 seconds
        try:
            time.sleep(2)
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(query=query, max_results=max_results, search_depth="basic")
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")[:300],
                    "content": item.get("content", "")
                })
            logger.info(f"[WEB SEARCH] Retry successful — {len(results)} results.")
            return results
        except Exception as retry_err:
            logger.error(f"[WEB SEARCH] Retry also failed: {retry_err}")
            return []


def format_sources_for_llm(results: List[Dict]) -> str:
    """
    Format web search results into a structured context block for the LLM.
    """
    if not results:
        return ""

    sections = ["=== WEB SEARCH RESULTS ===\n"]
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Source {i}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Content: {r['content']}\n"
        )
    sections.append("=== END OF WEB SOURCES ===")
    return "\n".join(sections)


def format_sources_for_display(results: List[Dict]) -> str:
    """
    Format web sources as markdown links for display in the UI.
    """
    if not results:
        return ""

    lines = ["\n\n---\n**📚 Sources:**"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['title']}]({r['url']})")
    return "\n".join(lines)


def get_source_urls(results: List[Dict]) -> List[Dict]:
    """
    Extract minimal source info for database storage (JSON serializable).
    """
    return [{"title": r["title"], "url": r["url"]} for r in results]
