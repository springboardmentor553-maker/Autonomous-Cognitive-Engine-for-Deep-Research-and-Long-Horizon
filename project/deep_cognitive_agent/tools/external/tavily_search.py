"""
Tavily Search Tool - External web search integration.

Placeholder for future milestones where the agent can
perform real-time web searches using the Tavily API.
"""

import os


def tavily_search(query: str, max_results: int = 3) -> str:
    """Search the web using Tavily API.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Search results as a formatted string.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "Tavily API key not configured. Set TAVILY_API_KEY in .env."

    # Placeholder — will integrate with TavilySearchResults in future milestones
    return f"Search for '{query}' (not yet implemented — set TAVILY_API_KEY)."
