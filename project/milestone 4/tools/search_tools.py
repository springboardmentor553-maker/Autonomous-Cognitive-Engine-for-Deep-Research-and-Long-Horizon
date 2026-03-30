from __future__ import annotations

import json
from urllib import request

from langchain_core.tools import tool

from app.config import TAVILY_API_KEY


def build_search_tools() -> list:
    @tool
    def tavily_search_tool(query: str, max_results: int = 5) -> str:
        """Run a real web search using Tavily and return a compact result summary."""
        if not TAVILY_API_KEY:
            return "TAVILY_API_KEY is not configured, so live web search is unavailable."

        payload = json.dumps(
            {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            }
        ).encode("utf-8")

        req = request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return f"Live web search failed: {exc}"

        results = data.get("results", [])
        if not results:
            return "No web results returned."

        lines = []
        for item in results[:max_results]:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"TITLE: {title}\nURL: {url}\nSNIPPET: {content}")

        return "\n\n".join(lines)

    return [tavily_search_tool]
