"""
Web Search Sub-Agent — Milestone 3.

A specialized LangGraph agent that:
  1. Performs a Tavily web search on the given task/query
  2. Synthesises the raw results into a structured research output
  3. Returns a clean, cited research block to the main agent

Registered in the sub-agent registry and invoked via delegate_task.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from core.llm import get_llm


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WebSearchState(TypedDict):
    """State for the web search sub-agent."""

    query: str            # search query derived from the task
    raw_results: str      # raw Tavily output
    research_output: str  # final synthesised research block


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis assistant.

You will receive raw web search results for a query.
Your job is to synthesise them into a clean, structured research block.

Format your output as:
## Research: <topic>

### Key Findings
- Bullet point each distinct fact or insight

### Sources
- List each source as: [Title](URL) — one-line description

### Summary
Two to three sentences summarising the overall picture.

Rules:
- Only include information present in the search results.
- Do not hallucinate facts.
- If results are sparse, say so honestly.
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def search_node(state: WebSearchState) -> dict:
    """
    Execute a Tavily search for ``state["query"]``.

    Parameters
    ----------
    state : WebSearchState

    Returns
    -------
    dict
        Partial state update setting ``raw_results``.
    """
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        return {"raw_results": f"ERROR: tavily-python not installed. {exc}"}

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"raw_results": "ERROR: TAVILY_API_KEY not set."}

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=state["query"],
            max_results=6,
            include_answer=True,
            include_raw_content=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {"raw_results": f"ERROR during Tavily search: {exc}"}

    lines: list[str] = []
    answer = response.get("answer", "")
    if answer:
        lines.append(f"ANSWER: {answer}\n")

    for i, r in enumerate(response.get("results", []), 1):
        lines.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    {r.get('content', '')[:500]}"
        )

    return {"raw_results": "\n\n".join(lines) if lines else "No results found."}


def synthesise_node(state: WebSearchState) -> dict:
    """
    Use the LLM to synthesise ``state["raw_results"]`` into a structured block.

    Parameters
    ----------
    state : WebSearchState

    Returns
    -------
    dict
        Partial state update setting ``research_output``.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Query: {state['query']}\n\n"
                f"Raw search results:\n{state['raw_results']}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"research_output": response.content}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_web_search_graph():
    """
    Build and compile the web search sub-agent graph.

    Flow: search_node → synthesise_node → END

    Returns
    -------
    CompiledGraph
    """
    builder = StateGraph(WebSearchState)
    builder.add_node("search", search_node)
    builder.add_node("synthesise", synthesise_node)
    builder.add_edge(START, "search")
    builder.add_edge("search", "synthesise")
    builder.add_edge("synthesise", END)
    return builder.compile()


# Module-level compiled graph — imported by the registry
web_search_graph = build_web_search_graph()


def run_web_search_agent(task: str) -> str:
    """
    Public entry point used by the delegate_task tool.

    Parameters
    ----------
    task : str
        The research task / query string.

    Returns
    -------
    str
        A structured research block with findings and sources.
    """
    result = web_search_graph.invoke(
        {"query": task, "raw_results": "", "research_output": ""}
    )
    return result.get("research_output", "ERROR: Web search agent returned no output.")
