"""
sub_agents/web_search_agent.py
================================
Web Search Sub-Agent

Mentor spec:
  - Specific purpose    : search the web and return structured findings
  - Small focused prompt: one PromptTemplate for structuring results
  - Limited toolset     : Tavily search only — nothing else
  - Clear responsibility: receive query -> search -> return findings

Same pattern as summarization_agent but for web research tasks.
"""

import os

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

import config

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model=config.MODEL_NAME,
    api_key=config.GROQ_API_KEY,
    temperature=0.1,
    max_tokens=600,
)

# ── Small focused prompt ──────────────────────────────────────────────────────
research_prompt = PromptTemplate(
    input_variables=["query", "search_results"],
    template="""
You are a specialized web search agent.
Your task is to extract the most relevant findings from these search results.

Query: {query}

Search Results:
{search_results}

Provide structured findings with:
- Summary (2-3 sentences of what was found)
- Key Facts (3 bullet points)
- Source Quality (reliable / mixed / limited)
""",
)


# ── Internal search helper (Tavily — this agent's only tool) ──────────────────
def _run_search(query: str) -> str:
    """Run Tavily search. This is the only external tool this agent uses."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
        response = client.search(query=query, max_results=3)
        parts = []
        for r in response.get("results", []):
            parts.append(
                f"Title: {r.get('title', '')}\n"
                f"URL:   {r.get('url', '')}\n"
                f"Info:  {r.get('content', '')[:250]}"
            )
        return "\n\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"Search unavailable: {e}"


# ── Agent function (same mentor pattern) ──────────────────────────────────────
def web_search_agent(query: str) -> str:
    """
    Single responsibility: receive query, search web, return findings.
    This agent does nothing else — no file writes, no planning, no state.
    """
    raw_results = _run_search(query)
    prompt      = research_prompt.format(query=query, search_results=raw_results)
    response    = llm.invoke(prompt)
    return response.content


# ── Wrap as RunnableLambda (same mentor pattern) ──────────────────────────────
web_searcher = RunnableLambda(web_search_agent)