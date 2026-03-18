"""
sub_agents/registry.py
========================
Sub-Agent Registry

Mentor exact spec:

    sub_agents = {
        "summarizer": summarizer
    }

We extend it with web_searcher — same pattern, second specialist.
The supervisor looks up an agent by name and calls .invoke() on it.
"""

from sub_agents.summarization_agent import summarizer
from sub_agents.web_search_agent import web_searcher

# ── Registry — mentor exact pattern (plain dict) ──────────────────────────────
sub_agents = {
    "summarizer":   summarizer,    # receives text   -> returns summary
    "web_searcher": web_searcher,  # receives query  -> returns findings
}


def describe_agents() -> str:
    """Return agent descriptions for the supervisor's context prompt."""
    return (
        'Available sub-agents:\n'
        '  "summarizer"   -> use when task says: summarize / condense / extract key points\n'
        '  "web_searcher" -> use when task says: search / find / look up / research'
    )