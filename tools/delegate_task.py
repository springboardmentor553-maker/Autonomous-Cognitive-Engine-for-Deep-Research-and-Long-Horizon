"""
delegate_task — Milestone 3 delegation tool.

Sub-Agent Registry
------------------
web_search_agent : performs focused Tavily search + synthesis (PRIMARY)

Note: summarization_agent is handled AUTOMATICALLY by graph.py during the
synthesis phase. The model must NOT call it manually — it will be intercepted
and redirected here to prevent tool_use_failed errors.
"""

from __future__ import annotations

import json
from typing import Callable

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Sub-agent registry — lazy to avoid circular imports
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[[str], str]] | None = None


def _get_registry() -> dict[str, Callable[[str], str]]:
    """Build (once) and return the sub-agent registry."""
    global _REGISTRY
    if _REGISTRY is None:
        from agents.web_search_agent import run_web_search_agent
        _REGISTRY = {
            "web_search_agent": run_web_search_agent,
        }
    return _REGISTRY


def get_available_agents() -> list[str]:
    """Return the list of currently registered sub-agent names."""
    return list(_get_registry().keys())


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def delegate_task(agent_name: str, task: str) -> str:
    """
    Delegate a research topic to the web_search_agent sub-agent.

    Args:
        agent_name: Must be "web_search_agent".
        task: Short research topic, under 60 characters.
            Example: "AI job displacement India 2024"

    Returns:
        Structured research block with key findings and sources.
    """
    if not agent_name or not agent_name.strip():
        return "ERROR: agent_name must not be empty."
    if not task or not task.strip():
        return "ERROR: task must not be empty."

    agent_name = agent_name.strip()

    # ------------------------------------------------------------------
    # Hard intercept: if the model tries to call summarization_agent,
    # redirect to web_search_agent instead of crashing with tool_use_failed.
    # Summarization is handled automatically by graph.py — model must not
    # call it directly.
    # ------------------------------------------------------------------
    if agent_name == "summarization_agent":
        # Extract a usable search query from the task string
        # (the model often passes filenames or notes here — clean it up)
        clean_task = task.strip()
        # If it looks like filenames, use a generic fallback
        if ".txt" in clean_task or ".md" in clean_task:
            clean_task = "research summary"
        # Truncate to safe length
        clean_task = clean_task[:60]
        agent_name = "web_search_agent"
        task = clean_task

    registry = _get_registry()

    if agent_name not in registry:
        available = ", ".join(f'"{k}"' for k in registry)
        return f"ERROR: Unknown sub-agent '{agent_name}'. Available: {available}"

    agent_fn = registry[agent_name]

    try:
        result = agent_fn(task.strip()[:60])
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Sub-agent '{agent_name}' failed: {exc}"

    payload = json.dumps({
        "action": "delegate_task",
        "agent_name": agent_name,
        "result": result,
    })
    return payload


# ---------------------------------------------------------------------------
# Helper used by graph.py to parse delegate_task output
# ---------------------------------------------------------------------------

def parse_delegation_output(raw_output: str) -> dict | None:
    """
    Parse the JSON payload returned by delegate_task.

    Returns dict with keys agent_name and result, or None if not a
    delegation payload.
    """
    try:
        data = json.loads(raw_output)
        if data.get("action") == "delegate_task":
            return {"agent_name": data["agent_name"], "result": data["result"]}
    except (json.JSONDecodeError, KeyError):
        pass
    return None
