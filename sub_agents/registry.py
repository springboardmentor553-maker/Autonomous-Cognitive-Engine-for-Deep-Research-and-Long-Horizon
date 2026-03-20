"""
sub_agents/registry.py - Sub-Agent Registry for Milestone 3 Delegation

Maps agent names (strings) to their runner functions so the supervision agent
can dynamically route sub-tasks without knowing implementation details.

Available agents:
    "summarization_agent"  — Summarizes content into structured overviews.
    "web_search_agent"     — Performs deep research and knowledge gathering.
    "code_analysis_agent"  — Reviews, analyzes, or drafts code artifacts.

Usage:
    from sub_agents.registry import SUB_AGENT_REGISTRY, run_sub_agent

    result = run_sub_agent("web_search_agent", task="Research quantum computing trends", context="")
"""

from sub_agents.summarization_agent import run_summarization_agent
from sub_agents.web_search_agent import run_web_search_agent
from sub_agents.code_analysis_agent import run_code_analysis_agent


# ─────────────────────────────────────────────────────────────────────────────
# Registry: maps agent_name → (runner_function, description)
# ─────────────────────────────────────────────────────────────────────────────

SUB_AGENT_REGISTRY: dict[str, dict] = {
    "summarization_agent": {
        "runner": run_summarization_agent,
        "description": (
            "Summarizes long or complex content into structured, bullet-pointed summaries. "
            "Use when you need to condense articles, reports, or large text blocks."
        ),
        "example_tasks": [
            "Summarize the following research paper on transformer models.",
            "Create a concise overview of the pros and cons I've gathered.",
        ],
    },
    "web_search_agent": {
        "runner": run_web_search_agent,
        "description": (
            "Performs deep research on topics using built-in knowledge and structured fact-gathering. "
            "Use when you need to investigate a topic, gather statistics, or find expert opinions. "
            "This agent records findings systematically and produces a research report."
        ),
        "example_tasks": [
            "Research the current state of quantum computing hardware.",
            "Find key facts about the EU AI Act and its implications.",
        ],
    },
    "code_analysis_agent": {
        "runner": run_code_analysis_agent,
        "description": (
            "Analyzes, reviews, explains, or drafts code. "
            "Use for code review, architecture analysis, security review, refactoring advice, "
            "or when you need to produce a code artifact."
        ),
        "example_tasks": [
            "Review this Python function for bugs and performance issues.",
            "Draft a LangGraph agent graph based on these requirements.",
        ],
    },
}


def run_sub_agent(agent_name: str, task: str, context: str = "") -> str:
    """
    Dispatch a task to a named sub-agent.

    Args:
        agent_name : One of the keys in SUB_AGENT_REGISTRY.
        task       : The specific sub-task description.
        context    : Optional context, code, or content to pass to the sub-agent.

    Returns:
        The sub-agent's output as a plain-text string.

    Raises:
        ValueError: If agent_name is not in the registry.
    """
    if agent_name not in SUB_AGENT_REGISTRY:
        available = list(SUB_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown sub-agent '{agent_name}'. "
            f"Available agents: {available}"
        )

    runner = SUB_AGENT_REGISTRY[agent_name]["runner"]
    return runner(task=task, context=context)


def list_available_agents() -> list[dict]:
    """
    Return a list of available sub-agents with their descriptions.

    Returns:
        List of dicts with 'name', 'description', and 'example_tasks'.
    """
    return [
        {
            "name": name,
            "description": info["description"],
            "example_tasks": info["example_tasks"],
        }
        for name, info in SUB_AGENT_REGISTRY.items()
    ]
