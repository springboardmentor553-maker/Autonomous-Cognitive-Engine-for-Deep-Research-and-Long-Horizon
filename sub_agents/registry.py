"""
sub_agents/registry.py
Sub-Agent Registry — Deep Cognitive Task Framework

Central registry mapping agent names to their runner functions and metadata.
The supervisor agent uses this (via delegation_tool.py) to discover and invoke
specialized sub-agents.

Registered agents:
  - web_search_agent      : Deep research and fact-finding
  - summarization_agent   : Condensing and structuring information
  - code_analysis_agent   : Technical review and code recommendations
"""

from sub_agents import web_search_agent, summarization_agent, code_analysis_agent

# ─────────────────────────────────────────────
# Registry: name → {runner, description, example_tasks}
# ─────────────────────────────────────────────

SUB_AGENT_REGISTRY = {
    "web_search_agent": {
        "run": web_search_agent.run,
        "description": (
            "Performs deep research on any topic. Use for fact-finding, "
            "gathering background knowledge, investigating current trends, "
            "or researching historical context."
        ),
        "example_tasks": [
            "Research the history of quantum computing hardware milestones",
            "Investigate the current state of EU AI regulation",
            "Find recent advances in transformer model efficiency (2022-2024)",
        ],
    },
    "summarization_agent": {
        "run": summarization_agent.run,
        "description": (
            "Condenses large or complex information into structured summaries. "
            "Use for distilling key points from long content, comparing options, "
            "or producing executive summaries."
        ),
        "example_tasks": [
            "Summarize the key differences between microservices and monolithic architecture",
            "Summarize the main concepts of retrieval augmented generation (RAG)",
            "Summarize the core principles of DevOps and CI/CD pipelines",
        ],
    },
    "code_analysis_agent": {
        "run": code_analysis_agent.run,
        "description": (
            "Reviews code, analyses technical architectures, and produces "
            "engineering recommendations. Use for code review, comparing "
            "technologies, security analysis, or design decisions."
        ),
        "example_tasks": [
            "Analyse the pros and cons of Python vs Go for a high-performance backend API",
            "Review best practices for securing a REST API in production",
            "Analyse the trade-offs of SQL vs NoSQL for a real-time analytics platform",
        ],
    },
}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def run_sub_agent(agent_name: str, task: str, context: str = "") -> str:
    """
    Invoke the named sub-agent with a task and optional context.

    Args:
        agent_name : Key in SUB_AGENT_REGISTRY.
        task       : Description of what the sub-agent should do.
        context    : Optional background content/code to pass to the agent.

    Returns:
        The sub-agent's output as a string.

    Raises:
        KeyError  : If agent_name is not in the registry.
        Exception : Propagates any error from the sub-agent runner.
    """
    if agent_name not in SUB_AGENT_REGISTRY:
        raise KeyError(
            f"Unknown sub-agent '{agent_name}'. "
            f"Available: {list(SUB_AGENT_REGISTRY.keys())}"
        )
    runner = SUB_AGENT_REGISTRY[agent_name]["run"]
    return runner(task=task, context=context)


def list_available_agents() -> list[dict]:
    """
    Return a list of dicts describing each registered sub-agent.
    Used by the `list_agents` tool so the supervisor can discover available agents.
    """
    return [
        {
            "name": name,
            "description": info["description"],
            "example_tasks": info["example_tasks"],
        }
        for name, info in SUB_AGENT_REGISTRY.items()
    ]