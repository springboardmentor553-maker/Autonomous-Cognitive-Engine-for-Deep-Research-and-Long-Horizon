"""
sub_agents/ — Specialized sub-agent package for Milestone 3.

Available agents:
    - summarization_agent : Summarizes content with a focused, isolated context.
    - web_search_agent    : Performs research / knowledge retrieval tasks.
    - code_analysis_agent : Analyzes, reviews, or drafts code.

Each agent is a standalone LangGraph graph that operates in its own context
and returns a plain-text result string to the supervisor agent.
"""

from sub_agents.registry import SUB_AGENT_REGISTRY, run_sub_agent

__all__ = ["SUB_AGENT_REGISTRY", "run_sub_agent"]
