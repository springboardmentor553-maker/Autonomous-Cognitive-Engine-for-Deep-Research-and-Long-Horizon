"""
sub_agents/
============
Specialized sub-agents for Milestone 3.

Each sub-agent has:
  - A specific purpose
  - A small focused prompt
  - A limited toolset
  - A clearly defined responsibility

The supervisor simply decides who handles each task.
"""

from sub_agents.registry import sub_agents, describe_agents
from sub_agents.summarization_agent import summarizer
from sub_agents.web_search_agent import web_searcher

__all__ = ["sub_agents", "describe_agents", "summarizer", "web_searcher"]