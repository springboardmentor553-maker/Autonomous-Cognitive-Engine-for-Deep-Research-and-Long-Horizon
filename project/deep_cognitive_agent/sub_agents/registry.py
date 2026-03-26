from project.deep_cognitive_agent.sub_agents.summarizer.summarizer_agent import summarization_agent

"""
Sub-Agent Registry

This file keeps track of all specialized agents
that the supervisor can delegate tasks to.
"""

sub_agents = {
    "summarizer": summarization_agent
}