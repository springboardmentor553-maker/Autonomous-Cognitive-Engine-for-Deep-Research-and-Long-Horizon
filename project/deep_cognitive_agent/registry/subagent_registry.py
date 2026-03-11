"""
Sub-Agent Registry - Milestone 3: Multi-Agent Collaboration

Central registry for managing available sub-agents.
Provides registration, discovery, instantiation, and listing
of specialized agents (researcher, summarizer, comparator, etc.).
"""

from typing import Dict, Optional, Type


class SubAgentRegistry:
    """Registry mapping agent names to their classes."""

    def __init__(self):
        self._agents: Dict[str, Type] = {}

    def register(self, name: str, agent_class: Type) -> None:
        """Register a sub-agent class by name."""
        self._agents[name] = agent_class

    def get(self, name: str) -> Optional[Type]:
        """Retrieve a registered agent class by name."""
        return self._agents.get(name)

    def list_agents(self) -> list:
        """Return a list of all registered agent names."""
        return list(self._agents.keys())

    def create(self, name: str, llm, **kwargs):
        """Instantiate a registered agent.

        Args:
            name: Registered agent name.
            llm: LLM instance to pass to the agent constructor.

        Returns:
            An instantiated agent, or None if not found.
        """
        agent_class = self._agents.get(name)
        if agent_class is None:
            return None
        return agent_class(llm, **kwargs)


# ── Global registry instance ─────────────────────────────────────────
registry = SubAgentRegistry()


def build_registry():
    """Register all available sub-agents in the global registry.

    This is called once at startup to populate the registry with
    all specialized sub-agents for Milestone 3.
    """
    from agents.subagents.research_agent import ResearchAgent
    from agents.subagents.summarizer_agent import SummarizerAgent
    from agents.subagents.comparator_agent import ComparatorAgent
    from agents.subagents.unifier_agent import UnifierAgent
    from agents.subagents.refiner_agent import RefinerAgent

    registry.register("researcher", ResearchAgent)
    registry.register("summarizer", SummarizerAgent)
    registry.register("comparator", ComparatorAgent)
    registry.register("unifier", UnifierAgent)
    registry.register("refiner", RefinerAgent)

    return registry
