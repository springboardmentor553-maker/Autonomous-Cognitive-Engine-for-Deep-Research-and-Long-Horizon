"""
Main Graph - Milestone 3: Multi-Agent Collaboration

Architecture:
  START ──► supervisor ──► execute_with_delegation ──► synthesize ──► END

  supervisor              – Creates structured TODOs (reuses plan_node from M2)
  execute_with_delegation – Delegates each step to specialized sub-agents
  synthesize              – Selective retrieval + final summary (reuses M2)

New in Milestone 3:
  ✔ Supervisor agent coordinates — does NOT execute itself
  ✔ Sub-agents (researcher, summarizer, comparator, unifier, refiner)
  ✔ Task delegation tool routes tasks to correct agent via registry
  ✔ SubAgentRegistry provides agent discovery and instantiation
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .supervisor_node import plan_node
from .execution_node_m3 import execute_with_delegation
from .synthesis_node import synthesize_node


def build_graph_m3(llm, registry):
    """
    Build and compile the LangGraph StateGraph for Milestone 3.

    Args:
        llm: The LLM instance (e.g., ChatGroq) to use in all nodes.
        registry: SubAgentRegistry with all sub-agents registered.

    Returns:
        Compiled LangGraph graph ready for ``graph.invoke(state)``.
    """
    def _supervisor(state: AgentState) -> dict:
        return plan_node(state, llm)

    def _execute(state: AgentState) -> dict:
        return execute_with_delegation(state, llm, registry)

    def _synthesize(state: AgentState) -> dict:
        return synthesize_node(state, llm)

    graph = StateGraph(AgentState)

    graph.add_node("supervisor", _supervisor)
    graph.add_node("execute", _execute)
    graph.add_node("synthesize", _synthesize)

    # Linear flow: supervisor delegates → agents execute → synthesize
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
