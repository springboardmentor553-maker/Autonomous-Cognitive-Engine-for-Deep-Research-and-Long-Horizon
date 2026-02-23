"""
Main Graph - Milestone 2: LangGraph StateGraph with VFS Context Offloading

Architecture:
  START ──► plan ──► execute ──► synthesize ──► END

  plan       – Creates structured TODOs from the user task (write_todos)
  execute    – Researches each topic with LLM, writes summaries to VFS
  synthesize – Reads all summaries from VFS, generates combined output
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .supervisor_node import plan_node
from .execution_node import execute_node
from .synthesis_node import synthesize_node


def build_graph(llm):
    """
    Build and compile the LangGraph StateGraph for Milestone 2.

    Args:
        llm: The LLM instance (e.g., ChatGroq) to use in all nodes.

    Returns:
        Compiled LangGraph graph ready for ``graph.invoke(state)``.
    """
    # Wrap node functions so they capture the LLM via closure
    def _plan(state: AgentState) -> dict:
        return plan_node(state, llm)

    def _execute(state: AgentState) -> dict:
        return execute_node(state, llm)

    def _synthesize(state: AgentState) -> dict:
        return synthesize_node(state, llm)

    # Build the StateGraph
    graph = StateGraph(AgentState)

    graph.add_node("plan", _plan)
    graph.add_node("execute", _execute)
    graph.add_node("synthesize", _synthesize)

    # Linear flow
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
