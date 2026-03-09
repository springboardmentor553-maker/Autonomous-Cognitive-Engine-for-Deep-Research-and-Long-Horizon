"""
graph/build_graph.py — Wires all nodes into a compiled LangGraph StateGraph.

Graph topology:

    START
      │
      ▼
   planner ──(tool_calls?)──► tools ──(write_todos done)──► executor
                                                               │
                                        ┌──────────────────────┤
                                        │ (tool_calls?)        │
                                        ▼                      ▼
                                      tools            task_complete
                                        │                      │
                                        └──────────┬───────────┘
                                                   │
                                        (all done? ▼ : executor)
                                               synthesiser
                                                   │
                                                  END
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

import config
from graph.nodes import (
    make_executor_node,
    make_planner_node,
    make_synthesiser_node,
    make_task_complete_node,
    make_tools_node,
)
from graph.router import (
    route_after_executor,
    route_after_planner,
    route_after_task_complete,
    route_after_tools,
)
from state import AgentState
from tools import ALL_TOOLS
from tools.filesystem import bind_vfs
from utils.logger import get_logger

logger = get_logger(__name__)


def build_graph():
    """
    Construct and compile the agent StateGraph.

    Returns:
        CompiledGraph ready to invoke with an initial AgentState.
    """
    config.validate_config()

    # ── LLM setup ─────────────────────────────────────────────────────────────
    base_llm = ChatGroq(
        model=config.MODEL_NAME,
        groq_api_key=config.GROQ_API_KEY,
        temperature=0,
    )
    llm_with_tools = base_llm.bind_tools(ALL_TOOLS)

    # ── Shared VFS container (mutable dict passed by reference) ──────────────
    # read_file and ls tools need live access to the VFS stored in state.
    # Since tool functions are stateless, we share a dict that nodes keep in sync.
    vfs_container: dict = {}
    bind_vfs(vfs_container)

    # ── Tool map for tools_node ───────────────────────────────────────────────
    tool_map = {t.name: t for t in ALL_TOOLS}

    # ── Node factories ────────────────────────────────────────────────────────
    planner_node = make_planner_node(llm_with_tools)
    executor_node = make_executor_node(llm_with_tools)
    tools_node = make_tools_node(tool_map, vfs_container)
    task_complete_node = make_task_complete_node(base_llm)
    synthesiser_node = make_synthesiser_node(base_llm)

    # ── Graph assembly ────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tools", tools_node)
    graph.add_node("executor", executor_node)
    graph.add_node("task_complete", task_complete_node)
    graph.add_node("synthesiser", synthesiser_node)

    # Edges
    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "synthesiser": "synthesiser"},
    )

    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"executor": "executor", "synthesiser": "synthesiser", "task_complete": "task_complete"},
    )

    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {"tools": "tools", "task_complete": "task_complete", "synthesiser": "synthesiser"},
    )

    graph.add_conditional_edges(
        "task_complete",
        route_after_task_complete,
        {"executor": "executor", "synthesiser": "synthesiser"},
    )

    graph.add_edge("synthesiser", END)

    compiled = graph.compile()
    logger.info("✅ Graph compiled successfully")
    return compiled