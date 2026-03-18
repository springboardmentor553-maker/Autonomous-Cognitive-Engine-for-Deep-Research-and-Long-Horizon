"""
graph/router.py — Conditional edge functions for the LangGraph state machine.

Routers inspect state and return the name of the next node to visit.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from state import AgentState
from utils.helpers import all_todos_done, next_pending_todo
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of agent iterations before forcing termination
MAX_ITERATIONS = 50


def route_after_planner(state: AgentState) -> str:
    """
    After the planner LLM responds:
    - If it called write_todos → go to tools_node to execute it
    - Otherwise → something went wrong, go to synthesiser
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        logger.debug("route_after_planner → tools")
        return "tools"
    logger.warning("route_after_planner → synthesiser (no tool call)")
    return "synthesiser"


def route_after_tools(state: AgentState) -> str:
    """
    After tools execute:
    - If todos were just written (no tasks done yet) → executor
    - If all todos done → synthesiser
    - Otherwise → executor for next task
    """
    todos = state.get("todos", [])

    if not todos:
        logger.debug("route_after_tools → executor (no todos yet)")
        return "executor"

    if all_todos_done(todos):
        logger.debug("route_after_tools → synthesiser (all tasks done)")
        return "synthesiser"

    # Check if current task is completed — go to task_complete first
    current_task_id = state.get("current_task_id")
    if current_task_id:
        current = next((t for t in todos if t["id"] == current_task_id), None)
        if current and current["status"] == "in_progress":
            logger.debug("route_after_tools → task_complete")
            return "task_complete"

    logger.debug("route_after_tools → executor (pending tasks remain)")
    return "executor"


def route_after_executor(state: AgentState) -> str:
    """
    After the executor LLM responds:
    - If no pending tasks remain → synthesiser
    - If it issued tool calls → tools_node
    - If no tool calls (free-form answer) → task_complete
    - If iteration limit reached → synthesiser
    """
    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"route_after_executor → synthesiser (iteration limit {MAX_ITERATIONS})")
        return "synthesiser"

    # If no pending tasks left, go straight to synthesiser
    todos = state.get("todos", [])
    if todos and all_todos_done(todos):
        logger.debug("route_after_executor → synthesiser (all tasks done)")
        return "synthesiser"

    # If there are no todos at all yet, go to synthesiser
    if not todos:
        logger.debug("route_after_executor → synthesiser (no todos)")
        return "synthesiser"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        logger.debug("route_after_executor → tools")
        return "tools"

    # No tool calls: mark task complete
    logger.debug("route_after_executor → task_complete")
    return "task_complete"


def route_after_task_complete(state: AgentState) -> str:
    """
    After marking a task complete:
    - If more tasks remain → executor
    - If all done → synthesiser
    """
    todos = state.get("todos", [])

    if all_todos_done(todos):
        logger.debug("route_after_task_complete → synthesiser")
        return "synthesiser"

    logger.debug("route_after_task_complete → executor")
    return "executor"

