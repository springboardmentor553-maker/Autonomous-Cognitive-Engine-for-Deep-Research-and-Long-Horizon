"""
LangGraph workflow definition for the Autonomous Cognitive Engine.

Graph topology
--------------

  [START]
     │
     ▼
 supervisor ──── (has tool calls?) ──Yes──► process_tool_calls
     ▲                                              │
     └──────────────────────────────────────────────┘
     │
  (no tool calls)
     │
     ▼
  [END]

The ``supervisor`` node reasons and either:
  - calls one or more tools  → ``process_tool_calls`` handles them, loops back
  - produces a plain text response → graph ends, final_output is set
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import ToolMessage, AIMessage

from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from agents.supervisor_agent import supervisor_node, ALL_TOOLS
from tools.write_todos import parse_write_todos_output
from tools.file_system_tools import apply_vfs_action


# ---------------------------------------------------------------------------
# Tool name → callable map (for dispatch)
# ---------------------------------------------------------------------------

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

VFS_TOOL_NAMES = {"ls", "read_file", "write_file", "edit_file"}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_supervisor(state: AgentState) -> Literal["process_tool_calls", "__end__"]:
    """
    Decide whether to execute tool calls or end the graph.

    If the last message contains tool_calls we route to the tool
    executor node; otherwise the agent has produced its final answer.
    """
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "process_tool_calls"
    return "__end__"


# ---------------------------------------------------------------------------
# Tool execution node
# ---------------------------------------------------------------------------

def process_tool_calls(state: AgentState) -> dict:
    """
    Execute every tool call present in the last AIMessage.

    Handles three categories of tool calls:
    1. ``write_todos``  – parse payload and update state["todos"]
    2. VFS tools        – apply mutation to state["files"] via apply_vfs_action
    3. All other tools  – invoke directly and return result

    Parameters
    ----------
    state : AgentState
        Current graph state.

    Returns
    -------
    dict
        Partial state update with new ToolMessages appended to messages,
        plus any mutations to ``todos``, ``files``, and
        ``intermediate_results``.
    """
    last_msg: AIMessage = state["messages"][-1]
    tool_calls = last_msg.tool_calls

    new_messages: list[ToolMessage] = []
    updated_todos = list(state.get("todos", []))
    updated_files = dict(state.get("files", {}))
    new_intermediate: list[str] = []

    for call in tool_calls:
        tool_name: str = call["name"]
        tool_args: dict = call["args"]
        call_id: str = call["id"]

        tool_fn = TOOL_MAP.get(tool_name)
        if tool_fn is None:
            result_str = f"ERROR: Unknown tool '{tool_name}'."
        else:
            try:
                # Invoke the tool with its arguments
                raw_result = tool_fn.invoke(tool_args)
            except Exception as exc:  # noqa: BLE001
                raw_result = f"ERROR executing '{tool_name}': {exc}"

            # ---- write_todos ------------------------------------------------
            if tool_name == "write_todos":
                parsed = parse_write_todos_output(str(raw_result))
                if parsed is not None:
                    updated_todos = parsed
                    result_str = (
                        f"TODO list created with {len(parsed)} tasks:\n"
                        + "\n".join(
                            f"  [{i+1}] {t['task']}" for i, t in enumerate(parsed)
                        )
                    )
                else:
                    result_str = str(raw_result)

            # ---- VFS tools --------------------------------------------------
            elif tool_name in VFS_TOOL_NAMES:
                result_str, updated_files = apply_vfs_action(str(raw_result), updated_files)
                # Store notable write/edit results as intermediate results
                if tool_name in {"write_file", "edit_file"} and "successfully" in result_str:
                    fname = tool_args.get("filename", "unknown")
                    content = tool_args.get("content") or tool_args.get("new_content", "")
                    new_intermediate.append(f"[{fname}]: {content[:200]}")

            # ---- Tavily search ----------------------------------------------
            else:
                result_str = str(raw_result)
                # Store search results as intermediate results
                new_intermediate.append(
                    f"[search: {tool_args.get('query', '')}]: {result_str[:300]}"
                )

        new_messages.append(
            ToolMessage(content=result_str, tool_call_id=call_id)
        )

    # -------------------------------------------------------------------------
    # TODO status tracking
    # Logic:
    #   - write_file / edit_file = task completed → mark "done", advance pointer
    #   - tavily_search alone    = task in progress
    #   - read_file calls        = synthesis phase → mark all remaining "done"
    # -------------------------------------------------------------------------
    current_task_idx = state.get("current_task", 0)
    tool_names_this_step = [c["name"] for c in tool_calls]

    if updated_todos:
        wrote_file = any(t in {"write_file", "edit_file"} for t in tool_names_this_step)
        searched = any(t == "tavily_search" for t in tool_names_this_step)
        reading_files = all(t == "read_file" for t in tool_names_this_step)

        if wrote_file and current_task_idx < len(updated_todos):
            # File written → task is done, advance pointer
            updated_todos[current_task_idx] = {
                **updated_todos[current_task_idx],
                "status": "done",
            }
            current_task_idx = min(current_task_idx + 1, len(updated_todos) - 1)

        elif searched and current_task_idx < len(updated_todos):
            # Searching → mark current task in_progress
            updated_todos[current_task_idx] = {
                **updated_todos[current_task_idx],
                "status": "in_progress",
            }

        elif reading_files:
            # Synthesis phase — mark all remaining tasks done
            for i, todo in enumerate(updated_todos):
                if todo["status"] != "done":
                    updated_todos[i] = {**todo, "status": "done"}
            current_task_idx = len(updated_todos) - 1

    return {
        "messages": new_messages,
        "todos": updated_todos,
        "files": updated_files,
        "intermediate_results": state.get("intermediate_results", []) + new_intermediate,
        "current_task": current_task_idx,
    }


# ---------------------------------------------------------------------------
# set_final_output node
# ---------------------------------------------------------------------------

def set_final_output(state: AgentState) -> dict:
    """
    Extract the agent's last plain-text message as the final output.

    Parameters
    ----------
    state : AgentState

    Returns
    -------
    dict
        Partial state update setting ``final_output``.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return {"final_output": msg.content}
    return {"final_output": ""}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready to be invoked with an initial state.
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("process_tool_calls", process_tool_calls)
    builder.add_node("set_final_output", set_final_output)

    # Edges
    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "process_tool_calls": "process_tool_calls",
            "__end__": "set_final_output",
        },
    )

    builder.add_edge("process_tool_calls", "supervisor")
    builder.add_edge("set_final_output", END)

    return builder.compile()
