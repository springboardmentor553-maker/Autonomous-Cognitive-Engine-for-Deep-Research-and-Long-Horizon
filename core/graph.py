"""
LangGraph workflow definition for the Autonomous Cognitive Engine.

Milestone 3: delegate_task support, automatic summarization after synthesis.

Graph topology
--------------
  [START] -> supervisor -> (tool calls?) -> process_tool_calls -> supervisor
                        -> (no tool calls) -> set_final_output -> [END]
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from core.state import AgentState, DelegationRecord
from agents.supervisor_agent import supervisor_node, ALL_TOOLS
from tools.write_todos import parse_write_todos_output
from tools.file_system_tools import apply_vfs_action
from tools.delegate_task import parse_delegation_output


TOOL_MAP = {t.name: t for t in ALL_TOOLS}
VFS_TOOL_NAMES = {"ls", "read_file", "write_file", "edit_file"}


def route_after_supervisor(state: AgentState) -> Literal["process_tool_calls", "__end__"]:
    """Route to tool executor if last message has tool calls, else end."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "process_tool_calls"
    return "__end__"


def process_tool_calls(state: AgentState) -> dict:
    """
    Execute every tool call in the last AIMessage and update state.

    Categories:
      1. write_todos   - updates state["todos"]
      2. delegate_task - runs sub-agent, logs to delegation_history + sub_agent_results
      3. VFS tools     - mutates state["files"]
      4. Everything else (tavily_search) - direct invoke

    Auto-summarization:
      When the agent reads back ALL saved files in a single pass (synthesis phase),
      the graph automatically calls the summarization_agent on the combined content
      and stores the result as "auto_summary.txt" — no model orchestration needed.
    """
    last_msg: AIMessage = state["messages"][-1]
    tool_calls = last_msg.tool_calls

    new_messages: list[ToolMessage] = []
    updated_todos = list(state.get("todos", []))
    updated_files = dict(state.get("files", {}))
    new_intermediate: list[str] = []
    updated_delegation_history: list[DelegationRecord] = list(state.get("delegation_history", []))
    updated_sub_agent_results: dict[str, str] = dict(state.get("sub_agent_results", {}))

    for call in tool_calls:
        tool_name: str = call["name"]
        tool_args: dict = call["args"]
        call_id: str = call["id"]

        tool_fn = TOOL_MAP.get(tool_name)
        if tool_fn is None:
            result_str = f"ERROR: Unknown tool '{tool_name}'."
        else:
            try:
                raw_result = tool_fn.invoke(tool_args)
            except Exception as exc:
                raw_result = f"ERROR executing '{tool_name}': {exc}"

            # ---- write_todos ------------------------------------------------
            if tool_name == "write_todos":
                parsed = parse_write_todos_output(str(raw_result))
                if parsed is not None:
                    updated_todos = parsed
                    lines = "\n".join(f"  [{i+1}] {t['task']}" for i, t in enumerate(parsed))
                    result_str = f"TODO list created with {len(parsed)} tasks:\n{lines}"
                else:
                    result_str = str(raw_result)

            # ---- delegate_task ----------------------------------------------
            elif tool_name == "delegate_task":
                delegation = parse_delegation_output(str(raw_result))
                if delegation is not None:
                    agent_name = delegation["agent_name"]
                    result = delegation["result"]
                    task_str = tool_args.get("task", "")
                    result_key = f"{agent_name}:{task_str[:60]}"
                    record: DelegationRecord = {
                        "agent_name": agent_name,
                        "task": task_str,
                        "result": result,
                    }
                    updated_delegation_history.append(record)
                    updated_sub_agent_results[result_key] = result
                    new_intermediate.append(f"[delegated to {agent_name}]: {result[:300]}")
                    result_str = (
                        f"Sub-agent '{agent_name}' completed successfully.\n\nResult:\n{result}"
                    )
                else:
                    result_str = str(raw_result)

            # ---- VFS tools --------------------------------------------------
            elif tool_name in VFS_TOOL_NAMES:
                result_str, updated_files = apply_vfs_action(str(raw_result), updated_files)
                if tool_name in {"write_file", "edit_file"} and "successfully" in result_str:
                    fname = tool_args.get("filename", "unknown")
                    content = tool_args.get("content") or tool_args.get("new_content", "")
                    new_intermediate.append(f"[{fname}]: {content[:200]}")

            # ---- tavily_search and everything else --------------------------
            else:
                result_str = str(raw_result)
                new_intermediate.append(
                    f"[search: {tool_args.get('query', '')}]: {result_str[:300]}"
                )

        new_messages.append(ToolMessage(content=result_str, tool_call_id=call_id))

    # ---- TODO status tracking -----------------------------------------------
    current_task_idx = state.get("current_task", 0)
    tool_names_this_step = [c["name"] for c in tool_calls]

    if updated_todos:
        wrote_file    = any(t in {"write_file", "edit_file"} for t in tool_names_this_step)
        delegated     = any(t == "delegate_task" for t in tool_names_this_step)
        searched      = any(t == "tavily_search" for t in tool_names_this_step)
        reading_files = all(t == "read_file" for t in tool_names_this_step)

        if (wrote_file or delegated) and current_task_idx < len(updated_todos):
            updated_todos[current_task_idx] = {
                **updated_todos[current_task_idx],
                "status": "done",
            }
            current_task_idx = min(current_task_idx + 1, len(updated_todos) - 1)

        elif searched and current_task_idx < len(updated_todos):
            updated_todos[current_task_idx] = {
                **updated_todos[current_task_idx],
                "status": "in_progress",
            }

        elif reading_files:
            # Synthesis phase — mark all remaining done
            for i, todo in enumerate(updated_todos):
                if todo["status"] != "done":
                    updated_todos[i] = {**todo, "status": "done"}
            current_task_idx = len(updated_todos) - 1

            # ---- Auto-summarization -----------------------------------------
            # If there are delegation results and no summary file yet,
            # automatically run the summarization sub-agent on all file content.
            # This replaces the fragile "model calls summarization_agent" pattern.
            if updated_delegation_history and "auto_summary.txt" not in updated_files:
                combined = "\n\n---\n\n".join(
                    f"## {fname}\n{content}"
                    for fname, content in updated_files.items()
                    if not fname.startswith("auto_")
                )
                if combined.strip():
                    try:
                        from agents.summarization_agent import run_summarization_agent
                        # Limit combined text to avoid token overflows
                        summary = run_summarization_agent(combined[:4000])
                        updated_files["auto_summary.txt"] = summary
                        # Log as a delegation record so it appears in the summary section
                        updated_delegation_history.append({
                            "agent_name": "summarization_agent",
                            "task": "auto-summarize all research files",
                            "result": summary,
                        })
                        updated_sub_agent_results["summarization_agent:auto"] = summary
                        new_intermediate.append(f"[auto_summary.txt]: {summary[:200]}")
                    except Exception:
                        pass  # summarization is optional — don't crash the run

    return {
        "messages": new_messages,
        "todos": updated_todos,
        "files": updated_files,
        "intermediate_results": state.get("intermediate_results", []) + new_intermediate,
        "current_task": current_task_idx,
        "delegation_history": updated_delegation_history,
        "sub_agent_results": updated_sub_agent_results,
    }


def set_final_output(state: AgentState) -> dict:
    """Extract the agent's last plain-text message as the final output."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return {"final_output": msg.content}
    return {"final_output": ""}


def build_graph() -> StateGraph:
    """Assemble and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("process_tool_calls", process_tool_calls)
    builder.add_node("set_final_output", set_final_output)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"process_tool_calls": "process_tool_calls", "__end__": "set_final_output"},
    )
    builder.add_edge("process_tool_calls", "supervisor")
    builder.add_edge("set_final_output", END)
    return builder.compile()
