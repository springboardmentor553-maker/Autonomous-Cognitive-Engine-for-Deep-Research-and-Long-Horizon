"""
graph/nodes.py — LangGraph node functions.

Each function takes AgentState and returns a dict of state updates.

Nodes:
  planner_node   — Calls LLM to decompose the request into TODOs
  executor_node  — Runs the ReAct loop: reasons + calls tools
  tools_node     — Executes the tool calls chosen by the LLM
  synthesiser_node — Reads VFS and generates the final answer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from state import AgentState
from utils.helpers import all_todos_done, mark_todo, next_pending_todo, utc_now
from utils.logger import get_logger
from utils.parser import safe_json_loads

logger = get_logger(__name__)

# Prompts directory
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPT_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ─── Planner Node ──────────────────────────────────────────────────────────────

def make_planner_node(llm_with_tools):
    """Factory: returns a planner_node bound to the given LLM."""

    system_prompt = _load_prompt("planner_prompt.txt")

    def planner_node(state: AgentState) -> dict:
        logger.info("▶ planner_node — decomposing request into TODOs")

        messages = [
            SystemMessage(content=system_prompt[:500]),
            HumanMessage(content=f"User request:\n\n{state['user_request']}"),
        ]

        response: AIMessage = llm_with_tools.invoke(messages)

        # The LLM should have called write_todos; we return the message so
        # the tools_node can execute it and update state.
        return {
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1,
        }

    return planner_node


# ─── Tools Node ────────────────────────────────────────────────────────────────

def make_tools_node(tool_map: dict[str, BaseTool], vfs_container: dict):
    """
    Factory: returns a tools_node that executes tool calls from the last
    AIMessage and applies side-effects (VFS mutations, TODO updates) to state.
    """

    def tools_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            logger.warning("tools_node called but no tool calls found")
            return {"iteration": state.get("iteration", 0) + 1}

        tool_messages: list[ToolMessage] = []
        state_updates: dict[str, Any] = {}

        virtual_fs: dict = dict(state.get("virtual_fs", {}))
        todos: list[dict] = list(state.get("todos", []))

        for tc in last_msg.tool_calls:
            name = tc["name"]
            args = tc["args"]
            call_id = tc["id"]

            logger.info(f"  🔧 {name}({list(args.keys())})")

            tool = tool_map.get(name)
            if tool is None:
                result = json.dumps({"error": f"Unknown tool: {name}"})
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    logger.error(f"Tool '{name}' raised: {e}")
                    result = json.dumps({"error": str(e)})

            # ── Side-effects for VFS tools ────────────────────────────────────
            parsed = safe_json_loads(result) if isinstance(result, str) else result

            if isinstance(parsed, dict):
                action = parsed.get("action")

                if action == "write_file":
                    path = parsed["path"]
                    virtual_fs[path] = {
                        "content": parsed["content"],
                        "created_at": parsed.get("created_at", utc_now()),
                        "updated_at": parsed.get("updated_at", utc_now()),
                    }
                    # Keep the live reference in sync
                    vfs_container.clear()
                    vfs_container.update(virtual_fs)
                    logger.info(f"  📝 VFS write: {path}")

                elif action == "edit_file":
                    path = parsed["path"]
                    mode = parsed.get("mode", "overwrite")
                    new_content = parsed.get("content", "")
                    old_text = parsed.get("old_text", "")
                    existing = virtual_fs.get(path, {})
                    current = existing.get("content", "") if isinstance(existing, dict) else ""

                    if mode == "append":
                        updated = current + "\n" + new_content
                    elif mode == "replace" and old_text:
                        updated = current.replace(old_text, new_content, 1)
                    else:  # overwrite
                        updated = new_content

                    virtual_fs[path] = {
                        "content": updated,
                        "created_at": existing.get("created_at", utc_now()) if isinstance(existing, dict) else utc_now(),
                        "updated_at": parsed.get("updated_at", utc_now()),
                    }
                    vfs_container.clear()
                    vfs_container.update(virtual_fs)
                    logger.info(f"  ✏️  VFS edit [{mode}]: {path}")

                # ── Side-effects for write_todos ──────────────────────────────
                elif parsed.get("status") == "todos_created":
                    todos = parsed.get("todos", todos)
                    state_updates["todos"] = todos
                    state_updates["current_task_id"] = todos[0]["id"] if todos else None
                    logger.info(f"  📋 TODOs set: {len(todos)} tasks")

            tool_messages.append(
                ToolMessage(content=result, tool_call_id=call_id, name=name)
            )

        state_updates["virtual_fs"] = virtual_fs
        state_updates["messages"] = tool_messages
        return state_updates

    return tools_node


# ─── Executor Node ─────────────────────────────────────────────────────────────

def make_executor_node(llm_with_tools):
    """
    The core ReAct reasoning step.  Picks the next pending TODO, builds context,
    and lets the LLM decide which tool to call.
    """
    system_prompt = _load_prompt("researcher_prompt.txt")

    def executor_node(state: AgentState) -> dict:
        todos = state.get("todos", [])
        current_todo = next_pending_todo(todos)

        if not current_todo:
            logger.info("▶ executor_node — no pending tasks, skipping")
            return {
                "iteration": state.get("iteration", 0) + 1,
            }

        task_id = current_todo["id"]
        task_desc = current_todo["description"]
        logger.info(f"▶ executor_node — working on [{task_id}]: {task_desc}")

        # Mark task as in_progress
        updated_todos = mark_todo(todos, task_id, "in_progress")

        # Build context — list VFS files but do NOT load their contents
        # The agent must reason about which files to read, not read all of them
        vfs = state.get("virtual_fs", {})
        vfs_file_list = []
        for path, entry in vfs.items():
            content = entry.get("content", "") if isinstance(entry, dict) else entry
            word_count = len(content.split())
            vfs_file_list.append(f"  - {path} ({word_count} words)")

        vfs_summary = "\n".join(vfs_file_list) if vfs_file_list else "  (empty)"

        context = (
            f"## Current Task\n**{task_id}**: {task_desc}\n\n"
            f"## Original Request\n{state['user_request']}\n\n"
            f"## Task Plan\n"
            + "\n".join(
                f"- [{t['status']}] {t['id']}: {t['description']}" for t in updated_todos
            )
            + f"\n\n## Virtual File System (files available — do NOT read all, only read what THIS task needs)\n"
            + vfs_summary
            + "\n\n## REMINDER: Before calling read_file, state which files you need for THIS task and why."
        )

        # Trim to last 2 messages only to stay within 12k TPM limit
        recent_messages = state.get("messages", [])[-2:]

        messages = [
            SystemMessage(content=system_prompt[:500]),
            *recent_messages,
            HumanMessage(content=context),
        ]

        response: AIMessage = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "todos": updated_todos,
            "current_task_id": task_id,
            "iteration": state.get("iteration", 0) + 1,
        }

    return executor_node


# ─── Task Completion Node ──────────────────────────────────────────────────────

def make_task_complete_node(llm):
    """
    After tool execution, mark the current TODO as completed.
    Uses the last ToolMessage content as the result summary.
    """

    def task_complete_node(state: AgentState) -> dict:
        task_id = state.get("current_task_id")
        if not task_id:
            return {"iteration": state.get("iteration", 0) + 1}

        todos = state.get("todos", [])

        # Gather the last tool result as a summary
        last_tool_msgs = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
        result_summary = ""
        if last_tool_msgs:
            raw = last_tool_msgs[-1].content
            parsed = safe_json_loads(raw)
            if parsed and isinstance(parsed, dict):
                result_summary = parsed.get("summary", parsed.get("content", str(raw)))[:300]
            else:
                result_summary = str(raw)[:300]

        updated_todos = mark_todo(todos, task_id, "completed", result_summary)
        next_task = next_pending_todo(updated_todos)

        logger.info(f"  ✅ Completed [{task_id}]. Next: {next_task['id'] if next_task else 'none'}")

        return {
            "todos": updated_todos,
            "current_task_id": next_task["id"] if next_task else None,
        }

    return task_complete_node


# ─── Synthesiser Node ──────────────────────────────────────────────────────────

def make_synthesiser_node(llm):
    """
    Reads all VFS files, combines with message history, and produces the
    final comprehensive answer.
    """

    def synthesiser_node(state: AgentState) -> dict:
        logger.info("▶ synthesiser_node — generating final output")

        vfs = state.get("virtual_fs", {})

        # ── Intelligent selective reading ─────────────────────────────────────
        # Priority order: drafts > compare > summaries > research
        # We read the most processed files first, avoid raw research if
        # higher-level summaries already exist
        priority_prefixes = ["/drafts/", "/compare/", "/summaries/", "/research/"]

        selected_files = {}
        for prefix in priority_prefixes:
            for path, entry in vfs.items():
                if path.startswith(prefix) and path not in selected_files:
                    content = entry.get("content", "") if isinstance(entry, dict) else entry
                    selected_files[path] = content

            # If we found drafts or compare files, skip lower-priority files
            if prefix in ("/drafts/", "/compare/") and selected_files:
                logger.info(f"synthesiser: using {len(selected_files)} files from {prefix}, skipping raw research")
                break

        vfs_contents = ""
        for path, content in selected_files.items():
            # Truncate each file to 600 chars to stay within token limits
            truncated = content[:600] + "..." if len(content) > 600 else content
            vfs_contents += f"\n\n### {path}\n{truncated}"
            logger.info(f"synthesiser: reading {path} ({len(content)} chars)")

        skipped = [p for p in vfs.keys() if p not in selected_files]
        if skipped:
            logger.info(f"synthesiser: skipped (not needed): {skipped}")

        todos_summary = "\n".join(
            f"- [{t['status']}] {t['description']}: {t.get('result', '')[:100]}"
            for t in state.get("todos", [])
        )

        synthesis_prompt = f"""You are synthesising the final answer for the user.

## Original Request
{state['user_request']}

## Completed Tasks
{todos_summary}

## Research & Notes (from Virtual File System)
{vfs_contents if vfs_contents else "(No files saved)"}

## Instructions
Write a comprehensive, well-structured final response that fully addresses the original request.
Use the research notes above as your primary source. Format with clear headings and sections.
"""

        response = llm.invoke([HumanMessage(content=synthesis_prompt)])
        final_output = response.content

        logger.info(f"  🏁 Final output: {len(final_output)} chars")

        return {
            "final_output": final_output,
            "messages": [AIMessage(content=final_output)],
        }

    return synthesiser_node
"""
graph/nodes.py — LangGraph node functions.

Milestone 3 — Supervisor pattern (mentor spec):

The executor IS the supervisor. Every iteration it:
  First  : decides what task needs to be done next
  Second : decides whether to perform it directly or delegate
  Third  : integrates the sub-agent result into the workflow

Mentor's LangSmith evaluation checks:
  1. Did supervisor recognize when delegation is required?
  2. Did supervisor correctly call task tool with correct sub-agent?
  3. Was the result returned by sub-agent integrated into the workflow?

tools_node handles delegate_task by:
  - Running agent.invoke(input_data) inside the tool
  - Recording the delegation in delegation_log (LangSmith trace)
  - Marking the TODO as delegated
  Supervisor then calls write_file to store the result (integration step).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from state import AgentState, DelegationRecord
from utils.helpers import all_todos_done, mark_todo, next_pending_todo, utc_now
from utils.logger import get_logger
from utils.parser import safe_json_loads

logger = get_logger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPT_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER NODE
# Converts user request into a structured TODO list via write_todos.
# ─────────────────────────────────────────────────────────────────────────────
def make_planner_node(llm_with_tools):
    system_prompt = _load_prompt("planner_prompt.txt")

    def planner_node(state: AgentState) -> dict:
        logger.info("planner_node: creating task plan")
        messages = [
            SystemMessage(content=system_prompt[:500]),
            HumanMessage(content=f"User request:\n\n{state['user_request']}"),
        ]
        response: AIMessage = llm_with_tools.invoke(messages)
        return {
            "messages":  [response],
            "iteration": state.get("iteration", 0) + 1,
        }

    return planner_node


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS NODE
# Executes every tool call from the last AIMessage.
#
# Milestone 3 addition:
#   When action == "delegate_task":
#     - sub-agent already ran inside the tool via agent.invoke()
#     - we record it in delegation_log for LangSmith
#     - we mark the TODO as delegated
#     - supervisor will next call write_file to integrate the result
# ─────────────────────────────────────────────────────────────────────────────
def make_tools_node(tool_map: dict[str, BaseTool], vfs_container: dict):

    def tools_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            logger.warning("tools_node: no tool calls found")
            return {"iteration": state.get("iteration", 0) + 1}

        tool_messages:  list[ToolMessage] = []
        state_updates:  dict[str, Any]    = {}
        virtual_fs:     dict              = dict(state.get("virtual_fs", {}))
        todos:          list[dict]        = list(state.get("todos", []))
        delegation_log: list[dict]        = list(state.get("delegation_log", []))

        for tc in last_msg.tool_calls:
            name    = tc["name"]
            args    = tc["args"]
            call_id = tc["id"]

            logger.info(f"  tool: {name}({list(args.keys())})")

            tool = tool_map.get(name)
            if tool is None:
                result = json.dumps({"error": f"Unknown tool: {name}"})
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    logger.error(f"Tool '{name}' raised: {e}")
                    result = json.dumps({"error": str(e)})

            parsed = safe_json_loads(result) if isinstance(result, str) else result

            if isinstance(parsed, dict):
                action = parsed.get("action")

                # ── M2: write_file ─────────────────────────────────────────
                if action == "write_file":
                    path = parsed["path"]
                    virtual_fs[path] = {
                        "content":    parsed["content"],
                        "created_at": parsed.get("created_at", utc_now()),
                        "updated_at": parsed.get("updated_at", utc_now()),
                    }
                    vfs_container.clear()
                    vfs_container.update(virtual_fs)
                    logger.info(f"  VFS write: {path}")

                # ── M2: edit_file ──────────────────────────────────────────
                elif action == "edit_file":
                    path        = parsed["path"]
                    mode        = parsed.get("mode", "overwrite")
                    new_content = parsed.get("content", "")
                    old_text    = parsed.get("old_text", "")
                    existing    = virtual_fs.get(path, {})
                    current     = existing.get("content", "") if isinstance(existing, dict) else ""

                    updated = (
                        current + "\n" + new_content
                        if mode == "append" else
                        current.replace(old_text, new_content, 1)
                        if mode == "replace" and old_text else
                        new_content
                    )
                    virtual_fs[path] = {
                        "content":    updated,
                        "created_at": existing.get("created_at", utc_now()) if isinstance(existing, dict) else utc_now(),
                        "updated_at": parsed.get("updated_at", utc_now()),
                    }
                    vfs_container.clear()
                    vfs_container.update(virtual_fs)
                    logger.info(f"  VFS edit [{mode}]: {path}")

                # ── M1: write_todos ────────────────────────────────────────
                elif parsed.get("status") == "todos_created":
                    todos = parsed.get("todos", todos)
                    for t in todos:
                        t.setdefault("delegated_to", "")
                    state_updates["todos"]           = todos
                    state_updates["current_task_id"] = todos[0]["id"] if todos else None
                    logger.info(f"  TODOs created: {len(todos)} tasks")

                # ── M3: delegate_task ──────────────────────────────────────
                # Mentor integration check:
                #   "check whether the result returned by the sub-agent
                #    is integrated into the workflow"
                #
                # Here we:
                #   1. Record in delegation_log (LangSmith sees this)
                #   2. Mark TODO as delegated with agent name
                #   Supervisor will then call write_file with the result.
                elif action == "delegate_task":
                    agent_name      = parsed.get("agent_name", "unknown")
                    input_data      = parsed.get("input_data", "")
                    sub_result      = parsed.get("result", "")
                    current_task_id = state.get("current_task_id", "")

                    # Record for LangSmith tracing (mentor evaluation method)
                    record: DelegationRecord = {
                        "task_id":      current_task_id,
                        "agent_name":   agent_name,
                        "input_data":   input_data,
                        "result":       sub_result,
                        "status":       parsed.get("status", "completed"),
                        "delegated_at": parsed.get("delegated_at", utc_now()),
                    }
                    delegation_log.append(record)
                    state_updates["delegation_log"] = delegation_log

                    # Mark TODO with agent name (supervisor can see who handled it)
                    if current_task_id:
                        for t in todos:
                            if t["id"] == current_task_id:
                                t["delegated_to"] = agent_name
                        state_updates["todos"] = todos

                    logger.info(
                        f"  DELEGATED [{current_task_id}] -> "
                        f"'{agent_name}' | {str(sub_result)[:80]}"
                    )

            tool_messages.append(
                ToolMessage(content=result, tool_call_id=call_id, name=name)
            )

        state_updates["virtual_fs"] = virtual_fs
        state_updates["messages"]   = tool_messages
        return state_updates

    return tools_node


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTOR NODE — the Supervisor (Milestone 3)
#
# Mentor spec — on every iteration the supervisor:
#   First  : decides what task needs to be done next
#   Second : decides whether to perform it directly or delegate
#   Third  : integrates the sub-agent result into the workflow
#
# Delegate when:
#   - task requires a specific skill (summarize, web search)
#   - task is repetitive and can be standardized
#   - task requires a focused prompt
#   - task can be isolated from the main reasoning flow
#
# Handle directly when:
#   - compare, analyze, synthesize, write reports, file operations
# ─────────────────────────────────────────────────────────────────────────────
def make_executor_node(llm_with_tools):
    system_prompt = _load_prompt("executor_prompt.txt")

    def executor_node(state: AgentState) -> dict:
        todos        = state.get("todos", [])
        current_todo = next_pending_todo(todos)

        if not current_todo:
            logger.info("executor_node: no pending tasks")
            return {"iteration": state.get("iteration", 0) + 1}

        task_id   = current_todo["id"]
        task_desc = current_todo["description"]
        logger.info(f"supervisor: [{task_id}] {task_desc[:80]}")

        updated_todos = mark_todo(todos, task_id, "in_progress")

        # VFS file list (what the supervisor can read)
        vfs      = state.get("virtual_fs", {})
        vfs_list = "\n".join(
            f"  {p} ({len((e.get('content','') if isinstance(e, dict) else e).split())} words)"
            for p, e in vfs.items()
        ) or "  (empty)"

        # Show past delegations so supervisor knows what's already done
        delegation_log = state.get("delegation_log", [])
        past_dels = ""
        if delegation_log:
            past_dels = "\nPast delegations:\n" + "\n".join(
                f"  [{r['task_id']}] -> {r['agent_name']}: completed"
                for r in delegation_log
            )

        # Supervisor context (mentor: first/second/third decision steps)
        context = (
            f"## Current Task\n{task_id}: {task_desc}\n\n"
            f"## Full Task Plan\n"
            + "\n".join(
                f"  [{t['status']}] {t['id']}: {t['description']}"
                + (f"  [done by: {t.get('delegated_to','')}]" if t.get("delegated_to") else "")
                for t in updated_todos
            )
            + f"\n\n## Virtual File System\n{vfs_list}"
            + past_dels
            + "\n\n## Sub-Agents Available\n"
            + '  "summarizer"   -> task says: summarize / condense / extract key points\n'
            + '  "web_searcher" -> task says: search / find / look up / research\n'
            + "\n## Your Three Decisions\n"
            + "1. What does this task need?\n"
            + "2. Delegate to sub-agent OR handle directly?\n"
            + "3. After delegation: call write_file to store result, then continue.\n"
        )

        fallback = (
            "You are the supervisor agent. Decide whether to delegate each task "
            "to a sub-agent or handle it directly. After delegation always store "
            "the result with write_file."
        )
        messages = [
            SystemMessage(content=system_prompt[:500] if system_prompt else fallback),
            *state.get("messages", [])[-2:],
            HumanMessage(content=context),
        ]

        response: AIMessage = llm_with_tools.invoke(messages)

        return {
            "messages":        [response],
            "todos":           updated_todos,
            "current_task_id": task_id,
            "iteration":       state.get("iteration", 0) + 1,
        }

    return executor_node


# ─────────────────────────────────────────────────────────────────────────────
# TASK COMPLETION NODE
# Marks current TODO as completed after tools have run.
# ─────────────────────────────────────────────────────────────────────────────
def make_task_complete_node(llm):

    def task_complete_node(state: AgentState) -> dict:
        task_id = state.get("current_task_id")
        if not task_id:
            return {"iteration": state.get("iteration", 0) + 1}

        todos = state.get("todos", [])

        # Build result summary from last tool message
        last_tool_msgs = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
        result_summary = ""
        if last_tool_msgs:
            raw    = last_tool_msgs[-1].content
            parsed = safe_json_loads(raw)
            if parsed and isinstance(parsed, dict):
                if parsed.get("action") == "delegate_task":
                    result_summary = (
                        f"Delegated to {parsed.get('agent_name', '')}: "
                        f"{str(parsed.get('result', ''))[:150]}"
                    )
                else:
                    result_summary = parsed.get(
                        "summary", parsed.get("content", str(raw))
                    )[:200]
            else:
                result_summary = str(raw)[:200]

        updated_todos = mark_todo(todos, task_id, "completed", result_summary)
        next_task     = next_pending_todo(updated_todos)

        logger.info(
            f"  completed [{task_id}]. "
            f"Next: {next_task['id'] if next_task else 'none'}"
        )

        return {
            "todos":           updated_todos,
            "current_task_id": next_task["id"] if next_task else None,
            "iteration":       state.get("iteration", 0) + 1,
        }

    return task_complete_node


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESISER NODE
# Reads highest-priority VFS files and writes the final answer.
# Milestone 3: includes delegation summary showing what each sub-agent did.
# ─────────────────────────────────────────────────────────────────────────────
def make_synthesiser_node(llm):

    def synthesiser_node(state: AgentState) -> dict:
        logger.info("synthesiser_node: generating final output")

        vfs = state.get("virtual_fs", {})

        # Selective reading — read most processed files, skip raw research
        priority = ["/drafts/", "/compare/", "/summaries/", "/research/"]
        selected = {}
        for prefix in priority:
            for path, entry in vfs.items():
                if path.startswith(prefix) and path not in selected:
                    content = entry.get("content", "") if isinstance(entry, dict) else entry
                    selected[path] = content
            if prefix in ("/drafts/", "/compare/") and selected:
                break

        vfs_section = ""
        for path, content in selected.items():
            snippet = content[:600] + "..." if len(content) > 600 else content
            vfs_section += f"\n\n### {path}\n{snippet}"

        # Delegation summary — what each sub-agent contributed
        delegation_log = state.get("delegation_log", [])
        del_section = ""
        if delegation_log:
            del_section = "\n\n## Sub-Agent Contributions\n" + "\n".join(
                f"- [{r['task_id']}] {r['agent_name']} handled: {str(r['result'])[:100]}"
                for r in delegation_log
            )

        todos_summary = "\n".join(
            f"- [{t['status']}] {t['description']}"
            + (f"  [by {t.get('delegated_to', '')}]" if t.get("delegated_to") else "")
            for t in state.get("todos", [])
        )

        prompt = (
            f"Synthesise the final answer.\n\n"
            f"## Original Request\n{state['user_request']}\n\n"
            f"## Completed Tasks\n{todos_summary}\n\n"
            f"## Research Notes\n{vfs_section or '(No files)'}"
            + del_section
            + "\n\nWrite a comprehensive, well-structured response with clear headings."
        )

        response     = llm.invoke([HumanMessage(content=prompt)])
        final_output = response.content
        logger.info(f"  final output: {len(final_output)} chars")

        return {
            "final_output": final_output,
            "messages":     [AIMessage(content=final_output)],
        }

    return synthesiser_node