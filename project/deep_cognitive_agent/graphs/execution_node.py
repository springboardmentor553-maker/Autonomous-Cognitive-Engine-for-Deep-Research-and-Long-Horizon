"""
Execution Node - Milestone 2: Smart Multi-Step Execution

Processes each enriched TODO step based on its step_type:
  - research  → LLM generates content → write_file(meaningful_name.txt)
  - compare   → read_file(dependencies) → LLM comparison → write_file
  - unify     → read_file(comparison) → LLM unified model → write_file
  - refine    → read_file(target) → LLM refinement → edit_file (not write!)

Key architectural principles:
  ✔ Selective retrieval: only reads files listed in depends_on
  ✔ Meaningful file names: derived from task content, not numbered
  ✔ edit_file for refinement: demonstrates read→modify→edit pattern
  ✔ Trace logging: every tool call recorded with purpose
  ✔ Memory offloading: content written to VFS, not kept in messages

Architecture:  START → plan → **execute** → synthesize → END
"""

import re
import time

from langchain_core.messages import AIMessage

from tools.vfs.write_file import write_file
from tools.vfs.read_file import read_file
from tools.vfs.edit_file import edit_file
from utils.helpers import (
    parse_retry_after,
    is_rate_limit_error,
    is_server_overload_error,
    sanitize_llm_output,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _invoke_llm_with_retry(llm, prompt: str, max_retries: int = 3) -> str:
    """Invoke the LLM with automatic rate-limit retry."""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return sanitize_llm_output(response.content)
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries - 1:
                if is_rate_limit_error(err_str):
                    wait = parse_retry_after(err_str)
                    print(f"  ⏳ Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if is_server_overload_error(err_str):
                    wait = min(2 ** attempt * 10, 60)
                    print(f"  ⏳ Server overloaded (503). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
            raise


def _log_trace(trace_log: list, action: str, filename: str,
               purpose: str, step: int):
    """Append a structured trace entry for evaluation visibility."""
    trace_log.append({
        "action": action,
        "file": filename,
        "purpose": purpose,
        "step": step,
    })


# ── Step Handlers ────────────────────────────────────────────────────

def _handle_research(todo: dict, step_num: int, vfs_state: dict,
                     trace_log: list, llm) -> str:
    """Execute a research step: LLM generates content → write_file."""
    task = todo["task"]
    output_file = todo["output_file"]

    prompt = (
        f"Write a detailed, substantive paragraph (at least 150 words) about "
        f"the following topic. Include specific facts, data points, examples, "
        f"and expert analysis where appropriate.\n\n"
        f"Topic: {task}\n\n"
        f"Write ONLY the content paragraph — no titles, headings, or extra "
        f"formatting."
    )
    content = _invoke_llm_with_retry(llm, prompt)

    # Write summary to VFS — offloads content from context
    result = write_file(vfs_state, output_file, content)
    _log_trace(trace_log, "write_file", output_file,
               f"Store research summary: {task[:60]}", step_num)
    print(f"    → write_file('{output_file}') — {len(content)} chars")

    return f"Researched '{task[:50]}' → {output_file}"


def _handle_compare(todo: dict, step_num: int, vfs_state: dict,
                    trace_log: list, llm) -> str:
    """Execute a comparison step: selective read → compare → write."""
    task = todo["task"]
    output_file = todo["output_file"]
    depends_on = todo.get("depends_on", [])

    # Selective retrieval: read ONLY the dependency files
    dep_contents = []
    for dep in depends_on:
        content = read_file(vfs_state, dep)
        _log_trace(trace_log, "read_file", dep,
                   f"Selective load for comparison (step {step_num})", step_num)
        dep_contents.append(f"--- {dep} ---\n{content}")
        print(f"    → read_file('{dep}') — selective retrieval")

    combined = "\n\n".join(dep_contents)

    prompt = (
        f"You are given individual research summaries on related topics. "
        f"Write a thorough comparison analysis that:\n"
        f"1. Identifies key differences between the topics\n"
        f"2. Highlights surprising similarities\n"
        f"3. Analyzes relative strengths and weaknesses\n"
        f"4. Notes complementary aspects\n\n"
        f"Task context: {task}\n\n"
        f"Research summaries:\n\n{combined}\n\n"
        f"Write the comparison analysis now (at least 200 words):"
    )
    comparison = _invoke_llm_with_retry(llm, prompt)

    result = write_file(vfs_state, output_file, comparison)
    _log_trace(trace_log, "write_file", output_file,
               "Store comparison analysis", step_num)
    print(f"    → write_file('{output_file}') — {len(comparison)} chars")

    return f"Compared {len(depends_on)} sources → {output_file}"


def _handle_unify(todo: dict, step_num: int, vfs_state: dict,
                  trace_log: list, llm) -> str:
    """Execute a synthesis/unify step: read comparison → propose model."""
    task = todo["task"]
    output_file = todo["output_file"]
    depends_on = todo.get("depends_on", [])

    # Selective retrieval: read only the comparison (not raw summaries)
    source_content = ""
    for dep in depends_on:
        content = read_file(vfs_state, dep)
        _log_trace(trace_log, "read_file", dep,
                   f"Load comparison for unified model (step {step_num})",
                   step_num)
        source_content += f"--- {dep} ---\n{content}\n\n"
        print(f"    → read_file('{dep}') — selective retrieval")

    prompt = (
        f"Based on the following analysis, propose a comprehensive unified "
        f"model or framework that integrates the best elements.\n\n"
        f"Task: {task}\n\n"
        f"Source analysis:\n{source_content}\n"
        f"Create a well-structured proposal with:\n"
        f"1. Core principles of the unified model\n"
        f"2. Key components and how they integrate\n"
        f"3. Implementation approach\n"
        f"4. Expected benefits\n\n"
        f"Write the unified model proposal now (at least 200 words):"
    )
    unified = _invoke_llm_with_retry(llm, prompt)

    result = write_file(vfs_state, output_file, unified)
    _log_trace(trace_log, "write_file", output_file,
               "Store unified model proposal", step_num)
    print(f"    → write_file('{output_file}') — {len(unified)} chars")

    return f"Unified model proposed → {output_file}"


def _handle_refine(todo: dict, step_num: int, vfs_state: dict,
                   trace_log: list, llm) -> str:
    """
    Execute a refinement step using read→modify→edit pattern.

    This demonstrates edit_file usage:
    1. read_file — load existing content
    2. LLM modifies — generates improved version
    3. edit_file — updates the file (NOT write_file)
    """
    task = todo["task"]
    output_file = todo["output_file"]
    depends_on = todo.get("depends_on", [])

    # Step 1: Read existing content
    target_file = depends_on[0] if depends_on else output_file
    existing_content = read_file(vfs_state, target_file)
    _log_trace(trace_log, "read_file", target_file,
               f"Load existing content for refinement (step {step_num})",
               step_num)
    print(f"    → read_file('{target_file}') — for refinement")

    # Step 2: LLM generates refined version
    prompt = (
        f"You are given an existing document. Refine and enhance it based on "
        f"this instruction: {task}\n\n"
        f"Current content:\n{existing_content}\n\n"
        f"Improve the content by:\n"
        f"1. Adding depth and nuance based on the refinement instruction\n"
        f"2. Strengthening weak arguments\n"
        f"3. Adding practical considerations\n"
        f"4. Ensuring coherence and completeness\n\n"
        f"Provide the COMPLETE improved version (not just changes):"
    )
    refined_content = _invoke_llm_with_retry(llm, prompt)

    # Step 3: edit_file (NOT write_file) — demonstrates intentional editing
    result = edit_file(vfs_state, output_file, refined_content)
    _log_trace(trace_log, "edit_file", output_file,
               f"Refine content: {task[:60]} (read→modify→edit)",
               step_num)
    print(f"    → edit_file('{output_file}') — {len(refined_content)} chars "
          f"(read→modify→edit pattern)")

    return f"Refined '{output_file}' with: {task[:50]}"


# ── Node Function ────────────────────────────────────────────────────

def execute_node(state: dict, llm) -> dict:
    """
    Execution node: processes ALL enriched TODO steps sequentially,
    dispatching each to the appropriate handler based on step_type.

    Demonstrates:
      ✔ Selective retrieval — only reads dependency files
      ✔ Meaningful file names — derived from task content
      ✔ edit_file usage — for refinement steps
      ✔ Clean dependency chain — each step builds on prior outputs
      ✔ Memory offloading — content stored in VFS, not messages
      ✔ Trace logging — every tool call recorded with purpose

    Args:
        state: Current AgentState dict.
        llm:   ChatGroq (or compatible) LLM instance.

    Returns:
        Partial state update with files, todos, trace_log, and messages.
    """
    todos = list(state.get("todos", []))
    files = dict(state.get("files", {}))
    vfs_state = {"files": files}
    trace_log = list(state.get("trace_log", []))
    messages = []

    # Dispatch table for step types
    handlers = {
        "research": _handle_research,
        "compare":  _handle_compare,
        "unify":    _handle_unify,
        "refine":   _handle_refine,
    }

    print(f"\n{'='*60}")
    print(f"[Execute Node] Processing {len(todos)} steps")
    print(f"{'='*60}")

    for i, todo in enumerate(todos):
        step_type = todo.get("step_type", "research")
        step_num = i + 1

        print(f"\n  Step {step_num}/{len(todos)}: [{step_type:8s}] {todo['task']}")

        # Dispatch to appropriate handler
        handler = handlers.get(step_type, _handle_research)
        result_msg = handler(todo, step_num, vfs_state, trace_log, llm)

        # Mark step as done
        todos[i] = {**todo, "status": "done"}
        messages.append(AIMessage(content=result_msg))

        # Rate-limit courtesy delay between LLM calls
        if i < len(todos) - 1:
            time.sleep(2)

    # Summary of execution
    file_list = list(vfs_state["files"].keys())
    print(f"\n[Execute Node] Completed {len(todos)} steps")
    print(f"  Files in VFS: {file_list}")
    print(f"  Trace entries: {len(trace_log)}")

    return {
        "files": vfs_state["files"],
        "todos": todos,
        "trace_log": trace_log,
        "messages": messages,
    }
