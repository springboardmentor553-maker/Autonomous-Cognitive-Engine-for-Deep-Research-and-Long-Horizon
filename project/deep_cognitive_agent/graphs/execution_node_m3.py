"""
Execution Node - Milestone 3: Multi-Agent Collaboration via Task Delegation

Processes each enriched TODO step by delegating to specialized sub-agents
through the task delegation tool and SubAgentRegistry.

Step-type → Sub-agent mapping:
  - research  → "researcher"  agent → write_file(meaningful_name.txt)
  - compare   → "comparator"  agent → read dependencies → write_file
  - unify     → "unifier"     agent → read comparison → write_file
  - refine    → "refiner"     agent → read target → edit_file

Key Milestone 3 principles:
  ✔ Supervisor delegates — does NOT execute tasks itself
  ✔ Each sub-agent has a focused prompt and limited responsibility
  ✔ Task delegation tool routes to the correct agent via registry
  ✔ Selective retrieval preserved from Milestone 2
  ✔ Trace logging records which agent handled each step
  ✔ Memory offloading via VFS unchanged from Milestone 2

Architecture:
  START → supervisor → **execute_with_delegation** → synthesize → END
"""

import time

from langchain_core.messages import AIMessage

from tools.vfs.write_file import write_file
from tools.vfs.read_file import read_file
from tools.vfs.edit_file import edit_file
from tools.delegation.task import delegate_task
from utils.helpers import parse_retry_after, is_rate_limit_error, is_server_overload_error


# ── Helpers ──────────────────────────────────────────────────────────

def _log_trace(trace_log: list, action: str, filename: str,
               purpose: str, step: int, agent: str = None):
    """Append a structured trace entry with agent attribution."""
    entry = {
        "action": action,
        "file": filename,
        "purpose": purpose,
        "step": step,
    }
    if agent:
        entry["delegated_to"] = agent
    trace_log.append(entry)


# ── Step Handlers (delegate to sub-agents) ───────────────────────────

def _handle_research(todo, step_num, vfs_state, trace_log, registry, llm):
    """Delegate research to the 'researcher' sub-agent."""
    task = todo["task"]
    output_file = todo["output_file"]

    print(f"    → Delegating to 'researcher' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate research to researcher: {task[:60]}",
               step_num, agent="researcher")

    content = delegate_task(registry, llm, "researcher", task)

    write_file(vfs_state, output_file, content)
    _log_trace(trace_log, "write_file", output_file,
               f"Store research summary: {task[:60]}", step_num,
               agent="researcher")
    print(f"    → write_file('{output_file}') — {len(content)} chars")

    return f"[researcher] Researched '{task[:50]}' → {output_file}"


def _handle_compare(todo, step_num, vfs_state, trace_log, registry, llm):
    """Delegate comparison to the 'comparator' sub-agent."""
    task = todo["task"]
    output_file = todo["output_file"]
    depends_on = todo.get("depends_on", [])

    # Selective retrieval: read ONLY dependency files
    dep_contents = []
    for dep in depends_on:
        content = read_file(vfs_state, dep)
        _log_trace(trace_log, "read_file", dep,
                   f"Selective load for comparison (step {step_num})",
                   step_num)
        dep_contents.append(f"--- {dep} ---\n{content}")
        print(f"    → read_file('{dep}') — selective retrieval")

    combined_sources = "\n\n".join(dep_contents)

    # Delegate to comparator agent: task|||sources
    print(f"    → Delegating to 'comparator' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate comparison to comparator: {task[:60]}",
               step_num, agent="comparator")

    input_data = f"{task}|||{combined_sources}"
    comparison = delegate_task(registry, llm, "comparator", input_data)

    write_file(vfs_state, output_file, comparison)
    _log_trace(trace_log, "write_file", output_file,
               "Store comparison analysis", step_num,
               agent="comparator")
    print(f"    → write_file('{output_file}') — {len(comparison)} chars")

    return f"[comparator] Compared {len(depends_on)} sources → {output_file}"


def _handle_unify(todo, step_num, vfs_state, trace_log, registry, llm):
    """Delegate unification to the 'unifier' sub-agent."""
    task = todo["task"]
    output_file = todo["output_file"]
    depends_on = todo.get("depends_on", [])

    # Selective retrieval
    source_content = ""
    for dep in depends_on:
        content = read_file(vfs_state, dep)
        _log_trace(trace_log, "read_file", dep,
                   f"Load comparison for unified model (step {step_num})",
                   step_num)
        source_content += f"--- {dep} ---\n{content}\n\n"
        print(f"    → read_file('{dep}') — selective retrieval")

    # Delegate to unifier agent: task|||sources
    print(f"    → Delegating to 'unifier' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate unification to unifier: {task[:60]}",
               step_num, agent="unifier")

    input_data = f"{task}|||{source_content}"
    unified = delegate_task(registry, llm, "unifier", input_data)

    write_file(vfs_state, output_file, unified)
    _log_trace(trace_log, "write_file", output_file,
               "Store unified model proposal", step_num,
               agent="unifier")
    print(f"    → write_file('{output_file}') — {len(unified)} chars")

    return f"[unifier] Unified model proposed → {output_file}"


def _handle_refine(todo, step_num, vfs_state, trace_log, registry, llm):
    """Delegate refinement to the 'refiner' sub-agent (read→modify→edit)."""
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

    # Step 2: Delegate to refiner agent: task|||existing_content
    print(f"    → Delegating to 'refiner' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate refinement to refiner: {task[:60]}",
               step_num, agent="refiner")

    input_data = f"{task}|||{existing_content}"
    refined_content = delegate_task(registry, llm, "refiner", input_data)

    # Step 3: edit_file (NOT write_file) — demonstrates read→modify→edit
    edit_file(vfs_state, output_file, refined_content)
    _log_trace(trace_log, "edit_file", output_file,
               f"Refine content: {task[:60]} (read→modify→edit)",
               step_num, agent="refiner")
    print(f"    → edit_file('{output_file}') — {len(refined_content)} chars "
          f"(read→modify→edit pattern)")

    return f"[refiner] Refined '{output_file}' with: {task[:50]}"


# ── Node Function ────────────────────────────────────────────────────

def execute_with_delegation(state: dict, llm, registry) -> dict:
    """
    Milestone 3 Execution Node: processes all enriched TODO steps by
    delegating to specialized sub-agents through the task delegation tool.

    The supervisor does NOT execute tasks itself — it routes each step
    to the appropriate sub-agent based on step_type.

    Args:
        state: Current AgentState dict.
        llm: ChatGroq (or compatible) LLM instance.
        registry: SubAgentRegistry with registered sub-agents.

    Returns:
        Partial state update with files, todos, trace_log, and messages.
    """
    todos = list(state.get("todos", []))
    files = dict(state.get("files", {}))
    vfs_state = {"files": files}
    trace_log = list(state.get("trace_log", []))
    messages = []

    # Dispatch table: step_type → handler
    handlers = {
        "research": _handle_research,
        "compare":  _handle_compare,
        "unify":    _handle_unify,
        "refine":   _handle_refine,
    }

    print(f"\n{'='*60}")
    print(f"[Execute Node] Processing {len(todos)} steps via multi-agent delegation")
    print(f"  Available agents: {registry.list_agents()}")
    print(f"{'='*60}")

    for i, todo in enumerate(todos):
        step_type = todo.get("step_type", "research")
        step_num = i + 1

        print(f"\n  Step {step_num}/{len(todos)}: [{step_type:8s}] {todo['task']}")

        handler = handlers.get(step_type, _handle_research)
        result_msg = handler(todo, step_num, vfs_state, trace_log, registry, llm)

        # Mark step as done
        todos[i] = {**todo, "status": "done"}
        messages.append(AIMessage(content=result_msg))

        # Rate-limit courtesy delay between delegations
        if i < len(todos) - 1:
            time.sleep(2)

    # Summary
    file_list = list(vfs_state["files"].keys())
    print(f"\n[Execute Node] Completed {len(todos)} steps via delegation")
    print(f"  Files in VFS: {file_list}")
    print(f"  Trace entries: {len(trace_log)}")

    # Count delegations for logging
    delegation_count = sum(
        1 for t in trace_log if t.get("action") == "delegate_task"
    )
    print(f"  Agent delegations: {delegation_count}")

    return {
        "files": vfs_state["files"],
        "todos": todos,
        "trace_log": trace_log,
        "messages": messages,
    }
