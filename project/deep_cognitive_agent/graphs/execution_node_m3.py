"""
Execution Node - Milestone 3: Multi-Agent Collaboration via Task Delegation

Processes each enriched TODO step using delegation reasoning to decide
whether the supervisor should handle a task itself or delegate to a
specialized sub-agent.

Three key decisions per step (Supervisor Architecture):
  1. Decide the next task (from enriched TODO list)
  2. Decide whether to perform it or delegate (_should_delegate)
  3. Integrate the result (_integrate_result)

Step-type → Sub-agent mapping (when delegating):
  - research  → "researcher"  agent → write_file(meaningful_name.txt)
  - compare   → "comparator"  agent → read dependencies → write_file
  - unify     → "unifier"     agent → read comparison → write_file
  - refine    → "refiner"     agent → read target → edit_file

Key Milestone 3 principles:
  ✔ Delegation reasoning — supervisor explicitly decides delegate vs self-handle
  ✔ Over-delegation prevention — trivial tasks handled by supervisor directly
  ✔ Clear structured instructions — subagents receive explicit requirements
  ✔ Result integration — supervisor validates subagent results before storing
  ✔ State management — only supervisor modifies VFS, subagents return results
  ✔ Selective retrieval preserved from Milestone 2
  ✔ Trace logging records delegation reasoning and agent attribution
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
               purpose: str, step: int, agent: str = None,
               delegation_reasoning: str = None):
    """Append a structured trace entry with agent attribution."""
    entry = {
        "action": action,
        "file": filename,
        "purpose": purpose,
        "step": step,
    }
    if agent:
        entry["delegated_to"] = agent
    if delegation_reasoning:
        entry["delegation_reasoning"] = delegation_reasoning
    trace_log.append(entry)


# ── Delegation Reasoning ─────────────────────────────────────────────

def _should_delegate(todo: dict, step_type: str) -> dict:
    """
    Delegation reasoning: decide whether the supervisor should handle
    a task itself or delegate it to a specialist sub-agent.

    The supervisor delegates when:
      - The task requires a specialized skill (research, comparison, etc.)
      - The task is repetitive and can be standardized
      - The task requires a focused prompt
      - The task can be isolated from the main reasoning flow

    The supervisor handles tasks itself when:
      - The task is simple enough (e.g., combining two short sentences)
      - The task is trivial reasoning that doesn't need a specialist
      - Delegating would introduce unnecessary communication overhead

    Returns:
        dict with keys:
          - "delegate": bool — True if should delegate, False if supervisor handles
          - "agent": str or None — sub-agent name if delegating
          - "reason": str — explanation for the decision
    """
    task_text = todo.get("task", "").lower()
    task_length = len(task_text)

    # Map step_type → preferred sub-agent
    agent_map = {
        "research": "researcher",
        "compare": "comparator",
        "unify": "unifier",
        "refine": "refiner",
    }

    # ── Over-delegation prevention ──
    # Simple tasks that the supervisor can handle itself:
    # 1. Very short tasks (< 30 chars) that are trivial reasoning
    trivial_keywords = ["list", "name", "count", "define briefly",
                        "state the", "mention"]
    is_trivial = (task_length < 30
                  and any(kw in task_text for kw in trivial_keywords))

    if is_trivial:
        return {
            "delegate": False,
            "agent": None,
            "reason": (f"Task is simple enough for supervisor to handle "
                       f"directly — avoids over-delegation overhead"),
        }

    # ── Delegation reasoning for specialized tasks ──
    agent_name = agent_map.get(step_type, "researcher")

    reasoning_map = {
        "research": (
            "Task requires deep research with specific facts and data — "
            "researcher agent has a focused prompt for 150+ word analysis"
        ),
        "compare": (
            "Task requires structured comparison across multiple sources — "
            "comparator agent specializes in identifying differences and similarities"
        ),
        "unify": (
            "Task requires proposing a unified framework from analysis — "
            "unifier agent has expertise in integration and model building"
        ),
        "refine": (
            "Task requires refining existing content with additional depth — "
            "refiner agent specializes in enhancement and practical considerations"
        ),
    }

    return {
        "delegate": True,
        "agent": agent_name,
        "reason": reasoning_map.get(step_type,
                                     f"Specialized {step_type} task → delegate to {agent_name}"),
    }


# ── Supervisor Self-Handling (for simple tasks) ──────────────────────

def _supervisor_handle(todo, step_num, vfs_state, trace_log, llm):
    """
    Supervisor handles a simple task itself — avoids over-delegation.

    This is called when _should_delegate() returns delegate=False,
    meaning the task is trivial enough that sending it to a sub-agent
    would introduce unnecessary communication overhead.
    """
    task = todo["task"]
    output_file = todo["output_file"]

    print(f"    → Supervisor handling directly (over-delegation prevention)")
    _log_trace(trace_log, "supervisor_handle", None,
               f"Supervisor handles simple task directly: {task[:60]}",
               step_num, delegation_reasoning="trivial task — no delegation needed")

    prompt = (
        f"Briefly and concisely address the following task in 2-3 sentences:\n\n"
        f"Task: {task}\n\n"
        f"Provide a direct, clear response:"
    )
    response = llm.invoke(prompt)
    content = response.content

    write_file(vfs_state, output_file, content)
    _log_trace(trace_log, "write_file", output_file,
               f"Store supervisor-handled result: {task[:60]}", step_num)
    print(f"    → write_file('{output_file}') — {len(content)} chars (supervisor-handled)")

    return f"[supervisor] Handled directly: '{task[:50]}' → {output_file}"


# ── Result Integration (validate sub-agent output) ───────────────────

def _integrate_result(content: str, agent_name: str, task: str,
                      step_num: int, trace_log: list) -> str:
    """
    Supervisor integrates the result returned by a sub-agent.

    After a sub-agent finishes:
      1. Supervisor receives the result
      2. Validates it (not empty, reasonable length)
      3. Processes it for the next step

    This ensures the supervisor always controls state management,
    and sub-agents only return results without modifying global state.

    Returns:
        The validated content (potentially with a warning appended).
    """
    # Validate: sub-agent returned something useful
    if not content or len(content.strip()) < 10:
        warning = (f"[WARNING] {agent_name} returned insufficient content "
                   f"for step {step_num}: '{task[:50]}'. "
                   f"Using fallback placeholder.")
        print(f"    ⚠ {warning}")
        _log_trace(trace_log, "result_validation", None,
                   f"Sub-agent {agent_name} returned empty/short result — flagged",
                   step_num, agent=agent_name)
        content = f"[Insufficient result from {agent_name}] Task: {task}"

    return content


# ── Step Handlers (delegate to sub-agents) ───────────────────────────

def _handle_research(todo, step_num, vfs_state, trace_log, registry, llm):
    """Delegate research to the 'researcher' sub-agent."""
    task = todo["task"]
    output_file = todo["output_file"]

    # Clear, structured instruction (not just "Summarize this")
    structured_input = (
        f"Research the following topic in detail. "
        f"Include specific facts, data points, and expert analysis. "
        f"Write at least 150 words in a single detailed paragraph.\n\n"
        f"Topic: {task}"
    )

    print(f"    → Delegating to 'researcher' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate research to researcher: {task[:60]}",
               step_num, agent="researcher",
               delegation_reasoning="requires deep research with specific facts")

    content = delegate_task(registry, llm, "researcher", structured_input)

    # Supervisor integrates the result (validate before storing)
    content = _integrate_result(content, "researcher", task, step_num, trace_log)

    # Supervisor controls state — stores result in VFS
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

    # Clear, structured instruction with explicit requirements
    structured_input = (
        f"Compare and contrast the following {len(depends_on)} research summaries. "
        f"Identify key differences, surprising similarities, relative strengths "
        f"and weaknesses, and complementary aspects. Write at least 200 words.\n\n"
        f"Context: {task}|||{combined_sources}"
    )

    print(f"    → Delegating to 'comparator' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate comparison to comparator: {task[:60]}",
               step_num, agent="comparator",
               delegation_reasoning="requires structured comparison across multiple sources")

    comparison = delegate_task(registry, llm, "comparator", structured_input)

    # Supervisor integrates the result
    comparison = _integrate_result(comparison, "comparator", task, step_num, trace_log)

    # Supervisor controls state — stores result in VFS
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

    # Clear, structured instruction
    structured_input = (
        f"Propose a unified model or framework that integrates the findings below. "
        f"Explain how different perspectives can be reconciled into a coherent whole. "
        f"Include practical implementation considerations.\n\n"
        f"Context: {task}|||{source_content}"
    )

    print(f"    → Delegating to 'unifier' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate unification to unifier: {task[:60]}",
               step_num, agent="unifier",
               delegation_reasoning="requires integration expertise to build unified framework")

    unified = delegate_task(registry, llm, "unifier", structured_input)

    # Supervisor integrates the result
    unified = _integrate_result(unified, "unifier", task, step_num, trace_log)

    # Supervisor controls state — stores result in VFS
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

    # Step 2: Clear, structured instruction for refinement
    structured_input = (
        f"Refine and enhance the following content. Add additional depth, "
        f"practical considerations, and improve coherence. "
        f"Preserve the original structure while strengthening the analysis.\n\n"
        f"Refinement goal: {task}|||{existing_content}"
    )

    print(f"    → Delegating to 'refiner' agent...")
    _log_trace(trace_log, "delegate_task", None,
               f"Delegate refinement to refiner: {task[:60]}",
               step_num, agent="refiner",
               delegation_reasoning="requires focused refinement with additional depth")

    refined_content = delegate_task(registry, llm, "refiner", structured_input)

    # Supervisor integrates the result
    refined_content = _integrate_result(refined_content, "refiner", task,
                                         step_num, trace_log)

    # Step 3: edit_file (NOT write_file) — demonstrates read→modify→edit
    # Supervisor controls state — only supervisor updates the VFS
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
    Milestone 3 Execution Node: processes all enriched TODO steps using
    delegation reasoning to decide whether to delegate or self-handle.

    Three key decisions per step (from supervisor architecture):
      1. Decide the next task (from enriched TODO list)
      2. Decide whether to perform it or delegate it (_should_delegate)
      3. Integrate the result (_integrate_result)

    The supervisor delegates specialized tasks to sub-agents but handles
    trivial tasks itself to avoid over-delegation overhead.

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

    # Dispatch table: step_type → handler (for delegated tasks)
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

    supervisor_handled = 0
    delegated_count = 0

    for i, todo in enumerate(todos):
        step_type = todo.get("step_type", "research")
        step_num = i + 1

        print(f"\n  Step {step_num}/{len(todos)}: [{step_type:8s}] {todo['task']}")

        # ── Decision 2: Delegate or self-handle? ──
        decision = _should_delegate(todo, step_type)
        print(f"    Delegation reasoning: {decision['reason']}")

        if decision["delegate"]:
            # Delegate to specialized sub-agent
            delegated_count += 1
            handler = handlers.get(step_type, _handle_research)
            result_msg = handler(todo, step_num, vfs_state, trace_log, registry, llm)
        else:
            # Supervisor handles directly (over-delegation prevention)
            supervisor_handled += 1
            result_msg = _supervisor_handle(todo, step_num, vfs_state, trace_log, llm)

        # Mark step as done
        todos[i] = {**todo, "status": "done"}
        messages.append(AIMessage(content=result_msg))

        # Rate-limit courtesy delay between delegations
        if i < len(todos) - 1:
            time.sleep(2)

    # Summary
    file_list = list(vfs_state["files"].keys())
    print(f"\n[Execute Node] Completed {len(todos)} steps")
    print(f"  Delegated to sub-agents: {delegated_count}")
    print(f"  Handled by supervisor:   {supervisor_handled}")
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
