"""
Milestone 3: Multi-Agent Collaboration with Task Delegation
=============================================================

Architecture
------------
LangGraph StateGraph with three nodes and multi-agent delegation:

    User Request
         ↓
    Supervisor Agent (plan node)
         ↓
    Task Delegation Tool
         ↓
    Sub-Agents (researcher / comparator / unifier / refiner)
         ↓
    LLM generates output per agent
         ↓
    Result returned to Supervisor
         ↓
    Supervisor stores result and continues workflow
         ↓
    Synthesis Node → Final Output

    START ──► supervisor ──► execute_with_delegation ──► synthesize ──► END

State structure (same as Milestone 2):
    state = {
        "todos":        [],       # enriched TODOs with step_type, output_file, depends_on
        "files":        {},       # virtual file system (filename → content)
        "messages":     [],       # conversation messages
        "final_output": "",       # combined structured summary
        "current_step": None,
        "trace_log":    [],       # ordered tool invocation trace (now includes delegations)
    }

What's new in Milestone 3:
    ✔ Supervisor agent — coordinates but does NOT execute tasks
    ✔ Sub-agents — researcher, summarizer, comparator, unifier, refiner
    ✔ Task delegation tool — routes tasks to the correct sub-agent
    ✔ SubAgentRegistry — central registry for agent discovery
    ✔ Agent attribution in trace log — which agent handled each step
    ✔ All Milestone 2 features preserved (VFS, selective retrieval, edit_file)
"""

import os
import json
import time
from typing import Dict
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables BEFORE any LangChain imports
load_dotenv()

# LangSmith tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_3_multi_agent")

# Validate API key early
_groq_key = os.getenv("GROQ_API_KEY", "")
if not _groq_key or _groq_key.startswith("your_"):
    raise SystemExit(
        "\n[ERROR] GROQ_API_KEY is missing or still set to the placeholder.\n"
        "       1. Go to https://console.groq.com/keys and create an API key.\n"
        "       2. Put it in project/deep_cognitive_agent/.env:\n"
        "          GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx\n"
        "       3. Or set it in PowerShell:  $env:GROQ_API_KEY = \"gsk_xxx\"\n"
    )

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ── Graph & Registry builders ──
from graphs.main_graph_m3 import build_graph_m3
from registry.subagent_registry import build_registry


# ── LLM Initialization ──────────────────────────────────────────────

_model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
print(f"[init] Using Groq model: {_model_name}")

llm = ChatGroq(
    model=_model_name,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ── Build Sub-Agent Registry ────────────────────────────────────────

_registry = build_registry()
print(f"[init] Registered sub-agents: {_registry.list_agents()}")


# ── Runner ───────────────────────────────────────────────────────────

def run_milestone3(task: str) -> Dict:
    """
    Run the Milestone 3 multi-agent pipeline:
        supervisor → execute_with_delegation → synthesize

    The supervisor creates a plan, then delegates each step to
    specialized sub-agents (researcher, comparator, unifier, refiner)
    via the task delegation tool and SubAgentRegistry.

    Args:
        task: The user's natural-language task description.

    Returns:
        The final AgentState dict with todos, files, trace_log, and final_output.
    """
    print("\n" + "=" * 70)
    print("  Milestone 3: Multi-Agent Collaboration")
    print("=" * 70)
    print(f"\n  Task: {task}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Available agents: {_registry.list_agents()}\n")

    # Build the LangGraph StateGraph with registry
    graph = build_graph_m3(llm, _registry)

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "todos": [],
        "files": {},
        "final_output": "",
        "current_step": None,
        "trace_log": [],
    }

    # Run the graph
    start_time = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start_time

    # Print results
    _print_results(final_state, task, elapsed)

    # Save to JSON
    _save_results(final_state, task)

    return final_state


def _print_results(final_state: dict, task: str, elapsed: float):
    """Print formatted results with multi-agent trace analysis."""
    print("\n" + "=" * 70)
    print("  EXECUTION RESULTS (Multi-Agent)")
    print("=" * 70)

    # ── TODOs ──
    print("\n--- Enriched Plan (TODOs with Dependencies) ---")
    for i, todo in enumerate(final_state.get("todos", []), 1):
        status = todo.get("status", "pending")
        stype = todo.get("step_type", "?")
        ofile = todo.get("output_file", "?")
        deps = todo.get("depends_on", [])
        icon = "✅" if status == "done" else "⬜"
        dep_str = f" ← reads: {deps}" if deps else ""
        print(f"  {i}. {icon} [{stype:8s}] {todo['task']}  [{status}]")
        print(f"              → {ofile}{dep_str}")

    # ── Virtual File System ──
    print("\n--- Virtual File System (state['files']) ---")
    files = final_state.get("files", {})
    if files:
        for fname in sorted(files.keys()):
            content = files[fname]
            print(f"\n  📄 {fname}  ({len(content)} chars)")
    else:
        print("  (empty)")

    # ── Tool Invocation Trace with Agent Attribution ──
    print("\n--- Tool Invocation Trace (with Agent Attribution) ---")
    trace_log = final_state.get("trace_log", [])
    if trace_log:
        for i, entry in enumerate(trace_log, 1):
            action = entry.get("action", "?")
            fname = entry.get("file", "—")
            purpose = entry.get("purpose", "")
            step = entry.get("step", "?")
            agent = entry.get("delegated_to", "")
            agent_str = f" [{agent}]" if agent else ""
            print(f"  {i:2d}. [{action:14s}]{agent_str:14s} "
                  f"{fname or '—':35s} (step {step}) {purpose[:55]}")
    else:
        print("  (no trace entries)")

    # ── Multi-Agent Analysis ──
    print("\n--- Multi-Agent Architecture Analysis ---")
    write_count = sum(1 for t in trace_log if t["action"] == "write_file")
    read_count = sum(1 for t in trace_log if t["action"] == "read_file")
    edit_count = sum(1 for t in trace_log if t["action"] == "edit_file")
    delegation_count = sum(1 for t in trace_log
                           if t["action"] == "delegate_task")
    supervisor_count = sum(1 for t in trace_log
                            if t["action"] == "supervisor_handle")
    ls_count = sum(1 for t in trace_log if t["action"] == "ls")

    # Count unique agents used
    agents_used = set(t.get("delegated_to", "")
                      for t in trace_log if t.get("delegated_to"))

    # Count delegation reasoning entries
    reasoned_entries = [t for t in trace_log if t.get("delegation_reasoning")]

    print(f"  delegate_task calls: {delegation_count}")
    print(f"  supervisor_handle:   {supervisor_count}")
    print(f"  Unique agents used:  {agents_used if agents_used else 'none'}")
    print(f"  write_file calls:    {write_count}")
    print(f"  read_file calls:     {read_count}")
    print(f"  edit_file calls:     {edit_count}")
    print(f"  ls calls:            {ls_count}")
    print(f"  Total tool calls:    {len(trace_log)}")
    print(f"  Files created:       {len(files)}")
    print(f"  Execution time:      {elapsed:.1f}s")

    # Verify key patterns
    patterns = []

    if delegation_count > 0:
        patterns.append(f"✔ Task delegation active ({delegation_count} delegations)")
    else:
        patterns.append("⚠ No task delegations detected")

    if len(agents_used) >= 2:
        patterns.append(f"✔ Multi-agent collaboration ({len(agents_used)} agents used)")
    else:
        patterns.append("⚠ Limited agent diversity")

    if reasoned_entries:
        patterns.append(f"✔ Delegation reasoning active ({len(reasoned_entries)} reasoned decisions)")
    else:
        patterns.append("⚠ No delegation reasoning recorded")

    if supervisor_count > 0:
        patterns.append(f"✔ Over-delegation prevention ({supervisor_count} tasks self-handled)")
    else:
        patterns.append("✔ All tasks required specialist delegation (no trivial tasks)")

    if edit_count > 0:
        patterns.append("✔ edit_file used (read→modify→edit pattern)")
    else:
        patterns.append("⚠ No edit_file usage detected")

    has_selective = any("selective" in t.get("purpose", "").lower()
                        for t in trace_log)
    if has_selective:
        patterns.append("✔ Selective retrieval demonstrated")
    else:
        patterns.append("⚠ No selective retrieval detected")

    if write_count > 0 and read_count > 0:
        patterns.append("✔ Memory offloading (write→read pattern)")

    for p in patterns:
        print(f"  {p}")

    # ── Final Output ──
    print("\n--- Final Structured Summary ---")
    final_output = final_state.get("final_output", "")
    if final_output:
        if len(final_output) > 500:
            print(final_output[:500])
            print(f"\n  ... ({len(final_output) - 500} more chars)")
            print(f"  Full output saved to outputs/milestone3_output.json")
        else:
            print(final_output)
    else:
        print("  (no output)")

    print(f"\n  Elapsed: {elapsed:.1f}s")


def _save_results(final_state: dict, task: str):
    """Save results to JSON file with multi-agent trace."""
    os.makedirs("outputs", exist_ok=True)

    # Clean filename from task
    task_slug = task[:40].lower()
    task_slug = "".join(c if c.isalnum() or c == " " else "" for c in task_slug)
    task_slug = task_slug.strip().replace(" ", "_")

    trace_log = final_state.get("trace_log", [])
    agents_used = list(set(
        t.get("delegated_to", "") for t in trace_log if t.get("delegated_to")
    ))

    serializable = {
        "task": task,
        "milestone": 3,
        "timestamp": datetime.now().isoformat(),
        "todos": final_state.get("todos", []),
        "files": final_state.get("files", {}),
        "trace_log": trace_log,
        "final_output": final_state.get("final_output", ""),
        "multi_agent_metrics": {
            "total_files": len(final_state.get("files", {})),
            "total_tool_calls": len(trace_log),
            "delegation_calls": sum(1 for t in trace_log
                                    if t["action"] == "delegate_task"),
            "supervisor_handled": sum(1 for t in trace_log
                                       if t["action"] == "supervisor_handle"),
            "delegation_reasoning_count": sum(
                1 for t in trace_log if t.get("delegation_reasoning")),
            "agents_used": agents_used,
            "write_calls": sum(1 for t in trace_log
                               if t["action"] == "write_file"),
            "read_calls": sum(1 for t in trace_log
                              if t["action"] == "read_file"),
            "edit_calls": sum(1 for t in trace_log
                              if t["action"] == "edit_file"),
        },
    }

    # Save main output
    output_path = os.path.join("outputs", "milestone3_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved results to {output_path}")

    # Save task-specific output
    task_path = os.path.join("outputs", f"m3_{task_slug}.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved task results to {task_path}")

    print("\n" + "=" * 70)
    print("  Milestone 3 complete.")
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Milestone 3: Multi-Agent Collaboration",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The task for the agent. If not provided, interactive mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even if --task is provided.",
    )
    args = parser.parse_args()

    if args.task and not args.interactive:
        run_milestone3(args.task)
    else:
        print("\n" + "=" * 70)
        print("  Deep Cognitive Agent — Milestone 3 (Multi-Agent)")
        print("  Interactive Mode")
        print("=" * 70)
        print(f"\n  Registered agents: {_registry.list_agents()}")
        print("\n  Enter your task below. Examples:")
        print("  • Analyze four AI ethics frameworks, identify differences,")
        print("    propose a unified model, then refine with sustainability.")
        print("  • Compare policy differences between Germany and India.")
        print("  • Research renewable energy trends and create a strategic plan.")
        print()

        while True:
            try:
                task = input("  Your task > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Goodbye!")
                break

            if not task:
                print("  (empty input — please enter a task or Ctrl+C to exit)\n")
                continue

            if task.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye!")
                break

            run_milestone3(task)
            print("\n  Enter another task or type 'quit' to exit.\n")
