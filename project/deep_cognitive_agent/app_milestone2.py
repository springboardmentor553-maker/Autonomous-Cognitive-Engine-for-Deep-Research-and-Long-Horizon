"""
Milestone 2: Deep Cognitive Agent with Context Offloading & Architectural Maturity
===================================================================================

Architecture
------------
LangGraph StateGraph with three nodes:

    START ──► plan ──► execute ──► synthesize ──► END

State structure:
    state = {
        "todos":        [],       # enriched TODOs with step_type, output_file, depends_on
        "files":        {},       # virtual file system  (filename → content)
        "messages":     [],       # conversation messages
        "final_output": "",       # combined structured summary
        "current_step": None,
        "trace_log":    [],       # ordered tool invocation trace for evaluation
    }

Architectural Principles:
    ✔ Selective retrieval — only reads files needed for each step
    ✔ Meaningful file names — derived from task content (not numbered)
    ✔ edit_file for refinement — demonstrates read→modify→edit pattern
    ✔ Clean dependency chain — each step builds on prior outputs
    ✔ Memory offloading — content stored in VFS, dropped from context
    ✔ Trace logging — every tool call recorded with purpose
    ✔ No duplication — write to file, return confirmation only
    ✔ Scaling stability — handles 3→20 files without architecture change

Workflow Example (AI ethics frameworks):
    1. Plan  → write_todos creates 6 enriched steps with dependencies
    2. Execute research → write_file("ethics_framework_A_summary.txt")
    3. Execute research → write_file("ethics_framework_B_summary.txt")
    4. Execute compare  → read_file(A,B) → write_file("comparison_analysis.txt")
    5. Execute unify    → read_file(comparison) → write_file("unified_model.txt")
    6. Execute refine   → read_file(unified) → edit_file("unified_model.txt")
    7. Synthesize       → read_file(unified,comparison) → final_output
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
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_2_vfs")

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

# ── Graph builder ──
from graphs.main_graph import build_graph


# ── LLM Initialization ──────────────────────────────────────────────

_model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
print(f"[init] Using Groq model: {_model_name}")

llm = ChatGroq(
    model=_model_name,
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# ── Runner ───────────────────────────────────────────────────────────

def run_milestone2(task: str) -> Dict:
    """
    Run the Milestone 2 agent pipeline:
        plan → execute (research/compare/unify/refine) → synthesize

    Args:
        task: The user's natural-language task description.

    Returns:
        The final AgentState dict with todos, files, trace_log, and final_output.
    """
    print("\n" + "=" * 70)
    print("  Milestone 2: Deep Cognitive Agent with Context Offloading")
    print("=" * 70)
    print(f"\n  Task: {task}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Build the LangGraph StateGraph
    graph = build_graph(llm)

    # Initial state — trace_log starts empty, populated by each node
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

    # ── Print results ────────────────────────────────────────────────
    _print_results(final_state, task, elapsed)

    # ── Save to JSON ─────────────────────────────────────────────────
    _save_results(final_state, task)

    return final_state


def _print_results(final_state: dict, task: str, elapsed: float):
    """Print formatted results with trace log analysis."""
    print("\n" + "=" * 70)
    print("  EXECUTION RESULTS")
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

    # ── Tool Invocation Trace (Critical for Evaluation) ──
    print("\n--- Tool Invocation Trace ---")
    trace_log = final_state.get("trace_log", [])
    if trace_log:
        for i, entry in enumerate(trace_log, 1):
            action = entry.get("action", "?")
            fname = entry.get("file", "—")
            purpose = entry.get("purpose", "")
            step = entry.get("step", "?")
            print(f"  {i:2d}. [{action:12s}] {fname or '—':35s} "
                  f"(step {step}) {purpose[:60]}")
    else:
        print("  (no trace entries)")

    # ── Architecture Analysis ──
    print("\n--- Architecture Analysis ---")
    write_count = sum(1 for t in trace_log if t["action"] == "write_file")
    read_count = sum(1 for t in trace_log if t["action"] == "read_file")
    edit_count = sum(1 for t in trace_log if t["action"] == "edit_file")
    ls_count = sum(1 for t in trace_log if t["action"] == "ls")
    print(f"  write_file calls: {write_count}")
    print(f"  read_file calls:  {read_count}")
    print(f"  edit_file calls:  {edit_count}")
    print(f"  ls calls:         {ls_count}")
    print(f"  Total tool calls: {len(trace_log)}")
    print(f"  Files created:    {len(files)}")
    print(f"  Read/Write ratio: {read_count}/{write_count}")
    print(f"  Execution time:   {elapsed:.1f}s")

    # Verify key patterns
    patterns = []
    if edit_count > 0:
        patterns.append("✔ edit_file used (read→modify→edit pattern)")
    else:
        patterns.append("⚠ No edit_file usage detected")

    has_selective = any("selective" in t.get("purpose", "").lower()
                        or "Selective" in t.get("purpose", "")
                        for t in trace_log)
    if has_selective:
        patterns.append("✔ Selective retrieval demonstrated")
    else:
        patterns.append("⚠ No selective retrieval detected")

    has_meaningful_names = all(
        not fname.startswith("summary") and not fname.startswith("step_")
        for fname in files.keys()
        if fname.endswith("_summary.txt")
    ) if files else False
    if has_meaningful_names:
        patterns.append("✔ Meaningful file names (not numbered)")
    else:
        patterns.append("⚠ File naming could be more descriptive")

    if write_count > 0 and read_count > 0:
        patterns.append("✔ Memory offloading (write→read pattern)")

    for p in patterns:
        print(f"  {p}")

    # ── Final Output (summary only, not duplicating file content) ──
    print("\n--- Final Structured Summary ---")
    final_output = final_state.get("final_output", "")
    if final_output:
        # Show first 500 chars to avoid duplication with files
        if len(final_output) > 500:
            print(final_output[:500])
            print(f"\n  ... ({len(final_output) - 500} more chars)")
            print(f"  Full output saved to outputs/milestone2_output.json")
        else:
            print(final_output)
    else:
        print("  (no output)")

    print(f"\n  Elapsed: {elapsed:.1f}s")


def _save_results(final_state: dict, task: str):
    """Save results to JSON file with trace log."""
    os.makedirs("outputs", exist_ok=True)

    # Create a clean filename from the task
    task_slug = task[:40].lower()
    task_slug = "".join(c if c.isalnum() or c == " " else "" for c in task_slug)
    task_slug = task_slug.strip().replace(" ", "_")

    serializable = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "todos": final_state.get("todos", []),
        "files": final_state.get("files", {}),
        "trace_log": final_state.get("trace_log", []),
        "final_output": final_state.get("final_output", ""),
        "architecture_metrics": {
            "total_files": len(final_state.get("files", {})),
            "total_tool_calls": len(final_state.get("trace_log", [])),
            "write_calls": sum(1 for t in final_state.get("trace_log", [])
                               if t["action"] == "write_file"),
            "read_calls": sum(1 for t in final_state.get("trace_log", [])
                              if t["action"] == "read_file"),
            "edit_calls": sum(1 for t in final_state.get("trace_log", [])
                              if t["action"] == "edit_file"),
        },
    }

    # Save main output
    output_path = os.path.join("outputs", "milestone2_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved results to {output_path}")

    # Save task-specific output
    task_path = os.path.join("outputs", f"m2_{task_slug}.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved task results to {task_path}")

    print("\n" + "=" * 70)
    print("  Milestone 2 complete.")
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Milestone 2: Deep Cognitive Agent with Context Offloading",
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
        # Direct execution with provided task
        run_milestone2(args.task)
    else:
        # Interactive mode — user provides input
        print("\n" + "=" * 70)
        print("  Deep Cognitive Agent — Milestone 2")
        print("  Interactive Mode")
        print("=" * 70)
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

            run_milestone2(task)

            print("\n  Enter another task or type 'quit' to exit.\n")
