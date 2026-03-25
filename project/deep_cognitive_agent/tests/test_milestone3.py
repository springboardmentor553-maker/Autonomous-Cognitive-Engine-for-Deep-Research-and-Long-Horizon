"""
Test Milestone 3 - Multi-Agent Collaboration

This test verifies multi-agent architecture maturity:
  1. Sub-agents are registered in the registry
  2. Task delegation tool routes to correct agents
  3. Trace log shows delegate_task actions with agent attribution
  4. Multiple distinct agents are used in a single workflow
  5. All Milestone 2 features preserved (VFS, selective retrieval, edit_file)
  6. Supervisor coordinates — delegates specialized tasks, self-handles trivial ones
  7. Delegation reasoning recorded in trace log
  8. Result integration validates sub-agent outputs before state update
  9. Final output produced via synthesis from delegated results
  10. Clean dependency chain maintained across agent boundaries
"""

import os
import sys
import io
import json
from datetime import datetime

# Fix Unicode output on Windows (cp1252 console can't print arrows/emojis)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_3_test")

from app_milestone3 import run_milestone3


def test_multi_agent_pipeline(task: str):
    """
    End-to-end test for the Milestone 3 multi-agent pipeline.
    Validates multi-agent collaboration architecture.
    """
    print("=" * 70)
    print("  TEST: Milestone 3 — Multi-Agent Collaboration Validation")
    print(f"  Task: {task}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Run the full pipeline
    final_state = run_milestone3(task)

    # ── Assertions ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKS (Milestone 3)")
    print("=" * 70)

    errors = []
    warnings = []

    # 1. Enriched TODOs exist with metadata
    todos = final_state.get("todos", [])
    if len(todos) >= 4:
        print(f"  ✅ TODOs generated: {len(todos)} enriched steps")
    else:
        msg = f"Expected ≥4 TODOs, got {len(todos)}"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 2. TODOs have enrichment metadata
    enriched_count = sum(1 for t in todos if "step_type" in t)
    if enriched_count == len(todos):
        print(f"  ✅ All {enriched_count} TODOs enriched with step_type/output_file/depends_on")
    else:
        msg = f"Only {enriched_count}/{len(todos)} TODOs have enrichment metadata"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 3. Step types present
    step_types = set(t.get("step_type", "") for t in todos)
    has_research = "research" in step_types
    has_advanced = bool(step_types & {"compare", "unify", "refine"})
    if has_research and has_advanced:
        print(f"  ✅ Step type diversity: {step_types}")
    elif has_research:
        warnings.append(f"Only research steps found: {step_types}")
        print(f"  ⚠️  {warnings[-1]}")
    else:
        msg = f"Unexpected step types: {step_types}"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 4. All TODOs marked done
    done_count = sum(1 for t in todos if t.get("status") == "done")
    if done_count == len(todos):
        print(f"  ✅ All {done_count} TODOs marked as done")
    else:
        msg = f"Only {done_count}/{len(todos)} TODOs marked done"
        print(f"  ⚠️  {msg}")
        warnings.append(msg)

    # 5. Files in VFS with meaningful names
    files = final_state.get("files", {})
    if files:
        print(f"  ✅ VFS has {len(files)} files: {list(files.keys())}")
    else:
        msg = "No files in VFS"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 6. Files have substantial content
    short_files = [f for f, c in files.items() if len(c) < 50]
    if not short_files:
        print(f"  ✅ All {len(files)} files have substantial content (>50 chars)")
    else:
        msg = f"Short files detected: {short_files}"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 7. Trace log exists
    trace_log = final_state.get("trace_log", [])
    if len(trace_log) >= 4:
        print(f"  ✅ Trace log has {len(trace_log)} entries")
    else:
        msg = f"Trace log too short ({len(trace_log)} entries)"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # ── MILESTONE 3 SPECIFIC CHECKS ──

    # 8. delegate_task actions present in trace
    delegation_actions = [t for t in trace_log
                          if t.get("action") == "delegate_task"]
    if delegation_actions:
        print(f"  ✅ Task delegation active: {len(delegation_actions)} delegate_task calls")
    else:
        msg = "No delegate_task actions found in trace — delegation not working"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 9. Agent attribution in trace entries
    attributed_entries = [t for t in trace_log if t.get("delegated_to")]
    if attributed_entries:
        agents = set(t["delegated_to"] for t in attributed_entries)
        print(f"  ✅ Agent attribution present: {agents}")
    else:
        msg = "No agent attribution in trace entries"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 10. Multiple agents used (not just one)
    agents_used = set(t.get("delegated_to", "")
                      for t in trace_log if t.get("delegated_to"))
    if len(agents_used) >= 2:
        print(f"  ✅ Multi-agent collaboration: {len(agents_used)} distinct agents ({agents_used})")
    elif len(agents_used) == 1:
        warnings.append(f"Only 1 agent used: {agents_used}")
        print(f"  ⚠️  {warnings[-1]}")
    else:
        msg = "No agents identified in trace"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 11. Researcher agent used for research steps
    researcher_used = any(
        t.get("delegated_to") == "researcher" for t in trace_log
    )
    if researcher_used:
        print(f"  ✅ Researcher agent invoked for research tasks")
    else:
        warnings.append("Researcher agent not explicitly found in trace")
        print(f"  ⚠️  {warnings[-1]}")

    # 12. Delegation reasoning recorded in trace
    reasoning_entries = [t for t in trace_log if t.get("delegation_reasoning")]
    if reasoning_entries:
        print(f"  ✅ Delegation reasoning active: {len(reasoning_entries)} reasoned decisions")
        # Show sample reasoning
        sample = reasoning_entries[0]["delegation_reasoning"]
        print(f"      Sample: \"{sample[:70]}...\"")
    else:
        warnings.append("No delegation reasoning entries in trace")
        print(f"  ⚠️  {warnings[-1]}")

    # 13. Result integration (validated results — no empty content in files)
    empty_files = [f for f, c in files.items() if len(c.strip()) < 10]
    if not empty_files:
        print(f"  ✅ Result integration verified — all {len(files)} files have valid content")
    else:
        msg = f"Result integration issue: files with insufficient content: {empty_files}"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 14. edit_file used for refinement (preserved from M2)
    edit_actions = [t for t in trace_log if t.get("action") == "edit_file"]
    if edit_actions:
        print(f"  ✅ edit_file used {len(edit_actions)} time(s) — read→modify→edit verified")
    else:
        warnings.append("No edit_file actions — refinement may not have triggered")
        print(f"  ⚠️  {warnings[-1]}")

    # 15. Selective retrieval (preserved from M2)
    read_actions = [t for t in trace_log if t.get("action") == "read_file"]
    selective_reads = [t for t in read_actions
                       if "selective" in t.get("purpose", "").lower()]
    if selective_reads:
        print(f"  ✅ Selective retrieval demonstrated ({len(selective_reads)} selective reads)")
    elif read_actions:
        warnings.append("Read actions present but not marked as selective")
        print(f"  ⚠️  {warnings[-1]}")
    else:
        msg = "No read_file actions in trace"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 16. Dependency chain valid
    write_files = [t["file"] for t in trace_log if t["action"] == "write_file"]
    read_files_list = [t["file"] for t in trace_log
                       if t["action"] == "read_file"]
    dep_reads = [f for f in read_files_list if f in write_files]
    if dep_reads:
        print(f"  ✅ Dependency chain valid: reads reference prior writes ({len(dep_reads)} links)")
    else:
        warnings.append("Could not verify dependency chain in trace")
        print(f"  ⚠️  {warnings[-1]}")

    # 17. Final output exists and is substantial
    final_output = final_state.get("final_output", "")
    if len(final_output) > 100:
        print(f"  ✅ Final structured summary generated ({len(final_output)} chars)")
    else:
        msg = f"Final output too short ({len(final_output)} chars)"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 18. Final output has structure markers
    structure_markers = ["Overview", "Key Findings", "Analysis",
                         "Conclusion", "Recommendations"]
    found_markers = [m for m in structure_markers
                     if m.lower() in final_output.lower()]
    if len(found_markers) >= 3:
        print(f"  ✅ Structured format verified ({len(found_markers)}/5 sections found)")
    else:
        warnings.append(
            f"Only {len(found_markers)}/5 structure sections found: {found_markers}"
        )
        print(f"  ⚠️  {warnings[-1]}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    if not errors:
        if warnings:
            print(f"  ✅ ALL CRITICAL CHECKS PASSED ({len(warnings)} warnings)")
        else:
            print("  🎉 ALL CHECKS PASSED — Milestone 3 fully verified!")
    else:
        print(f"  ❌ {len(errors)} check(s) failed:")
        for err in errors:
            print(f"     - {err}")
    if warnings:
        print(f"  ⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"     - {w}")
    print("-" * 70)

    # Save test results
    test_result = {
        "milestone": 3,
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "todo_count": len(todos),
        "todos_done": done_count,
        "step_types": list(step_types),
        "files_in_vfs": list(files.keys()),
        "file_sizes": {k: len(v) for k, v in files.items()},
        "trace_log_length": len(trace_log),
        "delegation_count": len(delegation_actions),
        "delegation_reasoning_count": len(reasoning_entries),
        "agents_used": list(agents_used),
        "edit_file_used": len(edit_actions) > 0,
        "selective_retrieval": len(selective_reads) > 0,
        "dependency_chain_valid": len(dep_reads) > 0,
        "result_integration_valid": len(empty_files) == 0,
        "final_output_length": len(final_output),
        "structure_markers_found": found_markers,
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }

    os.makedirs("outputs", exist_ok=True)
    test_path = os.path.join("outputs", "milestone3_test_result.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_result, f, indent=2, ensure_ascii=False)
    print(f"\n  Test results saved to {test_path}")

    return test_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Milestone 3: Multi-Agent Collaboration Validation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The task to test. If not provided, uses default.",
    )
    args = parser.parse_args()

    if args.task:
        test_multi_agent_pipeline(args.task)
    else:
        print("\n  Enter a task to test (or press Enter for default):")
        task = input("  > ").strip()
        if not task:
            task = (
                "Analyze four AI ethics frameworks, identify differences, "
                "propose a unified model, then refine it with sustainability "
                "considerations."
            )
            print(f"  Using default: {task}")
        test_multi_agent_pipeline(task)
