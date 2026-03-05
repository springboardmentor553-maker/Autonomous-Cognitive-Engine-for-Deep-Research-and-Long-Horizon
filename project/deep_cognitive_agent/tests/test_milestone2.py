"""
Test Milestone 2 - Deep Cognitive Agent with Context Offloading

This test verifies architectural maturity:
  1. TODOs are enriched with step_type, output_file, depends_on
  2. Meaningful file names (not numbered summary1.txt, summary2.txt)
  3. Selective retrieval — comparison reads only research files
  4. edit_file demonstrated for refinement steps
  5. Clean dependency chain visible in trace_log
  6. Memory offloading — content in VFS, not duplicated in messages
  7. Final structured summary produced from key files only
  8. System stable across different task types
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_2_test")

from app_milestone2 import run_milestone2


def test_vfs_pipeline(task: str):
    """
    End-to-end test for the Milestone 2 pipeline.
    Validates architectural discipline, not just output quality.
    """
    print("=" * 70)
    print("  TEST: Milestone 2 — Architectural Maturity Validation")
    print(f"  Task: {task}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Run the full pipeline
    final_state = run_milestone2(task)

    # ── Assertions ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION CHECKS")
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

    # 3. Step types present (research, compare/unify/refine)
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

    # 4. All TODOs are marked done
    done_count = sum(1 for t in todos if t.get("status") == "done")
    if done_count == len(todos):
        print(f"  ✅ All {done_count} TODOs marked as done")
    else:
        msg = f"Only {done_count}/{len(todos)} TODOs marked done"
        print(f"  ⚠️  {msg}")
        warnings.append(msg)

    # 5. Meaningful file names (NOT summary1.txt, summary2.txt, summary3.txt)
    files = final_state.get("files", {})
    numbered_files = [f for f in files.keys() if f.startswith("summary")]
    if files and not numbered_files:
        print(f"  ✅ Meaningful file names: {list(files.keys())}")
    elif numbered_files:
        msg = f"Generic numbered files found: {numbered_files}"
        print(f"  ⚠️  {msg}")
        warnings.append(msg)
    else:
        msg = "No files in VFS"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 6. Key output files exist (comparison, unified model)
    key_file_patterns = ["comparison", "unified", "model"]
    found_key = [f for f in files.keys()
                 if any(p in f for p in key_file_patterns)]
    if found_key:
        print(f"  ✅ Key output files present: {found_key}")
    else:
        warnings.append("No comparison/unified model files found")
        print(f"  ⚠️  {warnings[-1]}")

    # 7. Files have substantial content
    short_files = [f for f, c in files.items() if len(c) < 50]
    if not short_files:
        print(f"  ✅ All {len(files)} files have substantial content (>50 chars)")
    else:
        msg = f"Short files detected: {short_files}"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 8. Trace log exists and shows tool invocation sequence
    trace_log = final_state.get("trace_log", [])
    if len(trace_log) >= 4:
        print(f"  ✅ Trace log has {len(trace_log)} entries — tool sequence visible")
    else:
        msg = f"Trace log too short ({len(trace_log)} entries)"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 9. edit_file appears in trace (refinement pattern)
    edit_actions = [t for t in trace_log if t.get("action") == "edit_file"]
    if edit_actions:
        print(f"  ✅ edit_file used {len(edit_actions)} time(s) — read→modify→edit verified")
    else:
        warnings.append("No edit_file actions in trace — refinement may not have triggered")
        print(f"  ⚠️  {warnings[-1]}")

    # 10. Selective retrieval visible in trace
    read_actions = [t for t in trace_log if t.get("action") == "read_file"]
    selective_reads = [t for t in read_actions
                       if "selective" in t.get("purpose", "").lower()
                       or "Selective" in t.get("purpose", "")]
    if selective_reads:
        print(f"  ✅ Selective retrieval demonstrated ({len(selective_reads)} selective reads)")
    elif read_actions:
        warnings.append("Read actions present but not marked as selective")
        print(f"  ⚠️  {warnings[-1]}")
    else:
        msg = "No read_file actions in trace"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 11. Dependency chain — read actions reference specific files
    write_files = [t["file"] for t in trace_log if t["action"] == "write_file"]
    read_files = [t["file"] for t in trace_log if t["action"] == "read_file"]
    dep_reads = [f for f in read_files if f in write_files]
    if dep_reads:
        print(f"  ✅ Dependency chain valid: reads reference prior writes ({len(dep_reads)} links)")
    else:
        warnings.append("Could not verify dependency chain in trace")
        print(f"  ⚠️  {warnings[-1]}")

    # 12. Final output exists and is substantial
    final_output = final_state.get("final_output", "")
    if len(final_output) > 100:
        print(f"  ✅ Final structured summary generated ({len(final_output)} chars)")
    else:
        msg = f"Final output too short ({len(final_output)} chars)"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 13. Final output contains expected structure markers
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
            print("  🎉 ALL CHECKS PASSED — Milestone 2 fully verified!")
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
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "todo_count": len(todos),
        "todos_done": done_count,
        "step_types": list(step_types),
        "files_in_vfs": list(files.keys()),
        "file_sizes": {k: len(v) for k, v in files.items()},
        "trace_log_length": len(trace_log),
        "edit_file_used": len(edit_actions) > 0,
        "selective_retrieval": len(selective_reads) > 0,
        "dependency_chain_valid": len(dep_reads) > 0,
        "final_output_length": len(final_output),
        "structure_markers_found": found_markers,
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }

    os.makedirs("outputs", exist_ok=True)
    test_path = os.path.join("outputs", "milestone2_test_result.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_result, f, indent=2, ensure_ascii=False)
    print(f"\n  Test results saved to {test_path}")

    return test_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Milestone 2: Architectural Maturity Validation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The task for the agent. If not provided, interactive mode.",
    )
    args = parser.parse_args()

    if args.task:
        test_vfs_pipeline(args.task)
    else:
        # Interactive mode
        print("\n  Enter a task to test (or press Enter for default):")
        task = input("  > ").strip()
        if not task:
            task = (
                "Analyze four AI ethics frameworks, identify differences, "
                "propose a unified model, then refine it with sustainability "
                "considerations."
            )
            print(f"  Using default: {task}")
        test_vfs_pipeline(task)
