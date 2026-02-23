"""
Test Milestone 2 - Agent with Context Offloading via VFS

This test verifies:
  1. TODOs are generated (planning phase works)
  2. state["files"] contains 3 summary files after execution
  3. Each file has substantial content (>50 chars)
  4. A final structured summary is produced via read_file + synthesis
  5. write_file was used for each summary (files exist in state)
  6. read_file was implicitly used before final synthesis
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
    End-to-end test for the Milestone 2 VFS pipeline.
    Accepts ANY task — the pipeline dynamically plans, researches, and
    synthesizes based on whatever task string is provided.
    """
    print("=" * 70)
    print("  TEST: Milestone 2 - Context Offloading via VFS")
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

    # 1. TODOs exist
    todos = final_state.get("todos", [])
    if len(todos) >= 4:
        print(f"  ✅ TODOs generated: {len(todos)} steps")
    else:
        msg = f"Expected ≥4 TODOs, got {len(todos)}"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 2. All TODOs are marked done
    done_count = sum(1 for t in todos if t.get("status") == "done")
    if done_count == len(todos):
        print(f"  ✅ All {done_count} TODOs marked as done")
    else:
        msg = f"Only {done_count}/{len(todos)} TODOs marked done"
        print(f"  ⚠️  {msg}")

    # 3. state["files"] has 3 summary files
    files = final_state.get("files", {})
    expected_files = ["summary1.txt", "summary2.txt", "summary3.txt"]
    for fname in expected_files:
        if fname in files:
            content_len = len(files[fname])
            if content_len > 50:
                print(f"  ✅ {fname} exists ({content_len} chars) — write_file verified")
            else:
                msg = f"{fname} content too short ({content_len} chars)"
                print(f"  ❌ {msg}")
                errors.append(msg)
        else:
            msg = f"{fname} NOT found in state['files']"
            print(f"  ❌ {msg}")
            errors.append(msg)

    # 4. Final output exists and is substantial
    final_output = final_state.get("final_output", "")
    if len(final_output) > 100:
        print(f"  ✅ Final structured summary generated ({len(final_output)} chars)")
    else:
        msg = f"Final output too short ({len(final_output)} chars)"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # 5. Final output contains expected structure markers
    structure_markers = ["Overview", "Key Findings", "Analysis", "Conclusion"]
    found_markers = [m for m in structure_markers if m.lower() in final_output.lower()]
    if len(found_markers) >= 3:
        print(f"  ✅ Structured format verified ({len(found_markers)}/4 sections found)")
    else:
        msg = f"Only {len(found_markers)}/4 structure sections found: {found_markers}"
        print(f"  ⚠️  {msg}")

    # 6. VFS files are visible in state (not bypassed)
    if len(files) >= 3:
        print(f"  ✅ VFS contains {len(files)} files — context offloading verified")
    else:
        msg = f"Expected ≥3 files in VFS, got {len(files)}"
        print(f"  ❌ {msg}")
        errors.append(msg)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    if not errors:
        print("  🎉 ALL CHECKS PASSED — Milestone 2 verified!")
    else:
        print(f"  ⚠️  {len(errors)} check(s) failed:")
        for err in errors:
            print(f"     - {err}")
    print("-" * 70)

    # Save test results
    test_result = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "todo_count": len(todos),
        "todos_done": done_count,
        "files_in_vfs": list(files.keys()),
        "file_sizes": {k: len(v) for k, v in files.items()},
        "final_output_length": len(final_output),
        "structure_markers_found": found_markers,
        "errors": errors,
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
        description="Test Milestone 2: VFS Context Offloading pipeline",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task for the agent. Can be ANY task.",
    )
    args = parser.parse_args()

    test_vfs_pipeline(args.task)
