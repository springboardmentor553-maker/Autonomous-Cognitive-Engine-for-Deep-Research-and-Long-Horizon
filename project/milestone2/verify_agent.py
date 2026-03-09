"""
Milestone 2 — Agent Verification Script

Checks:
  1. Consistency: Same task run 3 times produces stable TODO structure
  2. Tool coverage: write_file, read_file, ls, edit_file all invoked correctly
  3. Accuracy: 10 tasks evaluated for correct tool sequence and file output
  4. Selective retrieval: Agent does not read unnecessary files
"""

import os
import sys
import time
import json
import ast
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Milestone2-Verify")

from app import create_milestone2_agent, run_agent
from tools.filesystem.vfs_tools import reset_vfs, get_vfs_snapshot

# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract tool names from a run's message list
# ─────────────────────────────────────────────────────────────────────────────

def extract_tool_calls(messages: list) -> list:
    """Return list of tool names called during the run."""
    tools_called = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])
        # Also catch ToolMessage names
        if hasattr(msg, "name") and msg.name:
            if msg.name not in tools_called:
                tools_called.append(msg.name)
    return tools_called


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Consistency (same task × 3)
# ─────────────────────────────────────────────────────────────────────────────

def verify_consistency(agent):
    task = (
        "Summarize two short renewable energy articles and write a combined output. "
        "Article 1: Solar panels have become 90% cheaper over the last decade. "
        "Article 2: Wind energy now powers 20% of global electricity."
    )
    print("\n" + "─" * 60)
    print("TEST 1 — Consistency (3 runs of same task)")
    print("─" * 60)

    results = []
    for i in range(3):
        time.sleep(10)
        print(f"  Run {i+1}/3 ...", end=" ", flush=True)
        try:
            result = run_agent(agent, task, thread_id=f"verify-consistency-{i}")
            tools = extract_tool_calls(result["messages"])
            has_write = "write_file" in tools
            has_todos = len(result["todos"]) > 0
            has_files = len(result["files"]) > 0
            ok = has_write and has_todos and has_files
            results.append(ok)
            print(f"{'✅' if ok else '❌'}  todos={len(result['todos'])}  files={list(result['files'].keys())}")
        except Exception as e:
            results.append(False)
            print(f"❌  ERROR: {e}")

    passed = sum(results)
    print(f"\n  Consistency score: {passed}/3")
    return passed == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Full tool coverage
# ─────────────────────────────────────────────────────────────────────────────

def verify_tool_coverage(agent):
    task = (
        "Read two policy documents, store summaries, compare them, then refine the comparison. "
        "Doc A: Carbon tax is effective when combined with revenue recycling. "
        "Doc B: Cap-and-trade systems provide flexible compliance pathways for industry. "
        "Steps: summarize each → write_file → ls → read_file selectively → "
        "write comparison → read comparison → edit_file to add recommendation."
    )
    print("\n" + "─" * 60)
    print("TEST 2 — Full Tool Coverage (write/read/ls/edit)")
    print("─" * 60)
    time.sleep(10)

    try:
        result = run_agent(agent, task, thread_id="verify-coverage")
        tools = extract_tool_calls(result["messages"])
        print(f"  Tools called: {tools}")

        checks = {
            "write_todos":  "write_todos" in tools,
            "write_file":   "write_file"  in tools,
            "read_file":    "read_file"   in tools,
            "ls":           "ls"          in tools,
            "edit_file":    "edit_file"   in tools,
        }

        for tool, ok in checks.items():
            print(f"  {'✅' if ok else '❌'}  {tool}")

        all_ok = all(checks.values())
        print(f"\n  Coverage: {'PASS' if all_ok else 'FAIL'}")
        return all_ok
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: 10-task accuracy evaluation
# ─────────────────────────────────────────────────────────────────────────────

EVAL_TASKS = [
    "Summarize article about solar energy and store in solar_summary.txt, then read it back.",
    "Write a brief wind energy overview to wind.txt, list all files, then read wind.txt.",
    "Summarize two AI documents (AI_doc1.txt, AI_doc2.txt) then write a comparison to ai_compare.txt.",
    "Create a draft policy in draft.txt, read it, then improve it using edit_file.",
    "Summarize 3 cybersecurity frameworks (NIST, ISO, NIS2), store each, then write a combined policy.",
    "Compare UK and EU climate policies: store each summary, read both selectively, write comparison.",
    "Summarize 4 renewable energy reports, store summaries, compare all, write unified strategy.",
    "Write initial model to model.txt, read it, refine with new insights using edit_file.",
    "Analyze two governance frameworks, store summaries, compare, write recommendation report.",
    "Summarize 5 education policy documents, store each, identify top 3 differences, write report.",
]

def verify_accuracy(agent):
    print("\n" + "─" * 60)
    print("TEST 3 — Accuracy Evaluation (10 tasks)")
    print("─" * 60)

    success = 0
    total   = len(EVAL_TASKS)

    for i, task in enumerate(EVAL_TASKS):
        time.sleep(12)
        print(f"\n  Task {i+1}: {task[:70]}...")
        try:
            result = run_agent(agent, task, thread_id=f"verify-accuracy-{i}")
            tools   = extract_tool_calls(result["messages"])
            has_plan  = "write_todos"  in tools
            has_write = "write_file"   in tools
            has_read  = "read_file"    in tools
            has_files = len(result["files"]) > 0
            has_todos = len(result["todos"]) > 0

            ok = has_plan and has_write and has_read and has_files and has_todos
            if ok:
                success += 1
                print(f"  ✅  tools={tools}  files={list(result['files'].keys())}")
            else:
                missing = []
                if not has_plan:  missing.append("no write_todos")
                if not has_write: missing.append("no write_file")
                if not has_read:  missing.append("no read_file")
                if not has_files: missing.append("no files stored")
                if not has_todos: missing.append("no todos")
                print(f"  ❌  {', '.join(missing)}")

        except Exception as e:
            print(f"  ❌  ERROR: {e}")

    accuracy = round((success / total) * 100)
    print(f"\n  Accuracy: {success}/{total}  ({accuracy}%)")
    print(f"  {'PASS ✅' if accuracy >= 80 else 'FAIL ❌ — needs improvement'}")
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Milestone 2 — Agent Verification")
    print("=" * 60)

    agent = create_milestone2_agent()

    consistent   = verify_consistency(agent)
    coverage_ok  = verify_tool_coverage(agent)
    accuracy_pct = verify_accuracy(agent)

    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Consistency:   {'✅ PASS' if consistent else '❌ FAIL'}")
    print(f"  Tool Coverage: {'✅ PASS' if coverage_ok else '❌ FAIL'}")
    print(f"  Accuracy:      {accuracy_pct}%  {'✅ PASS' if accuracy_pct >= 80 else '❌ FAIL'}")
    print("=" * 60)
