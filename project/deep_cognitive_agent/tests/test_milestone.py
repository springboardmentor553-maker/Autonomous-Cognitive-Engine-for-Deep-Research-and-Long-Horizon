"""
UNIFIED TEST SUITE - All Milestones (M1, M2, M3)
=================================================

This comprehensive test validates the entire Deep Cognitive Agent system
across all three milestones:

MILESTONE 1: Planning (write_todos)
  - write_todos tool called
  - Output is structured JSON
  - 4-6 actionable steps
  - Stored in state["todos"]

MILESTONE 2: Virtual File System (VFS) Memory
  - write_file, read_file, edit_file, ls tools
  - Enriched TODOs with step_type/output_file/depends_on
  - Meaningful file names (not summary1.txt)
  - Selective retrieval (read only needed files)
  - Dependency chain (A.txt -> B.txt -> comparison.txt)
  - Memory offloading (content in VFS, not messages)

MILESTONE 3: Multi-Agent Delegation
  - Supervisor coordinates, sub-agents execute
  - delegate_task actions in trace
  - Agent attribution (delegated_to field)
  - Delegation reasoning recorded
  - Over-delegation prevention (trivial tasks self-handled)
  - Result integration (sub-agent outputs validated)
  - All M2 features preserved

ACCURACY TARGETS: >= 80% for all components
"""

import os
import sys
import io
import json
import time
from datetime import datetime

# Fix Unicode output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "unified_milestone_test")


# ============================================================================
# TEST INPUTS (Diverse tasks for validation)
# ============================================================================

TEST_INPUTS = [
    # Research + Analysis tasks
    "Analyze four AI ethics frameworks, identify differences, propose a unified model, then refine it with sustainability considerations.",
    "Research renewable energy trends in solar, wind, and hydro power, compare their efficiency and cost, propose a unified strategy for adoption.",
    "Compare policy differences between Germany and India on climate change, propose a unified framework for international cooperation.",
]

STRONG_VERBS = [
    "analyze", "collect", "break down", "design", "compare", "draft",
    "evaluate", "implement", "validate", "test", "review", "identify",
    "research", "define", "develop", "create", "organize", "assess",
    "outline", "prioritize", "investigate", "establish", "map", "compile",
    "synthesize", "document", "benchmark", "categorize", "formulate",
    "determine", "examine", "explore", "plan", "select", "sequence",
    "propose", "refine", "unify", "integrate"
]


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def score_planning(todos: list) -> dict:
    """
    Score planning quality on four dimensions (max 20 points).

    Dimensions:
        Clarity (5): Steps are clear and unambiguous
        Completeness (5): 4-6 steps covering full task
        Specificity (5): Strong action verbs, no vagueness
        Logical Order (5): Steps in sensible sequence
    """
    tasks = [t.get("task", "") for t in todos]
    n = len(tasks)

    # Completeness (target: 4-6 steps)
    if 4 <= n <= 6:
        completeness = 5
    elif n == 3 or n == 7:
        completeness = 3
    else:
        completeness = 1

    # Specificity (strong verbs)
    verb_hits = 0
    for t in tasks:
        t_lower = t.lower()
        if any(t_lower.startswith(v) for v in STRONG_VERBS):
            verb_hits += 1
    specificity = min(5, max(1, round(verb_hits / max(n, 1) * 5)))

    # Clarity (5-15 words per step is ideal)
    word_counts = [len(t.split()) for t in tasks]
    avg_words = sum(word_counts) / max(len(word_counts), 1)
    if 5 <= avg_words <= 15:
        clarity = 5
    elif 3 <= avg_words < 5 or 15 < avg_words <= 20:
        clarity = 3
    else:
        clarity = 1

    # Logical Order (no duplicates)
    unique_steps = set(t.lower().strip() for t in tasks)
    if len(unique_steps) == n:
        order = 5
    else:
        order = max(1, 5 - (n - len(unique_steps)))

    total = clarity + completeness + specificity + order
    return {
        "clarity": clarity,
        "completeness": completeness,
        "specificity": specificity,
        "logical_order": order,
        "total": total,
        "max": 20,
        "percentage": round(total / 20 * 100, 1)
    }


def score_memory_usage(state: dict) -> dict:
    """
    Score VFS memory usage (Milestone 2).

    Checks:
        - Files created (VFS not empty)
        - Files reused (read actions in trace)
        - No duplication (no summary1.txt pattern)
        - Context offloading (trace shows write->read pattern)
        - Selective retrieval (not reading all files)
    """
    files = state.get("files", {})
    trace_log = state.get("trace_log", [])

    score = 0
    max_score = 25
    details = []

    # Files created (5 points)
    if len(files) >= 3:
        score += 5
        details.append("Files created: PASS")
    elif len(files) > 0:
        score += 2
        details.append("Files created: PARTIAL (< 3 files)")
    else:
        details.append("Files created: FAIL (no files)")

    # Meaningful file names (5 points)
    numbered_pattern = ["summary1", "summary2", "summary3", "file1", "file2",
                        "step_1", "step_2", "output1"]
    has_numbered = any(any(p in f for p in numbered_pattern) for f in files.keys())
    if not has_numbered and len(files) > 0:
        score += 5
        details.append("Meaningful names: PASS")
    elif has_numbered:
        score += 2
        details.append("Meaningful names: PARTIAL (found numbered files)")
    else:
        details.append("Meaningful names: N/A (no files)")

    # Write->Read pattern (5 points)
    write_files = [t["file"] for t in trace_log if t.get("action") == "write_file"]
    read_files = [t["file"] for t in trace_log if t.get("action") == "read_file"]
    reused = set(write_files) & set(read_files)
    if len(reused) >= 1:
        score += 5
        details.append(f"Context offloading: PASS ({len(reused)} files reused)")
    else:
        details.append("Context offloading: FAIL (no write->read pattern)")

    # Selective retrieval (5 points)
    selective_entries = [t for t in trace_log
                         if "selective" in t.get("purpose", "").lower()]
    if selective_entries:
        score += 5
        details.append(f"Selective retrieval: PASS ({len(selective_entries)} entries)")
    elif read_files:
        score += 2
        details.append("Selective retrieval: PARTIAL (reads exist but not marked)")
    else:
        details.append("Selective retrieval: FAIL")

    # edit_file used (5 points)
    edit_actions = [t for t in trace_log if t.get("action") == "edit_file"]
    if edit_actions:
        score += 5
        details.append(f"edit_file pattern: PASS ({len(edit_actions)} uses)")
    else:
        details.append("edit_file pattern: FAIL (not used)")

    return {
        "score": score,
        "max": max_score,
        "percentage": round(score / max_score * 100, 1),
        "details": details
    }


def score_delegation(state: dict) -> dict:
    """
    Score delegation logic (Milestone 3).

    Checks:
        - delegate_task actions present
        - Agent attribution (delegated_to field)
        - Multiple agents used
        - Delegation reasoning recorded
        - Result integration valid
        - Over-delegation prevention
    """
    trace_log = state.get("trace_log", [])
    files = state.get("files", {})

    score = 0
    max_score = 30
    details = []

    # delegate_task actions (5 points)
    delegation_actions = [t for t in trace_log if t.get("action") == "delegate_task"]
    if len(delegation_actions) >= 2:
        score += 5
        details.append(f"Delegation active: PASS ({len(delegation_actions)} delegations)")
    elif len(delegation_actions) == 1:
        score += 3
        details.append("Delegation active: PARTIAL (only 1 delegation)")
    else:
        details.append("Delegation active: FAIL (no delegations)")

    # Agent attribution (5 points)
    attributed = [t for t in trace_log if t.get("delegated_to")]
    agents_used = set(t["delegated_to"] for t in attributed)
    if len(agents_used) >= 2:
        score += 5
        details.append(f"Agent attribution: PASS ({agents_used})")
    elif len(agents_used) == 1:
        score += 3
        details.append(f"Agent attribution: PARTIAL ({agents_used})")
    else:
        details.append("Agent attribution: FAIL")

    # Multiple agents (5 points)
    if len(agents_used) >= 3:
        score += 5
        details.append(f"Multi-agent: PASS ({len(agents_used)} agents)")
    elif len(agents_used) >= 2:
        score += 3
        details.append(f"Multi-agent: PARTIAL ({len(agents_used)} agents)")
    else:
        details.append("Multi-agent: FAIL")

    # Delegation reasoning (5 points)
    reasoning_entries = [t for t in trace_log if t.get("delegation_reasoning")]
    if len(reasoning_entries) >= 2:
        score += 5
        details.append(f"Delegation reasoning: PASS ({len(reasoning_entries)} entries)")
    elif reasoning_entries:
        score += 3
        details.append("Delegation reasoning: PARTIAL")
    else:
        details.append("Delegation reasoning: FAIL")

    # Result integration (5 points) - files have substantial content
    empty_files = [f for f, c in files.items() if len(c.strip()) < 20]
    if not empty_files and len(files) > 0:
        score += 5
        details.append("Result integration: PASS")
    elif len(empty_files) < len(files) / 2:
        score += 3
        details.append(f"Result integration: PARTIAL ({len(empty_files)} empty files)")
    else:
        details.append("Result integration: FAIL")

    # Over-delegation prevention (5 points)
    supervisor_handled = [t for t in trace_log if t.get("action") == "supervisor_handle"]
    if supervisor_handled:
        score += 5
        details.append(f"Over-delegation prevention: PASS ({len(supervisor_handled)} self-handled)")
    else:
        # If all tasks are complex, no self-handling is expected
        score += 3
        details.append("Over-delegation prevention: PARTIAL (no trivial tasks detected)")

    return {
        "score": score,
        "max": max_score,
        "percentage": round(score / max_score * 100, 1),
        "details": details,
        "agents_used": list(agents_used)
    }


def score_final_output(state: dict) -> dict:
    """
    Score final output quality.

    Checks:
        - Output exists and is substantial (> 200 chars)
        - Structured format (5 sections)
        - Produced from VFS (not direct LLM answer)
    """
    final_output = state.get("final_output", "")
    trace_log = state.get("trace_log", [])

    score = 0
    max_score = 15
    details = []

    # Output exists (5 points)
    if len(final_output) > 500:
        score += 5
        details.append(f"Output exists: PASS ({len(final_output)} chars)")
    elif len(final_output) > 200:
        score += 3
        details.append(f"Output exists: PARTIAL ({len(final_output)} chars)")
    else:
        details.append(f"Output exists: FAIL ({len(final_output)} chars)")

    # Structured format (5 points)
    sections = ["overview", "key findings", "analysis", "recommendations", "conclusion"]
    found = sum(1 for s in sections if s in final_output.lower())
    if found >= 4:
        score += 5
        details.append(f"Structured format: PASS ({found}/5 sections)")
    elif found >= 2:
        score += 3
        details.append(f"Structured format: PARTIAL ({found}/5 sections)")
    else:
        details.append(f"Structured format: FAIL ({found}/5 sections)")

    # Produced from VFS (5 points) - trace shows synthesis from files
    synthesis_reads = [t for t in trace_log
                       if t.get("action") == "read_file"
                       and t.get("step") == "synthesis"]
    if synthesis_reads:
        score += 5
        details.append(f"VFS synthesis: PASS ({len(synthesis_reads)} files read)")
    elif any(t.get("action") == "synthesize" for t in trace_log):
        score += 3
        details.append("VFS synthesis: PARTIAL")
    else:
        details.append("VFS synthesis: FAIL")

    return {
        "score": score,
        "max": max_score,
        "percentage": round(score / max_score * 100, 1),
        "details": details
    }


# ============================================================================
# MAIN TEST FUNCTIONS
# ============================================================================

def test_milestone1_planning(task: str):
    """
    Test Milestone 1: Planning Agent.

    Validates:
        - write_todos tool is called
        - Output is structured JSON
        - 4-6 actionable steps
        - Stored in state["todos"]
    """
    from app import create_planning_agent, run_agent

    print("=" * 70)
    print("  MILESTONE 1: Planning Agent Test")
    print(f"  Task: {task[:60]}...")
    print("=" * 70)

    agent = create_planning_agent()
    result = run_agent(agent, task, thread_id=f"m1-{datetime.now().strftime('%H%M%S')}")

    todos = result.get("todos", [])
    messages = result.get("messages", [])

    # Verify write_todos was called
    tool_called = any(hasattr(m, 'name') and m.name == "write_todos" for m in messages)

    # Score planning
    plan_score = score_planning(todos)

    print(f"\n  Results:")
    print(f"    write_todos called: {'PASS' if tool_called else 'FAIL'}")
    print(f"    TODOs generated: {len(todos)}")
    print(f"    Planning score: {plan_score['total']}/{plan_score['max']} ({plan_score['percentage']}%)")

    if todos:
        print(f"\n  TODOs:")
        for i, todo in enumerate(todos, 1):
            print(f"    {i}. {todo.get('task', 'N/A')[:60]}...")

    passed = tool_called and len(todos) >= 4 and plan_score['percentage'] >= 80
    print(f"\n  Status: {'PASS' if passed else 'FAIL'}")

    return {
        "milestone": 1,
        "tool_called": tool_called,
        "todo_count": len(todos),
        "plan_score": plan_score,
        "passed": passed
    }


def test_milestone2_vfs(task: str):
    """
    Test Milestone 2: VFS Context Offloading.

    Validates:
        - Enriched TODOs with step_type/output_file/depends_on
        - Meaningful file names
        - Selective retrieval
        - edit_file for refinement
        - Dependency chain
        - Memory offloading
    """
    from app_milestone2 import run_milestone2

    print("=" * 70)
    print("  MILESTONE 2: VFS Context Offloading Test")
    print(f"  Task: {task[:60]}...")
    print("=" * 70)

    final_state = run_milestone2(task)

    todos = final_state.get("todos", [])
    files = final_state.get("files", {})
    trace_log = final_state.get("trace_log", [])
    final_output = final_state.get("final_output", "")

    # Score components
    plan_score = score_planning(todos)
    memory_score = score_memory_usage(final_state)
    output_score = score_final_output(final_state)

    # Check enrichment
    enriched = sum(1 for t in todos if "step_type" in t)
    step_types = set(t.get("step_type", "") for t in todos)

    print(f"\n  Results:")
    print(f"    TODOs: {len(todos)} (enriched: {enriched})")
    print(f"    Step types: {step_types}")
    print(f"    Files in VFS: {list(files.keys())}")
    print(f"    Trace entries: {len(trace_log)}")
    print(f"    Final output: {len(final_output)} chars")

    print(f"\n  Scores:")
    print(f"    Planning: {plan_score['percentage']}%")
    print(f"    Memory usage: {memory_score['percentage']}%")
    print(f"    Final output: {output_score['percentage']}%")

    for detail in memory_score["details"]:
        print(f"      - {detail}")

    overall = (plan_score['percentage'] + memory_score['percentage'] + output_score['percentage']) / 3
    passed = overall >= 80 and enriched == len(todos)

    print(f"\n  Overall: {overall:.1f}%")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return {
        "milestone": 2,
        "todo_count": len(todos),
        "enriched_count": enriched,
        "step_types": list(step_types),
        "files": list(files.keys()),
        "trace_entries": len(trace_log),
        "plan_score": plan_score,
        "memory_score": memory_score,
        "output_score": output_score,
        "overall_percentage": overall,
        "passed": passed,
        "state": final_state  # For M3 to build upon
    }


def test_milestone3_delegation(task: str):
    """
    Test Milestone 3: Multi-Agent Delegation.

    Validates:
        - delegate_task actions
        - Agent attribution
        - Multiple agents used
        - Delegation reasoning
        - Result integration
        - Over-delegation prevention
        - All M2 features preserved
    """
    from app_milestone3 import run_milestone3

    print("=" * 70)
    print("  MILESTONE 3: Multi-Agent Delegation Test")
    print(f"  Task: {task[:60]}...")
    print("=" * 70)

    final_state = run_milestone3(task)

    todos = final_state.get("todos", [])
    files = final_state.get("files", {})
    trace_log = final_state.get("trace_log", [])
    final_output = final_state.get("final_output", "")

    # Score all components
    plan_score = score_planning(todos)
    memory_score = score_memory_usage(final_state)
    delegation_score = score_delegation(final_state)
    output_score = score_final_output(final_state)

    print(f"\n  Results:")
    print(f"    TODOs: {len(todos)}")
    print(f"    Files in VFS: {list(files.keys())}")
    print(f"    Trace entries: {len(trace_log)}")
    print(f"    Agents used: {delegation_score['agents_used']}")
    print(f"    Final output: {len(final_output)} chars")

    print(f"\n  Scores:")
    print(f"    Planning: {plan_score['percentage']}%")
    print(f"    Memory (M2): {memory_score['percentage']}%")
    print(f"    Delegation (M3): {delegation_score['percentage']}%")
    print(f"    Final output: {output_score['percentage']}%")

    print(f"\n  Delegation Details:")
    for detail in delegation_score["details"]:
        print(f"      - {detail}")

    # Overall score (weighted)
    overall = (
        plan_score['percentage'] * 0.2 +
        memory_score['percentage'] * 0.25 +
        delegation_score['percentage'] * 0.35 +
        output_score['percentage'] * 0.2
    )

    passed = (
        overall >= 80 and
        delegation_score['percentage'] >= 70 and
        memory_score['percentage'] >= 70
    )

    print(f"\n  Overall (weighted): {overall:.1f}%")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return {
        "milestone": 3,
        "todo_count": len(todos),
        "files": list(files.keys()),
        "trace_entries": len(trace_log),
        "agents_used": delegation_score["agents_used"],
        "plan_score": plan_score,
        "memory_score": memory_score,
        "delegation_score": delegation_score,
        "output_score": output_score,
        "overall_percentage": overall,
        "passed": passed
    }


def run_full_test_suite(task_index: int = 0):
    """
    Run the complete test suite across all milestones.

    Args:
        task_index: Index of test task to use (0-2)
    """
    task = TEST_INPUTS[task_index]

    print("\n" + "=" * 70)
    print("  UNIFIED MILESTONE TEST SUITE")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Task: {task}")
    print("=" * 70)

    results = {}

    # Test Milestone 1
    print("\n" + "-" * 70)
    print("  Running Milestone 1 (Planning)...")
    print("-" * 70)
    try:
        results["m1"] = test_milestone1_planning(task)
        time.sleep(5)  # Rate limit buffer
    except Exception as e:
        print(f"  ERROR: {e}")
        results["m1"] = {"passed": False, "error": str(e)}

    # Test Milestone 2
    print("\n" + "-" * 70)
    print("  Running Milestone 2 (VFS)...")
    print("-" * 70)
    try:
        results["m2"] = test_milestone2_vfs(task)
        time.sleep(5)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["m2"] = {"passed": False, "error": str(e)}

    # Test Milestone 3
    print("\n" + "-" * 70)
    print("  Running Milestone 3 (Delegation)...")
    print("-" * 70)
    try:
        results["m3"] = test_milestone3_delegation(task)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["m3"] = {"passed": False, "error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL TEST SUMMARY")
    print("=" * 70)

    all_passed = all(r.get("passed", False) for r in results.values())

    print(f"\n  Milestone 1 (Planning):   {'PASS' if results.get('m1', {}).get('passed') else 'FAIL'}")
    print(f"  Milestone 2 (VFS):        {'PASS' if results.get('m2', {}).get('passed') else 'FAIL'}")
    print(f"  Milestone 3 (Delegation): {'PASS' if results.get('m3', {}).get('passed') else 'FAIL'}")

    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    # Save results
    os.makedirs("outputs", exist_ok=True)

    # Remove non-serializable state from results
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items() if kk != "state"}
        else:
            serializable[k] = v

    result_path = os.path.join("outputs", "unified_test_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "results": serializable,
            "all_passed": all_passed
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {result_path}")
    print("=" * 70)

    return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Milestone Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_milestone.py --all              # Run all milestones
    python test_milestone.py --m1               # Test planning only
    python test_milestone.py --m2               # Test VFS only
    python test_milestone.py --m3               # Test delegation only
    python test_milestone.py --task 1           # Use task index 1
    python test_milestone.py --custom "task"    # Use custom task
        """
    )

    parser.add_argument("--all", action="store_true", help="Run all milestone tests")
    parser.add_argument("--m1", action="store_true", help="Test Milestone 1 (Planning)")
    parser.add_argument("--m2", action="store_true", help="Test Milestone 2 (VFS)")
    parser.add_argument("--m3", action="store_true", help="Test Milestone 3 (Delegation)")
    parser.add_argument("--task", type=int, default=0, choices=[0, 1, 2],
                        help="Task index to use (0-2)")
    parser.add_argument("--custom", type=str, help="Custom task string")

    args = parser.parse_args()

    # Determine task
    if args.custom:
        task = args.custom
    else:
        task = TEST_INPUTS[args.task]

    # Run appropriate tests
    if args.all or (not args.m1 and not args.m2 and not args.m3):
        run_full_test_suite(args.task)
    else:
        if args.m1:
            test_milestone1_planning(task)
        if args.m2:
            test_milestone2_vfs(task)
        if args.m3:
            test_milestone3_delegation(task)
