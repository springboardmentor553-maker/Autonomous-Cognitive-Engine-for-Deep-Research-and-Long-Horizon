"""
Test Planning Agent - Milestone 1

This script tests the ReAct planning agent with 7 complex inputs,
scores each plan on Clarity/Completeness/Specificity/Order,
and saves the generated todos to outputs/*.json files.

Test Inputs:
1. Create a research outline for renewable energy trends
2. Design a structured learning roadmap for data science
3. Break down the steps for developing a web application
4. Plan a comparative study between electric and hydrogen vehicles
5. Create a technical writing outline for AI ethics
6. Develop a marketing strategy for a new SaaS product
7. Design a disaster recovery plan for a cloud-based system
"""

import os
import sys
import json
import re
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable LangSmith Tracing (disabled by default — set to "true" + add LANGCHAIN_API_KEY to enable)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "milestone_1_planning")

from app import create_planning_agent, run_agent, save_result_to_json


# ── Strong action verbs expected in high-quality plans ──
STRONG_VERBS = [
    "analyze", "collect", "break down", "design", "compare", "draft",
    "evaluate", "implement", "validate", "test", "review", "identify",
    "research", "define", "develop", "create", "organize", "assess",
    "outline", "prioritize", "investigate", "establish", "map", "compile",
    "synthesize", "document", "benchmark", "categorize", "formulate",
    "determine", "examine", "explore", "plan", "select", "sequence",
]

# Test inputs — 7 tasks covering technical, research, strategic & creative domains
TEST_INPUTS = [
    "Create a research outline for renewable energy trends",
    "Design a structured learning roadmap for data science",
    "Break down the steps for developing a web application",
    "Plan a comparative study between electric and hydrogen vehicles",
    "Create a technical writing outline for AI ethics",
    "Develop a marketing strategy for a new SaaS product",
    "Design a disaster recovery plan for a cloud-based system",
    "Create a deployment checklist for Kubernetes microservices" 
    
]


# ── Plan Quality Scoring ──────────────────────────────────────────────
def score_plan(todos: list) -> dict:
    """
    Score a TODO plan on four dimensions (1-5 each, max 20).

    Dimensions:
        Clarity       – Are steps clear and unambiguous?
        Completeness  – Does the plan cover the full task (4-6 steps)?
        Specificity   – Do steps use strong action verbs and avoid vagueness?
        Logical Order – Are steps in a sensible sequence?

    Returns dict with per-dimension scores and total.
    """
    tasks = [t["task"] for t in todos]
    n = len(tasks)

    # --- Completeness (target: 4-6 steps) ---
    if 4 <= n <= 6:
        completeness = 5
    elif n == 3 or n == 7:
        completeness = 3
    else:
        completeness = 1

    # --- Specificity (strong verbs) ---
    verb_hits = 0
    for t in tasks:
        t_lower = t.lower()
        if any(t_lower.startswith(v) for v in STRONG_VERBS):
            verb_hits += 1
    specificity = min(5, max(1, round(verb_hits / max(n, 1) * 5)))

    # --- Clarity (average word count per step — 5-15 words is ideal) ---
    word_counts = [len(t.split()) for t in tasks]
    avg_words = sum(word_counts) / max(len(word_counts), 1)
    if 5 <= avg_words <= 15:
        clarity = 5
    elif 3 <= avg_words < 5 or 15 < avg_words <= 20:
        clarity = 3
    else:
        clarity = 1

    # --- Logical Order (simple heuristic: no duplicate steps) ---
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
    }


# ── Test runners ──────────────────────────────────────────────────────
def run_all_tests():
    """
    Run the planning agent on all 7 test inputs, score each plan,
    and save results + summary to outputs/.
    """
    print("=" * 70)
    print("MILESTONE 1 - PLANNING AGENT TEST SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Tests: {len(TEST_INPUTS)}")
    print("=" * 70)

    print("\nInitializing Planning Agent...")
    agent = create_planning_agent()
    print("Agent initialized successfully!\n")

    all_results = []

    for i, task in enumerate(TEST_INPUTS, 1):
        print("-" * 70)
        print(f"TEST {i}/{len(TEST_INPUTS)}: {task}")
        print("-" * 70)

        try:
            thread_id = f"test-{i}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            result = run_agent(agent, task, thread_id=thread_id)

            n_todos = len(result["todos"])

            # ── Task 19: Verify tool was called, output structured, stored in state ──
            tool_called = False
            for msg in result["messages"]:
                if hasattr(msg, 'name') and msg.name == "write_todos":
                    tool_called = True
                    break
            assert tool_called, "write_todos tool was NOT called!"
            assert n_todos >= 4, f"Expected 4-6 todos, got {n_todos}"
            assert all("task" in t and "status" in t for t in result["todos"]), "Todos not structured correctly"
            print(f"  ✓ Verified: tool called, {n_todos} structured todos in state")

            print(f"\nGenerated {n_todos} TODOs:")
            for j, todo in enumerate(result["todos"], 1):
                icon = "⬜" if todo["status"] == "pending" else "✅"
                print(f"  {j}. {icon} {todo['task']}")

            # Score the plan
            scores = score_plan(result["todos"])
            print(f"\n  Plan Score: {scores['total']}/{scores['max']}")
            print(f"    Clarity:      {scores['clarity']}/5")
            print(f"    Completeness: {scores['completeness']}/5")
            print(f"    Specificity:  {scores['specificity']}/5")
            print(f"    Order:        {scores['logical_order']}/5")

            # Save to JSON
            filename = f"test_{i}_{task.lower().replace(' ', '_')[:30]}.json"
            filepath = save_result_to_json(result, filename)

            all_results.append({
                "test_number": i,
                "task": task,
                "todo_count": n_todos,
                "scores": scores,
                "output_file": filepath,
                "success": True,
            })
            print(f"\n✅ Test {i} completed successfully!")

            # Small delay between tests to avoid Groq rate limits
            if i < len(TEST_INPUTS):
                print("  \u23f3 Waiting 12s before next test (rate-limit safety)...")
                time.sleep(12)

        except Exception as e:
            print(f"\n❌ Test {i} failed with error: {e}")
            all_results.append({
                "test_number": i,
                "task": task,
                "todo_count": 0,
                "scores": None,
                "output_file": None,
                "success": False,
                "error": str(e),
            })

        print()

    # ── Summary ──
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in all_results if r["success"])
    total_score = sum(r["scores"]["total"] for r in all_results if r["scores"])
    max_score = 20 * len(TEST_INPUTS)

    print(f"\nTotal Tests:  {len(TEST_INPUTS)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {len(TEST_INPUTS) - successful}")
    print(f"Total Score:  {total_score}/{max_score}")
    accuracy_pct = round(total_score / max_score * 100, 1) if max_score > 0 else 0
    print(f"Accuracy:     {accuracy_pct}% {'✅ PASS (≥80%)' if accuracy_pct >= 80 else '❌ BELOW 80%'}")

    print("\nDetailed Results:")
    for r in all_results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        sc = f" (score {r['scores']['total']}/20)" if r["scores"] else ""
        print(f"  {r['test_number']}. {status} - {r['task'][:45]}...{sc} ({r['todo_count']} todos)")

    # Save summary JSON
    summary_file = os.path.join("outputs", "test_summary.json")
    os.makedirs("outputs", exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(TEST_INPUTS),
                "successful": successful,
                "failed": len(TEST_INPUTS) - successful,
                "total_score": total_score,
                "max_score": max_score,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nSummary saved to: {summary_file}")
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        print(f"\nCheck LangSmith dashboard for traces:")
        print(f"  Project: {os.getenv('LANGCHAIN_PROJECT', 'milestone_1_planning')}")
    print("=" * 70)

    return all_results


def run_single_test(test_number: int):
    """
    Run a single test by number (1-7).
    """
    if test_number < 1 or test_number > len(TEST_INPUTS):
        print(f"Invalid test number. Please choose 1-{len(TEST_INPUTS)}")
        return

    task = TEST_INPUTS[test_number - 1]
    print(f"\nRunning single test: {task}")
    print("-" * 50)

    agent = create_planning_agent()

    try:
        result = run_agent(agent, task, thread_id=f"single-test-{test_number}")
    except Exception as e:
        print(f"\n❌ Test {test_number} failed: {e}")
        return None

    print(f"\nGenerated TODOs:")
    for i, todo in enumerate(result["todos"], 1):
        print(f"  {i}. {todo['task']} [{todo['status']}]")

    # Score
    scores = score_plan(result["todos"])
    print(f"\n  Plan Score: {scores['total']}/{scores['max']}")
    print(f"    Clarity:      {scores['clarity']}/5")
    print(f"    Completeness: {scores['completeness']}/5")
    print(f"    Specificity:  {scores['specificity']}/5")
    print(f"    Order:        {scores['logical_order']}/5")

    filename = f"single_test_{test_number}.json"
    save_result_to_json(result, filename)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the Planning Agent")
    parser.add_argument(
        "--test",
        type=int,
        choices=range(1, len(TEST_INPUTS) + 1),
        metavar=f"1-{len(TEST_INPUTS)}",
        help=f"Run a single test by number (1-{len(TEST_INPUTS)})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )

    args = parser.parse_args()

    if args.test:
        run_single_test(args.test)
    else:
        # Default: run all tests
        run_all_tests()
