"""
test_runner.py - Milestone 2 Evaluation Script
Metric: Correct File System Tool Usage

Per-test checks:
  1. write_todos invoked (planning first)
  2. write_file called at least once (context offloaded)
  3. read_file called at least once (context retrieved before synthesis)
  4. At least 2 files saved in virtual file system
  5. Final output is non-empty (synthesis happened)

Success Criteria: ≥ 80% of test cases pass ALL checks.
"""

import json
import time
from main import run_agent, get_filesystem_tool_calls, LANGCHAIN_TRACING
from langchain_core.messages import AIMessage, ToolMessage

# ─────────────────────────────────────────────
# 10 Multi-Step Test Cases requiring context offloading
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC01",
        "request": (
            "Research the history and current state of quantum computing. "
            "Summarize the key hardware milestones, then analyze the top three software frameworks, "
            "and finally write a combined report with conclusions."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["quantum", "hardware", "software", "report"],
    },
    {
        "id": "TC02",
        "request": (
            "Compare microservices and monolithic architectures for a fintech startup. "
            "First gather the pros and cons of each, save them, then analyze which fits better "
            "for high-transaction environments, and produce a final recommendation document."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["microservice", "monolith", "fintech", "recommend"],
    },
    {
        "id": "TC03",
        "request": (
            "Create a 6-month machine learning roadmap for a Python developer. "
            "Research required skills and tools, save notes on each phase, "
            "then synthesize into a week-by-week structured plan."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["machine learning", "python", "roadmap", "plan"],
    },
    {
        "id": "TC04",
        "request": (
            "Analyze the environmental impact of large-scale AI model training. "
            "Gather data on energy consumption, save findings, analyze alternatives like "
            "efficient architectures or renewable energy, then write a policy brief."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["energy", "ai", "environment", "policy"],
    },
    {
        "id": "TC05",
        "request": (
            "Produce a technical comparison of PostgreSQL vs MongoDB for a real-time analytics platform. "
            "Research each database's strengths, save the notes, analyze performance trade-offs, "
            "and draft a final technical decision document."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["postgresql", "mongodb", "analytic", "performance"],
    },
    {
        "id": "TC06",
        "request": (
            "Research recent advances in transformer model efficiency (2022-2024). "
            "Save summaries on sparse attention, mixture-of-experts, and quantization, "
            "then synthesize the findings into a research summary document."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["transformer", "attention", "efficient", "summar"],
    },
    {
        "id": "TC07",
        "request": (
            "Design a cybersecurity incident response plan for a mid-sized SaaS company. "
            "Research best practices, save notes on detection and containment phases, "
            "then draft the full incident response playbook."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["cybersecurity", "incident", "response", "playbook"],
    },
    {
        "id": "TC08",
        "request": (
            "Investigate the current landscape of AI regulation globally. "
            "Save separate notes on EU AI Act, US executive orders, and China's AI governance, "
            "then analyze the differences and write a comparative policy brief."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["regulation", "eu", "policy", "governance"],
    },
    {
        "id": "TC09",
        "request": (
            "Create a technical architecture document for a scalable e-commerce platform. "
            "Research microservices patterns, save notes on each component (auth, catalog, payment, orders), "
            "analyze failure points, then draft the full architecture design document."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["architecture", "ecommerce", "microservice", "scalab"],
    },
    {
        "id": "TC10",
        "request": (
            "Write a comprehensive guide on building production-ready RAG (Retrieval Augmented Generation) systems. "
            "Research chunking strategies, embedding models, and vector stores, save notes on each, "
            "then synthesize into a complete implementation guide."
        ),
        "expected_files_min": 2,
        "expected_keywords": ["rag", "retrieval", "embedding", "vector"],
    },
]


# ─────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────

def evaluate_test_case(state: dict, test_case: dict) -> dict:
    """Run all Milestone 2 checks on the final agent state."""
    todos = state.get("todos", [])
    vfs = state.get("virtual_files", {})
    fs_calls = get_filesystem_tool_calls(state)
    write_todos_invoked = state.get("write_todos_invoked", False)

    # Extract final output text
    final_output = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_output = msg.content
            break

    # ── Check 1: write_todos invoked (planning first) ──────────────
    check_planning = write_todos_invoked

    # ── Check 2: write_file called at least once ───────────────────
    check_write_file = fs_calls["write_file"] >= 1

    # ── Check 3: read_file called at least once ────────────────────
    check_read_file = fs_calls["read_file"] >= 1

    # ── Check 4: At least N files saved in VFS ────────────────────
    check_files_saved = len(vfs) >= test_case["expected_files_min"]

    # ── Check 5: Final output is non-empty (synthesis happened) ───
    check_final_output = len(final_output.strip()) > 100

    # ── Check 6: Final output contains expected keywords ──────────
    output_lower = final_output.lower()
    matched_kw = sum(1 for kw in test_case["expected_keywords"] if kw.lower() in output_lower)
    keyword_coverage = round(matched_kw / len(test_case["expected_keywords"]), 2)
    check_keywords = keyword_coverage >= 0.5

    # Overall pass: ALL checks must pass
    passed = (
        check_planning
        and check_write_file
        and check_read_file
        and check_files_saved
        and check_final_output
        and check_keywords
    )

    return {
        "tc_id": test_case["id"],
        "passed": passed,
        "checks": {
            "write_todos_invoked":  check_planning,
            "write_file_called":    check_write_file,
            "read_file_called":     check_read_file,
            "files_saved":          check_files_saved,
            "final_output_present": check_final_output,
            "keyword_coverage_50pct": check_keywords,
        },
        "stats": {
            "todo_count":       len(todos),
            "todos_completed":  sum(1 for t in todos if t["status"] == "completed"),
            "files_in_vfs":     len(vfs),
            "vfs_filenames":    list(vfs.keys()),
            "write_file_calls": fs_calls["write_file"],
            "read_file_calls":  fs_calls["read_file"],
            "keyword_coverage": keyword_coverage,
            "output_length":    len(final_output),
        }
    }


def print_result(eval_result: dict, request: str):
    status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
    checks = eval_result["checks"]
    stats = eval_result["stats"]

    print(f"\n  {status} | {eval_result['tc_id']}")
    print(f"  Request : {request[:70]}...")
    print(f"  Checks  :")
    print(f"    write_todos invoked    : {'✅' if checks['write_todos_invoked']    else '❌'}")
    print(f"    write_file called      : {'✅' if checks['write_file_called']      else '❌'}  ({stats['write_file_calls']} calls)")
    print(f"    read_file called       : {'✅' if checks['read_file_called']       else '❌'}  ({stats['read_file_calls']} calls)")
    print(f"    files saved in VFS     : {'✅' if checks['files_saved']            else '❌'}  ({stats['files_in_vfs']} files: {stats['vfs_filenames']})")
    print(f"    final output present   : {'✅' if checks['final_output_present']   else '❌'}  ({stats['output_length']} chars)")
    print(f"    keyword coverage ≥50%  : {'✅' if checks['keyword_coverage_50pct'] else '❌'}  ({stats['keyword_coverage']*100:.0f}%)")
    print(f"  TODOs   : {stats['todos_completed']}/{stats['todo_count']} completed")


# ─────────────────────────────────────────────
# Main Evaluation Runner
# ─────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 65)
    print("  MILESTONE 2 EVALUATION — Context Offloading via VFS")
    print(f"  LangSmith Tracing: {'ENABLED ✅' if LANGCHAIN_TRACING else 'DISABLED ℹ️'}")
    print("=" * 65)

    results = []
    all_data = []

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] Running {tc['id']}...")
        try:
            state = run_agent(tc["request"], run_name=f"m2-eval-{tc['id']}")
            eval_result = evaluate_test_case(state, tc)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            import traceback
            traceback.print_exc()
            eval_result = {
                "tc_id": tc["id"],
                "passed": False,
                "checks": {k: False for k in [
                    "write_todos_invoked", "write_file_called", "read_file_called",
                    "files_saved", "final_output_present", "keyword_coverage_50pct"
                ]},
                "stats": {
                    "todo_count": 0, "todos_completed": 0, "files_in_vfs": 0,
                    "vfs_filenames": [], "write_file_calls": 0, "read_file_calls": 0,
                    "keyword_coverage": 0.0, "output_length": 0
                },
                "error": str(e),
            }

        print_result(eval_result, tc["request"])
        results.append(eval_result)
        all_data.append({"test_case": tc, "evaluation": eval_result})

        if i < len(TEST_CASES) - 1:
            time.sleep(1.5)

    # ── Summary ────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total * 100
    milestone_passed = pass_rate >= 80.0

    # Per-check breakdown
    check_keys = [
        "write_todos_invoked", "write_file_called", "read_file_called",
        "files_saved", "final_output_present", "keyword_coverage_50pct"
    ]
    check_counts = {k: sum(1 for r in results if r["checks"].get(k)) for k in check_keys}

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Total test cases              : {total}")
    print(f"  Passed (all checks)           : {passed}")
    print(f"  Failed                        : {total - passed}")
    print(f"  Pass Rate                     : {pass_rate:.1f}%")
    print(f"  Target                        : 80.0%")
    print(f"\n  Per-Check Breakdown:")
    for k, v in check_counts.items():
        print(f"    {k:<30}: {v}/{total}")
    print(f"\n  Milestone 2 Status: {'✅ PASSED' if milestone_passed else '❌ NOT YET PASSING'}")
    print("=" * 65 + "\n")

    if LANGCHAIN_TRACING:
        print("🔗 View traces at: https://smith.langchain.com")
        print(f"   Project: milestone2-deep-agent\n")

    output = {
        "milestone": 2,
        "langsmith_tracing_enabled": LANGCHAIN_TRACING,
        "pass_rate_percent": round(pass_rate, 1),
        "target_percent": 80.0,
        "milestone_passed": milestone_passed,
        "per_check_summary": {k: f"{v}/{total}" for k, v in check_counts.items()},
        "results": all_data,
    }

    with open("milestone2_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("📄 Full results saved to milestone2_eval_results.json\n")

    return output


if __name__ == "__main__":
    run_evaluation()