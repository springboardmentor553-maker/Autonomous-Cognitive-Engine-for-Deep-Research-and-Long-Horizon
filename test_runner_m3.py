"""
test_runner_m3.py - Milestone 3 Evaluation Script
Metric: Successful Delegation and Result Integration

Per-test checks:
  1. write_todos invoked (planning first)
  2. task() (delegation) tool called ≥ 1 time
  3. Correct sub-agent used (agent_name appears in delegation log)
  4. Result saved to VFS (write_file called ≥ 1 time after delegation)
  5. Final output is non-empty (synthesis happened, ≥ 100 chars)
  6. Keyword coverage ≥ 50% of expected keywords in final output

Success Criteria: ≥ 80% of test cases pass ALL checks.
"""

import json
import time
from main import run_agent, get_filesystem_tool_calls, get_delegation_tool_calls, LANGCHAIN_TRACING
from langchain_core.messages import AIMessage, ToolMessage


# ─────────────────────────────────────────────
# 10 Delegation Test Cases
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC01",
        "request": (
            "Research the history and key milestones of quantum computing hardware. "
            "Delegate the research to the appropriate sub-agent, save the findings, "
            "and then write a final summary report."
        ),
        "expected_agent": "web_search_agent",
        "expected_files_min": 1,
        "expected_keywords": ["quantum", "hardware", "qubit", "milestone"],
    },
    {
        "id": "TC02",
        "request": (
            "Summarize the key differences between microservices and monolithic architecture. "
            "Delegate the summarization task, then produce a final recommendation for a startup."
        ),
        "expected_agent": "summarization_agent",
        "expected_files_min": 1,
        "expected_keywords": ["microservice", "monolith", "recommend"],
    },
    {
        "id": "TC03",
        "request": (
            "Research the current state of large language model training techniques (2023-2024). "
            "Use the research sub-agent to gather facts, save to the file system, "
            "then synthesize a technical overview."
        ),
        "expected_agent": "web_search_agent",
        "expected_files_min": 1,
        "expected_keywords": ["language model", "training", "llm", "technique"],
    },
    {
        "id": "TC04",
        "request": (
            "Analyze the pros and cons of using Python vs Go for a high-performance backend API. "
            "Delegate the code analysis to a specialist agent, save the results, "
            "and write a final technical decision document."
        ),
        "expected_agent": "code_analysis_agent",
        "expected_files_min": 1,
        "expected_keywords": ["python", "go", "performance", "api"],
    },
    {
        "id": "TC05",
        "request": (
            "Research the EU AI Act and its implications for AI companies. "
            "Delegate to the research specialist, save findings, "
            "then draft a policy compliance brief."
        ),
        "expected_agent": "web_search_agent",
        "expected_files_min": 1,
        "expected_keywords": ["eu", "ai act", "compliance", "regulation"],
    },
    {
        "id": "TC06",
        "request": (
            "Summarize the main concepts of retrieval augmented generation (RAG). "
            "Delegate summarization to the appropriate sub-agent, "
            "then produce a beginner's guide to RAG systems."
        ),
        "expected_agent": "summarization_agent",
        "expected_files_min": 1,
        "expected_keywords": ["rag", "retrieval", "generation", "vector"],
    },
    {
        "id": "TC07",
        "request": (
            "Review the best practices for securing a REST API in production. "
            "Delegate to the code analysis specialist to cover authentication, "
            "rate limiting, and input validation. Save findings and write a security guide."
        ),
        "expected_agent": "code_analysis_agent",
        "expected_files_min": 1,
        "expected_keywords": ["security", "api", "authentication", "rest"],
    },
    {
        "id": "TC08",
        "request": (
            "Research the environmental impact of cryptocurrency mining globally. "
            "Delegate deep research to the research sub-agent. "
            "Save findings and produce an analytical report with policy recommendations."
        ),
        "expected_agent": "web_search_agent",
        "expected_files_min": 1,
        "expected_keywords": ["crypto", "energy", "environment", "mining"],
    },
    {
        "id": "TC09",
        "request": (
            "Summarize the core principles of DevOps and CI/CD pipelines. "
            "Delegate summarization work to the relevant sub-agent, "
            "then draft an implementation checklist for a team adopting DevOps."
        ),
        "expected_agent": "summarization_agent",
        "expected_files_min": 1,
        "expected_keywords": ["devops", "ci", "cd", "pipeline"],
    },
    {
        "id": "TC10",
        "request": (
            "Research recent advances in multi-agent AI systems and collaborative LLMs (2023-2025). "
            "Delegate research to the appropriate agent, save structured notes, "
            "then synthesize a research briefing document."
        ),
        "expected_agent": "web_search_agent",
        "expected_files_min": 1,
        "expected_keywords": ["multi-agent", "collaboration", "llm", "autonomou"],
    },
]


# ─────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────

def evaluate_test_case(state: dict, test_case: dict) -> dict:
    """Run all Milestone 3 checks on the final agent state."""
    todos = state.get("todos", [])
    vfs = state.get("virtual_files", {})
    delegation_log = state.get("delegation_log", [])
    fs_calls = get_filesystem_tool_calls(state)
    del_calls = get_delegation_tool_calls(state)
    write_todos_invoked = state.get("write_todos_invoked", False)

    # Extract final output text
    final_output = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_output = msg.content
            break

    # ── Check 1: write_todos invoked (planning first) ────────────
    check_planning = write_todos_invoked

    # ── Check 2: task() delegation tool called ≥ 1 time ─────────
    check_delegation_called = del_calls["task"] >= 1

    # ── Check 3: Expected sub-agent was used ────────────────────
    expected_agent = test_case.get("expected_agent", "")
    agents_used = [entry["agent_name"] for entry in delegation_log]
    check_correct_agent = expected_agent in agents_used if expected_agent else len(agents_used) > 0

    # ── Check 4: Result saved to VFS (write_file ≥ 1) ───────────
    check_result_saved = fs_calls["write_file"] >= 1

    # ── Check 5: Final output non-empty ─────────────────────────
    check_final_output = len(final_output.strip()) >= 100

    # ── Check 6: Keyword coverage ≥ 50% ─────────────────────────
    output_lower = final_output.lower()
    matched_kw = sum(1 for kw in test_case["expected_keywords"] if kw.lower() in output_lower)
    keyword_coverage = round(matched_kw / len(test_case["expected_keywords"]), 2)
    check_keywords = keyword_coverage >= 0.5

    passed = (
        check_planning
        and check_delegation_called
        and check_correct_agent
        and check_result_saved
        and check_final_output
        and check_keywords
    )

    return {
        "tc_id": test_case["id"],
        "passed": passed,
        "checks": {
            "write_todos_invoked":   check_planning,
            "delegation_called":     check_delegation_called,
            "correct_agent_used":    check_correct_agent,
            "result_saved_to_vfs":   check_result_saved,
            "final_output_present":  check_final_output,
            "keyword_coverage_50pct": check_keywords,
        },
        "stats": {
            "todo_count":          len(todos),
            "todos_completed":     sum(1 for t in todos if t["status"] == "completed"),
            "task_calls":          del_calls["task"],
            "list_agents_calls":   del_calls["list_agents"],
            "agents_used":         agents_used,
            "expected_agent":      expected_agent,
            "files_in_vfs":        len(vfs),
            "vfs_filenames":       list(vfs.keys()),
            "write_file_calls":    fs_calls["write_file"],
            "read_file_calls":     fs_calls["read_file"],
            "keyword_coverage":    keyword_coverage,
            "output_length":       len(final_output),
        },
    }


def print_result(eval_result: dict, request: str):
    status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
    checks = eval_result["checks"]
    stats = eval_result["stats"]

    print(f"\n  {status} | {eval_result['tc_id']}")
    print(f"  Request  : {request[:70]}...")
    print(f"  Checks   :")
    print(f"    write_todos invoked     : {'✅' if checks['write_todos_invoked']    else '❌'}")
    print(f"    delegation tool called  : {'✅' if checks['delegation_called']      else '❌'}  ({stats['task_calls']} call(s))")
    print(f"    correct agent used      : {'✅' if checks['correct_agent_used']     else '❌'}  (expected: {stats['expected_agent']}, got: {stats['agents_used']})")
    print(f"    result saved to VFS     : {'✅' if checks['result_saved_to_vfs']    else '❌'}  ({stats['files_in_vfs']} file(s): {stats['vfs_filenames']})")
    print(f"    final output present    : {'✅' if checks['final_output_present']   else '❌'}  ({stats['output_length']} chars)")
    print(f"    keyword coverage ≥50%   : {'✅' if checks['keyword_coverage_50pct'] else '❌'}  ({stats['keyword_coverage']*100:.0f}%)")
    print(f"  TODOs    : {stats['todos_completed']}/{stats['todo_count']} completed")


# ─────────────────────────────────────────────
# Main Evaluation Runner
# ─────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 65)
    print("  MILESTONE 3 EVALUATION — Sub-Agent Delegation")
    print(f"  LangSmith Tracing: {'ENABLED ✅' if LANGCHAIN_TRACING else 'DISABLED ℹ️'}")
    print("=" * 65)

    results = []
    all_data = []

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] Running {tc['id']}...")
        try:
            state = run_agent(tc["request"], run_name=f"m3-eval-{tc['id']}")
            eval_result = evaluate_test_case(state, tc)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            import traceback
            traceback.print_exc()
            eval_result = {
                "tc_id": tc["id"],
                "passed": False,
                "checks": {k: False for k in [
                    "write_todos_invoked", "delegation_called", "correct_agent_used",
                    "result_saved_to_vfs", "final_output_present", "keyword_coverage_50pct"
                ]},
                "stats": {
                    "todo_count": 0, "todos_completed": 0,
                    "task_calls": 0, "list_agents_calls": 0,
                    "agents_used": [], "expected_agent": tc.get("expected_agent", ""),
                    "files_in_vfs": 0, "vfs_filenames": [],
                    "write_file_calls": 0, "read_file_calls": 0,
                    "keyword_coverage": 0.0, "output_length": 0,
                },
                "error": str(e),
            }

        print_result(eval_result, tc["request"])
        results.append(eval_result)
        all_data.append({"test_case": tc, "evaluation": eval_result})

        if i < len(TEST_CASES) - 1:
            wait_s = 45  # 45s lets the 30 RPM window reset before the next test
            print(f"\n  ⏸️  Cooling down {wait_s}s before next test", end="", flush=True)
            for _ in range(wait_s):
                time.sleep(1)
                print(".", end="", flush=True)
            print(" ready!")

    # ── Summary ────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total * 100
    milestone_passed = pass_rate >= 80.0

    check_keys = [
        "write_todos_invoked", "delegation_called", "correct_agent_used",
        "result_saved_to_vfs", "final_output_present", "keyword_coverage_50pct"
    ]
    check_counts = {k: sum(1 for r in results if r["checks"].get(k)) for k in check_keys}

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY — Milestone 3")
    print("=" * 65)
    print(f"  Total test cases              : {total}")
    print(f"  Passed (all checks)           : {passed}")
    print(f"  Failed                        : {total - passed}")
    print(f"  Pass Rate                     : {pass_rate:.1f}%")
    print(f"  Target                        : 80.0%")
    print(f"\n  Per-Check Breakdown:")
    for k, v in check_counts.items():
        print(f"    {k:<30}: {v}/{total}")
    print(f"\n  Milestone 3 Status: {'✅ PASSED' if milestone_passed else '❌ NOT YET PASSING'}")
    print("=" * 65 + "\n")

    if LANGCHAIN_TRACING:
        print("🔗 View traces at: https://smith.langchain.com")
        print(f"   Project: milestone3-deep-agent\n")

    output = {
        "milestone": 3,
        "langsmith_tracing_enabled": LANGCHAIN_TRACING,
        "pass_rate_percent": round(pass_rate, 1),
        "target_percent": 80.0,
        "milestone_passed": milestone_passed,
        "per_check_summary": {k: f"{v}/{total}" for k, v in check_counts.items()},
        "results": all_data,
    }

    with open("milestone3_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("📄 Full results saved to milestone3_eval_results.json\n")

    return output


if __name__ == "__main__":
    run_evaluation()
