"""
test_runner.py - Milestone 1 Evaluation Script
10-task evaluation set measuring Task Decomposition Accuracy.

Per-test checks:
  1. write_todos was invoked (verified via state flag + ToolMessage scan)
  2. Exactly 5 TODO items were created
  3. Each task starts with a required action verb
  4. Keyword coverage ≥ 50% confirms plan relevance

Success Criteria: ≥ 80% of test cases pass all checks.
"""

import json
import time
from main import run_agent, LANGCHAIN_TRACING

# ─────────────────────────────────────────────
# 10 Varied Complex Test Cases
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC01",
        "request": "Research the current state of quantum computing and write a detailed report covering hardware advances, software ecosystems, and practical applications.",
        "expected_keywords": ["research", "hardware", "software", "report", "application"],
    },
    {
        "id": "TC02",
        "request": "Analyze the pros and cons of microservices vs monolithic architecture for a startup and provide a recommendation with justification.",
        "expected_keywords": ["microservice", "monolithic", "analys", "recommend", "architect"],
    },
    {
        "id": "TC03",
        "request": "Create a learning plan for someone who wants to become a machine learning engineer in 6 months starting from a basic Python background.",
        "expected_keywords": ["learn", "plan", "python", "machine learning", "skill"],
    },
    {
        "id": "TC04",
        "request": "Write a comprehensive competitive analysis of the top 5 cloud providers (AWS, Azure, GCP, Oracle, IBM) focusing on pricing, features, and market share.",
        "expected_keywords": ["aws", "azure", "gcp", "pric", "analys", "cloud"],
    },
    {
        "id": "TC05",
        "request": "Develop a strategy for migrating a legacy monolithic Java application to a cloud-native microservices architecture on Kubernetes.",
        "expected_keywords": ["migrat", "java", "kubernetes", "microservice", "strateg"],
    },
    {
        "id": "TC06",
        "request": "Research the environmental impact of cryptocurrency mining and propose sustainable alternatives with supporting data.",
        "expected_keywords": ["environment", "crypto", "mining", "sustain", "impact"],
    },
    {
        "id": "TC07",
        "request": "Create a detailed project plan for building and launching a SaaS product from idea to MVP in 3 months with a team of 4.",
        "expected_keywords": ["plan", "saas", "mvp", "launch", "team"],
    },
    {
        "id": "TC08",
        "request": "Investigate the current state of AI regulation globally and produce a policy brief comparing approaches in the EU, US, and China.",
        "expected_keywords": ["regulat", "ai", "policy", "eu", "china"],
    },
    {
        "id": "TC09",
        "request": "Produce a technical deep-dive report on transformer architecture improvements since the original attention paper, covering efficiency, scalability, and new variants.",
        "expected_keywords": ["transformer", "attention", "architecture", "efficienc", "report"],
    },
    {
        "id": "TC10",
        "request": "Design a cybersecurity incident response plan for a mid-sized e-commerce company, including threat detection, containment, recovery, and post-incident review.",
        "expected_keywords": ["cybersecurity", "incident", "response", "detect", "recover"],
    },
]

# Action verbs the system prompt enforces
REQUIRED_VERBS = ["RESEARCH", "ANALYZE", "SYNTHESIZE", "DRAFT", "REVIEW"]


# ─────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────

def evaluate_test_case(state: dict, test_case: dict) -> dict:
    """Run all checks on the final agent state for one test case."""
    todos = state.get("todos", [])
    write_todos_invoked = state.get("write_todos_invoked", False)

    # Check 1: write_todos was invoked
    check_invoked = write_todos_invoked

    # Check 2: Exactly 5 TODOs created
    check_five_todos = len(todos) == 5

    # Check 3: Each task starts with a required action verb
    verb_results = []
    for todo in todos:
        task_upper = todo["task"].strip().upper()
        starts_with_verb = any(task_upper.startswith(verb) for verb in REQUIRED_VERBS)
        verb_results.append(starts_with_verb)
    check_action_verbs = all(verb_results) and len(verb_results) == 5

    # Check 4: Keyword coverage ≥ 50%
    all_task_text = " ".join(t["task"].lower() for t in todos)
    matched_keywords = sum(
        1 for kw in test_case["expected_keywords"]
        if kw.lower() in all_task_text
    )
    keyword_coverage = round(matched_keywords / len(test_case["expected_keywords"]), 2)
    check_keywords = keyword_coverage >= 0.5

    # Overall pass: ALL checks must pass
    passed = check_invoked and check_five_todos and check_action_verbs and check_keywords

    return {
        "tc_id": test_case["id"],
        "passed": passed,
        "todo_count": len(todos),
        "check_invoked": check_invoked,
        "check_five_todos": check_five_todos,
        "check_action_verbs": check_action_verbs,
        "verb_details": verb_results,
        "keyword_coverage": keyword_coverage,
        "check_keywords": check_keywords,
        "todos": todos,
    }


def print_result(eval_result: dict, request: str):
    """Pretty-print a single test case result."""
    status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
    print(f"\n  {status} | {eval_result['tc_id']}")
    print(f"  Request : {request[:75]}...")
    print(f"  Checks  :")
    print(f"    write_todos invoked : {'✅' if eval_result['check_invoked']    else '❌'}")
    print(f"    Exactly 5 todos     : {'✅' if eval_result['check_five_todos'] else '❌'} (got {eval_result['todo_count']})")
    print(f"    Action verbs used   : {'✅' if eval_result['check_action_verbs'] else '❌'} ({sum(eval_result['verb_details'])}/5 tasks)")
    print(f"    Keyword coverage    : {'✅' if eval_result['check_keywords']   else '❌'} ({eval_result['keyword_coverage']*100:.0f}%)")
    if eval_result["todos"]:
        print(f"  Tasks generated:")
        for i, t in enumerate(eval_result["todos"], 1):
            verb_ok = "✅" if i <= len(eval_result["verb_details"]) and eval_result["verb_details"][i-1] else "❌"
            print(f"    {verb_ok} {i}. {t['task'][:70]}")


# ─────────────────────────────────────────────
# Main Evaluation Runner
# ─────────────────────────────────────────────

def run_evaluation():
    print("\n" + "="*60)
    print("  MILESTONE 1 EVALUATION — Task Decomposition Accuracy")
    print(f"  LangSmith Tracing: {'ENABLED ✅' if LANGCHAIN_TRACING else 'DISABLED ℹ️'}")
    print("="*60)

    results = []
    all_data = []

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] Running {tc['id']}...")
        try:
            state = run_agent(tc["request"], run_name=f"eval-{tc['id']}")
            eval_result = evaluate_test_case(state, tc)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            eval_result = {
                "tc_id": tc["id"],
                "passed": False,
                "todo_count": 0,
                "check_invoked": False,
                "check_five_todos": False,
                "check_action_verbs": False,
                "verb_details": [],
                "keyword_coverage": 0.0,
                "check_keywords": False,
                "todos": [],
                "error": str(e),
            }

        print_result(eval_result, tc["request"])
        results.append(eval_result)
        all_data.append({"test_case": tc, "evaluation": eval_result})

        # Brief pause between calls to avoid rate limiting
        if i < len(TEST_CASES) - 1:
            time.sleep(1.5)

    # ─── Summary ───
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total * 100
    milestone_passed = pass_rate >= 80.0

    # Per-check breakdown
    inv_count  = sum(1 for r in results if r["check_invoked"])
    five_count = sum(1 for r in results if r["check_five_todos"])
    verb_count = sum(1 for r in results if r["check_action_verbs"])
    kw_count   = sum(1 for r in results if r["check_keywords"])

    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)
    print(f"  Total test cases          : {total}")
    print(f"  Passed (all checks)       : {passed}")
    print(f"  Failed                    : {total - passed}")
    print(f"  Pass Rate                 : {pass_rate:.1f}%")
    print(f"  Target                    : 80.0%")
    print(f"\n  Per-Check Breakdown:")
    print(f"    write_todos invoked     : {inv_count}/{total}")
    print(f"    Exactly 5 todos         : {five_count}/{total}")
    print(f"    Action verbs correct    : {verb_count}/{total}")
    print(f"    Keyword coverage ≥50%   : {kw_count}/{total}")
    print(f"\n  Milestone 1 Status: {'✅ PASSED' if milestone_passed else '❌ NOT YET PASSING'}")
    print("="*60 + "\n")

    if LANGCHAIN_TRACING:
        print("🔗 View full traces at: https://smith.langchain.com")
        print(f"   Project: milestone1-deep-agent\n")

    # Save full results
    output = {
        "milestone": 1,
        "langsmith_tracing_enabled": LANGCHAIN_TRACING,
        "pass_rate_percent": round(pass_rate, 1),
        "target_percent": 80.0,
        "milestone_passed": milestone_passed,
        "per_check_summary": {
            "write_todos_invoked": f"{inv_count}/{total}",
            "exactly_5_todos": f"{five_count}/{total}",
            "action_verbs_correct": f"{verb_count}/{total}",
            "keyword_coverage_50pct": f"{kw_count}/{total}",
        },
        "results": all_data,
    }

    with open("generated_todos.json", "w") as f:
        json.dump(output, f, indent=2)
    print("📄 Full evaluation results saved to generated_todos.json\n")

    return output


if __name__ == "__main__":
    run_evaluation()