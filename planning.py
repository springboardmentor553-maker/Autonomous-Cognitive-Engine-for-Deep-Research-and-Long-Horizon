"""
Milestone 1 Test Suite — OpenAI
"""
import os
import sys
from langchain_core.messages import HumanMessage
from workflow.flow import create_agent_executor, create_system_prompt

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "milestone1-refinement"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print(f"✓ OpenAI Model: {OPENAI_MODEL}")

TEST_CASES = [
    {"id": 1, "input": "Create a comprehensive business plan for a sustainable urban farming startup",          "category": "Business Planning"},
    {"id": 2, "input": "Analyze the impact of remote work on employee productivity across different industries","category": "Research & Analysis"},
    {"id": 3, "input": "Develop a machine learning model to predict customer churn for a SaaS company",        "category": "Technical Development"},
    {"id": 4, "input": "Investigate the relationship between social media usage and mental health in teenagers","category": "Scientific Research"},
    {"id": 5, "input": "Design a marketing strategy for launching a new eco-friendly product line",            "category": "Marketing Strategy"},
    {"id": 6, "input": "Compare different cloud infrastructure providers for enterprise deployment",            "category": "Technical Comparison"},
    {"id": 7, "input": "Evaluate the feasibility of implementing blockchain technology in supply chain management", "category": "Technology Evaluation"},
]

ACTION_VERBS = [
    'research','analyze','create','compile','investigate','examine','evaluate',
    'gather','identify','develop','write','review','compare','assess','design',
    'collect','synthesize','summarize','build','implement','outline','establish',
    'select','test','validate'
]


def validate_todos(todos):
    checks = {
        "has_todos":           len(todos) > 0,
        "correct_count":       4 <= len(todos) <= 6,
        "all_have_ids":        all("id" in t for t in todos),
        "all_have_descriptions": all("description" in t and len(t["description"]) > 10 for t in todos),
        "all_have_status":     all("status" in t for t in todos),
        "unique_descriptions": len(set(t["description"] for t in todos)) == len(todos),
    }
    return checks, sum(checks.values()) / len(checks)


def check_verbs(todos):
    count = 0
    for todo in todos:
        first = todo.get("description","").lower().strip().split()
        if first and any(first[0].startswith(v) for v in ACTION_VERBS):
            count += 1
    return count


def run_tests():
    print("=" * 90)
    print("MILESTONE 1 — COMPREHENSIVE TEST SUITE")
    print("=" * 90)
    print(f"Model: {OPENAI_MODEL} | Tests: {len(TEST_CASES)}")
    print("=" * 90)

    agent         = create_agent_executor()
    system_prompt = create_system_prompt()
    results       = []

    for test in TEST_CASES:
        print(f"\n[{test['id']}/7] {test['category']}")
        print(f"Input: {test['input'][:75]}...")
        print("-" * 90)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=f"{system_prompt}\n\nTask: {test['input']}")]},
                {"configurable": {"thread_id": f"test-{test['id']}"}, "recursion_limit": 10}
            )
            todos    = result.get("todos", [])
            messages = result.get("messages", [])

            tool_called = any(
                tc["name"] == "write_todos"
                for msg in messages if hasattr(msg, "tool_calls") and msg.tool_calls
                for tc in msg.tool_calls
            )
            checks, score = validate_todos(todos)
            verb_count    = check_verbs(todos)
            verb_pct      = (verb_count / len(todos) * 100) if todos else 0

            results.append({"test_id": test['id'], "category": test['category'],
                             "tool_called": tool_called, "todo_count": len(todos),
                             "quality_score": score, "verb_pct": verb_pct})

            print(f"  Tool Called:   {'YES ✓' if tool_called else 'NO ✗'}")
            print(f"  TODO Count:    {len(todos)} {'✓' if 4 <= len(todos) <= 6 else '✗'}")
            print(f"  Quality Score: {score:.0%}")
            print(f"  Action Verbs:  {verb_count}/{len(todos)} ({verb_pct:.0f}%)")
            for i, todo in enumerate(todos, 1):
                print(f"    {i}. {todo['description'][:85]}")

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            results.append({"test_id": test['id'], "category": test['category'],
                             "error": str(e)[:100], "tool_called": False,
                             "quality_score": 0.0, "todo_count": 0, "verb_pct": 0})

    successful = [r for r in results if not r.get("error")]
    failed     = [r for r in results if r.get("error")]

    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)
    print(f"  Passed: {len(successful)}/7  |  Failed: {len(failed)}/7")

    if successful:
        tool_rate   = sum(1 for r in successful if r["tool_called"]) / len(successful) * 100
        avg_quality = sum(r["quality_score"] for r in successful)   / len(successful) * 100
        avg_todos   = sum(r["todo_count"]     for r in successful)   / len(successful)
        avg_verbs   = sum(r["verb_pct"]       for r in successful)   / len(successful)

        print(f"\n  Tool Call Rate:    {tool_rate:.0f}%  {'✓' if tool_rate == 100 else '✗'}")
        print(f"  Quality Score:     {avg_quality:.0f}%  {'✓' if avg_quality >= 90 else '✗'}")
        print(f"  Avg TODO Count:    {avg_todos:.1f}   {'✓' if 4 <= avg_todos <= 6 else '✗'}")
        print(f"  Action Verb Usage: {avg_verbs:.0f}%  {'✓' if avg_verbs >= 80 else '✗'}")

    if failed:
        print("\n  Failed:")
        for r in failed:
            print(f"    Test {r['test_id']} ({r['category']}): {r.get('error','')[:60]}")
    print("=" * 90)
    return results


if __name__ == "__main__":
    run_tests()
